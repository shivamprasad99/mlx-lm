# Copyright © 2023-2024 Apple Inc.
#
# MLX implementation of Nemotron Labs Diffusion.
#
# The checkpoint uses `encoder.*` and `diffusion_head.*` module names, so this
# file keeps that structure while supporting AR, diffusion, and self-speculative
# generation paths.

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .pipeline import PipelineMixin
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    rope_parameters: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    layer_types: Optional[List[str]] = None
    sliding_window: Optional[int] = None

    # Nemotron diffusion-specific config fields.
    mask_token_id: int = 100
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    dlm_paradigm: str = "bidirectional"
    block_size: int = 32
    dlm_loss_weight: Optional[float] = None
    ar_loss_weight: float = 1.0
    dp_varying_mask_ratio: bool = False
    use_cache: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers


def _get_llama_4_attn_scale(size, offset, beta: float, max_position_embeddings: int):
    if isinstance(offset, mx.array) and offset.ndim > 0:
        offset = offset[:, None]

    scaling = 1 + beta * mx.log(
        1 + mx.floor((mx.arange(size) + offset) / max_position_embeddings)
    )
    if scaling.ndim == 2:
        return scaling[:, None, :, None]
    else:
        return scaling[:, None]


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_parameters["rope_theta"],
            False,
            args.rope_parameters,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        attn_scale: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        update_cache: bool = True,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        offset = 0
        if cache is not None:
            offset = cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            if update_cache:
                keys, values = cache.update_and_fetch(keys, values)
            elif not cache.empty():
                cached_keys, cached_values = cache.state
                keys = mx.concatenate([cached_keys, keys], axis=2)
                values = mx.concatenate([cached_values, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries * attn_scale
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SwitchableLoRALinear(nn.Module):
    """LoRA wrapper that can be enabled only for diffusion draft phases.

    The wrapped layer computes:

        base(x) + scale * ((x @ lora_a) @ lora_b)

    when enabled, and just ``base(x)`` when disabled. This lets
    self-speculation use the LoRA adapter for diffusion drafting while keeping
    AR verification on the base model.
    """

    @staticmethod
    def from_base(
        linear: nn.Module,
        r: int,
        scale: float,
        dropout: float = 0.0,
        enabled: bool = False,
    ):
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims = input_dims * 32 // linear.bits

        lora_linear = SwitchableLoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            scale=scale,
            dropout=dropout,
            enabled=enabled,
        )
        lora_linear.linear = linear
        return lora_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int,
        scale: float,
        dropout: float = 0.0,
        enabled: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale
        self.enabled = enabled

        init_scale = input_dims**-0.5
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x: mx.array) -> mx.array:
        y = self.linear(x)
        if not self.enabled:
            return y
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool = False):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.use_sliding = use_sliding
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        attn_scale: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        update_cache: bool = True,
    ) -> mx.array:
        r = self.self_attn(
            self.input_layernorm(x), attn_scale, mask, cache, update_cache
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class LanguageModel(PipelineMixin, nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args, use_sliding=layer_type == "sliding_attention")
            for layer_type in self.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = self.layer_types.index("full_attention")
        self.swa_idx = None
        for e, layer in enumerate(self.layers):
            if layer.use_sliding:
                self.swa_idx = e
                break

    def pipeline(self, group):
        super().pipeline(group)
        self.fa_idx = None
        self.swa_idx = None
        for e, layer in enumerate(self.pipeline_layers):
            if self.swa_idx is None and layer.use_sliding:
                self.swa_idx = e
            elif self.fa_idx is None and not layer.use_sliding:
                self.fa_idx = e
            if self.fa_idx is not None and self.swa_idx is not None:
                break

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        causal: bool = True,
        update_cache: bool = True,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        pipeline_rank = self.pipeline_rank
        pipeline_size = self.pipeline_size

        if cache is None:
            cache = [None] * len(self.pipeline_layers)
            offset = 0
        else:
            offset = cache[0].offset

        swa_mask = fa_mask = None
        if causal and self.fa_idx is not None:
            fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if causal and self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        attn_scale = _get_llama_4_attn_scale(
            inputs.shape[1],
            offset,
            self.args.rope_parameters["llama_4_scaling_beta"],
            self.args.rope_parameters["original_max_position_embeddings"],
        ).astype(h.dtype)

        # Receive from the previous process in the pipeline.
        if pipeline_rank < pipeline_size - 1:
            h = mx.distributed.recv_like(h, (pipeline_rank + 1))

        for layer, layer_cache in zip(self.pipeline_layers, cache):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(
                h, attn_scale, mask, cache=layer_cache, update_cache=update_cache
            )

        # Send to the next process in the pipeline.
        if pipeline_rank != 0:
            h = mx.distributed.send(h, (pipeline_rank - 1) % pipeline_size)
            if cache[-1] is not None:
                cache[-1].keys = mx.depends(cache[-1].keys, h)

        # Broadcast h while keeping it in the graph.
        if pipeline_size > 1:
            h = mx.distributed.all_gather(h)[: h.shape[0]]

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.encoder = LanguageModel(args)
        self.diffusion_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.linear_spec_lora_loaded = False
        self.linear_spec_lora_enabled = False
        self.linear_spec_lora_layers = 0

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        causal: bool = True,
        update_cache: bool = True,
    ):
        # Default to causal attention so the generic MLX-LM generation stack
        # uses this model as a normal AR causal LM. Diffusion paths call this
        # with causal=False.
        out = self.encoder(
            inputs,
            cache=cache,
            input_embeddings=input_embeddings,
            causal=causal,
            update_cache=update_cache,
        )
        return self.diffusion_head(out)

    def diffusion_generate_one_block(
        self,
        prompt_ids: mx.array,
        block_length: int = 32,
        steps: Optional[int] = None,
        temperature: float = 0.0,
        threshold: Optional[float] = None,
        seed_first_token: bool = True,
    ):
        """Generate a single diffusion block after the prompt.

        Returns ``(output_ids, nfe)`` where ``output_ids`` is the prompt plus
        the denoised block and ``nfe`` is the number of denoising forwards.
        """
        if prompt_ids.ndim != 2:
            raise ValueError("prompt_ids must have shape [batch, seq_len]")
        if block_length <= 0:
            raise ValueError("block_length must be positive")

        steps = steps or block_length
        mask_token_id = self.args.mask_token_id
        batch_size = prompt_ids.shape[0]

        cache = self.make_cache()
        prompt_logits = self(
            prompt_ids,
            cache=cache,
            causal=True,
            update_cache=True,
        )
        mx.eval(prompt_logits)

        seed_token = None
        if seed_first_token:
            seed_token = mx.argmax(prompt_logits[:, -1, :], axis=-1)[:, None]
        block = _make_diffusion_block(
            batch_size,
            block_length,
            mask_token_id,
            prompt_ids.dtype,
            seed_token,
        )

        initial_mask = block == mask_token_id
        num_transfer_tokens = _get_num_transfer_tokens(initial_mask, steps)

        nfe = 0
        for step_idx in range(steps):
            mask_index = block == mask_token_id
            mx.eval(mask_index)
            if not bool(mx.any(mask_index).item()):
                break

            logits = self(
                block,
                cache=cache,
                causal=False,
                update_cache=False,
            )
            nfe += 1

            x0, transfer_index = _get_transfer_index(
                logits,
                temperature,
                mask_index,
                block,
                num_transfer_tokens[:, step_idx],
                threshold=threshold,
            )
            block = mx.where(transfer_index, x0.astype(block.dtype), block)
            mx.eval(block)

        return mx.concatenate([prompt_ids, block], axis=1), nfe

    def diffusion_generate(
        self,
        prompt_ids: mx.array,
        max_new_tokens: int,
        block_length: Optional[int] = None,
        steps_per_block: Optional[int] = None,
        temperature: float = 0.0,
        threshold: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        seed_first_token: bool = True,
    ):
        """Generate tokens with block diffusion decoding.

        Returns ``(output_ids, nfe)`` where ``output_ids`` includes the prompt.
        """
        if prompt_ids.ndim != 2:
            raise ValueError("prompt_ids must have shape [batch, seq_len]")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        block_length = block_length or self.args.block_size
        steps_per_block = steps_per_block or block_length
        if block_length <= 0:
            raise ValueError("block_length must be positive")
        if max_new_tokens % block_length != 0:
            raise ValueError("max_new_tokens must be divisible by block_length")
        if eos_token_id is None:
            eos_token_id = self.args.eos_token_id

        mask_token_id = self.args.mask_token_id
        batch_size = prompt_ids.shape[0]
        num_blocks = max_new_tokens // block_length

        cache = self.make_cache()
        output = self(
            prompt_ids,
            cache=cache,
            causal=True,
            update_cache=True,
        )
        mx.eval(output)

        next_token = None
        if seed_first_token:
            next_token = mx.argmax(output[:, -1, :], axis=-1)[:, None]
            mx.eval(next_token)

        generated_blocks = []
        nfe = 0

        for _ in range(num_blocks):
            block = _make_diffusion_block(
                batch_size,
                block_length,
                mask_token_id,
                prompt_ids.dtype,
                next_token if seed_first_token else None,
            )

            initial_mask = block == mask_token_id
            num_transfer_tokens = _get_num_transfer_tokens(
                initial_mask, steps_per_block
            )

            for step_idx in range(steps_per_block):
                mask_index = block == mask_token_id
                mx.eval(mask_index)
                if not bool(mx.any(mask_index).item()):
                    break

                logits = self(
                    block,
                    cache=cache,
                    causal=False,
                    update_cache=False,
                )
                nfe += 1

                x0, transfer_index = _get_transfer_index(
                    logits,
                    temperature,
                    mask_index,
                    block,
                    num_transfer_tokens[:, step_idx],
                    threshold=threshold,
                )
                block = mx.where(transfer_index, x0.astype(block.dtype), block)
                mx.eval(block)

            generated_blocks.append(block)

            # Causal post-block forward: commit the finalized block into the KV
            # cache and produce the optional seed token for the next block.
            output = self(
                block,
                cache=cache,
                causal=True,
                update_cache=True,
            )
            nfe += 1
            mx.eval(output)

            if seed_first_token:
                next_token = mx.argmax(output[:, -1, :], axis=-1)[:, None]
                mx.eval(next_token)

            if eos_token_id is not None:
                generated = mx.concatenate(generated_blocks, axis=1)
                mx.eval(generated)
                generated_list = generated.tolist()
                eos_positions = []
                for row in generated_list:
                    try:
                        eos_positions.append(row.index(eos_token_id))
                    except ValueError:
                        eos_positions.append(None)
                if all(pos is not None for pos in eos_positions):
                    stop_len = max(pos for pos in eos_positions) + 1
                    generated = generated[:, :stop_len]
                    return mx.concatenate([prompt_ids, generated], axis=1), nfe

        generated = mx.concatenate(generated_blocks, axis=1)
        return mx.concatenate([prompt_ids, generated], axis=1), nfe

    def self_spec_generate(
        self,
        prompt_ids: mx.array,
        max_new_tokens: int = 128,
        block_length: Optional[int] = None,
        temperature: float = 0.0,
        threshold: float = 0.0,
        eos_token_id: Optional[int] = None,
        use_adapter: bool = False,
        draft_steps: int = 1,
        profile: bool = False,
    ):
        """Generate tokens with Nemotron's diffusion/AR self-speculation.

        Diffusion drafts a block, AR verifies it, then the longest matching
        prefix plus one AR bonus token is accepted. When ``use_adapter=True``,
        the linear-speculation LoRA is enabled only for diffusion drafting.
        """
        if prompt_ids.ndim != 2:
            raise ValueError("prompt_ids must have shape [batch, seq_len]")
        if prompt_ids.shape[0] != 1:
            raise ValueError("self_spec_generate currently requires batch size 1")
        if temperature != 0.0:
            raise NotImplementedError(
                "self_spec_generate currently supports greedy decoding only"
            )
        if threshold != 0.0:
            raise NotImplementedError(
                "thresholded self-spec draft is not implemented yet"
            )
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if draft_steps <= 0:
            raise ValueError("draft_steps must be positive")

        total_start = time.perf_counter() if profile else None

        if eos_token_id is None:
            eos_token_id = self.args.eos_token_id
        block_length = block_length or self.args.block_size
        mask_token_id = self.args.mask_token_id

        if use_adapter and not self.linear_spec_lora_loaded:
            raise RuntimeError(
                "use_adapter=True requires load_linear_spec_lora(...) first."
            )
        if self.linear_spec_lora_loaded:
            self.set_linear_spec_lora_enabled(False)

        cache = self.make_cache()
        prefill_start = time.perf_counter() if profile else None
        output = self(prompt_ids, cache=cache, causal=True, update_cache=True)
        mx.eval(output)
        prefill_time = time.perf_counter() - prefill_start if profile else 0.0

        next_token = mx.argmax(output[:, -1, :], axis=-1)[:, None]
        mx.eval(next_token)

        generated = []
        stats = {
            "nfe": 1,
            "draft_forwards": 0,
            "verify_forwards": 0,
            "accepted_tokens": 0,
            "drafted_tokens": 0,
        }
        if profile:
            stats.update(
                {
                    "prefill_time": prefill_time,
                    "draft_time": 0.0,
                    "verify_time": 0.0,
                    "accept_time": 0.0,
                    "crop_time": 0.0,
                    "total_time": 0.0,
                    "accepted_per_iter": [],
                    "cache_len_per_iter": [],
                    "unresolved_masks_per_iter": [],
                }
            )

        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            return mx.concatenate([prompt_ids, next_token], axis=1), stats

        generated.append(next_token.astype(prompt_ids.dtype))
        total_generated = 1

        while total_generated < max_new_tokens:
            cache_len = cache[0].offset
            remaining = max_new_tokens - total_generated

            block = _make_diffusion_block(
                1,
                block_length,
                mask_token_id,
                prompt_ids.dtype,
                next_token,
            )

            # Diffusion draft phase. The optional LoRA adapter is enabled only
            # while drafting. With draft_steps>1, the block is progressively
            # filled over multiple diffusion forwards.
            initial_mask = block == mask_token_id
            num_transfer_tokens = _get_num_transfer_tokens(initial_mask, draft_steps)
            stats["drafted_tokens"] += block_length

            for draft_step_idx in range(draft_steps):
                mask_index = block == mask_token_id
                mx.eval(mask_index)
                if not bool(mx.any(mask_index).item()):
                    break

                if use_adapter:
                    self.set_linear_spec_lora_enabled(True)
                draft_start = time.perf_counter() if profile else None
                try:
                    logits = self(block, cache=cache, causal=False, update_cache=False)
                    mx.eval(logits)
                finally:
                    if use_adapter:
                        self.set_linear_spec_lora_enabled(False)
                if profile:
                    stats["draft_time"] += time.perf_counter() - draft_start

                stats["nfe"] += 1
                stats["draft_forwards"] += 1

                x0, transfer_index = _get_transfer_index(
                    logits,
                    temperature,
                    mask_index,
                    block,
                    num_transfer_tokens[:, draft_step_idx],
                    threshold=None,
                )
                block = mx.where(transfer_index, x0.astype(block.dtype), block)
                mx.eval(block)

            if profile:
                unresolved_masks = int(mx.sum(block == mask_token_id).item())
                stats["unresolved_masks_per_iter"].append(unresolved_masks)

            # AR verification phase. This updates cache for the whole draft;
            # rejected tokens are removed by cropping below. LoRA must remain
            # disabled so verification matches base AR semantics.
            verify_start = time.perf_counter() if profile else None
            verify_logits = self(block, cache=cache, causal=True, update_cache=True)
            ar_tokens = mx.argmax(verify_logits, axis=-1).astype(block.dtype)
            mx.eval(ar_tokens, block)
            if profile:
                stats["verify_time"] += time.perf_counter() - verify_start
            stats["nfe"] += 1
            stats["verify_forwards"] += 1

            accept_start = time.perf_counter() if profile else None
            block_list = block.tolist()[0]
            ar_list = ar_tokens.tolist()[0]

            accepted = 0
            for i in range(block_length - 1):
                if ar_list[i] == block_list[i + 1]:
                    accepted += 1
                else:
                    break
            # Bonus AR token: even on mismatch, use the verifier's next token
            # to guarantee progress.
            accepted += 1
            accepted = min(accepted, remaining)

            accepted_toks = ar_tokens[:, :accepted]
            generated.append(accepted_toks)
            total_generated += accepted
            stats["accepted_tokens"] += accepted
            if profile:
                stats["accepted_per_iter"].append(accepted)
                stats["cache_len_per_iter"].append(cache_len)
                stats["accept_time"] += time.perf_counter() - accept_start

            # Cache should contain only tokens up to the accepted draft prefix.
            # The final bonus token becomes the seed for the next block and is
            # intentionally cached during the next verification pass.
            crop_start = time.perf_counter() if profile else None
            _crop_cache(cache, cache_len + accepted)
            next_token = ar_tokens[:, accepted - 1 : accepted]
            mx.eval(next_token)
            if profile:
                stats["crop_time"] += time.perf_counter() - crop_start

            if eos_token_id is not None:
                accepted_list = accepted_toks.tolist()[0]
                if eos_token_id in accepted_list:
                    generated_all = mx.concatenate(generated, axis=1)
                    generated_row = generated_all.tolist()[0]
                    stop_len = generated_row.index(eos_token_id) + 1
                    generated_all = generated_all[:, :stop_len]
                    stats["acceptance_rate"] = (
                        stats["accepted_tokens"] / max(1, stats["drafted_tokens"])
                    )
                    if profile:
                        stats["total_time"] = time.perf_counter() - total_start
                    if use_adapter:
                        self.set_linear_spec_lora_enabled(False)
                    return mx.concatenate([prompt_ids, generated_all], axis=1), stats

        if use_adapter:
            self.set_linear_spec_lora_enabled(False)
        generated_all = mx.concatenate(generated, axis=1)[:, :max_new_tokens]
        stats["acceptance_rate"] = (
            stats["accepted_tokens"] / max(1, stats["drafted_tokens"])
        )
        if profile:
            stats["total_time"] = time.perf_counter() - total_start
        return mx.concatenate([prompt_ids, generated_all], axis=1), stats

    def load_linear_spec_lora(self, adapter_path: str):
        """Load the PEFT-style LoRA adapter used for linear speculation.

        The local adapter is stored in HuggingFace/PEFT format, not MLX-LM's
        tuner adapter format. It targets only ``self_attn.o_proj`` in every
        layer and is disabled by default after loading.
        """
        adapter_path = Path(adapter_path)
        with open(adapter_path / "adapter_config.json", "r") as f:
            config = json.load(f)

        if config.get("peft_type") != "LORA":
            raise ValueError("linear_spec_lora adapter must be a LoRA adapter")
        if config.get("target_modules") != ["o_proj"]:
            raise ValueError("linear_spec_lora currently only supports o_proj")
        if config.get("lora_dropout", 0.0) != 0.0:
            raise ValueError("linear_spec_lora with dropout is not supported")

        rank = int(config["r"])
        scale = float(config["lora_alpha"]) / rank
        weights = mx.load(str(adapter_path / "adapter_model.safetensors"))

        loaded_layers = 0
        for layer_idx, layer in enumerate(self.encoder.layers):
            o_proj = layer.self_attn.o_proj
            if not isinstance(o_proj, SwitchableLoRALinear):
                o_proj = SwitchableLoRALinear.from_base(
                    o_proj,
                    r=rank,
                    scale=scale,
                    dropout=0.0,
                    enabled=False,
                )
                layer.self_attn.o_proj = o_proj

            prefix = (
                f"base_model.model.encoder.layers.{layer_idx}."
                "self_attn.o_proj"
            )
            a_key = f"{prefix}.lora_A.weight"
            b_key = f"{prefix}.lora_B.weight"
            if a_key not in weights or b_key not in weights:
                raise KeyError(f"Missing LoRA weights for layer {layer_idx}")

            # PEFT stores A as [rank, input_dims] and B as
            # [output_dims, rank]. SwitchableLoRALinear expects A as
            # [input_dims, rank] and B as [rank, output_dims].
            o_proj.lora_a = weights[a_key].T
            o_proj.lora_b = weights[b_key].T
            o_proj.enabled = False
            loaded_layers += 1

        self.linear_spec_lora_loaded = True
        self.linear_spec_lora_layers = loaded_layers
        return self

    def set_linear_spec_lora_enabled(self, enabled: bool):
        """Enable or disable all loaded linear-speculation LoRA modules."""
        count = 0
        for layer in self.encoder.layers:
            o_proj = layer.self_attn.o_proj
            if isinstance(o_proj, SwitchableLoRALinear):
                o_proj.enabled = enabled
                count += 1
        if count == 0:
            raise RuntimeError(
                "No linear-speculation LoRA modules are loaded. "
                "Call load_linear_spec_lora(...) first."
            )
        self.linear_spec_lora_enabled = enabled
        return count

    def sanitize(self, weights):
        # Remove unused precomputed rotary frequencies if present.
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }

        # Keep `diffusion_head.*`; this checkpoint is untied and does not use
        # `lm_head.*` names.
        new_weights = {}
        for k, v in weights.items():
            if "weight_scale_inv" in k:
                scale_inv = v
                wk = k.replace("_scale_inv", "")
                weight = weights[wk]
                new_weights[wk] = weight * scale_inv
            elif "activation_scale" in k:
                continue
            elif k not in new_weights:
                new_weights[k] = v
        return new_weights

    def shard(self, group: Optional[mx.distributed.Group] = None):
        group = group or mx.distributed.init()
        N = group.size()
        for layer in self.encoder.layers:
            layer.self_attn.q_proj = shard_linear(
                layer.self_attn.q_proj, "all-to-sharded", group=group
            )
            layer.self_attn.k_proj = shard_linear(
                layer.self_attn.k_proj, "all-to-sharded", group=group
            )
            layer.self_attn.v_proj = shard_linear(
                layer.self_attn.v_proj, "all-to-sharded", group=group
            )
            layer.self_attn.o_proj = shard_linear(
                layer.self_attn.o_proj, "sharded-to-all", group=group
            )
            layer.self_attn.n_heads //= N
            layer.self_attn.n_kv_heads //= N

            layer.mlp.gate_proj = shard_linear(
                layer.mlp.gate_proj, "all-to-sharded", group=group
            )
            layer.mlp.down_proj = shard_linear(
                layer.mlp.down_proj, "sharded-to-all", group=group
            )
            layer.mlp.up_proj = shard_linear(
                layer.mlp.up_proj, "all-to-sharded", group=group
            )

    @property
    def layers(self):
        return self.encoder.pipeline_layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.encoder.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]


# Diffusion/self-speculation helpers.
def _make_diffusion_block(
    batch_size: int,
    block_length: int,
    mask_token_id: int,
    dtype,
    seed_token: Optional[mx.array] = None,
) -> mx.array:
    """Create a masked diffusion block, optionally seeding position 0."""
    if seed_token is not None and block_length == 1:
        return seed_token.astype(dtype)

    block = mx.full((batch_size, block_length), mask_token_id, dtype=dtype)
    if seed_token is None:
        return block
    return mx.concatenate([seed_token.astype(dtype), block[:, 1:]], axis=1)


def _crop_cache(cache, max_length: int):
    """Crop every layer cache to an absolute sequence length.

    Self-speculation verifies a full draft block, but may accept only a prefix.
    The verifier cache must then be shortened to remove rejected draft tokens.
    """
    for layer_cache in cache:
        if layer_cache is None:
            continue
        if not layer_cache.is_trimmable():
            raise ValueError(f"Cache type {type(layer_cache).__name__} is not trimmable")
        trim_count = max(0, layer_cache.offset - max_length)
        if trim_count:
            layer_cache.trim(trim_count)
    return cache


def _add_gumbel_noise(logits: mx.array, temperature: float) -> mx.array:
    """Apply Gumbel-max style noise used by the HF diffusion sampler."""
    if temperature == 0:
        return logits
    logits = logits.astype(mx.float32)
    noise = mx.random.uniform(shape=logits.shape, low=1e-9, high=1.0)
    gumbel_noise = (-mx.log(noise)) ** temperature
    return mx.exp(logits) / gumbel_noise


def _get_num_transfer_tokens(mask_index: mx.array, steps: int) -> mx.array:
    """Evenly split masked positions across denoising steps.

    Returns an int array with shape ``[batch, steps]``. Remainders are assigned
    to earlier steps, matching the HF helper.
    """
    mask_num = mx.sum(mask_index, axis=1, keepdims=True)
    base = mask_num // steps
    remainder = mask_num % steps
    step_ids = mx.arange(steps)[None, :]
    return (base + (step_ids < remainder)).astype(mx.int32)


def _get_transfer_index(
    logits: mx.array,
    temperature: float,
    mask_index: mx.array,
    x: mx.array,
    num_transfer_tokens: mx.array,
    threshold: Optional[float] = None,
):
    """Select which masked positions to commit during one denoising step.

    Returns:
        ``(x0, transfer_index)`` where ``x0`` contains candidate token ids and
        ``transfer_index`` is a boolean mask over positions to update.
    """
    logits_with_noise = _add_gumbel_noise(logits, temperature)
    x0 = mx.argmax(logits_with_noise, axis=-1)

    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    x0_probs = mx.take_along_axis(probs, x0[..., None], axis=-1).squeeze(-1)

    x0 = mx.where(mask_index, x0, x)
    confidence = mx.where(mask_index, x0_probs, -mx.inf)

    mx.eval(confidence, num_transfer_tokens)
    transfer_rows = []
    for batch_idx in range(confidence.shape[0]):
        row = confidence[batch_idx]
        if threshold is not None:
            k = int(mx.sum(mask_index[batch_idx]).item())
        else:
            k = int(num_transfer_tokens[batch_idx].item())
        k = max(0, min(k, row.shape[0]))

        row_mask = mx.zeros(row.shape, dtype=mx.bool_)
        if k > 0:
            selected = mx.argpartition(-row, kth=k - 1, axis=-1)[:k]
            selected_conf = row[selected]
            if threshold is not None:
                selected = selected[selected_conf >= threshold]
            row_mask[selected] = True
        transfer_rows.append(row_mask)

    transfer_index = mx.stack(transfer_rows, axis=0)
    return x0, transfer_index
