# Nemotron Labs Diffusion: AR Mode Implementation Plan for MLX-LM

## Scope

This document defines the first implementation milestone for running `Nemotron-Labs-Diffusion-3B-4bit` in MLX-LM: **autoregressive mode only**.

Nemotron Diffusion is tri-mode at inference time:

1. Autoregressive decoding
2. Diffusion/block-parallel decoding
3. Self-speculative decoding

This plan intentionally implements only mode 1 first. The goal is to prove that the model architecture, quantized weights, tokenizer, RoPE, output head, and KV cache work correctly before adding diffusion-specific generation.

## Target checkpoint facts

Local checkpoint:

```text
/Users/shivam/Desktop/mlx-models/Nemotron-Labs-Diffusion-3B-4bit
```

Relevant config values:

```json
{
  "model_type": "nemotron_labs_diffusion",
  "architectures": ["NemotronLabsDiffusionModel"],
  "hidden_size": 3072,
  "intermediate_size": 9216,
  "num_hidden_layers": 26,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "vocab_size": 131072,
  "rms_norm_eps": 1e-05,
  "tie_word_embeddings": false,
  "mask_token_id": 100,
  "eos_token_id": 11,
  "dlm_paradigm": "bidirectional",
  "block_size": 32,
  "rope_parameters": {
    "rope_type": "yarn",
    "rope_theta": 1000000.0,
    "original_max_position_embeddings": 16384,
    "llama_4_scaling_beta": 0.1,
    "factor": 16.0
  },
  "quantization": {
    "bits": 4,
    "group_size": 64,
    "mode": "affine"
  }
}
```

Checkpoint weight naming is already MLX-style quantized, but under Nemotron-specific module names:

```text
encoder.embed_tokens.weight
encoder.embed_tokens.scales
encoder.embed_tokens.biases
encoder.layers.0.self_attn.q_proj.weight
encoder.layers.0.self_attn.q_proj.scales
encoder.layers.0.self_attn.q_proj.biases
...
encoder.norm.weight
diffusion_head.weight
diffusion_head.scales
diffusion_head.biases
```

The AR implementation should preserve these names to avoid unnecessary weight remapping.

## Desired AR behavior

AR mode should match the Hugging Face `ar_generate()` path in `modeling_nemotron_labs_diffusion.py`:

```python
for layer in self.encoder.layers:
    if hasattr(layer.self_attn, "diffusion_lm"):
        layer.self_attn.diffusion_lm = False

enc_out = self.encoder(
    input_ids=prompt_ids,
    position_ids=position_ids,
    past_key_values=past_key_values,
    use_cache=True,
    cache_position=cache_position,
)
next_logit = self.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)
```

Translated to MLX-LM:

- use causal attention masks
- update KV cache normally
- apply RoPE using the cache offset
- apply Llama-4 query attention scaling
- project final hidden state through `diffusion_head`
- sample the next token from the last-step logits

No diffusion denoising, mask-token block filling, or self-speculation should be implemented in this milestone.

## Files to add/change

### 1. Add `mlx_lm/models/nemotron_labs_diffusion.py`

Recommended starting point: copy/adapt `mlx_lm/models/ministral3.py`.

The file should expose:

```python
@dataclass
class ModelArgs(BaseModelArgs):
    ...

class Attention(nn.Module):
    ...

class MLP(nn.Module):
    ...

class TransformerBlock(nn.Module):
    ...

class LanguageModel(PipelineMixin, nn.Module):
    ...

class Model(nn.Module):
    ...
```

The model file name must match `model_type`:

```text
nemotron_labs_diffusion.py
```

Then MLX-LM's existing loader should be able to import it via:

```python
importlib.import_module("mlx_lm.models.nemotron_labs_diffusion")
```

No `MODEL_REMAPPING` entry should be necessary if this file exists.

### 2. Optional later: add CLI integration for explicit AR mode

Initial AR support can rely on MLX-LM's normal autoregressive generation path if `Model.__call__` behaves like a standard causal LM.

Later, expose an explicit generation selector such as:

```bash
--generation-mode ar
```

But for the first milestone, the dedicated model class is enough.

## Model architecture details

### `ModelArgs`

Start from `ministral3.ModelArgs`, then add Nemotron-specific fields.

Required base fields:

```python
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
```

Nemotron fields to parse, even if unused by AR initially:

```python
mask_token_id: int = 100
dlm_paradigm: str = "bidirectional"
block_size: int = 32
dlm_loss_weight: Optional[float] = None
ar_loss_weight: float = 1.0
dp_varying_mask_ratio: bool = False
use_cache: bool = True
```

Important default:

```python
tie_word_embeddings: bool = False
```

The checkpoint has a separate `diffusion_head`; it should not tie output embeddings.

### Top-level module names

Use these exact module names:

```python
class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.encoder = LanguageModel(args)
        self.diffusion_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
```

Do **not** name the backbone `self.model`, because weights are stored as `encoder.*`.

Do **not** name the output head `self.lm_head`, because weights are stored as `diffusion_head.*`.

### `Model.__call__`

For AR mode, the standard call should be causal:

```python
def __call__(self, inputs: mx.array, cache=None, input_embeddings=None):
    out = self.encoder(
        inputs,
        cache=cache,
        input_embeddings=input_embeddings,
        causal=True,
    )
    return self.diffusion_head(out)
```

The return shape should be:

```text
[batch, sequence_length, vocab_size]
```

For a prompt of shape `[1, 10]`, logits should be `[1, 10, 131072]`.

## Attention implementation for AR mode

### Base from `ministral3.Attention`

Use the existing Ministral3 attention structure:

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- GQA with 32 Q heads and 8 KV heads
- `head_dim=128`
- YaRN RoPE via `initialize_rope`
- Llama-4 query scaling via `_get_llama_4_attn_scale`

### AR attention mask

In AR mode, attention must be causal.

Current `ministral3.LanguageModel.__call__` computes:

```python
fa_mask = create_attention_mask(h, cache[self.fa_idx])
```

This is appropriate for AR mode and should be retained.

For this checkpoint, `sliding_window` is `null`, so there should be only full causal attention.

### Cache update

In AR mode, each attention layer should mutate cache using:

```python
keys, values = cache.update_and_fetch(keys, values)
```

This is exactly what existing `ministral3.Attention.__call__` does.

No non-mutating cache path is needed in this AR milestone. That is required for diffusion mode later.

### RoPE offset

The cache offset must be used for incremental decoding:

```python
offset = cache.offset if cache is not None else 0
queries = self.rope(queries, offset=offset)
keys = self.rope(keys, offset=offset)
```

This exists in `ministral3.Attention` and should be preserved.

### Query attention scaling

Preserve MLX `ministral3._get_llama_4_attn_scale` behavior:

```python
attn_scale = _get_llama_4_attn_scale(
    inputs.shape[1],
    offset,
    self.args.rope_parameters["llama_4_scaling_beta"],
    self.args.rope_parameters["original_max_position_embeddings"],
).astype(h.dtype)
```

This must be applied to queries before attention.

## LanguageModel changes

`LanguageModel` should mostly mirror `ministral3.LanguageModel`, but with a configurable `causal` flag for future modes.

For AR milestone, default to causal:

```python
def __call__(
    self,
    inputs: mx.array,
    cache=None,
    input_embeddings: Optional[mx.array] = None,
    causal: bool = True,
):
    ...
```

Mask logic:

```python
fa_mask = None
if causal and self.fa_idx is not None:
    fa_mask = create_attention_mask(h, cache[self.fa_idx])
```

For now, always call with `causal=True`.

This small design choice avoids rewriting the model later when diffusion mode requires `causal=False`.

## Weight loading and sanitize

### Preserve quantized linear names

Because the checkpoint uses MLX-style quantized tensors, the loader should see matching module names:

```text
encoder.layers.N.self_attn.q_proj.{weight,scales,biases}
encoder.layers.N.mlp.gate_proj.{weight,scales,biases}
diffusion_head.{weight,scales,biases}
```

### `sanitize()`

Add a `sanitize()` method similar to `ministral3.Model.sanitize()`.

Minimum behavior:

```python
def sanitize(self, weights):
    weights = {
        k: v for k, v in weights.items()
        if "self_attn.rotary_emb.inv_freq" not in k
    }
    return weights
```

Do **not** drop `diffusion_head.weight`.

Do **not** try to remap `diffusion_head` to `lm_head`.

Do **not** drop output head weights based on `tie_word_embeddings`; this checkpoint is untied.

Potential extra compatibility:

If some future checkpoint stores `lm_head.*` instead of `diffusion_head.*`, add optional remapping later. It is not needed for the current local checkpoint.

## Cache construction

Add `make_cache()` to the top-level `Model`, mirroring `ministral3.Model` but using `self.encoder.layers`.

For this checkpoint, all layers are full attention, so normal `KVCache()` is sufficient.

```python
def make_cache(self):
    return [KVCache() for _ in self.layers]
```

If retaining `layer_types` and sliding support from `ministral3.py`, use the existing logic:

```python
return [
    RotatingKVCache(max_size=self.encoder.sliding_window)
    if layer.use_sliding
    else KVCache()
    for layer in self.layers
]
```

Also expose:

```python
@property
def layers(self):
    return self.encoder.pipeline_layers
```

## Pipeline/distributed considerations

If copying from `ministral3.py`, retain `PipelineMixin` support inside `LanguageModel`.

Required naming adjustments:

- `self.model` becomes `self.encoder` at top level
- inside `LanguageModel`, keep `self.layers`, `self.pipeline_layers`, `pipeline_rank`, etc.

`shard()` can be copied from `ministral3.Model.shard()` with references changed from:

```python
self.model.layers
```

to:

```python
self.encoder.layers
```

or omitted for the first non-distributed milestone if not needed.

## Testing plan

### 1. Import test

Expected:

```python
from mlx_lm.models.nemotron_labs_diffusion import Model, ModelArgs
```

No import errors.

### 2. Config parse test

Run:

```python
import json
from mlx_lm.models.nemotron_labs_diffusion import ModelArgs

cfg = json.load(open("/Users/shivam/Desktop/mlx-models/Nemotron-Labs-Diffusion-3B-4bit/config.json"))
args = ModelArgs.from_dict(cfg)
print(args)
```

Expected:

- `args.model_type == "nemotron_labs_diffusion"`
- `args.hidden_size == 3072`
- `args.num_hidden_layers == 26`
- `args.tie_word_embeddings is False`
- `args.mask_token_id == 100`

### 3. MLX-LM load test

Try:

```python
from mlx_lm import load

model, tokenizer = load("/Users/shivam/Desktop/mlx-models/Nemotron-Labs-Diffusion-3B-4bit")
```

Expected:

- model loads without unsupported model type error
- no missing critical weights
- quantized layers are created correctly

### 4. Forward shape test

```python
import mlx.core as mx

x = mx.array([[1, 2, 3, 4]])
logits = model(x)
mx.eval(logits)
print(logits.shape)
```

Expected:

```text
(1, 4, 131072)
```

### 5. Cache prefill test

```python
cache = model.make_cache()
x = mx.array([[1, 2, 3, 4]])
logits = model(x, cache=cache)
mx.eval(logits)
print(cache[0].offset)
```

Expected:

```text
4
```

### 6. Incremental decode test

```python
next_x = mx.array([[5]])
logits2 = model(next_x, cache=cache)
mx.eval(logits2)
print(logits2.shape)
print(cache[0].offset)
```

Expected:

```text
(1, 1, 131072)
5
```

### 7. Basic generation smoke test

Use the existing MLX-LM generation path:

```bash
python -m mlx_lm.generate \
  --model /Users/shivam/Desktop/mlx-models/Nemotron-Labs-Diffusion-3B-4bit \
  --prompt "Hello" \
  --max-tokens 16
```

Expected:

- no crash
- tokens are generated
- output may not be optimal because this is AR mode, not diffusion mode

### 8. Optional HF-vs-MLX first-token comparison

If PyTorch dependencies are available, compare the first next-token logits/top-k from HF `ar_generate` prefill against MLX.

Compare:

- same tokenized prompt
- same quantized or equivalent weights if possible
- top-10 next tokens

This is optional because HF and MLX quantization/runtime details may produce small numeric differences.

## Acceptance criteria for AR milestone

AR mode is considered complete when:

- `mlx_lm.models.nemotron_labs_diffusion` imports successfully
- `mlx_lm.load(local_path)` works
- the local 4-bit checkpoint loads with matching `encoder.*` and `diffusion_head.*` weights
- a forward pass returns logits with correct shape
- KV cache offsets update correctly during prompt prefill and one-token decode
- existing MLX-LM autoregressive generation can produce tokens without crashing
- code structure leaves a clear path for `causal=False` diffusion mode later

## Explicit non-goals for this milestone

Do not implement these yet:

- diffusion block denoising generation
- confidence-based unmasking
- non-mutating cache reads during denoising
- block-diff flex-attention masks
- self-speculative draft/verify generation
- LoRA adapter toggling from `linear_spec_lora/`
- training or loss computation

## Follow-up after AR mode

Once AR mode passes, the next milestone should add diffusion mode by extending the same model with:

```python
model(inputs, cache=prefix_cache, causal=False, update_cache=False)
```

That will require attention support for:

- bidirectional/no causal mask
- cached prefix concatenation without cache mutation
- block-level denoising generation

Self-speculation should come after diffusion mode, because it depends on both AR verification and diffusion drafting.
