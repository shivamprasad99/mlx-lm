# Nemotron Labs Diffusion: Self-Speculation + LoRA Adapter Plan

## Goal

Implement the third Nemotron inference mode in MLX-LM: **self-speculation**.

Self-speculation uses the same base model in two roles:

```text
Diffusion draft mode  -> proposes a block of tokens
AR verifier mode      -> checks/accepts the longest causal prefix
```

The checkpoint also includes a dedicated LoRA adapter:

```text
linear_spec_lora/
  adapter_config.json
  adapter_model.safetensors
```

This adapter should be used only during the **diffusion draft** phase, not during AR prefill or AR verification.

## Current implementation status

Already available:

- AR mode
- diffusion forward with `causal=False`
- non-mutating cached-prefix attention with `update_cache=False`
- multi-block diffusion generation via `model.diffusion_generate(...)`

Still missing:

- self-speculation loop
- AR verification/acceptance logic
- cache cropping after rejected draft tail
- loading HF/PEFT-style `linear_spec_lora`
- adapter on/off toggling per phase

## Local adapter facts

Adapter config:

```json
{
  "peft_type": "LORA",
  "target_modules": ["o_proj"],
  "r": 128,
  "lora_alpha": 512,
  "lora_dropout": 0.0,
  "bias": "none"
}
```

So the LoRA scale is:

```text
scale = lora_alpha / r = 512 / 128 = 4
```

Adapter tensor examples:

```text
base_model.model.encoder.layers.0.self_attn.o_proj.lora_A.weight  (128, 4096)
base_model.model.encoder.layers.0.self_attn.o_proj.lora_B.weight  (3072, 128)
...
base_model.model.encoder.layers.25.self_attn.o_proj.lora_A.weight (128, 4096)
base_model.model.encoder.layers.25.self_attn.o_proj.lora_B.weight (3072, 128)
```

The base `o_proj` maps:

```text
4096 -> 3072
```

MLX-LM `LoRALinear` stores:

```python
lora_a: [input_dims, r]    # [4096, 128]
lora_b: [r, output_dims]   # [128, 3072]
```

PEFT stores:

```text
lora_A.weight: [r, input_dims]      # [128, 4096]
lora_B.weight: [output_dims, r]     # [3072, 128]
```

Therefore loading requires transposes:

```python
mlx_lora_a = peft_lora_A.T
mlx_lora_b = peft_lora_B.T
```

## Why MLX-LM's existing adapter loader is not enough

MLX-LM's tuner adapter loader expects roughly:

```text
adapter_config.json with MLX tuning keys
adapters.safetensors
```

The local adapter is HF/PEFT style:

```text
adapter_model.safetensors
adapter_config.json with PEFT keys
```

So we should implement a small model-specific PEFT adapter loader instead of trying to force this through the generic tuner loader.

## Desired runtime behavior

Self-speculation phases:

```text
1. AR prefill       : adapter OFF, causal=True, update_cache=True
2. diffusion draft  : adapter ON,  causal=False, update_cache=False
3. AR verify        : adapter OFF, causal=True, update_cache=True
4. crop cache       : remove rejected draft tail
5. repeat
```

Why adapter OFF for AR phases?

The adapter is intended to improve the diffusion draft distribution. If left on during AR verification, the verifier would no longer match the base AR model semantics.

## Implementation chunks

### Step 1: Add adapter-capable LoRA wrapper

We need a way to enable/disable LoRA at runtime.

Current `mlx_lm.tuner.lora.LoRALinear` always applies LoRA:

```python
y = self.linear(x)
z = (x @ self.lora_a) @ self.lora_b
return y + scale * z
```

For self-speculation, we need either:

```python
module.adapters_enabled = True / False
```

or a model-level flag that all adapter modules read.

Recommended minimal approach:

- define a local `SwitchableLoRALinear` in `nemotron_labs_diffusion.py`, or
- extend/reuse `LoRALinear` carefully if acceptable globally

For minimal model-specific implementation, define:

```python
class SwitchableLoRALinear(nn.Module):
    linear: original o_proj
    lora_a
    lora_b
    scale
    enabled: bool = True

    def __call__(self, x):
        y = self.linear(x)
        if not self.enabled:
            return y
        return y + scale * ((x @ lora_a) @ lora_b).astype(x.dtype)
```

This avoids affecting other MLX-LM models.

### Step 2: Add PEFT adapter loader

Add a method to `Model`:

```python
def load_linear_spec_lora(self, adapter_path: str):
    ...
```

It should:

1. read `adapter_config.json`
2. validate:
   - `peft_type == "LORA"`
   - `target_modules == ["o_proj"]`
   - `lora_dropout == 0.0`
3. wrap each layer's `self_attn.o_proj` with `SwitchableLoRALinear`
4. load weights from `adapter_model.safetensors`
5. transpose PEFT A/B matrices into MLX layout
6. set all adapters disabled by default

Expected key conversion:

```text
PEFT key:
base_model.model.encoder.layers.{i}.self_attn.o_proj.lora_A.weight

MLX module:
encoder.layers.{i}.self_attn.o_proj.lora_a = A.T
```

and:

```text
PEFT key:
base_model.model.encoder.layers.{i}.self_attn.o_proj.lora_B.weight

MLX module:
encoder.layers.{i}.self_attn.o_proj.lora_b = B.T
```

### Step 3: Add adapter toggle helper

Add:

```python
def set_linear_spec_lora_enabled(self, enabled: bool):
    ...
```

It should iterate through modules and set:

```python
module.enabled = enabled
```

for `SwitchableLoRALinear` modules.

Default after loading:

```python
adapter OFF
```

### Step 4: Add cache cropping helper

Self-speculation verifies a full draft block, but may accept only part of it.

After verification, cache contains too many tokens:

```text
cache = prompt + full draft block
```

If accepted length is shorter, crop cache to:

```text
cache = prompt + accepted prefix
```

Need method:

```python
def _crop_cache(cache, max_length: int):
    ...
```

For `KVCache`, cropping can be simple:

```python
cache.offset = max_length
```

because `KVCache.state` and future `update_and_fetch` respect offset.

For `RotatingKVCache`, be more careful. Current checkpoint has no sliding window, so `KVCache` is enough initially.

### Step 5: Implement self-speculation generation without LoRA first

Before introducing the adapter, implement base self-speculation:

```python
def self_spec_generate(
    prompt_ids,
    max_new_tokens=128,
    block_length=32,
    temperature=0.0,
    threshold=0.0,
    eos_token_id=None,
    use_adapter=False,
):
    ...
```

Initial loop:

1. adapter OFF
2. causal prefill prompt
3. get AR seed token
4. while not done:
   - create draft block: `[seed] [MASK] ...`
   - adapter ON if requested
   - diffusion-denoise the block
   - adapter OFF
   - AR-verify the block with `causal=True, update_cache=True`
   - compare draft shifted by one against AR predictions
   - accept longest matching prefix plus one AR bonus token
   - crop cache to accepted length
   - continue from the AR bonus token

### Step 6: Add LoRA to draft phase only

Once base self-speculation works, enable adapter usage:

```python
self.set_linear_spec_lora_enabled(True)
# diffusion draft
self.set_linear_spec_lora_enabled(False)
# AR verify
```

This should match the HF reference behavior:

```python
_toggle_adapters(True)   # draft
_toggle_adapters(False)  # prefill / verify
```

### Step 7: Test adapter load only

Test before generation:

```python
model.load_linear_spec_lora(".../linear_spec_lora")
```

Checks:

- all 26 `o_proj` modules are wrapped
- LoRA shapes match
- adapters disabled by default
- AR generation output is unchanged when disabled
- enabling adapter changes logits for a diffusion forward

### Step 8: Test self-speculation without adapter

Small test:

```python
out, nfe = model.self_spec_generate(
    prompt_ids,
    max_new_tokens=16,
    block_length=8,
    use_adapter=False,
)
```

Check:

- no crash
- no mask tokens remain
- cache cropping works
- generated length <= prompt + max_new_tokens
- acceptance statistics are printed/returned

### Step 9: Test self-speculation with adapter

After adapter loading:

```python
model.load_linear_spec_lora(".../linear_spec_lora")
out, stats = model.self_spec_generate(..., use_adapter=True)
```

Compare:

- acceptance rate with adapter OFF
- acceptance rate with adapter ON
- output quality
- speed

## Acceptance statistics to return

Self-speculation should return more than just `nfe`.

Recommended:

```python
return output_ids, {
    "nfe": nfe,
    "draft_forwards": draft_forwards,
    "verify_forwards": verify_forwards,
    "accepted_tokens": accepted_tokens,
    "drafted_tokens": drafted_tokens,
    "acceptance_rate": accepted_tokens / drafted_tokens,
}
```

This is important because self-speculation only helps if acceptance rate is high.

## Risks / tricky parts

### 1. Token alignment during verification

The HF reference compares:

```python
ar_tokens[0, i] == block[0, i + 1]
```

because the AR prediction at position `i` predicts the next token.

This off-by-one alignment is easy to get wrong.

### 2. Bonus AR token

The reference does:

```python
accepted += 1
```

This means even at a mismatch, one AR token is accepted as the next seed/progress token.

### 3. Cache cropping

Verification updates cache for the entire draft block. Rejected tokens must be removed by cropping cache offset.

### 4. Adapter key format

The local adapter is PEFT-style, not MLX-LM tuner-style.

### 5. Adapter target shape

Only `o_proj` is adapted. Shape conversion must transpose both A and B matrices.

### 6. Quality depends on adapter toggle correctness

If adapter remains ON during AR verification, the verifier changes and acceptance semantics are wrong.

## Recommended next implementation step

Do **not** implement the full self-speculation loop immediately.

First implement only:

```python
SwitchableLoRALinear
Model.load_linear_spec_lora(...)
Model.set_linear_spec_lora_enabled(...)
```

Then test adapter loading and toggling in isolation.

Only after that should we implement the self-speculation loop.
