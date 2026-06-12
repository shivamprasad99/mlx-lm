# Nemotron Labs Diffusion: Diffusion Mode Implementation Plan

## Goal

Add the second inference mode for Nemotron Labs Diffusion in MLX-LM: **diffusion/block-parallel decoding**.

AR mode is already working. Diffusion mode should reuse the same weights and same transformer backbone, but switch the attention/cache behavior during denoising.

## Important distinction from AR mode

In AR mode:

```text
causal attention + cache update every forward
```

In diffusion mode:

```text
causal prefill prompt -> build prefix KV cache
bidirectional denoise generated block -> read prefix cache but do not mutate it
causal post-block forward -> update KV cache with finalized block
```

So diffusion mode is not just `causal=False`. It also needs careful cache behavior.

## HF reference behavior

The HF implementation's `generate()` does roughly this:

1. Causal prefill over the prompt.
2. Create a block of mask tokens.
3. Optionally seed the first token from the causal next-token prediction.
4. Repeatedly denoise the current block:
   - run model on only the current block
   - let the block attend to cached prefix
   - do not update the cache
   - predict replacements for masked positions
   - commit high-confidence predictions
5. After the block is finalized, run causal forward over the block.
6. Update KV cache.
7. Repeat for the next block.

## Implementation chunks

We will implement this in small steps.

### Step 1: Add explicit forward-mode flags

Add arguments through the model stack:

```python
causal: bool = True
update_cache: bool = True
```

Thread them through:

```text
Model.__call__
  -> LanguageModel.__call__
  -> TransformerBlock.__call__
  -> Attention.__call__
```

Expected behavior after this step:

- existing AR generation remains unchanged
- `model(inputs, causal=True, update_cache=True)` works like today
- `model(inputs, causal=False, update_cache=True)` runs bidirectional attention with no causal mask

Do not implement non-mutating cache reads yet in this step.

### Step 2: Add non-mutating cached-prefix attention

Support:

```python
model(block_tokens, cache=prefix_cache, causal=False, update_cache=False)
```

Expected behavior:

- current block can attend to prefix cache
- current block can attend bidirectionally within itself
- prefix cache offset does not change

This requires attention to concatenate cached keys/values with current keys/values without calling `cache.update_and_fetch()`.

### Step 3: Add small diffusion helper functions

Port the lightweight generation helpers from HF to MLX:

- choose how many mask tokens to reveal per step
- compute token confidence
- select top-k masked positions
- replace selected mask tokens

No full generation loop yet.

### Step 4: Add a minimal one-block denoise function

Implement a function that takes:

```python
prompt_ids
block_length
steps
```

and returns one generated block.

This function should:

1. prefill prompt causally
2. create one mask block
3. denoise the block
4. return the finalized block

### Step 5: Add full multi-block diffusion generation

Extend one-block denoising to support:

- `max_new_tokens`
- multiple blocks
- EOS stopping
- temperature
- threshold
- causal context between blocks

### Step 6: Integrate with CLI later

Initially, diffusion generation can be tested from a small Python function/script.

CLI integration can come after correctness is proven.

## First code change we will make next

The next code step should be small:

```python
# Model.__call__
def __call__(self, inputs, cache=None, input_embeddings=None, causal=True, update_cache=True):
    ...
```

and pass these flags down to attention.

At the end of that step we should test:

```python
logits_ar = model(x, causal=True)
logits_bi = model(x, causal=False)
```

Both should return shape:

```text
[batch, seq_len, vocab_size]
```

AR generation should still work exactly as before.

## Non-goals for the first diffusion step

Do not yet implement:

- full diffusion generation
- self-speculation
- LoRA speculation adapter support
- block-diff flex attention
- CLI flags

The first code step is only about adding explicit mode flags safely.
