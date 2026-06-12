# Nemotron Diffusion / Self-Speculation Performance Notes

## Context

We implemented three inference paths for `Nemotron-Labs-Diffusion-3B-4bit` in MLX-LM:

1. AR generation
2. Diffusion generation
3. Self-speculation generation, optionally with the bundled `linear_spec_lora` adapter

Initial benchmark showed:

```text
AR greedy:              ~32.95 tok/s
Pure diffusion:         ~11.13 tok/s
Self-spec adapter OFF:  ~20.20 tok/s
Self-spec adapter ON:   ~21.66 tok/s
Forced accept-all:      ~36.08 tok/s
```

The forced accept-all test shows self-speculation *can* beat AR slightly if acceptance is near-perfect, but the speedup is small. This document explains why.

---

## 1. AR with KV cache is very cheap

Autoregressive generation normally sounds slow because it generates one token at a time.

But with KV cache, each next-token step is cheap.

### What AR does

For each new token:

```text
q_len = 1
```

The model only computes attention for one new query token.

The previous prompt/generated tokens are already stored in KV cache:

```text
cache = keys/values for all previous tokens
```

So the forward pass does not recompute the whole sequence.

### Example

If the context is:

```text
A B C D E F
```

and we generate token `G`, AR does:

```text
new input: G
q_len: 1
attend against cached A B C D E F + G
```

So each step is small.

### Why this matters

Self-speculation has to beat a highly optimized baseline:

```text
cached AR, one-token forward, no draft overhead
```

That is harder than beating naive full-sequence AR.

---

## 2. Diffusion draft forwards use larger `q_len`

In self-speculation with block size 8, the diffusion draft processes an 8-token block at once.

```text
q_len = 8
```

That is bigger than AR's:

```text
q_len = 1
```

### What q_len means

`q_len` is the number of query positions being processed in the current attention call.

- AR next-token decode: `q_len = 1`
- block draft of 8 tokens: `q_len = 8`
- block draft of 32 tokens: `q_len = 32`

Larger `q_len` means more hidden states, projections, attention work, and output logits in one forward.

### Why this matters

Even if self-speculation uses fewer forward passes, each forward is heavier.

Example for 32 generated tokens with block length 8:

```text
AR:
  32 forwards, each q_len=1

Self-spec ideal:
  4 draft forwards, each q_len=8
  4 verify forwards, each q_len=8
```

That is fewer forwards, but each one processes 8 positions.

---

## 3. Diffusion draft attention is bidirectional

In AR mode, attention is causal.

For tokens:

```text
A B C D
```

token `C` can attend only to:

```text
A B C
```

In diffusion mode, attention is bidirectional inside the draft block.

Token `C` can attend to:

```text
A B C D
```

### Code path

Diffusion draft calls:

```python
model(block, cache=cache, causal=False, update_cache=False)
```

`causal=False` means we do not apply a causal mask.

### Why this can be heavier

Bidirectional block attention requires computing interactions across all block positions.

For a block of 8, each token can see all 8 block positions.

For a block of 32, each token can see all 32 block positions.

This is appropriate for diffusion, but it is not the same cheap q_len=1 cached decode as AR.

---

## 4. Verification also processes a whole block

Self-speculation does not only draft. It must verify.

After diffusion drafts a block, AR verifier runs:

```python
model(block, cache=cache, causal=True, update_cache=True)
```

For block length 8:

```text
q_len = 8
```

For block length 32:

```text
q_len = 32
```

### Why verification is needed

The verifier checks which draft tokens match what AR would have produced.

Without verification, output quality can degrade.

### Why this costs time

Each speculative block needs both:

```text
1 diffusion draft forward
1 AR verification forward
```

So even with perfect acceptance, block length 8 needs roughly:

```text
4 draft forwards + 4 verify forwards = 8 block forwards
```

for 32 generated tokens.

---

## 5. Low acceptance wastes draft work

Self-speculation only helps when many drafted tokens are accepted.

### Example

If block length is 8:

```text
draft 8 tokens
accept 2 tokens
reject 6 tokens
```

Then most of the draft work was wasted.

### Our measured acceptance

With adapter ON:

```text
accepted_tokens: 31
drafted_tokens: 80
acceptance_rate: ~0.39
```

This means many draft tokens did not survive AR verification.

### Important nuance

The raw acceptance rate can look harsh because once the first mismatch happens, later draft tokens are ignored even if some would coincidentally match.

But from a speed perspective, ignored tokens still cost compute.

---

## 6. Python loop control overhead

Our current self-spec implementation is Python-level orchestration.

It repeatedly does:

```python
while total_generated < max_new_tokens:
    create block
    draft
    verify
    compare Python lists
    crop cache
```

### Why this matters

Each transition between Python and MLX work has overhead.

The model forward is fast, but the loop around it is not fused or compiled.

For small blocks, Python overhead becomes more visible.

### Example

For block length 4 or 8, the model work per iteration is smaller, so Python overhead becomes a larger fraction of runtime.

---

## 7. Repeated `mx.eval` calls

`mx.eval(...)` forces MLX to execute pending lazy computations.

MLX is lazy by default: operations are queued and executed later.

### Why we call `mx.eval`

We need values for Python control flow:

```python
if mask is empty: break
accepted = compare token ids
if eos found: stop
```

To use MLX arrays in Python conditionals, we must materialize them.

### Why this slows things down

Frequent `mx.eval` calls reduce MLX's ability to batch/fuse work.

Instead of letting MLX schedule larger chunks, we force synchronization points.

Synchronization is expensive because Python waits for the device to finish.

---

## 8. Cache concatenation during diffusion draft

During diffusion denoising, we need:

```text
[prefix cache] + [current draft block]
```

But we must not mutate the cache because the draft block is temporary.

So our implementation does:

```python
cached_keys, cached_values = cache.state
keys = mx.concatenate([cached_keys, keys], axis=2)
values = mx.concatenate([cached_values, values], axis=2)
```

### Why this matters

Concatenation creates additional work and memory movement.

AR decode does not need this same temporary concatenation pattern. It just updates/fetches cache normally.

### Potential optimization

Avoid materializing concatenated K/V if MLX attention/cache APIs can support separate prefix K/V plus current K/V directly.

Currently we take the simple/correct route.

---

## 9. LoRA matmuls add cost during draft

When adapter is ON, each adapted `o_proj` computes:

```text
base_o_proj(x) + scale * ((x @ A) @ B)
```

The base projection already runs.

LoRA adds two extra matrix multiplications per adapted layer:

```text
x @ A
then result @ B
```

### Adapter facts

The bundled adapter targets every layer's:

```text
self_attn.o_proj
```

There are 26 transformer layers.

So during draft, LoRA adds work in 26 places.

### Why this is still useful

LoRA may increase acceptance rate.

If acceptance improves enough, fewer speculative iterations are needed.

But if acceptance improves only slightly, the extra LoRA cost may not pay for itself.

In our small benchmark, adapter ON was slightly better than OFF:

```text
OFF: ~20.20 tok/s
ON:  ~21.66 tok/s
```

So it helped, but not dramatically.

---

## 10. No fused/optimized self-spec loop

The current implementation prioritizes correctness and readability.

It is not a fused kernel-level implementation.

### Current structure

```text
Python loop
  MLX draft forward
  Python acceptance logic
  MLX verify forward
  Python cache crop
```

### Optimized structure would reduce

- Python round trips
- synchronization points
- repeated small array conversions
- repeated shape/control overhead

### Why this matters

Speculative decoding speedups often depend on careful runtime engineering, not just the algorithm.

A naive implementation can be correct but still slower than optimized AR.

---

## 11. No streaming/kernel-level optimization

Our diffusion/self-spec code is a model method, not integrated into MLX-LM's generation engine.

It does not use a specialized streaming path.

It does not have custom kernels for:

- draft/verify fusion
- acceptance checking
- cache cropping
- block-level decode scheduling

### Why this matters

The generic AR path in MLX-LM is already optimized and battle-tested.

Our self-spec path is new and Python-heavy.

---

## 12. Block length tradeoff

Block length controls how many tokens are drafted per speculative iteration.

### Smaller block

Example:

```text
block_length = 4 or 8
```

Pros:

- less wasted work when acceptance is low
- easier debugging
- lower per-forward cost

Cons:

- less amortization
- more Python loop iterations
- smaller theoretical speedup

### Larger block

Example:

```text
block_length = 16 or 32
```

Pros:

- fewer speculative iterations if acceptance is high
- better theoretical speedup

Cons:

- more wasted work when first mismatch occurs early
- heavier draft/verify forwards
- acceptance may drop

### Key point

Large blocks only help if acceptance is high.

---

## 13. Why forced accept-all was only slightly faster

Forced accept-all with block length 8 gave:

```text
~36.08 tok/s
```

AR gave:

```text
~32.95 tok/s
```

So even ideal acceptance only slightly beat AR.

Why?

Because block length 8 still requires:

```text
1 prefill
4 diffusion draft forwards
4 AR verify forwards
```

for 32 generated tokens.

Each forward has `q_len=8`, not `q_len=1`.

So the reduced forward count is partly offset by heavier forwards.

---

## 14. What would need to improve for speedups

Likely optimization directions:

### Improve acceptance

- use LoRA adapter during draft
- tune block length
- tune draft denoising behavior
- match HF reference more closely

### Reduce draft cost

- fewer denoising steps
- better one-shot draft quality
- avoid unnecessary logits computation if possible

### Reduce Python overhead

- fewer `mx.eval` calls
- vectorize acceptance logic
- avoid Python list conversions

### Improve cache handling

- avoid explicit K/V concatenation for non-mutating prefix attention
- optimize cache crop

### Tune block length

Run sweeps:

```text
block_length = 4, 8, 16, 32
```

Compare:

- tokens/sec
- acceptance rate
- output quality
- NFE

---

## Current interpretation

The implementation is functionally correct enough to demonstrate all three modes:

```text
AR
Diffusion
Self-speculation with LoRA draft adapter
```

But it is not yet optimized.

The current bottleneck is a combination of:

```text
expensive block forwards
low acceptance
Python/MLX synchronization overhead
LoRA draft overhead
cache concatenation overhead
```

So the main conclusion is:

```text
Correctness path: good progress.
Performance path: needs tuning and optimization.
```
