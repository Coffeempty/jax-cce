# jax-cce

Memory-efficient fused linear cross-entropy loss for JAX. Instead of the standard two-step approach that first materializes the full `[N, V]` logit matrix and then computes the loss, `jax-cce` fuses the linear projection and cross-entropy into a single pass that processes the vocabulary in small chunks via `lax.scan`. For large-vocabulary models (128K+ tokens), this reduces peak activation memory from O(N·V) to O(N·D + V·D), eliminating the multi-GB logit buffer that typically dominates memory usage during LLM training. The implementation is fully compatible with `jax.jit`, `jax.grad`, and `jax.value_and_grad`, supports both `float32` and `bfloat16`, and includes optional features for causal sequence shifting, vocabulary reordering for better cache locality, and gradient filtering to skip negligible chunks in the backward pass.

## Install

```bash
pip install jax-cce
```

## Quick usage

```python
import jax
import jax.numpy as jnp
from jax_cce import linear_cross_entropy

# Typical LLM shapes: N tokens, D hidden dim, V vocab size
N, D, V = 4096, 4096, 128_000

key = jax.random.key(0)
x      = jax.random.normal(key, (N, D), dtype=jnp.bfloat16)   # hidden states
w      = jax.random.normal(key, (V, D), dtype=jnp.bfloat16)   # lm_head weights
labels = jax.random.randint(key, (N,), 0, V)                   # target token ids

# Naive approach — allocates ~4 GB for bfloat16 logits at N=4096, V=128K
# logits = x @ w.T          # [N, V]  <-- this is what we avoid
# loss = cross_entropy(logits, labels)

# jax-cce — never allocates the [N, V] tensor
loss = linear_cross_entropy(x, w, labels, chunk_size=4096)

# Gradients work the same way
loss, (dx, dw) = jax.value_and_grad(
    lambda x_, w_: linear_cross_entropy(x_, w_, labels),
    argnums=(0, 1),
)(x, w)

# Next-token prediction (shift=1): x[t] predicts labels[t+1]
# Works on batched sequences [B, T, D] / [B, T] too
loss = linear_cross_entropy(x, w, labels, shift=1)
```

**Memory savings** at bfloat16 with V=128K vocab:

| N (tokens) | Naive logits | jax-cce (chunk=4096) |
|------------|-------------|----------------------|
| 1,024      | 256 MB      | ~32 MB               |
| 4,096      | 1 GB        | ~32 MB               |
| 16,384     | 4 GB        | ~32 MB               |

## API

```python
jax_cce.linear_cross_entropy(
    x,
    w,
    labels,
    *,
    shift=0,
    chunk_size=4096,
    vocab_sort_indices=None,
    filter_eps=None,
)
```

**Parameters**

- **x** `jax.Array` — Input activations, shape `[N, D]`. For sequence models with `shift > 0`, accepts `[..., T, D]`.
- **w** `jax.Array` — Vocabulary weight matrix, shape `[V, D]`.
- **labels** `jax.Array` — Integer class labels in `[0, V)`, shape `[N]` or `[..., T]`.
- **shift** `int` (default `0`) — When > 0, applies a causal sequence shift: `x[..., :-shift, :]` predicts `labels[..., shift:]`. Set `shift=1` for standard next-token prediction.
- **chunk_size** `int` (default `4096`) — Vocabulary entries per scan step. Larger values use more memory but fewer kernel launches. Must be a positive integer.
- **vocab_sort_indices** `jax.Array | None` (default `None`) — Integer array of shape `[V]`. When provided, reorders weight rows as `w[vocab_sort_indices]` before chunking, grouping related tokens for better cache locality. Labels are remapped internally so the loss is identical. Gradients are correctly unsorded back to original row order via JAX's gather VJP.
- **filter_eps** `float | None` (default `None`) — Gradient filtering threshold. Chunks whose per-sample max logit is more than `filter_eps` below the global max are zeroed out in the backward pass (their softmax contribution is negligible). The forward loss is always exact. Use `float('inf')` to enable the filter code path without actually filtering (useful for benchmarking).

**Returns** — Scalar mean cross-entropy loss in `float32`.

## Reference

This package implements the algorithm from:

> **"Cut Your Losses in Large-Vocabulary Language Models"**
> arxiv.org/abs/2411.09009

The implementation was contributed to JAX as part of [JAX issue #35906](https://github.com/google/jax/issues/35906) and reviewed by a JAX maintainer. See the upstream PR for additional context on the design and memory analysis.

The original reference implementation in Triton is available at [github.com/apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy).
