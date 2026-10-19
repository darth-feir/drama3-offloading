# Why the lm_head slice pads with finfo.min and not -inf

**Where:** `bin/offload.py`, `slice_lm_head`

`--slice-lm-head` keeps only the 4,097 output rows decode can actually
sample, of 252,288, and scatters them back into a full-width buffer so
`decode_step` needs no changes. The buffer is filled with
`torch.finfo(dtype).min` — the most negative **finite** value — rather than
`-inf`.

## What breaks if you change it back

Every chunk dies with `non-finite logits; refusing to sample invalid audio`,
but **only under `--cuda-graph 1`**. Without graphs the identical
configuration is fine, which is what made this look like a CUDA-graph bug
for a day.

`Drama3DecodeGraphs.run` (`generate.py:376-377`) does:

```python
logits = backbone.compute_logits(self.hidden)
self.finite.copy_(torch.isfinite(logits).all())
```

over the **full 252,288-wide** tensor. Fill 248,191 of those positions with
`-inf` and the check is False on the first step and every step after. **The
slice was never producing bad numbers. It was tripping a guard with its own
sentinel.**

Eager decode does not care because the guard that runs there lives in
`sample_top_k_top_p` and only ever sees `logits[:, candidates]`
(`tower.py:349-353`) — the 4,097 real values, all finite.

## Why finfo.min is safe

Nothing reads the padded positions except that check; the only other
consumer indexes by `candidates`. `finfo.min` softmaxes to exactly 0 the
same way `-inf` does, so the sampled distribution is unchanged — while a
genuine Inf or NaN out of the GEMM still trips the guard, which is what it
is for.

## The other half, which is a separate hazard

The wide buffer is **pre-allocated**, not created per call. The first
version built a fresh `torch.full(-inf)` inside `compute_logits`, which sits
inside the captured region; `fill_` and index-assign are capturable,
allocation is not safely so.

⚠️ That was fixed first, on its own, and **did not help** — because it was
not the cause. Two bugs, one symptom, and fixing the visible one left the
behaviour unchanged. If you are debugging something here and a correct fix
appears to do nothing, consider that there may be two.

`compute_logits` is only ever called with one row (`generate.py:252` does
`hidden = hidden[-1:]`, and the graph path uses a `(1, hidden)` buffer), so
one buffer suffices. Anything wider falls back to allocating and is not
capture-safe — fine, because it cannot occur in the decode path.

## Worth

0.95 GiB, lossless. On this box that bought 62 → 70 experts per layer of
cache and took RTF 0.919 → 0.872 — mostly a **decode** win, because
`compute_logits` stops reading a [252288, 2048] matrix out of VRAM every
frame. See `notes/10-prefill-gather.md`.
