# Handoff — what the NixOS session established, and what it did not

Written 2026-10-16 by the NixOS-config session, which had drifted well outside
its scope (it owns `.nix` files). Everything below is either measured or
arithmetic; nothing in the Drama 3 plan has been executed.

## Verified by reading the model's own metadata

Fetched `config.json` and `model.safetensors.index.json` from
`huggingface.co/fishaudio/drama-3`. Not downloaded; not run.

- 62,245 tensors, **61,600 of them expert weights**, total **36.9 GiB**
- **256 experts per layer × 40 layers**, every layer MoE, **8 active per token**
- **FP8 block-quantized** — every weight has a `weight_scale_inv` companion
- 40 layers = **30 `linear_attention` + 10 `full_attention`**
- `hidden_size` 2048, `moe_intermediate_size` 512, vocab 252,288
- **Not gated.** Licence: Fish Audio Research (non-commercial)

## Why this is worth trying when S2-Pro was not

S2-Pro is dense (`use_moe: false`, `num_experts: 1`, zero expert tensors among
358). Offloading a dense model means reading every weight every token: 196 GiB/s
needed against ~25 GiB/s of PCIe 4.0 x16. Eight times short — arithmetic, not
opinion.

Drama 3's 32:1 sparsity touches only ~0.94 GiB per token at FP8. At 21.5
tokens/sec that is 20.2 GiB/s with a **zero percent** cache hit rate, already
borderline inside PCIe. Residency alone should give ~67 % before routing skew.

Two conveniences: it needs **no quantisation work** (FP8 with
`weight_block_size [128,128]` is the one format `fish_omni/config.py:230`
accepts — which is exactly why the S2-Pro `w4a16` GPTQ checkpoint is unusable
natively), and the 4090 is **sm_89** so FP8 runs on tensor cores without a
dequant pass.

## Corrections made along the way — do not re-derive these

- First feasibility pass assumed **bf16** and doubled the per-token traffic.
  The model is FP8; active-set traffic is 0.94 GiB/token, not 1.88.
- **Swap is the wrong tool and is unnecessary.** 36.9 GiB fits in 125 GiB of
  RAM with ~88 GiB spare. Expert fetches are small scattered reads (~3 MiB),
  swap's worst case; a swap miss costs tens of ms against ~0.1 ms for a
  RAM→VRAM copy. Use `mmap` + page cache, pinned.
- Upstream gives you **nothing** here: it states 48 GB minimum, and its
  `--device` list is multi-GPU model parallelism ("the last device holds tower
  + codec"), not CPU offload.

## Open questions, in the order worth answering

1. **What does the router actually pick?** Log expert-selection counts per
   layer over a real utterance. Near-uniform routing devalues the whole plan;
   skewed routing improves it. This is cheap and decides everything after it.
2. Does the stock loader even start with weights it cannot fit, or does it OOM
   before any hook can intervene? Determines whether this is a hook or a fork.
3. What hit rate does a plain per-layer LRU achieve at ~20 GiB of VRAM budget?
4. Only then: can expert copies overlap compute on a second CUDA stream? The
   router's choice for layer N+1 is unknown until N completes, so prefetch
   depth is inherently limited.

Prior art to read before writing anything: `ktransformers`, llama.cpp's
`--override-tensor`, vLLM's expert-offload work.

## Expectations

It will probably run. **It will not be realtime** — 35B on this card lands well
above RTF 1.0, and the realtime job is already done by dots.tts at RTF 0.28.
The deliverable is a number: tokens/sec plus the cache hit rate. If the answer
is "PCIe-bound", that is a real result and was the predicted failure mode.

## Not started

Weights (36.9 GiB + 1.87 GiB codec) are **not downloaded**. No venv for this
model yet. `bin/` and `notes/` are empty scaffolding.

Read `../CLAUDE.md` first — it carries the NixOS CUDA/Python traps, every one of
which cost time to find.
