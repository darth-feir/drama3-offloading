# Static findings — read before touching the cache

Written 2026-10-16 by the drama3 session. Everything here is **verified from
the model's own safetensors headers and the fish-speech source on disk**, not
from arithmetic over a config. Where something is still unmeasured it says so.

Reproduce with `bin/fetch_headers.py` then `bin/budget2.py`. Raw headers for
all 62,245 tensors are kept in `notes/drama3-safetensors-headers.json.gz`.

## The big one: the code is already here, and it already supports Drama 3

`/mnt/unencrypted/ai/fish-s2` is a fish-speech checkout at commit **5415669**,
whose history includes **"Update for Drama-3 (#1350)"**. The whole
`fish_speech/models/text2semantic/fish_omni/` package — `moe.py`, `fp8_moe.py`,
`backbone.py`, `generate.py` — is Drama 3 support. Nothing needs to be cloned.

It is also **self-contained**: it imports only torch, safetensors, tokenizers,
loguru, tqdm and stdlib. The fish-speech `pyproject.toml` dependency list
(pyaudio, hydra, modelscope, descript-audio-codec) is **not** needed to run the
backbone. A minimal venv is enough — this avoids the portaudio build entirely.

## Verified: the config will load

`config.py:232` rejects any `quant_method` that is not `FP8_QUANT_METHOD` with
`weight_block_size [128,128]`. The brief cited line 230; the check is at 232.

The constant is **`FP8_QUANT_METHOD = "bsgemm-moe"`** (`config.py:9`) — *not*
the string `"fp8"`. Drama 3's `config.json` declares exactly
`"quant_method": "bsgemm-moe"` and `"weight_block_size": [128,128]`, so it
matches. `model_type`, `speech_template_version` (`reference-bracket-v1`) and
`audio_encoder_config: null` all satisfy the other three checks in
`from_dict`. **The config parses.** (Unverified: that the weights then load.)

## Corrected: residency is ~41–50 %, not 67 %

The brief said "~20 GiB of experts fit in VRAM, which is ~67 % residency". That
figure forgot to subtract everything that is *not* an expert. Exact bytes:

| | GiB | share |
|---|---|---|
| routed experts (+ scales) | **30.007** | 81.4 % |
| tower / mtp (audio decoder) | 2.278 | 6.2 % |
| linear attention (30 layers) | 1.884 | 5.1 % |
| embed_tokens | 0.962 | 2.6 % |
| lm_head (not tied) | 0.962 | 2.6 % |
| full attention (10 layers) | 0.508 | 1.4 % |
| shared expert (dense) | 0.235 | 0.6 % |
| router gates | 0.039 | 0.1 % |
| **total** | **36.876** | |

(Reconciles with `metadata.total_size` = 36.88 GiB. `codec.pth` is a *separate*
1.87 GiB file, so the real on-disk total is **38.75 GiB**.)

So experts are **30.0 GiB** and the non-expert remainder that must stay
resident is **6.87 GiB**. Against 23.52 GiB usable:

```
activations/KV   codec     resident    left for experts   residency
1.5 GiB          on GPU    10.24 GiB   13.28 GiB          44.3 %
1.5 GiB          on CPU     8.37 GiB   15.15 GiB          50.5 %
2.5 GiB          on GPU    11.24 GiB   12.28 GiB          40.9 %
2.5 GiB          on CPU     9.37 GiB   14.15 GiB          47.2 %
```

**The conclusion does not flip, but the margin is thinner than advertised.**
At a 44 % hit rate the PCIe demand is 11.3 GiB/s, still comfortably inside
PCIe 4.0 x16. The activations/KV figures above are *guesses* — not measured.

## Confirmed exactly: per-token traffic

One expert is **3,146,496 B = 3.001 MiB** (gate/up `[512,2048]` + down
`[2048,512]`, all `F8_E4M3`, plus three 256 B fp32 scale blocks). So

```
8 experts x 40 layers x 3,146,496 B = 0.9377 GiB per token
```

which matches the brief's 0.94 GiB. That number is right.

The **shared expert** (6.00 MiB/layer, bf16, 0.235 GiB total) runs for *every*
token and the brief never mentions it. It is small enough to stay resident, so
it costs no PCIe traffic — but do not forget it when sizing the resident set.

## Answered: hook, not fork (open question #2)

`model.py:100` builds the whole model under `with torch.device("meta")`, then
calls `backbone.materialize(devices)`. Construction allocates **nothing**, so
the stock loader does *not* OOM before a hook can intervene — it OOMs inside
`materialize()`, which is a single method doing `layer.to_empty(device=...)`
per layer (`backbone.py:128-155`). That is a clean override point.

Loading then streams tensors one at a time and `copy_()`s each into the
pre-allocated stacked buffers (`backbone.py:317`). Note the layout change:

- **checkpoint**: one tensor per expert — `experts.N.{gate,up,down}_proj.weight`
- **runtime**: stacked per layer — `w13_weight [256, 1024, 2048]`, `w2_weight
  [256, 2048, 512]` (`moe.py:87-114`)

The per-expert checkpoint layout is convenient for selective loading; the
stacked runtime layout is not, because it is one allocation per layer. Also
`fill()` tracks `expected`/`loaded` slots and **raises if any slot is unfilled**
(`backbone.py:392`), so a partial-load scheme must still satisfy it.

## ⚠️ WRONG — CORRECTED BELOW: "an expert cache kills CUDA graphs"

`Drama3DecodeGraphs` captures the decode step as CUDA graphs, because "eager
decode is bound by kernel launches (thousands of small kernels per step)"
(`generate.py:308`). `for_model` returns `None` — eager — whenever the model
does not sit on exactly one CUDA device (`generate.py:350-353`).

Any CPU-offload placement therefore **silently disables CUDA graphs**, and we
pay the launch-bound eager cost *on top of* the PCIe cost. Dynamic per-miss H2D
copies are also hostile to graph capture in principle. ⚠️ Unmeasured: how big
that penalty is. It must be measured separately from the PCIe penalty, or the
two will be confused.

`FISH_DRAMA3_CUDA_GRAPH=0` forces eager, which is the way to measure the
graph penalty alone on a configuration that *would* otherwise fit.

## Smaller notes

- `resolve_drama3_devices()` **rejects anything but `cuda:N`** — no CPU. It is
  only a CLI helper though; `FishOmniModel.from_pretrained(devices=[...])`
  takes devices directly, so CPU placement is reachable by bypassing it.
- `FISH_DRAMA3_FP8_EXPERTS=0/1` forces the FP8 path on or off (`model.py:88`).
  With `0` the loader dequantizes experts to bf16 on load — which would double
  expert bytes to 60 GiB. Do not set it.
- `MoEExperts.forward` dispatches to the Triton W8A8 kernel only when
  `self.fp8 and x.is_cuda` (`moe.py:133`). **Experts living on CPU fall off the
  fast path** into `forward_gathered`/`forward_grouped`, which call
  `dequantize_fp8_blocks` per forward. A cache must copy experts to the GPU
  *before* the FFN runs, not hand CPU tensors to it.
- The router gate is `nn.Linear(2048, 256)` per layer, 0.039 GiB for all 40.
  Routers are cheap enough to keep resident unconditionally — which is what
  makes instrumentation and any future prefetch possible at all.
- Routing is top-8 of 256 with `norm_topk_prob` renormalisation, from a plain
  softmax over the gate (`moe.py:217-220`). No noise, no aux-loss balancing at
  inference. **Instrumentation point: `topk_ids` at `moe.py:218`.**

## NixOS trap found here (new)

The uv-managed CPython has **no CA bundle**, so `urllib`/`requests` die with
`CERTIFICATE_VERIFY_FAILED`. `curl` works because it is the Nix build. Fix:

```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

## Status of the four open questions

1. **What does the router pick?** — still open, needs the weights. This is next.
2. **Does the stock loader OOM before a hook can run?** — **answered: no.**
   meta-device construction, then `materialize()`. Hook, not fork.
3. **Hit rate of a per-layer LRU at the VRAM budget?** — open. The budget is
   now known to be ~13–15 GiB, not ~20.
4. **Can copies overlap compute?** — open, and now complicated by the CUDA
   graph interaction above.


## CORRECTION (2026-10-17): CUDA graphs are NOT blocked by offloading

The section above claims `for_model` refuses graphs because the model spans
devices once experts sit in RAM. **That is false**, and it propagated into
`02-first-results.md` and the README as "the ceiling is eager mode".

`generate.py:350` reads:

```python
devices = {p.device for p in model.backbone.parameters()}
```

and in the FP8 path `MoEExperts.__init__` registers expert tensors with
**`register_buffer`**, not `nn.Parameter` (`moe.py:102-114`; the `nn.Parameter`
branch at `moe.py:90-95` is the non-FP8 path only). `Module.parameters()` does
not yield buffers. Verified by building the backbone on the meta device:

```
fp8_experts=True :  expert tensors as PARAMETERS: 0    as BUFFERS: 160
fp8_experts=False:  expert tensors as PARAMETERS: 80   as BUFFERS: 0
```

So with FP8 experts on the CPU, `devices == {cuda:0}` and **for_model returns
graphs**. What actually disables them is our own
`os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"` in `bin/offload.py` and
`bin/narrate.py`.

### What this changes

- The "19.34 tok/s compute floor" is an **eager-dispatch floor, not a compute
  floor**. Summing the weight bytes actually read per decode frame from the
  table above gives ~6.7 GiB/frame; at the 4090's ~939 GiB/s that is ~7.2 ms,
  i.e. **~139 tok/s** as the memory-bandwidth roof. The measured 51.7 ms/frame
  is ~14 % of that, so most of a step is kernel-launch latency.
- Graphs are still not free: the current MoE path **cannot be captured**,
  because `topk_ids.reshape(-1).cpu()` is a host sync and `copy_` from
  pageable host memory is not a legal graph node. Capture would raise, not
  silently misbehave. Making it capturable means a device-resident slot table
  and `cudaHostAllocMapped` host memory with a copy *kernel* (not a memcpy
  node — a captured memcpy bakes its source pointer and would replay the same
  experts forever, producing fluent wrong audio).
- ⚠️ Pinning was rejected in `02-first-results.md` as "+10 % for 30 GiB
  locked". That trade should be **re-decided**: pinning is the precondition
  for graph capture and for genuinely async copies, not a bandwidth tweak.

⚠️ Unmeasured on this box: the actual graph-vs-eager ratio. Published work on
a near-identical configuration (35B-A3B, 8-of-256, FP8, RTX 4090, PCIe 4.0)
reports 2.36x, and 104 tok/s at 45 % expert residency. Treat as a strong
prior, not as our result.
