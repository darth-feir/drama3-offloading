# Two traps in moving embed_tokens to host RAM

**Where:** `bin/offload.py`, `offload_embed`

The input embedding table is [252288, 2048] and decode only ever embeds a
token it just sampled, which is in the 4,097-id band. So: full table in host
RAM, the 4,097 rows decode can reach resident in VRAM (16 MiB), prefill
gathers what it needs. Worth **0.94 GiB**.

## Trap 1: the branch must be capture state, not row count

The tempting test is "one row means decode". It is wrong. A prefill whose
length is `1 mod --prefill-chunk` ends with a **single-row batch carrying an
arbitrary text id**, which would then read row −1 of the resident slice.

`torch.cuda.is_current_stream_capturing()` is exact. Inside capture the ids
are in-band by construction — the only thing decode embeds is
`sampled[:, 0]`, drawn from `main_candidates` (`tower.py:349-357`), including
during the warm-up passes. Replay does not call this at all.

### What breaks if you change it back

Not a crash. The model gets a **valid-looking embedding for the wrong word**
and narrates fluent nonsense. `bin/test_embed_offload.py` exercises the
one-row prefill case explicitly (id 248045).

## Trap 2: the host table must not be a Parameter

`Drama3DecodeGraphs.for_model` (`generate.py:350-353`) does:

```python
devices = {p.device for p in model.backbone.parameters()}
if len(devices) != 1 or ...: return None
```

so **one CPU parameter anywhere in the backbone silently turns CUDA graphs
off** — and graphs are worth RTF 1.71 → 0.97, roughly twenty times what this
change saves. The experts get away with living on the host because they are
registered as **buffers**; an embedding weight is a `Parameter`.

The host table is therefore held in a closure and the module is replaced
with a real 4,097-row `nn.Embedding` on the GPU. `offload_embed` raises if
any parameter ends up off-device rather than letting it pass quietly.

### What breaks if you change it back

The run still works and produces correct audio, roughly twice as slowly,
with no error and no log line saying graphs were declined.

## Worth, measured

0.94 GiB → cache 70 → 78 experts/layer → prefill gather 29.9 % → 33.4 %, and
**RTF unchanged**. It earns its place via the VRAM, not the clock: with
`--slice-lm-head` the two together free 1.89 GiB, and 1.94 GiB of VRAM trades
against 2.5 % of RTF. See `notes/10-prefill-gather.md`.
