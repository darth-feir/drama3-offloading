# Named voices — 2026-10-20

One reference clip per process was the limit until now, and the reason was
never a hard one: the prefix-KV snapshot was built fresh every boot and
lived only in memory. If it survives a process boundary, a server can hold
several and pick between them per request.

It does survive. This note is what that took and the two places the first
design was wrong.

## Does the state actually persist? (`bin/test_prefix_persist.py`)

Measured before building anything, because the feature is worthless if the
answer is no:

```
built-here vs saved   (CONTROL: cross-process determinism)  107/107
loaded     vs saved   (did the round trip preserve bytes)   107/107
built-here vs loaded  (what multi-voice would rely on)      107/107
```

⚠️ **The control is the part that makes this trustworthy.** If a freshly
*built* prefix in a new process did not already match, then "the reloaded
one matches" would be measuring nothing — and this project has been caught
by exactly that shape before (`test_prefix_kv.py` control E, where
`--prefill-chunk 128` vs `1024` diverges at frame 0). Build reproducibility
had to be established first.

End to end, the thing a user hears: a voice restored from disk scores
**0.9400** timbre against its reference clip, against **0.9426** for the
same clip prefilled live in the same process. The gap sits inside the
0.907–0.953 run-to-run spread at temperature 1.0 (see `17`).

## What a voice is

Prefilled state plus what is needed to decide whether reusing it is safe.

Size scales with the reference, at about **1.7 MiB per second**:

| reference | rows | snapshot |
|---|---|---|
| 28.5 s | 734 | **94 MiB** |
| 56 s | 1416 | **125 MiB** |
| 84.5 s | 2120 | **157 MiB** |

### Where it lives: pinned host RAM

⚠️ **This note has now been wrong in both directions.** The first draft
said host RAM when the loader did `.to(device)`; the correction said VRAM;
the right answer, once someone asked *why* it had to be on the card, was
that it never did. `restore()` only ever `copy_`s into the model's caches,
and a copy from host memory is an ordinary host-to-device transfer.

Measured on the real 125 MiB snapshot:

```
VRAM-resident        0.54 ms   226.6 GiB/s
host RAM, pinned     5.07 ms    24.1 GiB/s   <- at the PCIe roof
host RAM, pageable   9.02 ms    13.5 GiB/s
```

+4.5 ms per request against a ~16 s generation is **0.03 %**, paid once
per chunk rather than per token. In exchange a voice stops costing
125 MiB of a 24 GiB card — and VRAM is what actually runs out here: the
reference-length wall is prefill headroom, and the benchmark above OOM'd
on its first attempt purely because the server held the card. Freed VRAM
becomes expert cache, which `10` measured at roughly 1.94 GiB ≈ 2.5 % RTF.

⚠️ **`carry` stays on the GPU.** `generate_drama3_prefix` does
`torch.cat((carry, hidden))` against GPU hidden states, so a host-resident
carry breaks generation. It is one row.

⚠️ Pinned rather than pageable — 5.07 vs 9.02 ms for free, the same
reason `--pin-experts` pins the expert tables. Pinned pages cannot be
swapped, which is what the budget below is for.

### The budget: 1 GiB, growing a gigabyte at a time

Nothing is ever refused. Twenty voices simply use more RAM. But crossing
a gigabyte prints a banner at boot and after any live add — total, count,
a per-voice breakdown and the `--delvoice` command — because several
gigabytes of unswappable memory should never be a surprise.

```
0.12 GiB -> budget 1 GiB   quiet
0.88 GiB -> budget 1 GiB   quiet
1.20 GiB -> budget 2 GiB   LOUD
2.50 GiB -> budget 3 GiB   LOUD   (twenty voices)
```

⚠️ **Honest about the cost:** RTF measured 0.816/0.818 host-resident
against 0.783 in VRAM for the same voice — ~4 %, not the 0.03 % the copy
benchmark predicts. n=2 per arm, and that same voice spanned 0.764–0.801
across runs earlier, so it sits inside the noise. A small real regression
cannot be ruled out; the VRAM saving is still worth it.

**Order is the contract.** A live snapshot holds module references —
`(module, k, v)` — which cannot be serialised. On reload the tensors are
re-bound to the model's modules *by position*.

⚠️ **The reference audio codes travel with the voice.** This was missed
first time round: `build_drama3_prompt()` needs them to construct a prompt
for new text, and `prefix.applies_to()` checks the prompt's head equals
the snapshot's rows. A voice stored without its codes restores perfectly
and is then **illegal to use** — the failure arrives later and elsewhere.
Storing them also makes a voice independent of the original clip still
existing on disk.

⚠️ **The fingerprint compares resolved blobs, not paths.** Found by the
persistence test rather than reasoned about: HuggingFace keeps one
directory per revision but deduplicates blobs, so this machine has two
snapshot dirs whose `config.json`, `codec.pth` and all eight shards
resolve to identical files. A path comparison rejected a valid snapshot.

A voice failing its fingerprint is **skipped with a reason, never
restored**. State built against different weights would produce fluent
audio in the wrong voice, and no downstream check — duration or content —
would catch it.

## The restart that was not necessary

The first build had `--addvoice` stop the server, build in its own
process, and restart: **~70 s**. The justification written at the time was
that building needs ~20 GiB and so does the server, and a 24 GiB card
holds one.

That justification was wrong, and the user said so: *"why would it stop
it? why not just queue it."*

The constraint was real but self-inflicted. It existed because the builder
loaded its **own copy of the model** — an artifact of writing the builder
as a standalone script first. The running server already has a model, and
building a voice is ~3 s of prefill. It does not need a second one.

```
before: stop, load, build, restart     ~70 s
after:  POST /admin/voices             3.2 s, measured
```

The fix also removed duplication rather than adding it: `voices.build()`
is now the single implementation, used by both the server and the offline
builder.

> **What breaks if you change it back:** a builder that loads its own
> model while the server runs will OOM one of the two processes, and the
> error surfaces in whichever one loses — so it reads as a bug in
> generation rather than as two processes wanting the same card.

⚠️ **It queues, it does not run concurrently.** Prefilling writes the same
KV caches a decode reads. The build takes the same GPU lock every request
takes, so it waits behind an in-flight generation instead of corrupting
it. The client sends no timeout for that reason.

## Interfaces

`voice` on `/v1/audio/speech` was *accepted and ignored*; it now selects a
voice, so the standard field finally means something and no protocol had
to be invented. `/health` reports the voices too, so a monitor needs one
call rather than two. (The listing endpoint moved after review — see
*Interfaces, corrected* below.)

⚠️ **Admin routes are deliberately not under `/v1`.** OpenAI has no
voice-creation call; putting one in its namespace would tell a client this
is part of the standard. `POST /admin/voices` and
`POST /admin/voices/remove` are visibly ours.

An unknown voice is a **400 in ~5 ms**, validated *before* the GPU lock is
taken — a client's typo should not queue behind a 20 s generation to be
told it was a typo.

## Removal

`--delvoice` unloads from memory and **moves the files to
`voices/.trash/`** rather than erasing them, reassigning the default if it
removed it.

⚠️ It used to unlink. An unauthenticated `POST /admin/voices/remove`
destroyed a real voice during a CORS audit — the auditor fired the
destructive request to observe a *response header*. 125 MiB of state
derived from someone's recording, gone, with recovery requiring a GPU
rebuild that only works while the original clip still exists. CORS
exclusion does not help here: the caller was not a browser. An accidental
delete should cost a file move, not a recording. Deleting only the files would leave the
running server answering for a voice the user believes is gone, which for
state derived from somebody's recorded voice is the wrong way round to
fail.

## Commands

```
--start  --stop  --restart  --status  --config  log [N|-f]
--voices  --addvoice NAME CLIP  --delvoice NAME  --defaultvoice NAME
```

`bin/drama3-offloading` is the same script as `drama3ctl`; flag forms
exist because people reach for `--start` before `start`.

## ⚠️ voices/ is gitignored

A snapshot is not audio, but it exists solely to reproduce a particular
person's voice. That is the same personal data in a different format, and
it is excluded for the same reason `source clips/` and `out/` are.

## How long can a reference clip be?

Three different ceilings; the lowest one binds, and it is not the one the
arithmetic suggests.

**Measured, with the server at `--vram-reserve 2`:** 56 s works, and
everything longer fails during the build —

```
 56s  →  ok
 90s  →  OOM, wanted 456 MiB, 352 MiB free
112s  →  OOM, wanted 604 MiB
224s  →  failure inside the TorchScript codec
336s  →  failure inside the TorchScript codec
392s  →  OOM, wanted 2.06 GiB
```

The alternation between codec failure and allocator OOM is not two
different walls — it is the same shortage surfacing wherever the losing
allocation happens to be.

⚠️ **The binding limit is VRAM headroom during prefill, not KV capacity.**
Caches hold 10,240 rows and a reference costs ~24.8 rows/s, which implies
~385 s — that ceiling is never reached and should not be quoted as one.

**It is a trade, not a fixed number.** The same 90 s clip builds fine once
the expert cache is smaller:

| `--vram-reserve` | expert cache | free at peak | 90 s clip |
|---|---|---|---|
| 2 (default) | 87/layer, 10.2 GiB | ~2.0 GiB | OOM |
| 5 | 59/layer, 6.9 GiB | ~5.1 GiB | **builds** |

**And the headroom is only needed to BUILD.** A 90 s voice built at
reserve 5 loads and serves at reserve 2 with ~2 GiB free, because
restoring a snapshot does not need prefill activations. Build long voices
with a big reserve once, then run normally.

## Does a longer reference cost anything per request?

Same prompt (the 52-word Dune fixture), two runs each, three voices from
the same clip:

```
v30    (734 rows)   rtf 0.783  [0.764-0.801]
v56   (1416 rows)   rtf 0.783  [0.781-0.786]
len90 (2120 rows)   rtf 0.831  [0.829-0.833]
```

**Switching voices costs nothing**, and 28.5 s vs 56 s is identical at
0.783 despite nearly double the rows — the prefix is restored, not
re-prefilled, so its length does not enter per-request cost.

**len90 is ~6 % slower and it looks real:** its two runs span 0.004 and
sit entirely above the other two arms, whose ranges overlap completely.
At 2120 rows the decode KV length crosses from the 2048 CUDA-graph bucket
into 4096.

⚠️ n=2 per arm. The *measurement* is consistent; the bucket story is a
hypothesis, and `11` records a previous bucket explanation in this project
that was published from two points and contradicted by the third.

⚠️ A `rtf 0.99` figure was quoted for the 90 s voice earlier in the day
and wondered about as a prefix effect. It was a 40-byte request, where
fixed overhead dominates. On a real prompt the same voice runs 0.83.

## Interfaces, corrected after review

`GET /v1/audio/voices`, **not** `/v1/voices`. It shipped briefly at the
latter and was **moved, not aliased** — two paths for one resource is how
an interface that claims to be standard quietly stops being one.

⚠️ Checked rather than assumed: Kokoro-FastAPI does serve
`/v1/audio/voices`. **openedai-speech, cited alongside it, has no voices
endpoint at all** — it configures voices in YAML. The convention rests on
one implementation, not two.

**`voice` accepts three aliases** — `"drama3"`, `"default"` and `""` —
all meaning the default, none of them 400. They are load-bearing:
`drama3ctl config` emits `"voice": "drama3"`, which predates named
voices, so a client configured before this feature keeps working.

> **What breaks if you change it back:** delete the aliases and the
> backend block this project hands out starts returning 400.

## Not done

- Adding a voice with the server **stopped** still loads a model (~65 s).
  Unavoidable — there is no running model to borrow — but it is now the
  fallback rather than the default path.
- The model/codec/dev-cache load sequence exists in three files
  (`narrate.py`, `server.py`, `addvoice.py`) and has drifted once already.
  `voices.build()` fixed the *voice* duplication; the loader is still
  waiting for the same treatment.
- No measurement of many voices at once. Four is arithmetic, not
  observation.
