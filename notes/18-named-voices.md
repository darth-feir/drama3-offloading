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

118 MiB of prefilled state, plus what is needed to decide whether reusing
it is safe. Four voices is under half a gigabyte against 30 GiB of pinned
experts — the cost is nothing; the correctness is the whole problem.

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
to be invented. `GET /v1/voices` lists them in the shape
openedai-speech and Kokoro-FastAPI use. `/health` reports them too, so a
monitor needs one call rather than two.

⚠️ **Admin routes are deliberately not under `/v1`.** OpenAI has no
voice-creation call; putting one in its namespace would tell a client this
is part of the standard. `POST /admin/voices` and
`POST /admin/voices/remove` are visibly ours.

An unknown voice is a **400 in ~5 ms**, validated *before* the GPU lock is
taken — a client's typo should not queue behind a 20 s generation to be
told it was a typo.

## Removal

`--delvoice` unloads from memory **and** deletes from disk, and reassigns
the default if it removed it. Deleting only the files would leave the
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
