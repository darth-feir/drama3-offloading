# Drama 3 on 24 GB — expert offloading

**Narrate long documents in a cloned voice, on a GPU that officially cannot
hold the model.**

[fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) is a
text-to-speech model: give it ~30–60 seconds of someone speaking plus a
transcript, and it reads whatever you hand it in that voice. It is 35B-A3B
mixture-of-experts, FP8, **36.9 GiB of weights**, and upstream states a
**48 GB minimum**.

This runs it on a single **24 GB RTX 4090**, faster than it plays:

> **A 31-minute document took 27 minutes of wall clock.** 235 of 235
> paragraphs generated, nothing dropped, nothing truncated. Peak 20.3 GiB
> of VRAM and 38.9 GiB of system RAM.

In the usual shorthand that is **RTF 0.85** — real-time factor, seconds of
compute per second of audio, so below 1.0 means it generates faster than you
can listen. It started at 3.90.

<sub>That 0.85 is generation only; counting the 32.5 s of startup it is
0.869 all-in. Both verified against the PCM frame counts of the 235 output
files, not just the run's own bookkeeping.</sub>

The trick is that a mixture-of-experts model only uses a fraction of itself
per token. 30 GiB of expert weights live in **system RAM** and stream over
PCIe on demand; only the 6.9 GiB that every token needs stays resident.
Upstream's `--device` flag does multi-GPU model parallelism, not this.

**You provide:** a voice clip and a text file. **You get:** a wav per
paragraph plus a stitched whole, resumable if it dies halfway.

Two ways to run it:

| | |
|---|---|
| **batch** — `bin/narrate.py` | a whole document to wav files. Resumable; a crash at paragraph 140 of 235 costs one paragraph, not the run. |
| **server** — `bin/drama3ctl` | loads once, answers HTTP. An ordinary TTS endpoint: POST text, get wav. |

## How it works

256 experts per layer across 40 layers, **8 active per token** — 32:1
sparsity, so one token touches ~0.94 GiB of expert weights rather than all
30. Keep the non-expert weights resident, keep as many hot experts in VRAM
as fit, pin the rest in host RAM, and fetch misses with a `uint4` CUDA
kernel that runs *inside* a captured CUDA graph.

At 21.5 tokens/sec that is ~20 GiB/s of PCIe traffic with a zero percent
cache hit rate — already borderline inside PCIe 4.0 x16, and every hit
makes it comfortable. That arithmetic is why this model is worth offloading
and why a dense one of the same size is not.

## What it is not

Nothing here patches fish-speech. `bin/` hooks a **clean upstream checkout**
at runtime (verified against `5415669`), so there is no fork to maintain and
no patch series to rebase. No weights are redistributed — the model licence
is Fish Audio Research, non-commercial.

## Requirements

| | |
|---|---|
| GPU | **sm_89 or newer** (Ada / Hopper). 24 GB is enough; see below |
| RAM | **~40 GiB free.** `--pin-experts` page-locks 30 GiB of it |
| disk | ~50 GiB — 39 of weights, ~7 of venv, the rest output |
| python | **3.12.** Several upstream deps cap below 3.13 |
| **ffmpeg** | on `PATH`. Every reference clip goes through it; not optional |
| **CUDA toolkit** | `nvcc` **and headers**, plus `ninja` and a **g++ ≤ 14**, to build `csrc/fetch_experts.cu`. ⚠️ On NixOS `bin/cppfetch.py` finds these by globbing `/nix/store`, so the build's success depends on what your store happens to contain. There is a fallback — see below |
| speech-to-text | **optional, but see below.** Any OpenAI-compatible `/v1/audio/transcriptions` — [`parakeet-api`](https://github.com/likeablob/parakeet-api), whisper.cpp's server, faster-whisper-server, LocalAI |

⚠️ **Without an STT endpoint you lose the only check that catches
gibberish.** Generation occasionally produces fluent nonsense at the right
duration — real phonemes, wrong language — and nothing but transcribing the
output can see it. The server starts and runs fine without one; it just
cannot tell you when a paragraph came out as
`"sendoi shrueling report cat Sego and XC when it tofund Azure Compare."`

Point `DRAMA3_STT` at it:

```bash
export DRAMA3_STT=http://127.0.0.1:8816        # any OpenAI-compatible STT
```

Measured with `parakeet-api` serving NVIDIA Parakeet TDT 0.6B v2 int8 on
sherpa-onnx, CPU only: **RTF ~0.04**, roughly 27x realtime, ~0.17 s per
check. whisper.cpp's server or faster-whisper-server work the same way —
the endpoint is what matters, not the engine.

It is three steps, not one — `pip install` alone leaves you with no model:

```bash
python3.12 -m venv sttvenv            # its own venv; `pip` alone may not exist
sttvenv/bin/pip install parakeet-api
sttvenv/bin/parakeet-api download sherpa   # ~465 MB into ~/.local/share
sttvenv/bin/parakeet-api serve             # listens on 8816 by default
```

⚠️ Configure it with `STT__*` **environment variables**. Its `--stt.*`
command-line flags are accepted without complaint and then ignored.

⚠️ **On NixOS the STT venv hits the same `libstdc++.so.6` wall**, and
`bin/env.sh` cannot help because it only fixes processes it launches.
`source bin/env.sh` in the shell you start the STT server from, or copy
its `LD_LIBRARY_PATH`.

A unix socket path is also accepted for a local daemon that speaks
"write a wav path, read a line" — see `bin/paths.py`. Prefer the URL.

**The CUDA toolkit is only needed for `--fetch-backend cpp`**, which is the
default and what every measurement here used. With no compiler, pass
`--fetch-backend triton`: same job, pure Triton, no build step.

One clean-room data point, offered as weak evidence rather than a result:
**triton RTF 0.81 against cpp's 0.81** on the same 2 chunks, audio
identical. n=1 over two chunks is not a benchmark — it is enough to say
triton is not obviously slower, and not enough to say more.

⚠️ **g++ 15 will not work.** CUDA 12.9's `crt/host_config.h` hard-errors
above gcc 14. Point `DRAMA3_CXX` at an older one if your system compiler is
newer — this is the single most likely thing to stop the build.

⚠️ **sm_89 is not a preference, it is the premise.** The design rests on
native FP8 tensor cores. On Ampere (3090, sm_86) there are none, the loader
dequantises every expert to bf16 on load, and 30 GiB becomes 60 GiB — at
which point there is nothing to offload *to*. This has not been tried on
Hopper.

⚠️ **sm_89 means Ada or newer**: RTX 40-series, Ada workstation cards,
Hopper. **Not** Ampere — a 3090's 24 GB does not help, it has no FP8. Not
Turing, not AMD, not Apple. `notes/13-non-ada-gpus.md` sets out what it
would take to lift that, and why it is more tractable than it looks: the
per-expert dequantise needed already exists upstream, wired to the wrong
switch.

**VRAM floor**, measured, resident after load before any expert cache:
10.72 GiB plain, 9.77 with `--slice-lm-head`, **8.83 with `--offload-embed`
as well**. Add ~1.2–1.5 GiB of activation headroom, so ~10.3 GiB is the
minimum to run at all and 12 GB leaves room for ~14 experts/layer of cache
against the 80 these measurements used. 16 GB is where it stops hurting.

⚠️ **Everything here was measured on one machine** — a 4090, a 7950X3D and
125 GiB of RAM, on NixOS. The numbers are real but they are n=1.

## Setup

```bash
git clone https://github.com/darth-feir/drama3-offloading.git
cd drama3-offloading
./bin/setup.sh
```

That clones `fish-speech` beside this repo at its pinned commit, builds a
Python 3.12 venv, installs the exact versions every measurement here was
taken on, offers to fetch the 38 GiB of weights, and finishes by running
`bin/doctor.py` against your machine. Re-running it is safe; every step
checks before acting.

```
./bin/setup.sh --yes          # no prompts
./bin/setup.sh --no-weights   # you already have them, or want them later
```

⚠️ **It will not download 38 GiB without asking.** Non-interactive runs
are treated as "no" and it prints the command to run later.

⚠️ **It will not overwrite your work.** If `fish-speech` is at a
different commit *with local changes*, it says so and leaves it alone
rather than checking out over your edits.

Three steps in there exist only because of traps found the hard way, and
are worth knowing even though the script now handles them:

| | |
|---|---|
| `git checkout 5415669` | upstream moves. "Verified against" is worth nothing without the checkout |
| `pip install --no-deps` | fish-speech declares `pyaudio`: needs portaudio headers, fails to build, and nothing here records audio. Expect ~15 red `ERROR: fish-speech 3.0.0 requires ...` lines from the *next* install — correct, and alarming |
| `HF_HUB_DISABLE_XET=1` | without it the download can sit at **4.5 kB/s with no error on stdout** — 18 minutes of silence in a clean-room run, failures hidden in `$HF_HOME/xet/logs/`. With it: ~35 MB/s |

⚠️ If you interrupt a download, `hf` leaves `.incomplete` blobs behind and
never reaps them — a killed attempt left 44 G on disk for a 39 G
snapshot. Delete `$HF_HOME/hub/**/blobs/*.incomplete` by hand.

### Speech-to-text (optional, but read this)

The content check that catches **fluent gibberish** — nonsense at the
right duration, which every other check is blind to by construction —
needs an OpenAI-compatible `/v1/audio/transcriptions` endpoint.

Already run one? Point at it and you are done:

```bash
export DRAMA3_STT=http://127.0.0.1:8080
```

Don't have one? The second script installs a local one:

```bash
./bin/setup-stt.sh
```

It exits immediately if something already answers on the port. It
installs `parakeet-api` serving NVIDIA Parakeet TDT 0.6B v2 int8 on
sherpa-onnx, CPU — measured RTF ~0.04, ~0.17 s per check.

**The endpoint is what matters, not the engine.** whisper.cpp's server,
faster-whisper-server and LocalAI all work; that script exists so "I
don't have one" is not a dead end.

⚠️ **CPU on purpose.** ONNX Runtime's CUDA provider measured *slower*
than CPU for this model — 520 ms against 426 ms — and CPU means
transcription survives the TTS server owning the GPU.

### Doing it by hand

`./bin/setup.sh` is twelve commands in a trenchcoat; read it if you want
them. `bin/env.sh` documents every variable the project reads and why.
Most people need none of them.

⚠️ **The checkout must be writable.** `torch-ext-cache/` is built next to
the code — set `DRAMA3_EXT_CACHE` if your clone is read-only.


### ⚠️ If you are on NixOS

Two steps above fail, and the second fails in a way nothing else in this
repository would have told you about. Both were found by building this
from scratch in a clean room; neither had ever been hit here, for the
reason in the second note.

**There is no `python3.12` on `PATH`.** `python3.12 -m venv venv` gives
`command not found`. Use any 3.12 interpreter — `nix-shell -p python312`,
or a store path directly. `bin/cleanroom.sh` reads `DRAMA3_PYTHON` for
exactly this.

**`import torch` fails with `ImportError: libstdc++.so.6`** in a venv built
from a *nix-store* python. `source bin/env.sh` now fixes this by putting
the newest `gcc-*-lib` on `LD_LIBRARY_PATH`, so source it before anything
that imports torch.

⚠️ `nix-shell -p gcc` does **not** fix it — the compiler's lib output is
not on the runtime loader path. That guess costs an hour.

Why this was invisible for the life of the project: the venv here is built
on a **uv-managed CPython**, which is a foreign binary, so NixOS's
`programs.nix-ld` supplies `libstdc++` transparently. A nix-store python
uses the nix loader, nix-ld never fires, and the documented setup breaks.
The setup had therefore never been executed as written on the only machine
this has ever run on.

⚠️ **torch is pinned to 2.8.0 on purpose.** It is what fish-speech pins and
what every measurement here was taken on. Unpinned, `pip install torch`
currently resolves to 2.14.1+cu130.

2.14.1 does work — verified, correct audio — and **steady-state speed is
the same**: RTF excluding graph capture came out 0.873 on 2.14.1 against
0.861 on 2.8.0 over the same 20 chunks, a gap inside run-to-run variation.

What differs is **capture: 54.9 s and 82.4 s on two 2.14.1 runs, against
2.1–2.5 s on 2.8.0.** A one-time cost, so ~3 % of a 31-minute document and
80 % of a two-chunk test. If a short run reports an alarming RTF, check the
`captured in …s` line before concluding anything:

```
                        RTF    capture    RTF excluding capture
2 chunks,  torch 2.14   5.333    54.9 s           0.938
20 chunks, torch 2.14   1.206    82.4 s           0.873
20 chunks, torch 2.8    0.872     2.5 s           0.861
235 chunks, torch 2.8   0.851     2.1 s           0.850
```

### Check it works before committing an hour to it

**You need one thing this repo does not ship: a reference clip.**

The text is provided — `text for testing quote from Dune.txt`, 52 words.
At 310 bytes against the 400-byte `para_max_bytes` it is a **single**
chunk, so it exercises load, build, capture and generation but says
nothing about chunk-to-chunk behaviour; use a longer file for that. It is a deliberately awkward fixture: an
apostrophe inside small caps (MUAD'DIB), and a spaced ellipsis
(`. . .`), both of which have broken text handling before. Swap in your
own file whenever you like.

The clip is **30–60 s of one person speaking cleanly**, any format ffmpeg
reads, and it is the single biggest influence on output quality.

If you have no recording of your own, this builds one from LibriSpeech.
It is the only open-ended step in the whole setup, so here it is in full
— a clean room spent a 337 MB download and some guesswork getting here:

```bash
curl -O https://www.openslr.org/resources/12/dev-clean.tar.gz   # 337 MB
tar xf dev-clean.tar.gz
cd LibriSpeech/dev-clean/1272/128104
# 8 utterances from one speaker -> ~55 s
ls *.flac | head -8 | sed 's|^|file |' > list.txt
ffmpeg -f concat -safe 0 -i list.txt -ar 44100 -ac 1 ../../../../myvoice.wav
# the matching transcript, which --ref-text wants
head -8 *.trans.txt | cut -d' ' -f2- | tr '\n' ' ' > ../../../../myvoice.txt
```

⚠️ A 337 MB corpus download to obtain one clip is heavy. A single
LibriVox MP3 is far smaller if you only want to hear it work.

⚠️ **Longer than ~60 s may not build**, and the limit is VRAM headroom
during prefill rather than anything about the clip. At the default
`--vram-reserve 2` a 90 s clip fails with a CUDA OOM; at `--vram-reserve
5` it builds fine, because the expert cache shrinks and leaves room. The
headroom is only needed to *build* — a 90 s voice built at reserve 5
loads and serves at reserve 2. See `notes/18-named-voices.md`.

**`--ref-text` matters less than this README used to claim.** With an STT
endpoint set it is transcribed for you; a corpus transcript is better
still.

⚠️ The old text said errors in it were "inherited by every chunk of the
document". That was reasoned, not measured, and when finally measured it
did not hold: a transcript reading only `placeholder`, against 56 s of
unrelated speech, scored **no worse on timbre or content** than the real
transcript — run-to-run variance at temperature 1.0 is larger than the
difference between them. It did speak **~8 % slower**, which is the one
effect that separated cleanly. See `notes/17-ref-text.md`; the test was
one short single-chunk document, so it says nothing about drift
accumulating over a long one.

```bash
bin/run.sh bin/narrate.py \
  --text-file "text for testing quote from Dune.txt" --out-dir out/smoke \
  --ref myvoice.wav --ref-text "what the clip says" --ref-seconds 56 \
  --dev-cache --fetch-backend cpp --cuda-graph 1 --pin-experts \
  --vram-reserve 2 --prefill-chunk 1024 \
  --prefix-kv --slice-lm-head --offload-embed \
  --limit 2
```

⚠️ **`--ref-seconds` is not optional in spirit.** It defaults to 30, so a
55 s clip is silently cut to the first 30 s — a clean-room run logged
`reference: snapped 30.0s -> 27.2s` and never said why. Set it to your
clip's length.

⚠️ **`--fetch-backend cpp` needs `nvcc`, and on NixOS you probably do not
have it.** `bin/cppfetch.py` discovers the toolchain by globbing
`/nix/store/*cuda-merged*/bin/nvcc`, so it succeeds on a machine that
happens to have CUDA installed and fails on one that does not — the
failure is a property of your store, not of the command. Either provide a
toolchain (`nix-shell -p cudaPackages.cuda_nvcc gcc14 ninja`) or use
`--fetch-backend triton`, which needs no compiler.

`--limit 2` does the full load, the CUDA build, graph capture and two
paragraphs in about two minutes, and writes `out/smoke/full.wav`. If that
is right, drop `--limit` and run the document.

⚠️ **`!! TRUNCATED` on a chunk means it hit `--max-new-tokens`**, not
that anything is broken — the manifest records `"finish_reason": "length"`.
Raise that flag (or split the paragraph) and rerun. A clean-room run hit it
on 1 of its first 2 chunks with a long paragraph, which reads alarmingly
against the front page's "nothing dropped, nothing truncated" — that figure
is for the 235-chunk document at its default settings, not a promise about
arbitrary input.

⚠️ **The first run is slower than the table above, even on torch 2.8.**
Measured cold in a clean room: `captured in 45.2s`, `RTF 2.27`. The same
command second time: `captured in 2.5s`, `RTF 1.03`. The 45 s is a cold
`TRITON_CACHE_DIR` on top of the `.cu` build, not the torch-2.14 capture
problem — do not read a slow first capture as being on the wrong torch.
Set `TRITON_CACHE_DIR` somewhere persistent and you pay it once.

There is also `bin/check_botnotes.py`, which needs no GPU, no model and no
dependencies — a quick way to confirm the checkout itself is intact.

## Running it

You need a **reference clip** — 30–60 s of one person speaking cleanly — and
its transcript.

```bash
bin/run.sh bin/narrate.py \
  --text-file "text for testing quote from Dune.txt" --out-dir out/run1 \
  --ref myvoice.wav --ref-seconds 56 --ref-text "what the clip says" \
  --dev-cache --fetch-backend cpp --cuda-graph 1 --pin-experts \
  --vram-reserve 2 --prefill-chunk 1024 \
  --prefix-kv --slice-lm-head --offload-embed
```

**The first chunk of a fresh checkout will take ~45 s longer than the rest.**
That is `csrc/fetch_experts.cu` being compiled into `torch-ext-cache/`; it is
cached from then on. Nothing is wrong.

That is the configuration every measurement here was taken with. It is
resumable: each chunk is written as soon as it is made, so a crash at chunk
140 of 235 costs one chunk, not the run.

`--vram-reserve 2` holds 2 GiB free **for other programs while generating**,
because the measurement box is not headless. The expert cache takes what is
left. Raising it to 3.9 gives the desktop ~2 GiB more and costs 2.5 % of RTF
— that exchange rate is measured, not guessed.

## Running it as a server

Startup is 35 s — model load, page-locking 30 GiB of experts, the codec, and
the reference clip's KV snapshot. Paying that per paragraph would be absurd,
so it is a resident process.

```bash
bin/drama3-offloading --start     # ~35 s, then follows the log
bin/drama3-offloading --status
bin/drama3-offloading --config    # backend JSON for a client
bin/drama3-offloading --stop
```

**No environment variables are needed** once a voice is registered: the
voice *is* the reference. `--ref` / `DRAMA3_REF` remains available as an
unnamed fallback, and until 2026-10-21 the server demanded it even with a
voice present — a requirement that outlived the reason for it.

If your checkout does not sit beside its venv, fish-speech clone and
weights, write those paths once into **`bin/env.local`** rather than
exporting them every time:

```bash
cp bin/env.local.example bin/env.local    # then edit
```

`bin/env.sh` sources it first, so anything in it wins over the defaults.
⚠️ It is **gitignored**: it holds absolute paths from one machine, and
possibly the path to somebody's voice recording.

```
POST /v1/audio/speech   {"input": "..."}   ->  audio, OpenAI-compatible
GET  /v1/models                            ->  the usual model list
GET  /health                               ->  uptime, requests, running RTF
POST /tts   {"text": "..."}                ->  the older bespoke route, still served
```

The port is whatever `[server] port` in `config.toml` says (8123 by
default). You never have to work it out: the running server writes its
real address to `run/server.url`, and `drama3ctl status` and
`drama3ctl config` both report that rather than assuming.

**The voice is prepared once**, which is the point. Startup builds a prefix
KV snapshot (117 MiB for a 56 s clip) and every request restores it rather
than re-reading the clip — measured **1,244 of ~1,270 prompt rows reused,
98 %**. One voice per process; changing it means a restart.

Point a client at it:

```bash
bin/drama3ctl config | say -add-backend drama3
say -engine drama3 DOC.txt
```

`drama3ctl config` prints a bare JSON backend entry on stdout (explanation
goes to stderr, so it pipes cleanly). It sets **`"concurrency": 1`**, which
matters: this server holds one model on one GPU and serialises requests, so
two in flight do not finish sooner — they interleave and both finish later.
Measured by say's author against this backend, four paragraphs rendered
two-at-a-time took 57 s of generation for 44 s of audio where one-at-a-time
predicted 37 s. `"prefetch": 3` is the opposite knob and is the one you
want: queue ahead, run one at a time.

**`speed` is real.** It was once accepted and ignored, which was worse than
rejecting it: `speed` sits in a reading client's cache key, so changing it
invalidated a whole document and returned byte-identical audio. It is now
an ffmpeg `atempo` post-process with pitch preserved — **worst error 0.52 %
across 0.25–4.0×**, chained outside ffmpeg's 0.5–2.0 limit, and out-of-range
values are a 400 rather than a silent clamp. `speed: 1.0` stays on the
native path with no transcode.

`voice` is still accepted and ignored: the voice is whichever clip the
process was started with. It is accepted rather than rejected so a client
template that sends it works unmodified.

That is the whole protocol. Any reader, script or pipeline that can POST
JSON and play wav will work; `drama3ctl config` happens to print a ready-made
config block for the one this was developed against.

### Named voices

One reference clip per process was the old limit. A voice is now a stored
prefix-KV snapshot — about **1.7 MiB per second** of reference, so 94 MiB
for a 28 s clip and 157 MiB for 85 s — and several live in one server,
selected by the standard `voice` field.

```bash
bin/drama3-offloading --addvoice narrator /path/to/clip.mp3
bin/drama3-offloading --voices
bin/drama3-offloading --defaultvoice narrator
bin/drama3-offloading --delvoice narrator
```

```bash
curl localhost:8123/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"drama-3","input":"...","voice":"narrator"}'
```

**Measured:** a voice restored from disk scores **0.9400** timbre against
its reference clip, against **0.9426** for the same clip prefilled live in
the same process — a gap well inside the 0.907–0.953 run-to-run spread at
temperature 1.0. `bin/test_prefix_persist.py` checks the stronger claim:
107/107 tensors bit-identical, including a control proving the build is
reproducible across processes in the first place.

**Adding a voice to a running server takes ~3 s and needs no restart.**
`--addvoice` asks the server to build it with the model already loaded,
queued behind whatever it is doing. With the server stopped it loads a
model of its own instead, which costs the usual ~65 s.

⚠️ It used to stop and restart the server (~70 s) because the builder
loaded a *second* copy of the model and two will not fit on a 24 GiB
card. That was an artifact of writing the builder standalone, not a real
constraint.

**Voices live in pinned host RAM, not VRAM.** Measured on a 125 MiB
snapshot: restoring costs 5.07 ms from pinned host memory against 0.54 ms
from VRAM — +4.5 ms on a ~16 s generation, or 0.03 %, paid once per chunk.
In exchange the card keeps 125 MiB per voice, and VRAM is what runs out
here. Budget is **1 GiB, growing a gigabyte at a time**: nothing is ever
refused, but crossing a step prints a breakdown at boot, because pinned
pages cannot be swapped.

**`--delvoice` moves the voice to `voices/.trash/`**, it does not erase
it. Recovery is a file move rather than a GPU rebuild — which matters,
because a rebuild only works while the original clip still exists.

⚠️ **A voice whose fingerprint does not match is skipped with a reason,
never restored.** Torch version, weights identity, prefill chunk size and
cache capacity all have to agree. Restoring state built against different
weights would produce fluent audio in the wrong voice, and no downstream
check would catch it.

⚠️ `voices/` is **gitignored**. A snapshot is not audio, but it exists
solely to reproduce someone's voice — the same personal data in a
different format.

### Why the interfaces are boring on purpose

Both ends of this project speak interfaces somebody else defined — OpenAI
speech out, OpenAI transcription in — and that is the single rule the
design answers to:

> **You should be able to clone this, point it at the TTS client and the
> STT server you already run, and have it work — without reading our
> source and without wiring anything by hand.**

The interesting part of this repository is expert offloading. Everything
around it should be the part you already know. Concretely that means: no
adapter you have to write, no endpoint whose shape you have to discover by
reading `bin/server.py`, no "first run this helper to generate a config".
If your STT is whisper.cpp instead of Parakeet, you change one URL. If your
reader already drives OpenAI, Kokoro-FastAPI or LocalAI, it drives this
with an address and nothing else.

That rule is also how the bugs get found, because it is testable. Two were
caught by it after everything "worked": `config.toml` documented
`[voice] ref` while `drama3ctl` demanded `DRAMA3_REF` and failed with a
message naming only the variable — so following the documentation produced
an error explicable only from our shell script. And `drama3ctl` passed its
own default `--port` on every start, which silently beat `[server] port`,
left the launcher health-checking a port the server had not bound, and made
`drama3ctl config` hand clients a URL nothing was listening on. Neither was
a crash. Both were someone's wasted evening.

## Quality checks, and what they cost

**`config.toml` drives both the server and `bin/narrate.py`**, with
`defaults -> config.toml -> flag` in both. Point narrate at another file
with `--config`.

⚠️ It did not always. Until 2026-10-19 the file was server-only, so
setting `[voice] seconds = 56.0` and running narrate silently got you the
argparse default of 30 — `reference: snapped 30.0s -> 28.5s` from narrate
against `1206 codes (56.0s)` from the server on the same config. If older
notes disagree with this paragraph, that is why.

`--stt-every` is deliberately **not** in the file: the server checks every
piece, narrate samples one in N, and one knob for both would change what
the server does.

Generation is stochastic, and it occasionally produces **fluent gibberish** —
real phonemes, right duration, wrong language. Observed on a novel:
`"der pausbomwerk denisian"` where the text said *"their world"*.

⚠️ **A duration check cannot catch that**, because nonsense takes about as
long to say as the words it replaced. So there are two checks, configured in
`config.toml` and independently switchable:

| | catches | cost |
|---|---|---|
| `[checks.duration]` | stopped early, babbled 3x too long | free |
| `[checks.content]` | **gibberish** — transcribes each piece and compares | ~11 %, below |

The content check scores the whole piece *and its last quarter separately,
taking the worse*. Length dilutes: a 196-word chunk with clear gibberish in
it still scored 0.87 overall and 0.49 on its tail. The failure lives at the
end, which is where to look.

Measured over 170 paragraphs of a technical document:

```
verify off   RTF 0.798
verify on    RTF 0.884      ~11 % slower
   STT calls    51.9 s / 201 pieces = 0.26 s each   3.0 %
   9 retries    ~78 s                                4.5 %
```

⚠️ **Tune it to your text.** On a novel the check caught two real failures
nothing else could see. On the technical document above, the same document
scored 0/235 content loss by an independent measure — so some of those 9
retries were probably **false positives**, because STT mangles `BACPAC` into
"backpack" and a piece *ending* on one takes a tail-weighted hit. Jargon-heavy
text: lower `min_similarity` or turn it off. Prose: leave it on.

Needs `DRAMA3_STT` (see Requirements). Without it the check silently does
nothing and you keep duration-only behaviour — which will not catch
gibberish, because gibberish has a perfectly normal duration.

After `max_attempts` (default 3) the **best** draw is delivered anyway — an
imperfect paragraph beats a hole in the audiobook — and says so in the log.

## What each flag is worth

Measured on 20 chunks of the same document, cumulatively:

| | RTF | prefill |
|---|---|---|
| baseline (cache + graphs + pinning) | 1.014 | 39.4 s |
| `--prefix-kv` | 0.949 | 22.4 s |
| + hybrid gather (on by default) | 0.919 | 17.7 s |
| + `--slice-lm-head` | 0.872 | 17.3 s |
| + `--offload-embed` | 0.872 | 17.1 s |

- **`--prefix-kv`** prefills the system turn and reference clip **once**
  instead of once per chunk. The reference is 734–1,416 rows of a ~1,600-row
  prompt and byte-identical every time. Restore is bit-identical to a full
  prefill, verified tensor by tensor.
- **hybrid gather** lets prefill borrow experts already resident in VRAM
  instead of re-fetching them over PCIe. It was fetching 28 % of them twice.
- **`--slice-lm-head`** keeps only the 4,097 output rows decode can sample,
  of 252,288. Lossless.
- **`--offload-embed`** moves the input embedding table to host RAM, keeping
  decode's 4,097 rows resident.

## The two findings worth stealing

**Decode is dispatch-bound. Prefill is PCIe-bound. They are not the same
problem.** ~86 % of a decode step was the GPU idle waiting for kernel
launches — CUDA graphs took RTF 1.71 → 0.97 and expert streaming fits inside
time that was already being wasted. A *prefill* batch touches ~196 of 256
experts per layer, moves 23 GiB, and runs at 25.8 GB/s — the pinned PCIe
roof. Bring the decode intuition to prefill and you will optimise launches
that are not the problem.

**For prefill caching, capacity matters and policy does not.** A batch wants
76 % of all 256 experts, so almost any cached expert is a wanted one. An
oracle that knew the future beats a plain LRU by **3.6 points**. The 83 %
overlap between consecutive chunks' expert sets looks like a reusable
working set and is very nearly chance.

## An upstream bug you may be hitting

`fish_omni/tower.py` builds its nucleus filter as
`dropped = cumsum(softmax(values)) > top_p`, with no right-shift — so it
drops the token that *crosses* the threshold instead of keeping it.

At the shipped default `top_p=0.7` we measured **EOS drawn 0 times in 4,000
samples, at every temperature**. End-of-speech is unreachable, so
generations run to the length cap and fill with silence. This is not
specific to offloading and it is worth knowing about if your Drama 3 output
has long trailing gaps. Defaults here are `1.0 / 0.9 / 4097`.

## Layout

```
bin/narrate.py       long-form runner: chunking, retries, resumable, stage timing
bin/offload.py       expert storage in host RAM, the gather, lm_head + embed slicing
bin/devcache.py      device-resident LRU, no host sync, CUDA-graph capturable
bin/prefix_kv.py     prefill the reference once and restore it per chunk
bin/cppfetch.py      builds csrc/fetch_experts.cu
csrc/fetch_experts.cu  uint4 expert fetch that replays correctly inside a graph
bin/test_*.py        correctness: bit-identity of every lossless transform
bin/prefill_probe.py measure where a prefill batch's time actually goes
bin/server.py        resident HTTP server; one model load, many requests
bin/drama3ctl        start|stop|restart|status|log|config for that server
bin/setup.sh         clone, venv, deps, weights, doctor -- the whole install
bin/setup-stt.sh     OPTIONAL local speech-to-text, if you have none
bin/cleanroom.sh     build a fresh venv and check the setup above still works
bin/doctor.py        preflight: can this machine run it, and what is missing
bin/check_docs.py    do the docs still name flags, keys, routes and files that exist
config.toml          every quality check and threshold, with its reasoning
requirements.txt     the exact versions every measurement here was taken on
notes/               how all of it was measured, including the wrong turns
notes/botnotes/      the long version of short code comments
```

**`notes/` is the part with the longest shelf life.** 14 files of measured
results, with corrections left in place rather than deleted — several
confident conclusions in there turned out to be wrong, and the record of
*how* is more useful than a tidy summary. Start at `notes/00-INDEX.md`.

## Licence

Code: MIT, see `LICENSE`.

The **model** is not covered by that and is not distributed here. Drama 3 is
released under the Fish Audio Research licence — **non-commercial**. Check
it before you do anything with the output.
