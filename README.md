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

The trick is that a mixture-of-experts model only uses a fraction of itself
per token. 30 GiB of expert weights live in **system RAM** and stream over
PCIe on demand; only the 6.9 GiB that every token needs stays resident.
Upstream's `--device` flag does multi-GPU model parallelism, not this.

**You provide:** a voice clip and a text file. **You get:** a wav per
paragraph plus a stitched whole, resumable if it dies halfway.

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
| **CUDA toolkit** | `nvcc` **and headers**, plus `ninja` and a **g++ ≤ 14**, to build `csrc/fetch_experts.cu`. See below — there is a fallback |

**The CUDA toolkit is only needed for `--fetch-backend cpp`**, which is the
default and what every measurement here used. With no compiler, pass
`--fetch-backend triton`: same job, pure Triton, no build step. Verified to
run; **never benchmarked against `cpp`**, so no claim either way on speed.

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
# 1. fish-speech, used UNMODIFIED and not vendored here
git clone https://github.com/fishaudio/fish-speech
git clone git@github.com:darth-feir/drama3-offloading.git
cd drama3-offloading

# 2. a python 3.12 venv
python3.12 -m venv venv
venv/bin/pip install torch torchaudio torchcodec triton \
                     numpy soundfile loguru tokenizers tqdm \
                     "huggingface_hub[cli]"
venv/bin/pip install -e ../fish-speech

# 3. the weights: 36.9 GiB + a 1.87 GiB codec. Set HF_HOME first if you
#    do not want them in ~/.cache.
export HF_HOME=/somewhere/with/40GB
venv/bin/huggingface-cli download fishaudio/drama-3

# 4. environment. Defaults assume ../fish-speech and find the weights
#    under HF_HOME; override DRAMA3_FISH / DRAMA3_CKPT if not.
source bin/env.sh
```

`bin/env.sh` documents every variable it reads and why. Most people need
none of them.

### Check it works before committing an hour to it

```bash
bin/run.sh bin/narrate.py \
  --text-file document.txt --out-dir out/smoke \
  --ref myvoice.wav --ref-text "what the clip says" \
  --dev-cache --fetch-backend cpp --cuda-graph 1 --pin-experts \
  --vram-reserve 2 --prefill-chunk 1024 \
  --prefix-kv --slice-lm-head --offload-embed \
  --limit 2
```

`--limit 2` does the full load, the CUDA build, graph capture and two
paragraphs in about two minutes, and writes `out/smoke/full.wav`. If that
is right, drop `--limit` and run the document.

There is also `bin/check_botnotes.py`, which needs no GPU, no model and no
dependencies — a quick way to confirm the checkout itself is intact.

## Running it

You need a **reference clip** — 30–60 s of one person speaking cleanly — and
its transcript.

```bash
bin/run.sh bin/narrate.py \
  --text-file document.txt --out-dir out/run1 \
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
