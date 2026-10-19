# Notes for anyone picking this up

## The house rule the notes follow

Every number is marked measured, inferred, or someone else's. Where a claim
turned out wrong it is corrected **in place, with the wrong version kept**,
because the reasoning that produced it is usually the useful part.

There are a lot of those. A partial list, all of them things that were
stated confidently and then measured:

- "Residency will be ~67 %" → 41–50 %; the non-expert weights were forgotten.
- "19.34 tok/s is the compute ceiling" → it was the *dispatch* ceiling. The
  bandwidth roof is ~139 tok/s.
- "CUDA graphs are refused when experts are on CPU" → false. FP8 experts are
  registered as **buffers**, and the check walks `parameters()`.
- "~30 % of chunks lose content" → 2 % on real prose. The 30 % came from
  adversarially repetitive test sentences.
- "CPU codec decode will hide behind generation" → only if pipelined, which
  it was not. It made the run 50 % slower.
- "Cache capacity has hit diminishing returns" → the two arms compared
  differed in more than capacity. A clean A/B put it at 2.5 % per ~1.9 GiB.
- "A longer reference costs ~3 % via the CUDA-graph KV bucket" → a causal
  story fitted to two points; the third contradicted it.

If you add a number, say how you got it.

## Before trusting a speedup

Three of the bugs here produced **plausible output rather than an error**,
which is why the tests check bit-identity rather than "it still sounds
fine":

- A duplicated function definition meant every "fix" patched a copy that
  never ran.
- A cache was allocated and never consulted; the only symptom was a missing
  log line.
- `--slice-lm-head` tripped a finiteness guard with its own `-inf` padding,
  which read as a CUDA-graph bug for a day.

`bin/test_prefix_kv.py` is the model to copy: it separates determinism,
faithfulness, drift and cost, and it zeroes the caches before restoring so a
restore that copies nothing cannot pass by accident.

## Before you trust the setup instructions

`bin/cleanroom.sh DIR [pinned|latest]` builds a fresh venv and a fresh
fish-speech clone and installs per the README, so the instructions can be
executed rather than read.

They were written twice, believed twice, and were wrong six ways — an
unpinned torch, a `pip install` that fails on a C extension nothing here
uses, four packages missing for import, one more missing at runtime behind
a misleading hydra error, and a renamed CLI binary. None of that is
findable by rereading. **If you change the dependencies, run it.**

`latest` mode installs whatever pip resolves today instead of the pinned
versions. Useful as a forward-compatibility canary; the timings from it are
not project numbers.

## Things that are open

- `bin/prebuild.py` has never been run for real against a `say` installation.
- A pre-input text cleaner. The model reads a markdown `#` aloud as
  "hashtag", correctly, because that is what it was handed.
- Nothing here has run on any machine but the one it was developed on.
