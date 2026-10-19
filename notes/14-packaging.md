# Making it run somewhere else — 2026-10-18

Everything up to `13` is about making the model fast. This one is about
making the *repository* work on a machine that is not the one it was built
on, which turned out to be a separate problem with its own bugs.

## The method: a clean room

`bin/cleanroom.sh` builds a fresh venv and a fresh fish-speech clone and
installs per the README. Nothing from the development tree is on the path.

That is the entire trick, and it found **six** bugs in setup instructions
that had been written twice and believed twice. Rereading them found none.

## What was wrong

| | |
|---|---|
| torch unpinned | `pip install torch` resolves to **2.14.1+cu130**; fish-speech pins **2.8.0** in five places and every measurement used it |
| `pip install -e ../fish-speech` | **fails outright** — `pyaudio` needs a C compiler and portaudio headers |
| missing for import | `safetensors`, `hydra-core`, `transformers`, `einops` |
| missing at runtime | `descript-audio-codec`, behind a hydra `InstantiationException` naming a *class* rather than the real `ModuleNotFoundError` |
| `huggingface-cli` | does not exist in huggingface-hub >= 1.0; the binary is `hf` |
| `huggingface_hub[cli]` | the `cli` extra no longer exists either |

The fix is `--no-deps` plus an explicit list, derived by installing nothing
and adding whatever raised `ImportError` until it ran. fish-speech declares
**27** dependencies; this project needs that list minus most of them, plus
five. Dropping `pyaudio` removes the compiler requirement entirely.

## torch 2.14.1 works, and is the same speed

Worth stating because the version pin looks like it is about correctness or
throughput and is about neither.

```
                        RTF    capture    RTF excluding capture
2 chunks,  torch 2.14   5.333    54.9 s           0.938
20 chunks, torch 2.14   1.206    82.4 s           0.873
20 chunks, torch 2.8    0.872     2.5 s           0.861
235 chunks, torch 2.8   0.851     2.1 s           0.850
```

**Steady-state throughput is identical** — 0.873 against 0.861 on the same
20 chunks is inside run-to-run variation. The whole difference is
**CUDA-graph capture, 25–40x slower on 2.14.1 and variable** (54.9 s and
82.4 s on two runs of the same work).

That is a one-time cost: ~3 % of a 31-minute document, ~80 % of a two-chunk
test. ⚠️ **A short run on new torch will report an alarming RTF.** Grep the
log for `captured in` and subtract before concluding anything.

## The RTF was verified, not trusted

Prompted by that 5.333 looking incompatible with the headline 0.851.

1. **Arithmetic.** The manifest's audio total, the PCM frame counts of all
   235 output wavs read independently, and the stitched `full.wav` agree to
   the hundredth of a second: 1864.14 s. RTF 0.851 either way.
2. **The outlier.** Explained in full by capture, per the table above. Every
   configuration converges on ~0.85 once it is removed, with the full
   document at the asymptote — the shape a fixed cost spread over more
   audio should have.
3. **An external clock.** 342 s measured with `date` against 330 s of
   internal stage accounting. The 12 s gap is interpreter startup, imports
   and the final stitch, outside the instrumented region — 3.5 %, and in
   the direction where the script is not flattering itself.

⚠️ **0.851 is generation only. All-in it is 0.869**, counting the 32.5 s of
startup. The README quotes both the wall-clock figure and the RTF and
previously did not say they were measured differently.

## Two traps worth not re-learning

- **`pgrep -f "[p]ip install"` can hang forever.** A stale process from
  *another session* had `uv pip install` in its command line and matched
  the pattern indefinitely. The bracket trick guards against matching your
  own process, not someone else's. Do not gate a wait loop on command-line
  text; use a marker file.
- **A missing-module loop can declare success on a failure.** Forgetting
  `DRAMA3_FISH` makes `paths.py` exit with its "cannot find a fish-speech
  checkout" message, which is *not* a `ModuleNotFoundError` — so a loop
  grepping for missing modules sees nothing and reports resolved. Check for
  the success string, not the absence of an error.

## What is still untested

The clean room is on the **same machine**: same GPU, same driver, same
NixOS. It proves the repository is self-contained and the instructions are
executable. It proves nothing about a different distro, a different CUDA, a
different card. That remains n=1.
