# Does a wrong `--ref-text` actually hurt? — 2026-10-19

**Mostly no.** The claim this repo made in four places was not supported
when measured. There is a real effect, and it is not the one we warned
about.

## The claim, and where it came from

`README.md`, `bin/env.sh`, `bin/drama3ctl` and `bin/server.py` all said
some version of:

> ASR errors in the reference transcript are **inherited by every chunk**
> of the document.

Nobody measured that. It was reasoned from how the model conditions on the
(reference audio, reference text) pair, it sounded right, and it was
written down four times — which made it feel established rather than
assumed.

It came apart by accident. A smoke run was made with `--ref-text
"placeholder"` against 56 s of unrelated speech — about the worst input
the flag can take — and the person whose voice it is said it sounded
good, *better than expected*. If the claim were as strong as written,
that should have been obviously broken.

## The test

Same text (the 52-word Dune fixture), same clip, same flags. Only
`--ref-text` differs. Three runs per arm, because temperature is 1.0 and
every render is a draw.

| arm | `--ref-text` |
|---|---|
| `stt` | Parakeet's transcript of the clip — what the tool uses by default |
| `corrupted` | that transcript with 30 % of words replaced by unrelated ones |
| `placeholder` | the single word `placeholder` |

Scored on content (STT the output, word-level similarity to the input),
timbre (resemblyzer speaker-embedding cosine against the clip), and
duration.

⚠️ There is no human transcript of this clip, so `stt` is the baseline,
not ground truth. This measures **degradation relative to the default**.

## Results

```
timbre      stt [0.907 0.934 0.936]   corrupted [0.932 0.935 0.936]   placeholder [0.927 0.935 0.953]
content     stt [0.960 0.970 0.970]   corrupted [0.970 0.970 0.970]   placeholder [0.920 0.960 0.960]
seconds     stt [19.59 20.35 20.77]   corrupted [19.64 19.92 20.15]   placeholder [20.98 21.57 22.43]
```

| | widest within-arm spread | largest between-arm gap | verdict |
|---|---|---|---|
| timbre | 0.0287 | 0.0126 | **not separable** |
| content | 0.0400 | 0.0230 | **not separable** |
| seconds | — | 1.76 | **separable** |

**Timbre and content: no effect detectable.** The spread *within* a single
arm is larger than the gap *between* arms. `placeholder` had the highest
mean timbre of the three (0.938 vs 0.925 for the correct transcript) —
noise, but the opposite of a penalty.

**Speech rate: a real effect.** All three `placeholder` runs land above
all six others, with no overlap. Exact probability of that ordering by
chance is 1/C(9,3) = **0.012**. A degenerate reference transcript makes
the model speak about **8 % slower**.

That is worth keeping. It is also not what any of the four warnings said.

## What this does and does not establish

**Does:** for a single chunk, a reference transcript can be badly wrong —
even a single unrelated word — without measurable cost to voice similarity
or content fidelity. The strong form of the claim is dead.

**Does not:**

- ⚠️ **The claim was about accumulation across chunks, and this test used a
  ONE-CHUNK document.** The fixture is 310 bytes against a 400-byte
  `para_max_bytes`. Whatever "inherited by every chunk" might mean for a
  235-chunk book, this does not test it. That is the obvious next
  experiment and it has not been run.
- n=3 per arm detects a large effect and cannot rule out a small one.
- One clip, one voice, one language, one short text.
- `corrupted` replaced words with real English words. Transcripts garbled
  into non-words might behave differently.
- The timbre scorer is resemblyzer's GE2E encoder. The `0.9799` figure
  quoted elsewhere in this project came from different tooling and is
  **not** on the same scale as these numbers.

## What changed in the repo

All four warnings rewritten to say what was measured. `server.py` still
refuses to start without a transcript — the model needs *some* text to
align the clip against, and "approximate is fine" is not "none is fine".

## House rule, again

This is the fourth figure in this project asserted as fact before anyone
measured it, after three different speech-rate numbers that preceded
2.36 w/s. The pattern is identical: a plausible mechanism, written down
once, then cited by later documents as though the citation were evidence.
Writing it in four places made it look more established, not more true.

Harness: `reftext_ab.py`, kept out of the repo as a one-off; its logic is
reproduced above in enough detail to rerun.
