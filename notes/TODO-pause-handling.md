# ~~TODO~~ DONE: punctuation-aware pause clamping

Raised by the user 2026-10-17, to do after the current speed tests.

## The problem with what is there now

`clamp_internal_silence` (`bin/narrate.py`) caps **every** silent stretch
inside a chunk at `--max-gap` (0.5 s default). It was built to fix a real
measurement — 104 gaps of >=0.6 s totalling 134.8 s, 7 % of a 32-minute file,
worst single gap 9.8 s — but it is indiscriminate.

A pause at `;` or an em-dash is **performance**, not dead air. Flattening it to
0.5 s makes the reading worse, and the current code cannot tell the difference
because it only looks at the waveform.

## Why this is harder than it looks

The clamp operates on **audio**; the punctuation is in **text**. There is no
alignment between them, so we cannot say "this gap is the one at the em-dash".
Options, cheapest first:

### A. Per-chunk budget (recommended first attempt)
Count pause-worthy marks in the chunk text, then allow that many of the
**longest** gaps to keep a higher cap, and clamp the rest hard.

```
marks = count of  ;  --  —  –  …  :  in the chunk
keep the `marks` longest internal gaps at up to ~1.2s
clamp every other gap to ~0.4s
```

Cheap, no alignment, no extra model. Fails when a stall happens to be the
longest gap in a chunk that also contains an em-dash — it would preserve the
stall and clamp a real pause. Mitigated by the existing duration check, which
rejects chunks whose total length is far from expected.

### B. Scale the cap by chunk content
If the chunk contains any of those marks, raise `--max-gap` for the whole
chunk; otherwise clamp tight. Trivial to implement, cruder than A, and it
re-admits dead air in chunks that happen to contain one semicolon.

### C. Forced alignment (accurate, heavy)
Get word timestamps for the generated audio and map each gap to its position
in the text, then decide per gap. The resident Parakeet daemon returns text
only — no timestamps — so this needs a different ASR path
(`whisper.cpp` can emit word timings, but ⚠️ `../CLAUDE.md` records that there
is no binary able to run the 4.9 GB of ggml models in `../whisper-models/`,
so this would need a system change filed in `NEEDS-FROM-NIXOS.md`).

### D. Do nothing and expose the knob
`--max-gap` already exists. Setting it to 1.2 s keeps dramatic pauses and
still removes the worst dead air (measured: 0.5 s saves 1.39 min on the
32-minute file, 0.8 s saves 0.98 min).

## Which marks

Worth distinguishing, since they do not all carry the same weight:

| mark | role | suggested cap |
|---|---|---|
| `—` em dash, `–` en dash | strongest break, often dramatic | longest |
| `;` | clause break, deliberate | long |
| `:` | introduces, slight hold | medium |
| `…` | trailing off | long |
| `-` hyphen | ⚠️ **usually not a pause** — "post-migration". Only treat a *spaced* hyphen " - " as a dash. |
| `,` | already short; leave alone | clamp |

⚠️ The hyphen case matters for this document: the SQL script is full of
"Convenience - Setting up..." (spaced, a real dash) *and* "post-migration
fixes" (unspaced, not a pause). The rule must key on spacing.

## Interaction to watch

`;` and `:` are already in the line-splitting rule added for bullet lists
(a line ending in `.!?;:` is treated as finished and may stay joined). That is
a *splitting* decision; this is a *silence* decision. They are independent but
both read the same punctuation, so change them together or the behaviour will
drift apart.

## Verification

Take chunks from the SQL script containing em-dashes and semicolons, generate
with the clamp off and with each policy, and compare gap lengths at those
positions. ⚠️ The final judgement is a listening one — whether the pause
sounds deliberate or like a glitch — so this needs the user's ears, not a
metric.


---

## Implemented 2026-10-17 — option A, the per-chunk budget

`pause_budget(text)` counts marks that earn a pause; `clamp_internal_silence`
then lets that many of the **longest** silent runs keep up to `--long-gap`
(1.2 s) and clamps everything else to `--max-gap` (0.5 s).

The hyphen rule keys on spacing, as the note required:

```
budget 1   'Convenience - Setting up an Azure SQL platform'    spaced dash
budget 0   'post-migration fixes and breaking changes'         not a pause
budget 2   'He paused — then spoke; quietly, and left'
budget 2   'Phase 1: assess. Phase 2: migrate'
```

Regex: `[—–…;:]|(?<=\s)-(?=\s)|(?<=\s)--(?=\s)`

### Measured effect: real but small

40 of 192 chunks in the SQL narration earn at least one pause. Over the first
ten of them, original 121.2 s, flat clamp 119.8 s, punctuation-aware 121.0 s —
**1.2 s of deliberate pause preserved** that the flat clamp was removing.

⚠️ Do not oversell this. The effect is modest because most chunks had no long
internal gaps to begin with; the 134.8 s of dead air measured across the whole
file was concentrated in a few bad chunks, and those are stalls rather than
pauses. The value here is that the clamp no longer flattens performance — not
that it reclaims much time.

### Known failure case, unfixed

A stall that happens to be the longest gap in a chunk that *also* contains a
dash will be spared while a genuine pause gets clamped. The duration check in
`narrate.py` is what catches those chunks, and it is a different mechanism —
if it ever stops working, this gets worse silently.

### Not done

Weighting marks differently (an em dash arguably deserves longer than a
colon; "Phase 1: assess. Phase 2: migrate" scoring 2 is generous). Left flat
because there is no evidence yet that the distinction is audible.


---

## Measured outcome across three full runs of the same document

| run | treatment | gaps >=0.6 s | total | % of file | longest |
|---|---|---|---|---|---|
| `out/paragraph` | none (overnight) | 104 | 134.8 s | **7.0 %** | **9.8 s** |
| `out/paragraph-v3` | trim + clamp | 23 | 21.0 s | 1.1 % | 1.2 s |
| `out/paragraph-v4` | trim + punctuation-aware clamp | 26 | 22.0 s | 1.2 % | 1.2 s |

**7.0 % dead air -> 1.2 %, and no gap longer than 1.2 s.** The cap is doing
exactly what it was built to do: the 9.8 s and 7.2 s holes are gone, and the
26 remaining pauses are at or under the punctuation allowance, median 0.80 s
— a normal sentence pause.

⚠️ Earlier in this file I wrote "do not oversell this... the value is that
the clamp no longer flattens performance, not that it reclaims much time."
That was measured on ten chunks. Across the whole document it reclaims
**113 seconds**. The caution was right about the mechanism and wrong about
the magnitude.

## What is NOT fixed: the waste

Post-processing removes the silence after it has been generated. The model
still *produces* it, and that is the real remaining cost:

- chunk 84 of `paragraph-v4` burned **three full attempts** to yield 2.6 s of
  usable audio (`rtf 15.01` for that chunk alone).
- The run held RTF 0.86 mid-document but finished at **1.03**, and the
  stalled/retried chunks are most of the difference.

So the open problem is no longer "remove the gaps" — it is **stop generating
them**. The payoff would be an early-stop signal in the semantic token stream:
detect a stall as it begins, abort, retry immediately, rather than generating
600 frames of silence and discarding them. Whether such a signal exists is
the subject of the research in progress.
