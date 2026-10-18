# Long-form narration: what breaks

⚠️ **Much of this file describes a SAMPLER BUG, not model behaviour.**
See the section at the end, 'The stalls were a sampler bug'. The
mitigations documented here remain useful, but the root cause was
found on 2026-10-17 and is fixable by changing two parameters.


2026-10-17, overnight run against `test script/design a sql server migration
strategy.txt` (32,901 bytes, 170 blank-line paragraphs, 446 lines, ~4,768
words, ≈32 min of audio).

Tools: `bin/narrate.py` (resumable long-form runner), `bin/probe.py` (sweep
short cases in one model load).

## ⚠️ READ THIS FIRST: the scary number below is wrong for real text

The section that follows measures a ~30 % content-loss rate and was written
before the full run. **On the actual document the rate is 2 %.** Measured over
the first 51 paragraphs of the SQL migration script:

```
first draw accepted   50/51   98 %
needed a redraw        1/51    2 %
still wrong after 3    0/51    0 %
genuinely truncated    0
trailing silence       median 0.1s  (0.2 min trimmed across 51 chunks)
```

The 30 % figure came from synthetic test sentences that were adversarially
repetitive, which is the worst case for Repetition-Aware Sampling. It is a
real failure mode and the retry machinery earns its keep, but it is **not**
what ordinary prose does. The original section is kept below because the
failure modes it documents are real and the detector was built from them.

## The failure modes (measured on adversarial text, ~30 % there, 2 % on real prose)

This is the most important thing found tonight, and it affects the user's
existing workflow, not just this experiment.

Roughly **30 % of short multi-sentence chunks come out wrong**, in one of two
ways, with the *same input* succeeding on a retry:

| failure | what it looks like |
|---|---|
| **stall** | speaks part of the text, then emits pure digital silence (RMS exactly 0) until EOS or the token cap |
| **babble** | generates 3x the expected duration; STT recovers only the first sentence |

Measured examples, reference voice attached, default sampling (temperature 0.7,
top-p 0.7, top-k 30):

```
"This is the first sentence. This is the second sentence."
  -> 398 frames, 18.5s audio, speech ends at 1.8s, 16.7s of silence
  -> STT: "This is the first sentence."          SECOND SENTENCE LOST

"The first phase is assessment. The second phase is migration.
 The third phase is optimization."
  -> 425 frames, 19.7s audio for ~6s of text, no trailing silence
  -> STT: "The first phase is assessment."       BABBLED
```

**It is not content-specific.** A sentence pair that dropped its second
sentence was re-run three times and came back clean every time (141, 144, 135
frames; 6.3–6.7 s; full text). So any given chunk is a coin-flip weighted
toward success, and a 170-chunk document will contain tens of bad chunks
unless something checks.

### ⚠️ Correction: that 30 % is probably not the rate for real prose

Both failing texts above are **adversarially repetitive** — "This is the first
sentence. This is the second sentence.", "The first phase is... The second
phase is... The third phase is...". The tower samples with **Repetition-Aware
Sampling** (`tower.py:23`, `RAS_WINDOW_FRAMES = 16`), which suppresses main
tokens seen in the last 16 frames. Near-identical consecutive sentences are
precisely its pathological case, so these texts are far harder than ordinary
writing.

Treat 30 % as an upper bound on a worst case, **not** the expected rate for a
real document. The full paragraph run measures the honest number.

### Temperature is not the fix

24 generations, the same two repetitive texts, four draws at each temperature:

| temperature | duration ok | content ok |
|---|---|---|
| 0.3 | 2/8 (25 %) | 2/8 (25 %) |
| 0.5 | 5/8 (62 %) | 5/8 (62 %) |
| 0.7 (default) | 3/8 (38 %) | 2/8 (25 %) |

Lowering temperature did **not** help — 0.3 was the worst of the three. With
n=8 per cell these differences are mostly noise, and the honest reading is
"no clear effect", not "0.5 is best". Left at the default; retries do the work.

### The duration detector is sound

The same sweep doubles as a validation of the cheap check. Comparing "duration
within 0.55x–2.2x of expected" against "STT recovered >85 % of the text",
the two agreed in **23 of 24 generations** (one false accept at temperature
0.7, where a draw ran 1.97x long but still dropped a sentence).

So duration is a good proxy for correctness, it needs no STT, and it is what
`narrate.py` retries on. ⚠️ It is a proxy: the one false accept shows it can
pass a bad chunk.

### The detection that works

A bad chunk is obvious from **duration**: the text implies a length, and both
failure modes land far from it. Measured across many generations, speech runs
at **~16–18 UTF-8 bytes per second**, so `expected = bytes / 16`.

```
"This is the first sentence. ..."   56B -> expect 3.5s, got 1.8s  ratio 0.51
"The first phase is ..."            93B -> expect 5.8s, got 19.7s ratio 3.40
good chunks                                               ratio 0.81 .. 1.17
```

`bin/narrate.py` therefore redraws any chunk whose spoken duration falls
outside **0.55x–2.2x** of expected, up to `--max-attempts` times, and keeps the
draw closest to expected if all attempts fail. The check is skipped below 1.5 s
of expected audio, where the ratio is too noisy to mean anything.

This needs no STT and costs nothing on the ~70 % of chunks that are fine.

## Trailing silence is routine, and wastes most of the compute

Separately from dropped content, chunks frequently finish speaking and then
emit many seconds of digital silence before stopping:

```
paragraph 1 of the script (196B):  10s speech + 17.9s silence,
                                   stopped on 'length' -> 64% of compute wasted
same text, another draw:           9.9s speech + 12.7s silence, stopped on 'eos'
```

So it happens on both stop reasons. Two fixes, both in `narrate.py`:

- **Budget frames from the text length** (`bytes / 16 * 21.53 * 1.6 + 32`)
  instead of always allowing 600. Bounds the waste.
- **Trim trailing silence** from each chunk before writing it, keeping 0.15 s.
  Without this the stitched narration has 10–20 s gaps in it.

A chunk that stops on `length` *with* no trailing silence is a genuine cut-off
and is retried with the full budget; one that stops with a silent tail is just
trimmed. Conflating those two is how you either truncate speech or keep gaps.

## Isolated fragments are reliable

Short standalone text is clean — the problem appears when text is combined:

```
"Data platform modernization ... goals."       232f 10.8s eos 0.1s silence  OK
"Use case scenario"                             28f  1.3s eos 0.1s silence  OK
"Use case scenario."                            29f  1.3s eos 0.0s silence  OK
"<sentence> \n Use case scenario"              487f 22.6s eos 12.7s silence  heading dropped
```

The document is full of headings that blank-line splitting merges onto the end
of a paragraph ("...goals.\nUse case scenario"). Those merged chunks both drop
the heading and stall. ⚠️ Not yet tested: whether splitting on single newlines
(so headings generate alone) removes the problem. It would cost many more
chunks — 446 lines vs 170 paragraphs — and prefill dominates short chunks, so
it is not obviously a win.

## Cost note: prefill dominates short chunks

Prefill grows as history accumulates (576 -> 919 rows across two chunks) and
took 23 % of wall time on a two-chunk run. For a one-line heading the prefill
is ~10 s for 1.3 s of audio. This is why one-chunk-per-line is not free.


## One-shot: the prompt is not the limit — the generation is

I expected the whole script to overflow `DRAMA3_MAX_PROMPT_ROWS = 8192` and
raise `Drama3PromptTooLongError`. **It does not.** Measured with the real
tokenizer (no GPU needed, see the snippet in this file's history):

```
script 32,898 bytes
  no reference              6,072 rows   FITS
  with a 20s reference      6,565 rows   FITS
  largest prefix that fits  100% of the script
```

So roughly **44 KB of text fits in a single prompt** (~5.4 bytes per row).
Prompt length is not what stops you feeding a document in one go.

What stops you is **generation length**. 32,898 bytes is ~32 minutes of speech
= ~41,400 frames at 21.53 Hz. Against that:

- `DRAMA3_CHUNK_MAX_NEW_TOKENS` is 600 frames (27.9 s)
- `narrate.py`'s budget caps at `--max-new-tokens`, default 600

So a one-shot run produces **27.9 s and stops**, having silently dropped ~98 %
of the document. No exception, no warning — exactly the failure mode that is
worst to discover late. `narrate.py` flags it (`truncated: true`, stop reason
`length` with no trailing silence) rather than letting it pass.

Nothing in the architecture forbids a longer generation —
`max_position_embeddings` is 262,144, so 6,565 + 41,400 rows is legal. ⚠️
Untested whether the model stays coherent over 41,400 frames; at ~8 tok/s it
would also take ~86 minutes for one un-resumable call, which is a poor trade
against chunking that can be retried per paragraph.


## The real-world trigger: lines without terminal punctuation

Across the full paragraph run the failures are not random — they cluster on
one shape. Blank-line splitting glues a *heading* or a *bullet list* onto
neighbouring text, producing a chunk whose lines do not end in `.`/`?`/`!`:

```
chunk_0057  ratio 0.23  137B
  'Any potential migration blockers\n    Any breaking changes that require...'

paragraph 1 ratio 0.81  196B
  '...strategic objectives and goals.\nUse case scenario'
```

Both under-speak. Isolated, the same fragments are clean (`"Use case
scenario"` -> 28 frames, 1.3 s, eos). So it is the *combination* that breaks,
and it is a text-splitting problem, not a model defect.

`narrate.py` already sub-splits paragraphs over `--para-max-bytes`, but these
chunks are small — they are not caught by a size rule. ⚠️ Untested fix: split
on single newlines when a line does not end in terminal punctuation, so
headings and list items generate alone. Cost is more chunks, and prefill
dominates short chunks (~8 s of prefill for 1.3 s of audio), so it trades
wall-clock for correctness rather than being free.

## Measured cost of long-form narration

Over the first 81 of 192 chunks of a real document:

```
audio          15.4 min
compute        59.5 min
overall RTF    3.86
speed          5.72 tok/s
prompt rows    min 534, median 918, max 1248
stop reasons   eos 79, length 2
```

RTF 3.86 is worse than the 2.68 single-shot decode figure, and the gap is all
overhead that scales with chunking: prefill is re-paid per chunk over a
history that grows to ~1250 rows, and 4 % of chunks are drawn more than once.

Rough planning number: **~4 minutes of compute per minute of finished audio**,
at this (uncached) speed.


## One-shot, measured: fluent audio of the WRONG TEXT

Running the whole 32,899-byte script as a single prompt, transcribing the
output and diffing it against the source:

| prompt | budget | result |
|---|---|---|
| script + 20 s reference | 600 frames | reads the **reference transcript**, verbatim |
| script + 20 s reference | 4000 frames | same, drifting into invented text, stops on `eos` at ~360 frames |
| script, no reference | 600 frames | starts at **char 1,891 (6 % in)**, skipping the opening |

Every run reported success. None raised. None truncated in the 4000-frame
case — the model emitted a normal end-of-speech after ~17 s despite having
room for 4,000 frames, so the generation budget is **not** the binding
constraint; the model simply will not narrate a 32 KB prompt.

This supersedes the earlier section in this file, which assumed the one-shot
failure was budget truncation. It is not. It is content substitution, and it
is undetectable by ear.


## The bullet-list fix, verified

Regenerating the four paragraphs that lost content, with and without line
splitting, transcribing every chunk and measuring coverage against the source:

```
--no-split-lines (reproduces the bug)   6 chunks   3 wrong    56% coverage
default (split at unpunctuated lines)  20 chunks   0 wrong   102% coverage
```

Example, same source paragraph:

```
glued:  "Any potential migration blockers
         Any breaking changes that require post migration fixes
         Azure features that the workloads can use"
    ->  "Any potential migration blockers."                        26% coverage

split:  three chunks ->
        "Any potential migration blockers."
        "Any breaking changes that require post-migration fixes."
        "Azure features that the workloads can use."              100% coverage
```

Cost: 6 chunks became 20, and each pays its own prefill. On this sample that
is the correct trade — losing 44 % of the words is not a performance
optimisation. ⚠️ Verified on the four known-bad paragraphs only, not on a
fresh full-document run.


---

# The stalls were a sampler bug, and I missed it for a day

## The mechanism

`tower.py:57` filters the nucleus like this:

```python
dropped = (values.softmax(-1).cumsum(-1) > top_p) | (ranks >= top_k)
dropped[..., 0] = False
```

It **drops the token that crosses top_p** — its own docstring says so. Standard
nucleus sampling *keeps* that token. Only rank 0 is protected.

Consequence, measured directly against the installed sampler with a synthetic
distribution where EOS genuinely holds 25 % of the mass:

```
  top_p    temp 0.3    temp 0.7    temp 1.0
    0.7       0.000       0.000       0.000   <- our default
    0.9       0.000       0.000       0.315
    1.0       0.071       0.242       0.310
```

**At top_p 0.7, EOS was drawn 0 times in 4000 samples at every temperature.**
End-of-speech was mathematically unreachable, so the model had no way to stop
and emitted silence until the token cap.

This also explains the result in this file that never made sense: the
temperature sweep found **0.3 was worst**. Sharpening a distribution cannot
help when truncation has already collapsed the draw to argmax.

## ⚠️ Why I missed it

`narrate.py` exposed only `--temperature`. All 24 generations in the sweep ran
at `top_p=0.7`. I varied the one parameter that could not help and concluded
"temperature is not the fix" — which was true, and the wrong conclusion to
stop at.

## Measured improvement

40 generations, 4 texts x 5 draws x 2 configs, `--max-attempts 1`:

| config | ok | stall | short | babble | % ok |
|---|---|---|---|---|---|
| 0.7 / 0.7 / 30 (old default) | 7 | 10 | 0 | 3 | **35 %** |
| 1.0 / 0.9 / 4097 (new default) | **15** | 3 | 1 | 1 | **75 %** |

```
old: 203.4 s of trailing silence generated, 7,768 frames
new:  60.8 s                              , 4,728 frames   (-39 % frames)
```

Per text:

```
       dash:  2/5 -> 5/5    the real chunk 83 failure
 three_sent:  3/5 -> 5/5    the real chunk 231 failure
      three:  2/5 -> 3/5
        two:  0/5 -> 2/5    adversarially repetitive
```

**Both real-document failures go to 5/5.** The two that remain imperfect are
the synthetic repetitive probes, which are a separately-documented
unfixable regime — repeated phrases defeat repetition penalties, greedy
decoding and RAS alike.

## Upstream status

Reported independently in October 2026 and not yet merged:
sgl-project/sglang-omni #2823 (top_k collapse), #2853 and #2810 (the top_p
crossing token), vllm-project/vllm-omni #9044 (semantic code 0 sampled as a
main token). Upstream `fish-speech` has not acknowledged either issue.
Fish Audio's own vLLM recipe documents the symptom as "model behaviour" and
recommends capping `max_new_tokens` — i.e. the mitigation, not the fix.

## Still true after the fix

Keep the duration check and the retry loop. The Drama 3 technical report
(arXiv:2610.18010 §5.1) says Fish Audio regenerate takes that fail "silent
output, too little speech, runaway length" at a rate of 1.24 % (zh) /
1.84 % (en) — their published quality numbers assume a retry mechanism
exists. Ours is the same mechanism.
