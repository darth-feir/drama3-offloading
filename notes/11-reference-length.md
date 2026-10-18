# Reference length is now free — measured

2026-10-18, after `--prefix-kv`. 20 chunks of the real script each, identical
configuration apart from `--ref-seconds`.

```
arm                ref  prefix   n  audio s    gen s    RTF  mean kv  retry
28.5 s            28.5     734  20    237.7    207.2  0.872     1908      1
51.7 s            51.7    1309  20    235.1    211.8  0.901     2485      0
56.0 s            56.0    1416  20    232.3    197.8  0.851     2765      0
```

**Doubling the reference costs nothing measurable.** RTF lands between 0.851
and 0.901 with no monotonic relationship to reference length — the longest
reference produced the *fastest* run. That spread is run-to-run variation:
changing the reference changes what gets sampled, so the chunks are not the
same work. It is the answer prefix reuse was built for.

Before `--prefix-kv` a 56 s reference cost ~10 % on **every chunk**. Now the
prefix is prefilled once (1,416 rows, 2.79 s, a 125 MiB snapshot) and
restored for free thereafter.

## ⚠️ A causal story I got wrong, and should not have published

On the first two rows alone I claimed the cost of a longer reference had
moved into the CUDA-graph KV bucket, and gave an extrapolation table
predicting a 200 s reference would be badly hurt. **The third row
contradicts it**: 56.0 s has the highest mean attended KV (2,765) and the
*lowest* RTF.

What is true is only the mechanism, not the magnitude.
`DRAMA3_GRAPH_KV_BUCKETS = (1024, 2048, 4096, 10240)`, a step replays the
smallest bucket covering its position, and a longer prefix really does push
steps into bigger buckets:

```
28.5s ref   5169 decode steps   1024:14% 2048:86%   mean kv 1908
51.7s ref   5318 decode steps   2048:79% 4096:21%   mean kv 2485
```

But its effect on RTF is smaller than the noise between runs, and the "3.2 %"
I attributed to it was two points fitted to a line. Treat the bucket as a
thing to *check* if a very long reference ever does look slow, not as a known
cost.

The one part worth keeping: between 4096 and 10240 there is nothing, so a
prompt of ~5,500 rows (which is roughly what a 200 s reference would give)
would attend 10,240. That is real waste and it is cheap to fix if it ever
shows up — the bucket list is a module constant, and capturing all four
buckets takes **2.5 s**, so two more cost a second or two. **Unverified that
it matters.**

## A bug this found: the tail of a clip was being thrown away

`snap_to_silence` cuts the reference at a silence rather than mid-word, which
is what took the voice from "bad" at a hard 20 s cut to acceptable at 28 s.
It only closed a silence run when speech resumed *after* it, so a silence
running to the end of the file was never recorded, and the snap fell back to
the last **interior** gap.

On this project's clip — speech to 55.8 s, then 0.2 s of silence to the end —
`--ref-seconds 56` snapped to **51.7 s**, discarding 4.3 s (8 %) of the
reference for no reason. `clamp_internal_silence` had always handled the
trailing case; this function did not, and the asymmetry was the whole bug.

Second half of the fix: a trailing run still failed the 0.3 s minimum. That
threshold is right for an interior gap, where the cut lands in the *middle*
and must not clip speech on either side. The end of a file only has to show
the speaker finished, which 0.1 s does — and the cut goes at the file end,
not the run's midpoint.

```
  --ref-seconds 20    -> 17.70s      unchanged
  --ref-seconds 28.2  -> 26.65s      unchanged
  --ref-seconds 30    -> 28.48s      unchanged (matches every run before this)
  --ref-seconds 56    -> 56.00s      was 51.73
```

Interior behaviour is untouched, so no earlier measurement moves.

## For the longer clips being sourced

The user is obtaining ~90 s and ~200 s recordings.

- **Room in the prompt.** `DRAMA3_MAX_PROMPT_ROWS` is 8,192 and audio is
  21.53 code rows per second. 200 s is ~4,300 code rows plus transcript, call
  it 4,900. Observed prompts peak at 1,359 with a 734-row prefix, so the
  history-and-chunk part is at most ~625 rows. ~5,500 total. Fits.
- **Supply `--ref-text` by hand for the long ones.** The transcript is
  produced by STT today. A 200 s ASR transcript will contain errors, and
  those errors become the one thing every chunk in the document conditions
  on — a mis-transcribed word teaches the wrong text↔audio alignment
  everywhere. Tolerable at 28 s, not at 200 s.
- Check the KV bucket question then, rather than pre-emptively.
