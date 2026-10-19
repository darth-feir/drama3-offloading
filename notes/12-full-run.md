# Full 30-minute script, end to end — 2026-10-18

`out/full-v5/`. The whole production configuration on the whole document,
and the first stage-by-stage timing of a real run.

```bash
bin/narrate.py --mode paragraph --dev-cache --fetch-backend cpp \
  --cuda-graph 1 --pin-experts --vram-reserve 2 --prefill-chunk 1024 \
  --prefix-kv --slice-lm-head --offload-embed --ref-seconds 56
```

```
235 ok, 0 failed, 0 truncated of 235
audio 31.07 min, compute 26.45 min, RTF 0.85
VRAM 20.29 GiB peak of 23.52 | RAM 38.92 GiB peak
prefill gather: 37.6% served from VRAM
                3,585 GiB over PCIe instead of 5,745
```

**27.0 minutes of wall clock for 31 minutes of audio.**

⚠️ **0.851 is generation only; all-in it is 0.869**, counting the 32.5 s of
startup. Both figures were later verified against the PCM frame counts of
all 235 output files rather than the run's own bookkeeping — they agree to
the hundredth of a second. See `14-packaging.md`.

## The user's verdict

**"v5 is the best sounding so far by far."** (2026-10-18, on the full
`out/full-v5/full.wav`.) Earlier verdicts on reference length, same listener:
20 s "bad", 28.2 s "no longer a different voice", 56 s "a better yet but not
much", and on the 20-chunk 56 s slice "better, still needs some work for
consistency, close though".

**That improvement is attributable to the reference length, not to the speed
work**, and the reason is worth keeping: every other change made today is
numerically neutral by construction.

| change | effect on what the model computes |
|---|---|
| `--prefix-kv` | restored state is **bit-identical** (all 106 cache tensors, `bin/test_prefix_kv.py`) |
| hybrid gather | device-to-device copy of the **same bytes** as the PCIe fetch |
| `--slice-lm-head` | the retained rows are unchanged and the dropped ones were never read |
| `--offload-embed` | **bit-identical** embeddings on decode, prefill and mixed ids |
| `--ref-seconds` 28.2 → 56 | **the only change to what the model is conditioned on** |

The one caveat: moving a prefill chunk boundary shifts the last bits of the
hidden state, so v5 is a *different draw* from v4, not the same audio faster.
A different draw could flatter one render by luck — but across 235 chunks,
consistently, that is not a likely explanation.

So the evidence now points the same way the user's instinct did: a longer,
more consistent reference clip is the lever on voice consistency. The 90 s
and ~200 s recordings being sourced are the right next experiment.

## Against the same script yesterday

```
v4 (2026-10-17)  235/235 ok  audio 31.26min  gen 31.58min  RTF 1.010  retries 1  trunc 1
v5 (today)       235/235 ok  audio 31.07min  gen 26.45min  RTF 0.851  retries 0  trunc 0
```

16 % less compute, while carrying **twice the reference** — the median
prompt grew from 724 to 1,610 rows. v4's one truncation and one retry are
both gone, but at n=1 each that is suggestive, not evidence.

## Where the time goes

```
  --- stages (27.0 min wall) ---
  model load                      7.6s   0.5%
  pin experts                    14.4s   0.9%
  slice lm_head                   0.0s   0.0%
  offload embed                   0.5s   0.0%
  codec load                      4.8s   0.3%
  reference encode                2.4s   0.2%
  prefix KV build                 2.8s   0.2%
  generation (all chunks)      1587.8s  98.0%

  --- inside generation (26.5 min of chunk time) ---
  prefill (prompt)              179.8s  11.3%
  codec (codes -> audio)          8.1s   0.5%
  decode + per-chunk rest      1399.3s  88.2%
  40,195 frames, 34.81 ms/frame of decode+rest
```

- **Startup is 32.5 s, 2.0 %.** Over half is `pin_experts` page-locking
  30 GiB, which buys 1.47x on every DMA afterwards. Free on a job this
  length; it would dominate a one-sentence request, which matters for the
  `say` integration where the server is resident and pays it once.
- **The codec is 0.5 %.** Worth recording against the temptation to move it
  to the CPU for VRAM: that was tested and made the run **50 % slower**
  (`08`, item 3). Half a percent is the size of the prize that was being
  chased.
- `decode + per-chunk rest` is named for what it is. Prefill and codec are
  measured; the remainder is decode *plus* the duration check, the silence
  clamping, the wav write and the history re-encode. Calling it decode would
  assert something unmeasured.
- The prefill gather reached **37.6 %**, against 33.4 % on a 20-chunk slice:
  the cache keeps warming over a long document, so short benchmarks
  understate it.

## Content: nothing dropped

`bin/add_stt.py` then `bin/score_stt.py` over all 235 chunks.

```
word-similarity  median 1.000  mean 0.980
  >= 0.95:  219/235  (93.2%)
  >= 0.90:  229/235  (97.4%)
```

**Every chunk below 0.95 is the scorer or the ASR, not the model.** Checked
individually:

| chunk | source | STT heard | what it is |
|---|---|---|---|
| 210 | `1.` | "One." | model read the numeral aloud, correctly; word-level diff scores 0.00 |
| 215 | `2.` | "Two." | same |
| 89 | Database Mail | "Database Male" | homophone; ASR cannot distinguish |
| 112 | pre-migration | "premigration" | hyphenation in the transcript |
| 177/178 | vCore | "V-Core" / "V Cor" | ASR on a coined term |
| 148 | IaaS | "IIS" | ASR |
| 123 | BACPAC | "backpack" | ASR |
| 69 | `# Phase Action Description` | "Hashtag phase action description" | **real**, see below |

⚠️ The two 0.00 scores are a **scorer artefact worth remembering**: the
similarity is word-level after lowercasing, so a numeral spoken as a word is
a total mismatch. A future run that genuinely drops a short chunk will look
identical to these two. Read the pair before concluding anything.

### The one real finding: `#` is read as "hashtag"

Chunk 69 is a markdown table header, `# \tPhase \tAction \tDescription`,
and the model says "hashtag phase action description". That is the model
doing as it is told — the `#` is in the text we hand it. It is an argument
for the pre-input text cleaner the user raised earlier, and a concrete
example to design it against: markup that is layout, not speech.

Not fixed here. A cleaner is a behaviour change on every chunk and deserves
its own measurement rather than being bolted onto a results run.
