# Does Drama 3 route by language? No.

2026-10-17. The question: could an English-only (or English+Chinese) Drama 3
drop experts the way people strip languages out of LLMs?

**Both runs complete**: 30 minutes of audio each, 12,400,640 English and
12,447,680 Chinese decode routing decisions, same reference voice so language
is the only variable.

## How people strip languages, and what each is worth here

| technique | what it does | worth on Drama 3 |
|---|---|---|
| vocabulary pruning | drop unused tokens from embedding + output | ~0.76 GiB. Safe, lossless, tiny |
| layer pruning | remove whole layers | lossy, not considered |
| expert pruning | remove rarely-routed experts | **the only one that could matter** -- experts are 30 of 37 GiB |

## First: presence/absence is the wrong question

The saturation curve kills it. Over 30 minutes of varied English
(Austen, Carroll, Melville, a technical document), 12,400,640 decode routing
decisions:

```
experts touched per layer: min 220, median 232, max 254 of 256
experts never used in ANY layer: 0 of 256
```

**English uses every one of the 256 experts.** Not "nearly all" -- all. So
nothing can be dropped on the grounds that English never touches it. Only
frequency-based pruning is possible, and that changes the computed function.

The coverage curve (how much traffic the top N carry) converged well before
30 minutes -- it moved 0.09 points between the 8.3 and 14.5 minute samples:

| keep top N / layer | % of calls served | frees |
|---|---|---|
| 224 | 99.61 % | 3.8 GiB |
| 192 | 97.20 % | 7.5 GiB |
| 160 | 92.71 % | 11.3 GiB |
| 128 | 86.04 % | 15.0 GiB |
| 96 | 76.70 % | 18.8 GiB |

⚠️ Counterintuitive and worth remembering: coverage **rose** with more data
(keep-192 went 95.35 % at 1 min -> 97.20 % at 30 min). I expected the
opposite -- more text surfacing more rare experts. What actually happens is
the ranking stabilises and genuinely hot experts accumulate proportionally
more traffic while the tail stays rare. Short samples are *pessimistic* about
pruning, not optimistic.

## Then: the languages use the same experts

Same reference voice for both runs, so language is the only variable.

```
Spearman rank correlation of per-expert usage, per layer:
  mean 0.742   (min 0.516, max 0.864)

english's top 192 serve 93.14 % of chinese's calls
chinese's top 192 serve 93.43 % of english's calls
english's top 128 serve 78.50 % of chinese's calls
chinese's top 128 serve 76.99 % of english's calls

(layer, expert) slots used by english only:   174
(layer, expert) slots used by chinese only:    98
used by both:                                9,159
```

**The near-symmetry is what makes this convincing**: 93.14 % and 93.43 % in
both directions, from independent 30-minute samples. If either language had
private experts the two directions would not match like that.

A correlation of 0.733 means both languages lean on substantially the same
experts in substantially the same order. English's own top-192 serve 97.7 %
of English traffic; those *same* 192 serve **93.2 % of Chinese** -- only 4.5
points worse. Language-specific experts would collapse that number.

**Only 98 of 10,240 slots are Chinese-exclusive: 0.29 GiB.**

The earlier partial run showed 314 English-only vs 75 Chinese-only, which I
predicted was a sample-size artifact. With Chinese caught up to the same 30
minutes it fell to **174 vs 98** and the shared count rose from 9,019 to
9,159 — as predicted.

## Conclusion

An English-only Drama 3 saves roughly **0.29 GiB of experts plus ~0.76 GiB of
vocabulary -- about 1 GiB of 37**. Not worth the engineering, and not worth
any quality risk.

Whatever these experts specialise in, it is shared across languages:
phonetics, prosody or syntax rather than vocabulary. That matches what the
MoE literature generally reports for text models, and it now has a
measurement on this one.

**The expert cache already captures the benefit** that pruning was supposed
to deliver. Rarely-used experts simply never occupy cache, and they cost
nothing but host RAM, of which there is plenty (37.9 GiB of 125 used).
Pruning would only additionally shrink the RAM footprint -- which is not a
constraint -- at the cost of changing the model.

## An unexpected side result: prefill and decode differ

Recorded separately because mixing them is misleading:

```
decode   12,400,640 decisions
prefill  86,280,320 decisions   (87 % of all routing)
rank correlation decode vs prefill: mean 0.605 (min 0.394, max 0.764)
```

Reading the text and speaking it are measurably **different workloads inside
the same model** -- more different from each other (0.605) than English is
from Chinese (0.742). The split that matters inside this model is "reading
text vs producing speech", not "which language". Any future pruning or caching work has to say which
regime it is optimising for. It also explains why raising the prefill chunk
size paid so well: prefill is where most of the routing happens.

## Reproduce

```bash
bin/run.sh bin/route_census.py --label english --minutes 30 \
  --corpus corpus/pg1342-clean.txt corpus/pg11-clean.txt ...
bin/census_compare.py notes/census-english.json notes/census-chinese.json
```

`route_census.py` writes no audio, so the Chinese census needed separate
validation — nothing in it confirms the model produced intelligible Mandarin
rather than failing quietly, and Parakeet is English-only so transcription
cannot check it.

**Validated** via `bin/zh_validate.sh` -> `out/zh-sample/full.wav`:

```
3 chunks, 0 failed, 0 truncated
25.40 s of audio from 112 CJK characters = 4.4 chars/sec
fluent Mandarin runs 4-6 chars/sec
peak 0.816, rms 0.092
```

A model babbling, stalling, or spelling out pinyin would not land inside the
normal speech-rate band. ⚠️ This establishes that it is *speaking Mandarin at
a plausible rate*, not that the speech is correct or natural — that still
needs a listener. The sample also uses the **English** reference voice (held
constant so language was the only variable in the census), so the accent is
likely odd.
