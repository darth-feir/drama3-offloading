# Fluent gibberish, and the check that catches it — 2026-10-19

## The failure

Reading a novel through the server, the user reported "bugging out and
outputting gibberish every now and then at the end of longer paragraphs".

They were right, and it is not an ASR artefact — the transcripts are not
near-misses, they are a different language:

```
want  ...their world
got   ...der pausbomwerk denisian

want  ...far gone in its cooling and this world is still crowded with life
got   ...a gotiwerkte lartes setting passanting leit a king turtled plus the curia

want  ...restore an on-premises database backup to Azure?
got   ...sendoi shrueling report cat Sego and XC when it tofund Azure Compare.
```

⚠️ **The duration check cannot see this.** Gibberish takes about as long to
say as the words it replaced — measured at 16.2 and 16.5 chars/second
against the 16.0 the budget assumes — so `0.55 <= ratio <= 2.2` passes it
on the first attempt and no retry ever fires. Every quality gate the
project had was blind to it by construction.

**It is intermittent.** The exact paragraph that failed came back clean
3/3 an hour later. A clean re-run proves nothing here.

## Two wrong turns, recorded because both looked convincing

**"State leaking between pieces."** The failing piece came back perfect on
its own, 3/3, so the multi-piece path looked guilty. Then the whole
paragraph also came back perfect, 3/3. The piece-position theory was dead
and had been announced before the second test.

**"The retries on the technical document are false positives."** STT does
mangle jargon — `BACPAC` to "backpack", `vCore` to "V-Core" — and a piece
*ending* on one takes a tail-weighted hit, so this was plausible. It was
also wrong. Checked one by one, all five were real: a dropped trailing
heading, two substituted sentences, and the "sendoi shrueling" line above.

⚠️ **And the earlier "0/235 content loss" on that same document
(`12-full-run.md`) was optimistic.** It scored whole chunks, where one bad
piece is diluted by the good ones around it. Those failures were always
there.

## The detector

`stt_similarity` in `bin/server.py`: transcribe the generated piece,
compare to the text it was given, retry below a threshold.

Three parameters, each from a specific failure:

**`min_similarity = 0.75`.** Good audio scores ~0.95 — STT normalises
numbers and mangles proper nouns, so it never reaches 1.00. The observed
gibberish scored 0.04 and 0.49. Room on both sides.

**`tail_fraction = 0.25`, taking the WORSE of whole and tail.** Length
dilutes: a 196-word chunk with clear gibberish in it scored **0.87
overall** and would have passed. Its last quarter scored **0.49**. The
failure lives at the end, which is where to look.

**`min_words = 8`.** Below that, similarity is noise. The 8-byte warm-up
text "Warm up." scored 0.44 and triggered a pointless regeneration — the
duration check already had an `expected < 1.5 s` bypass for exactly this
reason and the content check did not.

### The give-up path

Bounded at `max_attempts` (3), then the **best** draw is delivered anyway —
an imperfect paragraph beats a hole in an audiobook — and it is logged and
counted in `/health`.

⚠️ That was broken when written. `best` was assigned unconditionally
rather than compared, so when every attempt failed verification the *last*
draw was delivered rather than the best one. Exactly backwards for a
fallback. Found only because the user asked what happens after repeated
failures.

## Cost

170 paragraphs of the technical document, same text both arms:

```
verify off   RTF 0.798
verify on    RTF 0.884      ~11 % slower
   STT calls    51.9 s / 201 pieces = 0.26 s each   3.0 %
   retries      5-9 fired across runs, ~78 s        4.5 %
```

The check is cheap; the regenerations it orders are most of the cost. Both
are configurable per document in `config.toml`, because the right answer
differs: on a novel the check caught failures nothing else could see, and
on technical prose the retry rate is higher.

## The STT connector

⚠️ **This was bespoke and should not have been.** `transcribe()` originally
spoke a unix socket with "write an absolute wav path, read one line of
text" — a protocol that exists on exactly one machine, which made the
content check unusable by anyone who cloned the repo.

`DRAMA3_STT` now takes either:

```
http://host:port     OpenAI-compatible POST /v1/audio/transcriptions
/path/to/sock        the bespoke socket, still supported, documented as the fallback
```

**The engine is a separate question from the interface.** The user runs
**NVIDIA Parakeet TDT 0.6B v2, int8, under sherpa-onnx, on the CPU** and
prefers it to Whisper. That preference costs nothing: `pip install
parakeet-api` (github.com/likeablob/parakeet-api) puts that exact model
behind the standard endpoint. Measured: **RTF ~0.04**, about 27x realtime
on 4 CPU threads.

⚠️ A hand-written server was built for this first, and deleted. It worked
and it was redundant — same engine, same model, same transcripts, same
speed (0.14/0.27 s against 0.15/0.31 s), 250 more lines to own. **Nobody
checked whether the thing already existed.** It did, under the name that
was chosen for the directory. Search before building a connector; that is
the same rule the rest of this file is about, applied one level up.

Anyone preferring whisper.cpp, faster-whisper-server or LocalAI points the
variable at those instead and nothing else changes.

### Why the STT stays on the CPU

Asked and answered with numbers, so it does not get re-litigated.

**It is slower on the GPU.** The existing dictation daemon's own docstring,
measured on this box: *"ONNX Runtime's CUDA provider is SLOWER than CPU for
models this size — 520 ms against 426 ms for Parakeet — because
per-inference launch and transfer overhead exceeds the compute saved."*

**The ceiling is 3 %.** Transcription is 51.9 s of 1747.9 s of runtime.
Making it infinitely fast wins 3 % and no more.

**It would cost more than it wins.** The card holds 20.3 GiB of Drama 3
with ~2 GiB free at peak. An ONNX CUDA session wants roughly 1 GiB, taken
straight out of the expert cache — and that exchange rate is measured
(`10-prefill-gather.md`): 1.94 GiB is worth 2.5 % of RTF, so ~1 GiB is
~1.3 % slower generation. Paying 1.3 % to win part of 3 %, on an operation
that benchmarks slower on the GPU, is not a trade.

And the same non-performance argument the daemon makes applies here: on the
CPU it keeps working while something else owns the GPU.

⚠️ Note that every STT measurement in this file — the 0.26 s per check, the
11 % cost, the scoring of both documents — was *already* Parakeet on the
CPU, through the socket daemon. 2026-10-19 changed the interface, not the
engine.

### What testing against a second implementation found

The HTTP client was written to the OpenAI spec and **did not work**. It
asked for `response_format=text`; faster-whisper-server 0.0.3 answers a
flat `400 Unsupported response_format: text`. The spec lists five formats
and implementations ship subsets. **json is the one everybody has.**

That bug survived a careful reading of the spec and died in thirty seconds
against somebody else's server. Writing to a standard is not the same as
being compatible with it.
