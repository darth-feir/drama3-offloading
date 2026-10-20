# Notes index

Read in this order. Later files correct earlier ones; corrections are marked
in place rather than deleted, so the reasoning survives.

| file | what is in it |
|---|---|
| `00-origin.md` | How the project started: the architecture facts verified from the model's own metadata before anything was run, and why a dense model (S2-Pro) cannot be offloaded the same way. Written before any of it was executed — read it as a plan, not a result. |
| `01-static-findings.md` | Read before touching anything. Byte-level budget from the safetensors headers, why this is a hook not a fork, and a ⚠️ **correction**: CUDA graphs are *not* blocked by offloading. |
| `02-first-results.md` | First working run. The per-expert DMA fix (+30 %), the measured transfer costs, and the dispatch floor. ⚠️ Its "19.34 tok/s ceiling" and its rejection of pinning are both superseded — see `06`. |
| `03-longform-findings.md` | Long documents. The content-dropping failure modes, the duration check that catches them, and ⚠️ a corrected failure rate (2 % on real prose, not 30 %). One-shot produces **fluent audio of the wrong text**. |
| `04-speedups.md` | Measured speed work: expert cache, prefill chunk size, lm_head slice. RTF 3.90 -> 1.88, with quality verified at each step. Also how the cache was broken and what that cost. |
| `05-say-integration.md` | Connecting to the user's `say` plugin. Use `say -chunks`; do not reimplement its splitting. The `clean` hazard and its fix. |
| `06-cuda-graphs-plan.md` | **Unmeasured.** The biggest remaining win, the pinned-base-address design, and the three experiments that should precede writing any of it. |
| `07-language-specialisation.md` | **Answers "can we make it English-only?"** — no. English uses all 256 experts; English and Chinese route to the same ones (correlation 0.733). Stripping Chinese frees 0.22 GiB of 37. |
| **`08-STATE-AND-NEXT.md`** | **Start here.** Where it got to, the working config, and the ranked next steps with the measurement behind each. |
| `09-prefix-kv.md` | The reference clip is now prefilled once, not once per chunk (`--prefix-kv`). 734 of 768 prompt rows reused, restore is bit-identical. ⚠️ Corrects `08`'s "608 rows / 84 %" to **734 rows / 96 %**. Makes reference length nearly free, so `--ref-seconds 56` costs almost nothing. |
| `10-prefill-gather.md` | **Prefill is PCIe-bound; decode is not.** 96.5 % of a prefill batch is weight transfer at the 25.8 GB/s roof, 23 GiB per chunk. The gather ignored 28 % of experts already in VRAM -- now it does not. Also: why `--slice-lm-head` NaN'd under graphs (it tripped a guard with its own `-inf` sentinel) and the one-value fix. RTF 1.014 -> 0.872. |
| `11-reference-length.md` | **Reference length is now free**: 28.5 s / 51.7 s / 56 s all land RTF 0.85-0.90 with no trend. Use the full clip. Also a `snap_to_silence` bug that silently discarded the tail of any cleanly-ended recording (8 % of this clip), and ⚠️ a CUDA-graph-bucket causal story I published from two points and the third contradicted. |
| `12-full-run.md` | **The whole 30-minute script, end to end.** 235/235 ok, RTF 0.85, 27 min wall for 31 min of audio, stage-by-stage timing. STT over every chunk: nothing dropped. ⚠️ Also a scorer artefact to know about — a numeral spoken as a word scores 0.00 and looks exactly like a dropped chunk. |
| `13-non-ada-gpus.md` | **DESIGN ONLY, nothing run.** Why this needs an Ada NVIDIA card today, and how to lift that: split the one `fp8` flag into storage vs compute, because the per-expert dequantise it needs already exists upstream and is wired to the wrong switch. Covers AMD, the ~10.3 GiB VRAM floor, and the one experiment that decides whether any of it is possible. |
| `14-packaging.md` | Making the **repository** work elsewhere, which turned out to be a separate problem. A clean room found six bugs in setup instructions written twice and believed twice. Also: torch 2.14.1 is the same speed as 2.8.0 (the pin is about graph-capture time, not throughput), and the three checks that verified RTF 0.851 rather than trusting it. |
| `15-gibberish-and-the-content-check.md` | Generation occasionally produces **fluent gibberish at the right duration**, which every existing check was blind to by construction. The detector, the three thresholds and the failure each came from, the measured 11 % cost, two wrong diagnoses, and why the STT connector is now a standard API with the user's Parakeet behind it. ⚠️ Corrects `12`'s "0/235 content loss". |
| `16-download-and-run.md` | **Can a stranger clone this and run it?** Two clean rooms, the second network-isolated and downloading all 38 GiB itself — it finished, and reproduced 8.83 GiB / 20.29 peak / 44.5 s capture to two decimals. ⚠️ `--fetch-backend cpp` built here only because this box happens to have CUDA in its Nix store — fine on a conventional distro, a trap on NixOS either way. Also the two silent config bugs stating the rule exposed, and the method for auditing whether an agent cheated. |
| `17-ref-text.md` | **A claim this repo made four times, measured and mostly withdrawn.** A `--ref-text` of the single word "placeholder" against 56 s of unrelated speech cost **nothing detectable** in timbre or content — within-arm spread exceeds between-arm gap. It did speak **8 % slower** (clean separation, p = 0.012). ⚠️ Tested on a ONE-chunk document, so the "inherited by every chunk" part remains untested rather than disproven. |
| `18-named-voices.md` | **Several voices in one server**, selected by the standard `voice` field. A voice is a 118 MiB prefix-KV snapshot; restored from disk it scores **0.9400** timbre against its clip vs **0.9426** prefilled live. ⚠️ Includes a restart that was never necessary — `--addvoice` stopped the server for ~70 s purely because the builder loaded its own model; asking the running server instead takes **3.2 s**. |
| `botnotes/` | The long version of short code comments. A `see notes/botnotes/<slug>.md` pointer in the code means the full story — what was tried, what broke, what the measurement was — is in there. Each one carries a **What breaks if you change it back** section. `bin/check_botnotes.py` fails on a dead pointer or an orphan. |
| `TODO-pause-handling.md` | Requested: punctuation-aware pause clamping (`;`, em dash) instead of a flat 0.5 s cap. |

## House rule followed here

Every number is marked measured, inferred, or someone else's. Where a claim
turned out wrong it is struck through in place with what replaced it — three
separate figures in this project were asserted as fact before anyone measured
them, and the cost of that is why the convention exists.
