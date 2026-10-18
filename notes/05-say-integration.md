# Connecting Drama 3 to `say`

2026-10-17. `say` lives at `/mnt/data/projects/say/elevenlabs-say/` and is
**not ours** — read only, never write. Answers below came from its maintainer
session; the parts marked "verified here" I checked myself.

## The two paths

| | live backend | **prebuild** |
|---|---|---|
| how | `say` POSTs each paragraph to a local server | we fill `say-<doc>/audio/` ahead of time |
| needs | a running HTTP server | nothing running at read time |
| speed | gappy at RTF 1.9 | instant — no backend call at all |
| code change in `say` | none (generic `http` kind) | none |

Prebuild is the one that fits. `say` reuses saved audio before calling any
backend, so RTF stops mattering.

## What the maintainer confirmed

- **The "8 seconds" is a prefetch lead, not a timeout** (`audio.go` `tick()`:
  `total-pos < 8` starts the next paragraph). A slow backend produces a
  **gap, not an error**. Real timeouts: `http` 300 s, `openai` 5 min.
- **Cache key**: `sha256("<backend>|<voiceID>|<model>|<speed %.2f>|<text>")`,
  first 10 bytes hex = 20 chars. ⚠️ `elevenlabs` alone gets **no prefix**
  (historical, so pre-registry files stay valid). Everything else, including
  `drama3`, is prefixed.
- **`index.json` is not needed** — `store.load` only tries
  `audio/<fp>.wav|.mp3`. The index is for humans and `:SayClean`.
- **No resampling needed.** 44.1 kHz mono wav is read from the header and
  converted on ingest.
- ⚠️ **The `clean` hazard was real.** Its `keep` set came from the document's
  *current* chain, so prebuilt drama3 audio would have been deleted whenever
  the chain pointed elsewhere. Now fixed upstream: `keep` covers every
  configured backend, and `:SayClean` is only ever user-initiated.
- **Prefetch depth is now configurable** (`"prefetch": 4`), and a backend that
  asks for depth starts as soon as the current paragraph is in hand rather
  than in its last 8 s.

## Do not reimplement the chunking

`say -engine drama3 -chunks FILE` prints one JSON object per paragraph with
the exact hashed text and the absolute destination path. Use it. The
splitting it performs is not something to guess at: CRLF -> LF, split on blank
lines, then `strings.Join(strings.Fields(...), " ")` per paragraph (internal
newlines become single spaces, whitespace runs collapse), then paragraphs over
2500 bytes split at the last sentence end, falling back to the last space,
then a rune boundary.

Verified here, on the SQL script:

```
170 chunks   bytes: min 4, median 133, max 1668
16 chunks over 400 B   longest 1668 B ~ 104 s of speech
```

⚠️ Small correction to the maintainer's note that `-chunks` "writes nothing":
it does create the `say-<doc>/` project directory. Harmless, but worth knowing
before running it against a tree you care about.

## ⚠️ The interaction that actually matters

say's chunker **joins a trailing heading onto the previous paragraph**:

```
"Data platform modernization ... objectives and goals. Use case scenario"
```

That is precisely the shape `notes/03` measured Drama 3 failing on — it speaks
the sentence, drops the heading, and stalls into silence.

We **cannot** split it, because that exact string is the cache key. So
`bin/prebuild.py` splits it **internally only**, generates each piece, and
concatenates them into the single file say expects. Correct bytes on the key,
correct audio inside.

## Running a prebuild

```bash
say -engine drama3 -chunks DOC.txt > chunks.json     # on the machine with DOC
bin/run.sh bin/prebuild.py --chunks chunks.json      # on this box
```

`--dry-run` lists what it would write. `--dest-root` re-roots the output when
generating here for another machine (say runs on Windows; `say.exe` is a PE32+
binary and there is no Linux build in the repo).

⚠️ The backend name must be `drama3`, not `elevenlabs`. Because elevenlabs is
the one backend with no key prefix, running `-chunks` without `-engine drama3`
yields fingerprints that `say` will never look up. `prebuild.py` refuses
quietly-wrong behaviour and warns.

## Building `say` locally (for `-chunks` only)

`say.exe` is Windows. A Linux build needs CGO and ALSA headers because
`oto/v3` does playback. This works without touching the system config or
their tree:

```bash
cp -r /mnt/data/projects/say/elevenlabs-say/. $SCRATCH/saybuild/
cd $SCRATCH/saybuild
nix-shell -p go alsa-lib pkg-config --run 'CGO_ENABLED=1 go build -o say-linux .'
```

Run it with `HOME` and `XDG_CONFIG_HOME` pointed at scratch so it cannot touch
the real `~/.config`.
