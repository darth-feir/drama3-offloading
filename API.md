# Connecting to this server

A text-to-speech HTTP server. It speaks the **OpenAI speech API**, so a
client that already talks to OpenAI, Kokoro-FastAPI, openedai-speech,
LocalAI or anything else in that family needs no new code — only an address.

Default address: `http://127.0.0.1:8123` — but do not hard-code it. The
port comes from `[server] port` in `config.toml`, and the running server
writes the address it actually bound to `run/server.url`. `drama3ctl
config` prints a ready-made backend entry containing it.

The intent is that adopting this costs you an address and nothing else: if
you already drive OpenAI-compatible speech, you should not have to read any
of our code, write an adapter, or wire anything up by hand.

---

## POST /v1/audio/speech

The one that matters.

```http
POST /v1/audio/speech
Content-Type: application/json

{
  "model": "drama-3",
  "input": "The text to speak.",
  "voice": "drama3",
  "response_format": "wav",
  "speed": 1.0
}
```

Returns the audio as the raw response body, with a matching `Content-Type`.
Errors return JSON: `{"error": "..."}` with 400 or 500.

| field | required | notes |
|---|---|---|
| `input` | yes | the text. `text` is accepted as a synonym |
| `response_format` | no | `wav` `mp3` `flac` `opus` `aac` `pcm`. **Defaults to `mp3`**, per the OpenAI spec |
| `model` | no | accepted and ignored; one model per process |
| `voice` | no | accepted and ignored; see *One voice per process* below |
| `speed` | no | **honoured.** `0.25`–`4.0`, default `1.0`. Outside that range is a `400`, not a clamp |

**`speed` is a real post-process**, not a hint: ffmpeg `atempo` with pitch
preserved, chained to get outside ffmpeg's own 0.5–2.0 limit. Measured in a
clean room: `1.5` gave a duration ratio of **0.680** against an ideal
0.667, and `0.5`→`2.0` spanned **3.95x** against an ideal 4.00x — under 2 %
either way. `speed: 1.0` skips the transcode entirely.

⚠️ This row previously read *"accepted and ignored"* and was wrong. If you
cache generated audio, note that `speed` genuinely changes the bytes, so it
belongs in your cache key.

**`wav` is the recommended format.** It is produced natively — everything
else is a transcode of it — and its header carries the sample rate, so
there is nothing for a client to misconfigure.

⚠️ **If you choose `pcm`, it is headerless 16-bit little-endian mono at
44100 Hz.** Nothing in the stream says so. A client that assumes 22050 will
play it at half speed with no error.

## GET /v1/models

```json
{"object": "list", "data": [{"id": "drama-3", "object": "model", "owned_by": "fishaudio"}]}
```

For clients that enumerate before their first call.

## Also accepted

Older shapes, so existing configuration keeps working:

| | |
|---|---|
| `POST /tts` | `{"text": ...}` → wav |
| `POST /tts_to_audio` | XTTS-api-server's path. `{"text": ...}` → wav |

## GET /health

Not part of any standard; useful for readiness and monitoring.

```json
{"ready": true, "requests": 170, "audio_seconds": 2010.5,
 "generate_seconds": 1720.4, "rtf": 0.856,
 "verify_seconds": 52.2, "verify_calls": 197, "retries": 5, "gave_up": 0,
 "load_seconds": 36.1, "uptime_seconds": 1758.4}
```

`ready` is false until the model is loaded. Everything else is cumulative
since start.

---

## Behaviour a client should know about

**Requests are serialised.** One model on one GPU. Two requests in flight do
not finish sooner — they interleave and both finish later. If your client
has a concurrency setting, **set it to 1**. If it has a prefetch or
read-ahead setting, that is the one to raise instead: queueing ahead hides
latency, running in parallel destroys it.

**Startup is about 35 seconds** — model load, page-locking 30 GiB of expert
weights, the codec, and the reference clip. Poll `/health` rather than
sleeping. Requests before `ready` will block, not fail.

**Generation is faster than playback but not instant.** Measured RTF ~0.85,
so a 30-second paragraph takes about 25 seconds to produce and arrives in
one piece at the end. There is no streaming: the body is complete when it
starts. A reader should fetch ahead.

**One voice per process.** The voice is the reference clip the server was
started with. `voice` in the request is accepted and ignored. To change
voice, restart the server.

**Long input is handled internally.** Paragraphs over a threshold are split
for generation and the audio is concatenated before it is returned — the
caller always gets exactly one audio stream per request, and the text is
never altered.

**Quality checks may add latency.** When enabled, generated audio is checked
against the request text and regenerated if it does not match; this is how
occasional gibberish is caught. A request that retries takes roughly twice
as long. It is bounded — after a few attempts the best attempt is returned
regardless, so a request always completes.

## Timeouts

A long paragraph can legitimately take over a minute, and a retry doubles
it. **Allow at least 300 seconds.** A client that times out at 30 will fail
on perfectly healthy requests.
