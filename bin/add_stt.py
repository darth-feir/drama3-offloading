"""Fill `stt` into an existing run's manifest, so bin/score_stt.py can read it.

`--stt-every` transcribes during generation, which costs time on the critical
path and is therefore usually left off. This does the same work afterwards
from the wavs on disk, so an A/B that was run for speed can still be scored
for content without regenerating anything.

  bin/run.sh bin/add_stt.py out/prefix-ab-off
  bin/run.sh bin/score_stt.py out/prefix-ab-off

⚠️ The STT daemon runs PrivateTmp=true, so the temporary wav must live in
this directory, not /tmp, or it transcribes as empty and looks like the
generating model failed. It also reads PCM_16 only.
"""
from __future__ import annotations

import json, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from narrate import transcribe

directory = Path(sys.argv[1])
manifest_path = directory / "manifest.json"
manifest = json.loads(manifest_path.read_text())
temporary = directory / "_stt.wav"

done = 0
for chunk in manifest["chunks"]:
    if not chunk.get("ok") or chunk.get("stt"):
        continue
    wav = directory / f"chunk_{chunk['index']:04d}.wav"
    if not wav.is_file():
        continue
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(wav),
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                    str(temporary)], check=True)
    chunk["stt"] = transcribe(temporary)
    done += 1
    print(f"  {chunk['index']:3d} {chunk['stt'][:80]}", flush=True)

temporary.unlink(missing_ok=True)
manifest_path.write_text(json.dumps(manifest, indent=1))
print(f"transcribed {done} chunk(s) into {manifest_path}")
