"""Summarise a bin/narrate.py run from its manifest. Read-only, no GPU."""
import json, sys, pathlib
from collections import Counter

d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/paragraph")
m = json.loads((d / "manifest.json").read_text())
cs = m["chunks"]
ok = [c for c in cs if c.get("ok")]
bad = [c for c in cs if not c.get("ok")]
if not ok:
    print(f"{d}: no successful chunks ({len(bad)} failed)"); sys.exit()

audio = sum(c["seconds"] for c in ok)
gen = sum(c.get("gen_seconds", 0) for c in cs)
att = sum(len(c.get("attempts", [])) for c in ok)
retried = [c for c in ok if len(c.get("attempts", [])) > 1]
unacc = [c for c in ok if not c.get("accepted", True)]
trunc = [c for c in ok if c.get("truncated")]
sil = sorted(c.get("trailing_silence", 0) for c in ok)

print(f"=== {d}  (mode={m.get('mode')}) ===\n")
print(f"chunks              {len(cs)}  ({len(ok)} ok, {len(bad)} failed)")
print(f"audio produced      {audio/60:.1f} min")
print(f"compute             {gen/60:.1f} min")
print(f"overall RTF         {gen/max(audio,1e-9):.2f}   (1.0 = realtime)")
print(f"speed               {sum(c['frames'] for c in ok)/max(gen,1e-9):.2f} tok/s\n")
print(f"first draw accepted {len(ok)-len(retried)}/{len(ok)} "
      f"({100*(len(ok)-len(retried))/len(ok):.0f}%)")
print(f"needed a redraw     {len(retried)} ({100*len(retried)/len(ok):.0f}%)")
print(f"wrong after retries {len(unacc)} ({100*len(unacc)/len(ok):.0f}%)")
print(f"truncated mid-word  {len(trunc)}")
print(f"avg attempts        {att/len(ok):.2f}\n")
print(f"trailing silence    median {sil[len(sil)//2]:.1f}s  p90 {sil[int(.9*len(sil))]:.1f}s  "
      f"max {max(sil):.1f}s  total trimmed {sum(sil)/60:.1f} min")
mem = m.get("memory_peak") or m.get("memory_after_load") or {}
if mem:
    print(f"VRAM peak           {mem.get('vram_peak_gib')} GiB of {mem.get('vram_total_gib')} "
          f"(reserved {mem.get('vram_reserved_gib')})")
    print(f"host RAM peak       {mem.get('ram_peak_gib')} GiB\n")
print(f"stop reasons        {dict(Counter(c.get('finish_reason') for c in ok))}")
rows = [c.get("prompt_rows", 0) for c in ok]
print(f"prompt rows         min {min(rows)}  median {sorted(rows)[len(rows)//2]}  max {max(rows)}")

if unacc:
    print(f"\n--- chunks still off after retries (listen to these) ---")
    for c in unacc[:12]:
        a = c["attempts"][-1]
        print(f"  chunk_{c['index']:04d}.wav  ratio {a.get('ratio')}  "
              f"{c['bytes']}B  {c['text'][:70]!r}")
if bad:
    print(f"\n--- failed chunks ---")
    for c in bad[:12]:
        print(f"  {c['index']}: {c.get('error')}  {c['text'][:60]!r}")

spot = [c for c in ok if c.get("stt")]
if spot:
    print(f"\n--- STT spot checks ({len(spot)}) ---")
    for c in spot[:8]:
        print(f"  [{c['index']}] in : {c['text'][:88]}")
        print(f"       out: {c['stt'][:88]}")
