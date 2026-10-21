"""Named voices: a prefix-KV snapshot per reference clip, stored on disk.

A voice is 118 MiB of prefilled state plus the metadata needed to decide
whether reusing it is safe. Four voices is under half a gigabyte of host
RAM, against the 30 GiB of pinned experts -- the cost is nothing; the
correctness is the whole problem.

WHY THIS IS SAFE TO DO AT ALL
-----------------------------
Measured by `bin/test_prefix_persist.py`, 107/107 tensors bit-identical
across three comparisons -- including the control that a *freshly built*
prefix in a new process already matches, so the round trip is what is
being tested rather than nondeterminism hiding behind it.

LAYOUT

    voices/
      alice.pt      the tensors, in module order
      alice.json    fingerprint + provenance
      default       one line: the name to use when a request names none

⚠️ ORDER IS THE CONTRACT. A snapshot holds live module references in
memory -- `(module, k, v)` -- and those cannot be serialised. On reload the
tensors are re-bound to the model's modules BY POSITION, so anything that
changes module enumeration (a different layer count, a different build)
invalidates the file. That is what the fingerprint is for.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

HOME = Path(__file__).resolve().parent.parent


def voices_dir() -> Path:
    return Path(os.environ.get("DRAMA3_VOICES", HOME / "voices"))


# ⚠️ Voices live in PINNED HOST RAM, not VRAM.
#
# They were on the card at first for no better reason than that a
# freshly-built prefix is cloned from GPU caches and the loader mirrored
# it. Measured on a real 125 MiB snapshot:
#
#     VRAM-resident      0.54 ms   226.6 GiB/s
#     host RAM, pinned   5.07 ms    24.1 GiB/s   <- at the PCIe roof
#     host RAM, pageable 9.02 ms    13.5 GiB/s
#
# So host RAM costs +4.5 ms per request against a ~16 s generation:
# 0.03 %, well under run-to-run variance, and paid once per chunk rather
# than per token. In exchange each voice stops taking 125 MiB off a
# 24 GiB card -- VRAM being the resource that actually runs out here (the
# reference-length wall is prefill headroom, and this benchmark itself
# OOM'd on the first attempt because the server held the card).
#
# Pinned rather than pageable: 5.07 vs 9.02 ms for free, same reason
# --pin-experts pins the expert tables. Pinned pages cannot be swapped,
# so the budget below exists to keep that honest.
BUDGET_STEP = 1 << 30          # 1 GiB


def budget_for(total_bytes: int) -> int:
    """How much host RAM we have agreed to spend on voices.

    One gigabyte, growing a gigabyte at a time. Twenty voices simply eat
    more RAM rather than hitting a wall -- but crossing a step is worth
    saying out loud, because this memory is locked and cannot be swapped.
    """
    steps = max(1, -(-int(total_bytes) // BUDGET_STEP))
    return steps * BUDGET_STEP


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()[:16]


def checkpoint_identity(checkpoint: str | Path) -> str:
    """Identify weights by CONTENT, not by the path they were reached through.

    ⚠️ A path comparison is wrong here and it was caught by the persistence
    test rather than reasoned about: HuggingFace keeps one directory per
    revision but deduplicates the blobs, so this machine has two snapshot
    dirs whose config.json, codec.pth and all eight shards resolve to the
    same files. Comparing paths rejected a snapshot that was perfectly
    valid. Resolve the symlink and hash the target name instead.
    """
    cfg = Path(checkpoint) / "config.json"
    try:
        return hashlib.sha256(
            str(cfg.resolve()).encode()).hexdigest()[:16]
    except OSError:
        return "unresolvable"


def tensors_of(prefix) -> list[tuple[str, torch.Tensor]]:
    """Every snapshot tensor in the stable order reload depends on."""
    out: list[tuple[str, torch.Tensor]] = [("carry", prefix.carry)]
    for i, (_, k, v) in enumerate(prefix.backbone_kv):
        out += [(f"backbone.{i}.k", k), (f"backbone.{i}.v", v)]
    for i, (_, c, r) in enumerate(prefix.gdn):
        out += [(f"gdn.{i}.conv", c), (f"gdn.{i}.recurrent", r)]
    for i, (_, k, v) in enumerate(prefix.tower_kv):
        out += [(f"tower.{i}.k", k), (f"tower.{i}.v", v)]
    return out


def fingerprint(model, prefix, checkpoint: str | Path,
                prefill_chunk: int) -> dict:
    """What must match for a stored snapshot to be safe to restore.

    Deliberately NOT data_ptr(): those differ in every process and would
    reject every reload. These are the things whose change makes restored
    state WRONG rather than merely relocated. A stale snapshot restored
    silently is the poisoned-cache failure this project already hit once,
    where a torch-version mismatch cost minutes of a process spinning at
    65 % CPU inside graph capture with no error.
    """
    return {
        "torch": torch.__version__,
        "checkpoint_id": checkpoint_identity(checkpoint),
        "rows_sha": _digest(prefix.rows),
        "P": int(prefix.P),
        "cache_capacity": int(model.cache_capacity),
        "n_backbone": len(prefix.backbone_kv),
        "n_gdn": len(prefix.gdn),
        "n_tower": len(prefix.tower_kv),
        "prefill_chunk": int(prefill_chunk),
        "dtype": str(prefix.carry.dtype),
    }


def save(name: str, prefix, model, checkpoint: str | Path,
         *, source: str, ref_text: str, seconds: float,
         prefill_chunk: int, prompt_tokens) -> Path:
    d = voices_dir(); d.mkdir(parents=True, exist_ok=True)
    pairs = tensors_of(prefix)
    payload = {n: t.detach().to("cpu").clone() for n, t in pairs}
    # ⚠️ The reference AUDIO CODES travel with the voice, not just the
    # prefilled state. build_drama3_prompt() needs them to construct the
    # prompt for new text, and prefix.applies_to() checks that the
    # prompt's head equals the snapshot's rows -- so a voice without its
    # codes can be restored and then never legally used. They are a few
    # thousand ints; storing them is what makes a voice self-contained and
    # independent of the original clip still existing.
    payload["__prompt_tokens__"] = prompt_tokens[0].detach().to("cpu").clone()
    torch.save(payload, d / f"{name}.pt")
    (d / f"{name}.json").write_text(json.dumps({
        "name": name,
        "fingerprint": fingerprint(model, prefix, checkpoint, prefill_chunk),
        "order": [n for n, _ in pairs],
        "rows": prefix.rows.tolist(),
        "source": str(source),
        "ref_text": ref_text,
        "seconds": round(float(seconds), 2),
        "bytes": int(prefix.bytes),
    }, indent=2))
    return d / f"{name}.pt"


def build(model, codec, device, name: str, clip: str | Path, *,
          ref_text: str | None, ref_seconds: float, prefill_chunk: int,
          checkpoint: str | Path, transcribe=None, workdir: Path | None = None):
    """Prefill a clip into a named voice. Needs a LOADED model, not a process.

    Shared by the offline builder and the running server, because the only
    reason a voice ever required a restart was that the builder loaded its
    own copy of the model. The server already holds one; it can do this
    under its existing GPU lock in about three seconds.

    Returns (prefix, info). Raises ValueError with something actionable.
    """
    # Imported here, not at module scope: this module is also imported by
    # tooling that has no model and must not pay for fish_speech.
    import narrate
    import prefix_kv
    from fish_speech.models.text2semantic.inference import encode_audio

    clip = Path(clip)
    if not clip.is_file():
        raise ValueError(f"no such clip: {clip}")
    work = Path(workdir or (voices_dir() / ".build"))
    work.mkdir(parents=True, exist_ok=True)
    wav = work / f"{name}.wav"

    seconds = narrate.snap_to_silence(clip, ref_seconds)
    narrate.to_pcm16(clip, wav, seconds)
    text = ref_text or (transcribe or narrate.transcribe)(wav)
    if not text or str(text).startswith("<stt"):
        raise ValueError(
            f"no transcript for the clip ({text!r}). Pass ref_text, or set "
            "DRAMA3_STT. An approximate transcript is fine -- measured no "
            "worse on timbre or content (notes/17) -- but some text is "
            "required to align the clip against.")

    prompt_tokens = (encode_audio(wav, codec, device),)
    rows = prefix_kv.prefix_rows(model.tokenizer, model.config,
                                 (text,), prompt_tokens)
    prefix = prefix_kv.PrefixKV(model, rows)
    prefix.build(prefill_chunk_size=prefill_chunk)
    prefix.prompt_texts = (text,)
    prefix.prompt_tokens = prompt_tokens

    save(name, prefix, model, checkpoint, source=str(clip), ref_text=text,
         seconds=seconds, prefill_chunk=prefill_chunk,
         prompt_tokens=prompt_tokens)
    return prefix, {
        "name": name, "seconds": round(float(seconds), 2),
        "rows": int(prefix.P), "bytes": int(prefix.bytes),
        "build_seconds": round(float(prefix.build_seconds), 2),
        "ref_text": text,
    }


def names() -> list[str]:
    d = voices_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json") if p.parent == d)


def meta(name: str) -> dict:
    return json.loads((voices_dir() / f"{name}.json").read_text())


def get_default() -> str | None:
    f = voices_dir() / "default"
    if f.is_file():
        want = f.read_text().strip()
        if want in names():
            return want
    found = names()
    return found[0] if len(found) == 1 else None


def set_default(name: str) -> None:
    if name not in names():
        raise SystemExit(f"no voice named {name!r}; have: "
                         f"{', '.join(names()) or '(none)'}")
    d = voices_dir(); d.mkdir(parents=True, exist_ok=True)
    (d / "default").write_text(name + "\n")


def remove(name: str) -> Path | None:
    """Move a voice to voices/.trash/ rather than erasing it.

    ⚠️ This used to unlink. An unauthenticated POST deleted a real voice
    during a CORS audit -- 125 MiB of state derived from someone's
    recording, gone, with the only recovery being a GPU rebuild that
    needs the original clip to still exist. A voice is personal data; an
    accidental delete should cost a file move, not a recording.

    Returns where it went, or None if there was nothing to move.
    """
    d = voices_dir()
    trash = d / ".trash"
    moved = None
    for suffix in (".pt", ".json"):
        src = d / f"{name}{suffix}"
        if not src.exists():
            continue
        trash.mkdir(parents=True, exist_ok=True)
        dst = trash / f"{name}{suffix}"
        if dst.exists():                      # keep the older copy too
            dst = trash / f"{name}.{int(src.stat().st_mtime)}{suffix}"
        src.replace(dst)
        moved = trash
    if (d / "default").is_file() and (d / "default").read_text().strip() == name:
        (d / "default").unlink()
    return moved


def load_into(name: str, model, prefix_cls, checkpoint: str | Path,
              prefill_chunk: int):
    """Rebuild a usable PrefixKV from disk, bound to THIS process's modules.

    Returns (prefix, why_not). On a fingerprint mismatch it returns
    (None, reason) rather than raising: the caller's correct response is
    usually to rebuild the voice, not to die.
    """
    d = voices_dir()
    info = json.loads((d / f"{name}.json").read_text())
    rows = torch.tensor(info["rows"], dtype=torch.long)
    prefix = prefix_cls(model, rows)

    want = info["fingerprint"]
    # Build a comparable fingerprint without having prefilled anything:
    # the fields that depend on the snapshot come from the file, the rest
    # from the live model.
    live = {
        "torch": torch.__version__,
        "checkpoint_id": checkpoint_identity(checkpoint),
        "rows_sha": _digest(rows),
        "P": int(prefix.P),
        "cache_capacity": int(model.cache_capacity),
        "prefill_chunk": int(prefill_chunk),
    }
    for key, value in live.items():
        if want.get(key) != value:
            return None, f"{key}: stored {want.get(key)!r} != now {value!r}"

    payload = torch.load(d / f"{name}.pt", map_location="cpu")
    if "__prompt_tokens__" not in payload:
        return None, ("saved before reference codes were stored; "
                      "rebuild this voice")
    full, gdn, towers = prefix._cache_modules()
    if (len(full), len(gdn), len(towers)) != (
            want["n_backbone"], want["n_gdn"], want["n_tower"]):
        return None, (f"module counts changed: stored "
                      f"{want['n_backbone']}/{want['n_gdn']}/{want['n_tower']}, "
                      f"now {len(full)}/{len(gdn)}/{len(towers)}")

    device = next(model.parameters()).device

    def host(key: str) -> torch.Tensor:
        """Stays on the host, pinned. restore() copies it in."""
        t = payload[key]
        return t.pin_memory() if not t.is_pinned() else t

    def gpu(key: str) -> torch.Tensor:
        return payload[key].to(device, non_blocking=True)

    # ⚠️ carry goes to the DEVICE, not the host: generate_drama3_prefix
    # does torch.cat((carry, hidden)) against GPU hidden states. It is one
    # row, so keeping it resident costs nothing. Everything else is only
    # ever copy_'d into a cache and can live on the host.
    prefix.carry = gpu("carry")
    prefix.backbone_kv = [(m, host(f"backbone.{i}.k"), host(f"backbone.{i}.v"))
                          for i, m in enumerate(full)]
    prefix.gdn = [(m, host(f"gdn.{i}.conv"), host(f"gdn.{i}.recurrent"))
                  for i, m in enumerate(gdn)]
    prefix.tower_kv = [(m, host(f"tower.{i}.k"), host(f"tower.{i}.v"))
                       for i, m in enumerate(towers)]
    prefix.host_resident = True
    prefix._buffer_ids = prefix._buffer_identity()
    prefix.bytes = int(info["bytes"])
    prefix.prompt_texts = (info["ref_text"],)
    prefix.prompt_tokens = (payload["__prompt_tokens__"].to(device),)
    return prefix, None
