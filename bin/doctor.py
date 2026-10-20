#!/usr/bin/env python
"""Check this machine can run the project, and say what is wrong if not.

    bin/run.sh bin/doctor.py          # inside the project environment
    venv/bin/python bin/doctor.py     # or bare, if env.sh is already sourced

WHY THIS EXISTS
---------------
Every setup problem found in two clean-room builds cost 20-40 minutes to
discover, and every one of them was a five-second question:

    no python3.12 on PATH                  -> 20 min, blocker
    import torch -> libstdc++.so.6         -> 40 min, blocker, undocumented
    parakeet venv hit the same libstdc++   -> ~20 min
    nvcc found via a /nix/store glob       -> build succeeded for a reason
                                              the user could not see
    hf download stalled behind xet         -> 18 min at 4.5 kB/s, no error

Three of those are invisible until something far downstream fails, and one
of them (the toolchain) failed *upwards* into a silent Triton fallback.
A checklist that runs before the 38 GiB download is worth more than any
amount of prose in the README.

EXIT CODES: 0 all good, 1 something is FAIL, 2 only WARNs.
Checks are ordered so the cheapest and most fundamental run first; nothing
here loads the model or touches the network.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HOME / "bin"))

OK, WARN, FAIL = "ok", "warn", "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))


def check_python() -> None:
    v = sys.version_info
    got = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 12):
        record("python", OK, got)
    elif (v.major, v.minor) < (3, 12):
        record("python", FAIL, f"{got}; need 3.12")
    else:
        # Not theoretical: numpy<=1.26.4, pydub's removed audioop and
        # dots.tts all cap below 3.13 elsewhere in this tree.
        record("python", WARN, f"{got}; 3.12 is what this was built on")


def check_torch() -> None:
    try:
        import torch
    except ImportError as exc:
        hint = ""
        if "libstdc++" in str(exc):
            # The blocker that had never been hit here, because this box's
            # venv runs a uv CPython and nix-ld patched it invisibly.
            hint = ("  <- NixOS: `source bin/env.sh` first. "
                    "`nix-shell -p gcc` does NOT fix this")
        record("torch import", FAIL, f"{exc}{hint}")
        return
    record("torch", OK, torch.__version__)

    if not torch.cuda.is_available():
        # Silent CPU execution is the expensive failure: no error, 50x
        # slower, and it looks like the model being slow.
        record("cuda", FAIL,
               "torch.cuda.is_available() is False -- everything would "
               "silently run on CPU. On NixOS: LD_LIBRARY_PATH needs "
               "/run/opengl-driver/lib (bin/env.sh does this)")
        return

    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name(0)
    gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    record("gpu", OK, f"{name}, sm_{cap[0]}{cap[1]}, {gib:.1f} GiB")

    if cap < (8, 9):
        record("fp8 support", FAIL,
               f"sm_{cap[0]}{cap[1]} has no FP8 tensor cores; needs sm_89 "
               "(Ada) or newer. See notes/13-non-ada-gpus.md")
    else:
        record("fp8 support", OK, f"sm_{cap[0]}{cap[1]}")

    if gib < 23.0:
        record("vram", WARN,
               f"{gib:.1f} GiB; measured peak here is 20.3 GiB, so this is "
               "tight but may work with a smaller --vram-reserve")
    else:
        record("vram", OK, f"{gib:.1f} GiB, peak measured 20.3")


def check_ram() -> None:
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        record("ram", WARN, "could not determine")
        return
    gib = total / 1024**3
    # --pin-experts page-locks 30 GiB; peak RSS measured at 38.6 GiB.
    if gib < 40:
        record("ram", WARN, f"{gib:.0f} GiB; --pin-experts page-locks 30 and "
                            "peak measured 38.6 GiB")
    else:
        record("ram", OK, f"{gib:.0f} GiB")


def check_imports() -> None:
    missing = [m for m in ("numpy", "soundfile", "loguru", "safetensors",
                           "hydra", "transformers", "einops", "tokenizers",
                           "dac", "huggingface_hub", "triton")
               if importlib.util.find_spec(m) is None]
    if missing:
        record("python packages", FAIL,
               f"missing {', '.join(missing)} -- "
               "venv/bin/pip install -r requirements.txt")
    else:
        record("python packages", OK, "all present")


def check_fish_speech() -> None:
    spec = importlib.util.find_spec("fish_speech")
    if spec is None:
        record("fish-speech", FAIL,
               "not importable; clone it and set DRAMA3_FISH "
               "(see README Setup step 1)")
    else:
        # ⚠️ submodule_search_locations, NOT origin. A PEP-660 editable
        # install leaves origin as None, and the previous `or "."` then
        # resolved the CURRENT DIRECTORY -- turning a visibly wrong "."
        # into an invisibly wrong absolute path, which is worse.
        locs = list(getattr(spec, "submodule_search_locations", None) or [])
        if locs:
            where = Path(locs[0]).resolve().parent
        elif spec.origin:
            where = Path(spec.origin).resolve().parent.parent
        else:
            record("fish-speech", WARN, "importable, location undeterminable")
            return
        record("fish-speech", OK, str(where))


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        record("ffmpeg", OK, shutil.which("ffmpeg") or "")
    else:
        record("ffmpeg", FAIL,
               "not on PATH; every reference clip goes through it")


def check_toolchain() -> None:
    """The one that failed upwards into a silent fallback."""
    try:
        import cppfetch
        cppfetch._resolve()
        nvcc, ninja = cppfetch.CUDA_MERGED, cppfetch.NINJA_BIN
    except Exception as exc:                       # noqa: BLE001
        record("cpp toolchain", WARN, f"could not probe: {exc}")
        return
    if nvcc and ninja:
        # ⚠️ Derive provenance from the RESOLVED PATH, never from which
        # env vars happen to be set. Checking the env var first claimed
        # "found via DRAMA3_CUDA_HOME" when that override had been
        # rejected as invalid and the store glob had supplied the answer
        # -- a true toolchain reported with a false origin, which is the
        # harder kind of wrong to notice.
        override = os.environ.get("DRAMA3_CUDA_HOME", "")
        on_path = shutil.which("nvcc")
        if override and Path(nvcc).resolve() == Path(override).resolve():
            via, status = "DRAMA3_CUDA_HOME", OK
        elif on_path and Path(on_path).resolve().parent.parent == Path(nvcc).resolve():
            via, status = "PATH", OK
        else:
            via, status = "a /nix/store glob", WARN
            if override:
                via += f" (DRAMA3_CUDA_HOME={override} was rejected: no bin/nvcc)"
                status = FAIL
        detail = f"nvcc at {nvcc}, found via {via}"
        if status is WARN:
            detail += ("  <- succeeds because your store happens to have "
                       "CUDA, not because you installed it")
        record("cpp toolchain", status, detail)
    else:
        record("cpp toolchain", WARN,
               f"no {'nvcc' if not nvcc else 'ninja'}; "
               "use --fetch-backend triton, which needs no compiler")


def check_weights() -> None:
    try:
        import paths
        ckpt = paths.checkpoint()
    except SystemExit as exc:
        # ⚠️ Keep ALL the lines. This took [0] of the split and discarded
        # the three that name the fix -- on the single failure a fresh
        # machine is most likely to hit, in a tool the README promises
        # "prints the fix beside whatever is wrong".
        detail = "\n".join(
            ("" if i == 0 else " " * 10) + line
            for i, line in enumerate(str(exc).strip().splitlines()))
        record("weights", FAIL, detail)
        return
    except Exception as exc:                       # noqa: BLE001
        record("weights", FAIL, f"{type(exc).__name__}: {exc}")
        return
    codec = Path(ckpt) / "codec.pth"
    detail = str(ckpt)
    if not codec.is_file():
        record("weights", WARN, f"{detail}  <- no codec.pth beside them")
    else:
        record("weights", OK, detail)


def check_writable() -> None:
    root = os.environ.get("DRAMA3_EXT_CACHE", str(HOME / "torch-ext-cache"))
    probe = Path(root)
    try:
        probe.mkdir(parents=True, exist_ok=True)
        t = probe / ".doctor-write-test"
        t.write_text("x")
        t.unlink()
        record("ext cache writable", OK, str(probe))
    except OSError as exc:
        record("ext cache writable", FAIL,
               f"{probe}: {exc}; set DRAMA3_EXT_CACHE to a writable path")


def check_stt() -> None:
    """Optional, but it is the only check that catches fluent gibberish."""
    ep = os.environ.get("DRAMA3_STT", "")
    if not ep:
        record("stt (optional)", WARN,
               "DRAMA3_STT unset -- the content check is off, so nonsense "
               "at the right duration will not be caught. See README")
        return
    if not ep.startswith("http"):
        record("stt (optional)", OK if Path(ep).exists() else WARN,
               f"socket {ep}")
        return
    try:
        import urllib.request
        with urllib.request.urlopen(ep.rstrip("/") + "/health", timeout=3) as r:
            record("stt (optional)", OK, f"{ep} -> HTTP {r.status}")
    except Exception as exc:                       # noqa: BLE001
        # /health is liveness not readiness on parakeet-api, and some
        # servers have no /health at all -- hence WARN, never FAIL.
        record("stt (optional)", WARN, f"{ep} unreachable: {exc}")


def check_disk() -> None:
    try:
        usage = shutil.disk_usage(HOME)
    except OSError as exc:
        record("disk", WARN, str(exc))
        return
    free = usage.free / 1024**3
    record("disk", OK if free > 50 else WARN,
           f"{free:.0f} GiB free at {HOME} (weights alone are ~39)")


def main() -> int:
    for fn in (check_python, check_torch, check_ram, check_imports,
               check_fish_speech, check_ffmpeg, check_toolchain,
               check_weights, check_writable, check_stt, check_disk):
        try:
            fn()
        except Exception as exc:                   # noqa: BLE001
            record(fn.__name__, WARN, f"check itself failed: {exc}")

    width = max(len(n) for n, _, _ in results)
    for name, status, detail in results:
        mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
        print(f"[{mark}] {name:<{width}}  {detail}")

    fails = sum(1 for _, s, _ in results if s == FAIL)
    warns = sum(1 for _, s, _ in results if s == WARN)
    print()
    if fails:
        print(f"{fails} failure(s), {warns} warning(s) -- "
              "fix the failures before running anything else.")
        return 1
    if warns:
        print(f"no failures, {warns} warning(s) -- should run; "
              "read the warnings for what you are giving up.")
        return 2
    print("all checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
