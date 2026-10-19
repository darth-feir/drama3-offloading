"""Build and load csrc/fetch_experts.cu.

Kept separate from devcache.py so the C++ path can be swapped in or out
without touching the cache logic, and so a build failure degrades to "use
Triton" rather than breaking everything.

The NixOS traps below were each hit for real during the Stage 0
investigation; none are theoretical.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

# Toolchain discovery: env var, then PATH, then a Nix store glob. Override
# with DRAMA3_CUDA_HOME (needs bin/nvcc AND include/), DRAMA3_CXX, or
# DRAMA3_CUDA_ARCH. Nothing is hardcoded -- Nix store paths are not gc roots.
# see notes/botnotes/toolchain-discovery.md

_ext = None
_err = None


def _which(name: str, nix_glob: str) -> str:
    """Directory containing `name`: PATH first, then a Nix store glob."""
    import glob
    import shutil
    found = shutil.which(name)
    if found:
        return os.path.dirname(os.path.realpath(found))
    hits = sorted(glob.glob(nix_glob))
    return os.path.dirname(hits[0]) if hits else ""


def _resolve() -> None:
    global GCC14_BIN, NINJA_BIN, CUDA_MERGED
    import glob
    CUDA_MERGED = os.environ.get("DRAMA3_CUDA_HOME", "")
    if not CUDA_MERGED:
        nvcc = _which("nvcc", "/nix/store/*cuda-merged*/bin/nvcc")
        CUDA_MERGED = os.path.dirname(nvcc) if nvcc else ""
    if not CUDA_MERGED:
        for candidate in ("/usr/local/cuda", "/opt/cuda", "/usr"):
            if os.path.isfile(os.path.join(candidate, "bin", "nvcc")):
                CUDA_MERGED = candidate
                break
    cxx = os.environ.get("DRAMA3_CXX", "")
    GCC14_BIN = (os.path.dirname(cxx) if cxx
                 else _which("g++", "/nix/store/*gcc-wrapper-14*/bin/g++"))
    NINJA_BIN = _which("ninja", "/nix/store/*ninja*/bin/ninja")


def cuda_arch() -> str:
    """"8.9" on this project's 4090. Explicit so the BUILD need not touch the GPU.

    cpp_extension's _get_cuda_arch_flags() calls get_device_capability() when
    TORCH_CUDA_ARCH_LIST is unset, which makes compilation depend on a
    visible device.
    """
    explicit = os.environ.get("DRAMA3_CUDA_ARCH")
    if explicit:
        return explicit
    import torch
    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


@contextmanager
def nvcc_env():
    """Narrowly set what the build needs, then put it all back.

    * TORCH_CUDA_ARCH_LIST must be set, or _get_cuda_arch_flags()
      (cpp_extension.py:2421) calls get_device_capability() and the *build*
      touches the GPU.
    * CC must be UNSET. cpp_extension.py:2715 does
      `if CC: cuda_flags = ['-ccbin', CC]`, overriding nvcc's own correct
      gcc-14 default from nvcc.profile. bin/run.sh runs inside
      `nix-shell -p gcc`, which exports CC pointing at gcc 15 -- and CUDA
      12.9's crt/host_config.h hard-errors above gcc 14. Our own wrapper
      breaks the build unless this is cleared.
    * CXX must point somewhere real; torch falls back to a bare `c++`.
    * cpp_extension caches CUDA_HOME in a module global at import time, so
      the env var alone is not enough -- the attribute is patched too.
    """
    _resolve()
    keys = ("CC", "CXX", "PATH", "CUDA_HOME", "TORCH_CUDA_ARCH_LIST",
            "TORCH_EXTENSIONS_DIR")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.pop("CC", None)
        if GCC14_BIN:
            os.environ["CXX"] = os.path.join(GCC14_BIN, "g++")
        os.environ["CUDA_HOME"] = CUDA_MERGED
        os.environ["TORCH_CUDA_ARCH_LIST"] = cuda_arch()
        os.environ.setdefault(
            "TORCH_EXTENSIONS_DIR",
            str(Path(__file__).resolve().parent.parent / "torch-ext-cache"))
        os.environ["PATH"] = ":".join(
            p for p in (os.path.join(CUDA_MERGED, "bin"), NINJA_BIN, GCC14_BIN,
                        os.environ.get("PATH", "")) if p)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def load(verbose: bool = False):
    """Return the compiled extension, or None with the reason in last_error().

    ~40 s on a cold cache, instant afterwards. Never raises: a missing
    toolchain should fall back to Triton, not stop the run.
    """
    global _ext, _err
    if _ext is not None or _err is not None:
        return _ext
    src = Path(__file__).resolve().parent.parent / "csrc" / "fetch_experts.cu"
    if not src.is_file():
        _err = f"missing {src}"
        return None
    try:
        with nvcc_env():
            import torch.utils.cpp_extension as cpp
            cpp.CUDA_HOME = CUDA_MERGED          # the cached module global
            _ext = cpp.load(name="drama3_fetch", sources=[str(src)],
                            extra_cuda_cflags=["-O3", "--use_fast_math"],
                            verbose=verbose)
    except Exception as exc:                      # noqa: BLE001 - must not raise
        _err = f"{type(exc).__name__}: {exc}"
        _ext = None
    return _ext


def last_error() -> str | None:
    return _err
