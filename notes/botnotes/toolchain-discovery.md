# Finding nvcc, g++ and ninja without hardcoding them

**Where:** `bin/cppfetch.py`, `_resolve` and `nvcc_env`

Order: explicit env var → whatever is on `PATH` (right on an ordinary
distro) → a Nix store glob (right on NixOS, where nvcc is usually not on
`PATH` at all) → `/usr/local/cuda`, `/opt/cuda`, `/usr`.

| | |
|---|---|
| `DRAMA3_CUDA_HOME` | a CUDA toolkit with **both** `bin/nvcc` and `include/` |
| `DRAMA3_CXX` | host compiler for nvcc |
| `DRAMA3_CUDA_ARCH` | `8.9` Ada, `9.0` Hopper. Detected from the GPU when unset |

## What breaks if you change it back

The original hardcoded three Nix store hashes. That is correct on exactly
one machine, and on that machine only until `nix-collect-garbage` runs —
they are not gc roots. The failure is a build error pointing at a path that
no longer exists, which reads as "the code broke" rather than "the store
path was collected".

## Four things the build environment has to get right

Each of these was a real failure, and `nvcc_env` sets and restores all of
them narrowly:

- **`TORCH_CUDA_ARCH_LIST` must be set**, or `_get_cuda_arch_flags()`
  (`cpp_extension.py:2421`) calls `get_device_capability()` and the *build*
  touches the GPU.
- **`CC` must be UNSET.** `cpp_extension.py:2715` does
  `if CC: cuda_flags = ['-ccbin', CC]`, overriding nvcc's own correct
  default from `nvcc.profile`. On NixOS `bin/run.sh` runs inside
  `nix-shell -p gcc`, which exports `CC` pointing at gcc 15 — and CUDA
  12.9's `crt/host_config.h` **hard-errors above gcc 14**. Our own wrapper
  breaks the build unless this is cleared.
- **`CXX` must point somewhere real**; torch falls back to a bare `c++`.
- **`CUDA_HOME` is cached in a module global at import time**, so setting
  the env var is not enough — the attribute is patched too.

⚠️ On NixOS the CUDA derivation must be the **merged** one. The nvcc-only
derivation has no headers and `cpp_extension` needs both.
