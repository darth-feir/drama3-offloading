# Download-and-run — 2026-10-19

`14` asked whether the repository worked on another machine and found six
bugs. This one asks a harder version of the same question, and the reason
the question changed is worth recording.

The framing used while standardising the APIs was *"engine preference
should cost nothing"*. That is a nice property and it is not the point.
The point, in the user's words:

> we want it so someone can dl this project and get it working with their
> own setup without needing to do a bunch of reverse engineering of our
> code or manual setup

That is a **testable** statement, where "the interfaces are standard" is
only an aesthetic one. Everything below came from testing it.

## Two bugs found by stating the rule

Both were silent. Both had survived every local run, because a local run
has the environment variables already set.

**The launcher contradicted the config file it ships.** `config.toml`
documents `[voice] ref`; `drama3ctl` demanded `DRAMA3_REF` and failed with
`DRAMA3_REF is unset` — a message naming only the variable. Following the
documentation produced an error whose explanation lived in our shell
script.

Fixed by **deleting** the check, not by teaching bash to parse TOML. A
launcher that re-validates what the server validates will eventually
disagree with it; `server.py` already resolves flag → env → config and
refuses with a message naming both, before the model loads, so a bad setup
still fails in one second rather than forty.

**The launcher's own default `--port` silently beat `[server] port`.** The
server obeyed the config, the launcher health-checked its own guess and
reported *ready*, and `drama3ctl config` handed clients a URL nothing was
listening on — the JSON a user pastes into another tool.

The root cause of both is the launcher **guessing** an address. The server
now publishes the one it actually bound to `run/server.url`, via
`getsockname()` so that `port = 0` reports a real number, and `health`,
`status` and `config` read it. Verified: `port = 8131` in config, started
with no environment variables at all, bound 8131, nothing on 8123.

> **What breaks if you change it back:** re-add `--host`/`--port` to the
> launcher's default argument list and the config file becomes
> decorative — with no error, because the launcher will be probing the
> port it chose.

## Clean room 1 — the setup path had never been run here

Method as `14`: fresh directory, fresh fish-speech clone, fresh venv,
README followed literally. Differences from `14`: an agent did it, it was
told to report rather than fix, and it was told that *needing to read our
source* counts as a failure even if it then succeeds.

Two blockers, and the second is the interesting one.

**`import torch` → `ImportError: libstdc++.so.6`** in a venv built from a
nix-store python. The obvious fix is wrong: `nix-shell -p gcc` does **not**
help, because the compiler's lib output is not on the runtime loader path.
Verified, because it is an hour of someone's life.

Why it was invisible for the life of the project: the venv on this machine
is built on a **uv-managed CPython**. That is a foreign binary, so NixOS's
`programs.nix-ld` supplies `libstdc++` transparently. A nix-store python
uses the nix loader, nix-ld never fires. `grep -rn libstdc` over the whole
repo returned **zero hits**, and `bin/env.sh` had a dedicated NixOS block
handling four other traps and missing this one.

**The documented setup had therefore never been executed on the only
machine this project has ever run on.** `env.sh` now picks the newest
`/nix/store/*-gcc-*-lib/lib` that actually contains the library, skipping
mingw, which matches the glob with the wrong libc.

The other blocker was mundane: no `python3.12` on `PATH`. `DRAMA3_PYTHON`
existed for exactly this and lived only in `bin/cleanroom.sh`, which Setup
never mentioned.

## Clean room 2 — isolated, and it downloaded everything

The first run was compromised: it was handed `HF_HOME`, a voice clip and a
running STT server, so it never exercised the parts a stranger hits. The
second got a copy of the repo and an internet connection, was forbidden to
read `/mnt/unencrypted`, the real repo, or the earlier clean rooms, and
fetched all ~38 GiB itself.

**It finished.** Empty directory to a cloned voice reading a document, no
blocker.

It also reproduced the README's own numbers, which had been asserted from
one machine:

| | predicted | clean room |
|---|---|---|
| resident VRAM | 8.83 GiB | **8.83** |
| peak VRAM | 20.3 GiB | **20.29** |
| cold graph capture | 45.2 s | **44.5 s** |
| warm capture | 2.5 s | 2.4 s |
| RTF, warm, 2 chunks | 0.872 | 0.81 |

### ⚠️ `--fetch-backend cpp` succeeds here for a reason you cannot ship

There is no `nvcc`, `g++` or `ninja` on this machine's `PATH`. The build
succeeded anyway, because `bin/cppfetch.py` discovers its toolchain by
globbing the Nix store:

```python
nvcc = _which("nvcc", "/nix/store/*cuda-merged*/bin/nvcc")
```

It found `cuda-merged-12.9`, which is present only because this machine's
owner installed it for something else.

Be precise about the scope, because the first write-up of this overstated
it. Discovery is `PATH` first, then the Nix glob, then `/usr/local/cuda`,
`/opt/cuda`, `/usr` — so on a conventional distro with CUDA installed,
`cpp` works normally and this is a non-issue. The problem is specific to
NixOS, and it is two problems:

1. **On a NixOS box that happens to have CUDA in its store**, the build
   succeeds without the user installing anything, via a path they cannot
   see and did not choose. Every `cpp` measurement here was taken that
   way.
2. **On a NixOS box that does not**, the README's default smoke command
   fails at the build and the user must know to pass
   `--fetch-backend triton`.

⚠️ **Unverified: what the failure in (2) actually looks like.** It was
never observed — this machine cannot produce it without uninstalling
CUDA. If it surfaces as a `cpp_extension` error about `CUDA_HOME` rather
than a sentence naming the fallback, that is the thing to fix, and it is
a few lines in `_resolve()`.

A side effect worth knowing: the build used **gcc-wrapper-15.3.0** despite
the README's "g++ 15 will not work", because `nvcc_env` unsets `CC` and
nvcc picks its own `ccbin`. The warning is not wrong, but it reads stronger
than it is.

### The rest

| sev | finding |
|---|---|
| FRICTION | **`hf download` stalls behind hf-xet.** 18 min, zero progress, no error on stdout — 99 failures went to `$HF_HOME/xet/logs/`. Measured write rate **4.5 kB/s**. With `HF_HUB_DISABLE_XET=1`: **~35 MB/s**, done in 17 min. |
| FRICTION | **No guidance on the reference clip** — one sentence, no source, no format, nothing on how exact `--ref-text` must be, though everything downstream depends on it. |
| FRICTION | **`!! TRUNCATED` was explained nowhere.** It is `--chunk-max-new-tokens`; the manifest says `finish_reason: length`. Hit on 1 of the first 2 chunks, which reads badly against the front page's "nothing truncated" — that figure is for one document at its settings, not a promise about arbitrary input. |
| FRICTION | **The libstdc++ fix does not reach the STT venv.** `env.sh` only fixes processes it launches. |
| COSMETIC | `pip install parakeet-api` is not enough — `download sherpa` and `serve` were missing. |
| COSMETIC | `--ref-seconds` defaults to 30, so a 55 s clip is silently cut; the only clue is `reference: snapped 30.0s -> 27.2s`. |
| COSMETIC | `--no-deps` prints ~15 red `ERROR:` lines that are correct and alarming. |
| COSMETIC | Interrupted `hf` downloads orphan `.incomplete` blobs — 44 G on disk for a 39 G snapshot — and are never reaped. |
| COSMETIC | `torch-ext-cache/` has no override, so the checkout must be writable. |
| COSMETIC | `drama3ctl status` printed whole-GPU vram unlabelled; right after `stop` it reads ~22 GiB and looks like a leak. |

**Found by auditing afterwards, not by either clean room:** the README's
smoke command passes `--text-file document.txt` and the repo ships no such
file. Both runs supplied their own text and neither noticed.

## The method that made run 2 trustworthy

A subagent inherits this project's `CLAUDE.md`, which lists every NixOS
trap on this box. That knowledge cannot be removed from its context, so an
isolated run is only as good as its honesty: an agent that silently applies
an insider fix makes the project look *more* installable than it is.

Two things addressed that, and both should be reused.

**Make the temptation reportable.** The instruction was not "do not use
it" but "if you catch yourself reaching for machine knowledge, stop and
log it as a finding". The portability bug above came out of that list, not
out of a failure — the agent recorded wanting to type
`nix-shell -p cudaPackages.cuda_nvcc`, which is what exposed that the build
had no business succeeding.

**Audit mechanically afterwards.** All 118 tool calls were scanned for
`/nix/store`, `nix-shell`, `/mnt/unencrypted`, `LD_LIBRARY_PATH` and the
other tells. Result: clean. Every `nix-shell -p python312` was the
documented step; the `LD_LIBRARY_PATH` it used was obtained by *sourcing
`bin/env.sh`* rather than from memory; it spotted the resident STT daemon
with `ps` and refused it, running its own on a different port.

The reason to script the audit rather than read the transcript: the
transcript is large enough to be unreadable, and "did it cheat" is a
grep, not a comprehension task.

## What is still unproven

- **Everything here is one machine.** Both clean rooms ran on the same
  4090 under the same NixOS. No non-NixOS, non-Ada, non-24 GB run exists.
  The `sm_89` floor in the README is inferred from the FP8 path, not
  observed to fail elsewhere.
- **`--fetch-backend cpp` has never been built on a machine without this
  machine's CUDA.** See above.
- **triton vs cpp: RTF 0.81 against 0.81**, identical audio, n=1 over two
  chunks. Recorded as weak evidence. It is enough to say triton is not
  obviously slower and not enough to say more.
- **`~/.cache/pip` was warm** in run 2, so its "57 s to install torch" is
  not a cold-machine number.
- **The fixed Setup has not itself been clean-roomed.** Run 2 validated
  the pre-fix text plus workarounds; the corrected README is unverified as
  a whole.

## House rule followed here

Every number above is measured unless marked otherwise. The two "fixed"
claims in `15` and `12` that later proved optimistic are why the rule
exists; the n=1 triton figure is labelled rather than rounded up into a
result.
