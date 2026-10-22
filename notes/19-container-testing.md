# Testing on other distros, in containers — 2026-10-21

`16` established that a stranger could build this, but every piece of
evidence came from one NixOS box. This note is about getting evidence
from somewhere else without a second machine, and about two days of
wrong diagnoses on the way.

## Containers, not VMs, and why that matters here

Rootless podman via `nix-shell -p podman` — no install, no daemon, no
system configuration. The processes run on **this** kernel (6.18.57)
under user namespaces; `/etc/subuid` maps uid 0 inside to 100000 on the
host. Two consequences that mattered:

- **Safe for the desktop.** A container shares the GPU through the
  driver; it does not unbind it. VRAM stayed at ~1 GiB throughout and the
  Wayland session never noticed. A VM would need VFIO passthrough, which
  takes the card from the host and kills the display.
- **It tests another distro's USERSPACE, not another kernel.** glibc,
  apt, python packaging — which is the part that breaks. Nothing here
  depends on a different kernel.

⚠️ Keep podman's state out of `~/.config` by pointing `HOME` at a scratch
directory. Cleanup then needs `podman unshare rm -rf`: the overlay files
belong to uid 100000+, and plain `rm` cannot touch them.

## ⚠️ The IPv6 trap — three wrong answers before the right one

Symptom: pip inside a container ran at **~30 KiB/s** while the host did
**3.7 MB/s**. Same network, same moment.

What it was NOT, each asserted confidently and then disproved:

1. **"Rootless networking is slow."** Plausible — pasta/slirp4netns is
   userspace. Switching to `--network=host` changed nothing.
2. **"PyPI is unreachable."** Measured `0 B/s` from
   `files.pythonhosted.org`… with a curl that had no `-L`. It was a 302.
   With redirects followed: 4.9 MB/s. The measurement was broken, not
   the network.
3. **"It has stalled."** The pip cache sat at 11 MiB for ten minutes, so
   it looked dead. pip writes a wheel into the cache only *after* the
   download completes, and torch is one 2.5 GB file. Watching the
   container's overlay layer instead showed 10 MB/s.

What it actually was: **this host has no working IPv6 egress**, and
inside a container `files.pythonhosted.org` resolves to IPv6 only.

```
host    curl -6  →  0 B/s (http 000)
host    curl -4  →  3,689,252 B/s
```

Every connection tried v6, hung until timeout, then retried. Nothing
errors; everything is merely glacial. Host tools fall back quickly;
the container's minimal resolver setup does not.

**The fix is two mounted files:**

```bash
# resolv.conf -- the image has no nameserver line at all
nameserver <your resolver>
options edns0 timeout:2 attempts:2

# gai.conf -- make glibc prefer IPv4 over RFC 3484's v6-first default
precedence ::ffff:0:0/96  100
```

Measured: **30 KiB/s → 3,962,994 B/s**, a ~130x difference.

> **What breaks if you drop them:** nothing errors. The build just takes
> hours instead of minutes, and every instinct will point at the wrong
> component. That is what makes it worth writing down.

## What the Ubuntu run proved

```
[  ok  ] python              3.12.3
[  ok  ] torch               2.8.0+cu128
[  ok  ] gpu                 NVIDIA GeForce RTX 4090, sm_89, 23.5 GiB
[  ok  ] python packages     all present
[  ok  ] weights             (mounted read-only from the host)
no failures, 2 warning(s)
```

The pinned `requirements.txt` resolves and installs on a distro that is
not NixOS, and torch sees the card from inside a rootless container.

**The toolchain warning fired for the first time ever:**

```
[ warn ] cpp toolchain   no nvcc; use --fetch-backend triton
```

Ubuntu has no CUDA toolkit, so this is the first environment capable of
producing it. The message was written blind in `16` after discovering
that `cpp` only ever succeeded here because this machine's Nix store
happens to contain CUDA. It says the right thing, and `doctor` grades it
`warn` rather than `FAIL`, which is correct — Triton needs no compiler.

## ⚠️ A bug only a GPU container could reveal

`env.sh` printed, on Ubuntu:

```
env.sh: no libstdc++.so.6 found in /nix/store; torch will not import
```

on a machine where torch imported perfectly well. Harmless, alarming,
and wrong. The NixOS gate was the problem, and **two attempted fixes were
also wrong**:

| test | why it fails |
|---|---|
| `-d /run/opengl-driver/lib` | the nvidia-container-toolkit **creates** it inside the container, on any distro |
| `-d /nix/store` | on NixOS the toolkit **mounts the driver libs from the store**, so the container has one too |
| `-e /etc/NIXOS` | ✅ present on the host, absent in the container |

This was unfindable before the toolkit was installed: it needs a GPU
container on a non-NixOS distro to appear at all.

> **What breaks if you widen it again:** every GPU container on every
> distro gets a scary message about a store it does not have.

## Method notes worth reusing

- **Measure the thing that moves.** The pip cache lies during a large
  download; the container's overlay layer does not.
- **Check the host before blaming the container.** One `curl -4` vs
  `curl -6` would have found the real cause in the first five minutes.
- **Mounted copies go stale.** One "fix verified" was tested against the
  container's rsync'd copy of the repo while the edit was in the source
  tree. The test passed for the wrong reason.

## Still unproven

- **Generation on another distro.** The install and GPU visibility are
  confirmed; an actual render is not.
- **Any distro other than Ubuntu 24.04.** Debian and Fedora ship
  different Python versions, and this project hard-requires 3.12 — a
  distro that ships only 3.13 may not be able to run it at all without
  extra work. That is the next thing to find out.
- **A different kernel.** Containers cannot test one, and nothing here
  appears to need it.
