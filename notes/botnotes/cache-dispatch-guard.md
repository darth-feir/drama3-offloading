# Why the MoE dispatch checks per-layer capacity, not a global

**Where:** `bin/offload.py`, `offloaded_forward`

The branch into the cached path tests `self._cap` / `self._dc_cap` — the
per-layer capacity — and nothing else.

## What breaks if you change it back

Checking the `CACHE_EXPERTS` global **as well** was a bug that shipped.
`--vram-reserve` sizes and allocates the cache without ever setting that
global, so the guard was false, the cache was never consulted, and **8.32
GiB of VRAM sat allocated and unused**. The run produced correct audio at
exactly the speed of having no cache at all.

The only symptom was a **missing line in the log** — the hit-rate report
that never printed. There is now an explicit check at the end of a run that
says `EXPERT CACHE WAS ALLOCATED BUT NEVER USED` rather than staying quiet.

⚠️ That check itself then had the mirror-image bug: it tested only `_cap`,
which `--dev-cache` deliberately zeroes when it installs `_dc_cap`, so it
printed "a cache was requested but none was allocated" on *every*
`--dev-cache` run including ones where a 7.27 GiB cache was working
perfectly. Both caches have to be considered, in both places.

## graphstub overrides everything

`COPY_MODE == "graphstub"` is tested first and unconditionally. It is the
only sync-free path, and a cache allocated by `--vram-reserve` would
otherwise shadow it and send CUDA-graph capture straight into
`cached_forward`'s `.cpu()` host sync — which is illegal during capture.
