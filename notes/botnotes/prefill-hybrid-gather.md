# Prefill borrows from the device cache, read-only

**Where:** `bin/offload.py`, `offloaded_forward`, the `direct` copy branch

For each distinct expert a prefill batch needs, copy from the device cache
(device-to-device, ~1.5 TB/s) when it is resident and over PCIe only when it
is not.

## Why it exists

Measured (`notes/10-prefill-gather.md`): a prefill batch is **96.5 % weight
transfer** running at **25.8 GB/s** — the pinned PCIe roof — and **28.1 % of
the experts it fetches are already sitting in VRAM**. This path never
looked at the cache, so its hit rate was zero *by construction*, not because
the experts were absent.

The slot lookup needs the table on the host, but `unique()` a few lines
above already pays a host sync, so the lookup rides one that exists.

## What breaks if you change it back

Nothing visibly — which is the problem. It silently moves 138 GiB more over
PCIe per 20 chunks and costs ~20 % of prefill time. The measured effect was
RTF 0.950 → 0.919. `--no-hybrid-gather` reproduces the old behaviour for
A/B.

## ⚠️ READ-ONLY, deliberately

No insert, no LRU stamp bump. A prefill batch wants **~196 experts per
layer** against a 62–80 slot cache, so letting it populate would evict
decode's working set wholesale — and decode is 80 % of runtime. Prefill
borrows whatever happens to be there and leaves the ordering to decode.

**If you change this, measure decode, not prefill.** Prefill will look
better and the run will be slower.

## Do not reach for a smarter policy

Also measured: a batch wants 76 % of all 256 experts, so almost any cached
expert is a wanted one. An oracle that knew the future beats the
decode-driven LRU by **3.6 points**. Capacity matters here; policy does not.
