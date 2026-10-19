# The graphstub MoE: timing only, the audio is garbage

**Where:** `bin/offload.py`, `_graph_stub`, selected by `--copy-mode graphstub`

Measures CUDA-graph capture against eager on the **real** attention, tower
and sampling path, with a synthetic MoE that performs no host sync.

Graphs cannot capture the real path as it stands, because every layer does
`topk_ids.cpu()` to decide what to fetch and a host sync is illegal during
capture. The stub replaces that with `topk_ids % cap`, a pure GPU op.

## What stays real, and why that is the point

The expert weights are junk and resident, so the routing is wrong and the
output is noise. What remains real is the **~4,600 small kernel launches per
decode step** across attention, the 13-layer tower and 10 sampling passes —
which is exactly where the dispatch overhead lives, and dispatch was the
thing being measured.

That measurement said 2.35×, and the real path later came in at 1.71 → 0.97.

## What breaks if you change it back

Nothing crashes. It produces **confident, fluent, completely wrong audio**.
This mode must never be reachable from a production configuration, which is
why `COPY_MODE == "graphstub"` is tested before every other dispatch branch
and the runner prints `!! graphstub: TIMING ONLY, the audio is garbage`.

⚠️ Do not use a stub run to conclude anything about quality, content loss
or EOS behaviour. Zeroed stub weights make capture NaN; random ones make the
model stop immediately. Both are artefacts of the stub.
