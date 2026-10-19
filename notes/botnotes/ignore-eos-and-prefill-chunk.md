# --ignore-eos must restore eos_token_id, and why prefill batches are big

**Where:** `bin/narrate.py`, the per-attempt generation block

## --prefill-chunk defaults to 1024, not upstream's 128

With experts offloaded, each prefill batch fetches every expert it routes
to, and that count **saturates near 256 regardless of batch size**. Small
batches therefore re-fetch the same experts over and over. Bigger batches
amortise it.

Measured: a 745-row prompt is one batch at 1024 and costs 1.35 s; an
1,180-row prompt is **two** batches and costs 2.42 s — almost exactly
double for 58 % more rows. See `notes/10-prefill-gather.md`.

## --ignore-eos is for timing only, and MUST be restored

Timing runs with a stub MoE cannot rely on generation behaviour: zeroed stub
weights made CUDA-graph capture produce NaN, and random ones made the model
emit EOS immediately. Disabling the EOS comparison forces exactly
`max_new_tokens` steps, so both arms of an A/B do identical work.

### What breaks if you change it back

`model.config.eos_token_id = -1` is set around the call and restored in a
`finally`. Leave it unrestored and **every subsequent chunk fails** —
`build_drama3_prompt` validates that
`tokenizer.token_to_id("<|im_end|>") == config.eos_token_id` and raises
`tokenizer/config mismatch`. The first chunk succeeds, which is what makes
this confusing: the error appears to come from the chunk after the one that
caused it.
