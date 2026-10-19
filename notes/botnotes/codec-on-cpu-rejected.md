# Why the codec stays on the GPU

**Where:** `bin/narrate.py`, `--codec-cpu`

The codec turns the model's codes into a waveform. On the GPU it holds
1.74 GiB that could instead cache ~14 more experts per layer, so moving it
to the CPU looks like free VRAM.

**It was tried. It made the whole run 50 % slower: RTF 0.99 → 1.49.**

## Why the reasoning was wrong

The prediction was that CPU decode at 0.64–0.73× realtime would hide behind
GPU generation at RTF ~1.0. It does not, because `decode_to_audio` is called
**inline** — nothing pipelines it. CPU decode lands directly in the critical
path and adds its full duration to every chunk.

⚠️ That prediction described software that had not been written. A
background-thread version might genuinely win; the serial version cannot.

## What breaks if you change it back

Measured on the full 31-minute script, the codec is **0.5 % of generation
time** (8.1 s of 1,588 s). That is the entire prize. Against it you are
adding a CPU-bound stage to the critical path, and the headroom shrinks as
generation gets faster — the faster the GPU side becomes, the sooner the
CPU caps the pipeline.

The flag is kept because the measurement is worth being able to repeat, not
because it is a good idea.
