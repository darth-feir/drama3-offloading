"""Prove the prefix snapshot reproduces the state a real prefill would build.

Four separate questions, deliberately not mixed. Conflating them is how you
end up arguing about whether 1e-3 is a bug.

  0. DETERMINISM -- run the same prefill twice and diff it. Everything below
     is measured against this floor, because fp8 MoE kernels and a changing
     expert-cache residency are not obliged to be bit-stable, and a
     "faithfulness error" smaller than the noise floor is not an error.

  A. FAITHFULNESS -- build the state in one pass, versus snapshot it, wipe
     the caches, restore, and resume. Same arithmetic in the same order, so
     any difference beyond the floor means the snapshot is missing a tensor
     or the tower lag is off by one. THIS is the test that matters.

  B. DRIFT -- how far the resumed path lands from an ordinary single-pass
     prefill. Not zero, and not a bug: GatedDeltaNet accumulates in whole
     64-row chunks and the tower is batched one row out of step, so moving a
     boundary moves the last bits. The same drift already exists between
     --prefill-chunk 128 and 1024. The number is here so nobody has to
     re-derive whether it is alarming.

  C. COST -- prefill seconds saved per chunk, the snapshot's footprint, and
     an end-to-end generation on both paths.

Run:
  bin/run.sh bin/test_prefix_kv.py --checkpoint "$DRAMA3_CKPT" \
      --dev-cache --fetch-backend cpp --pin-experts --vram-reserve 2 \
      --prefill-chunk 1024

CUDA graphs are OFF by default. They only touch decode, and leaving them off
keeps A honest: capture reallocates the caches and runs three decode steps
through them, which is exactly the corruption `PrefixKV.is_valid` exists to
survive. Use --cuda-graph 1 to exercise that recovery path instead.
"""
from __future__ import annotations

import argparse, os, sys, time
from pathlib import Path

import paths  # resolves DRAMA3_FISH / DRAMA3_CKPT; see bin/paths.py
FISH = paths.install()

import torch
import offload
import prefix_kv
from narrate import snap_to_silence, to_pcm16, transcribe

from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.generate import (
    build_drama3_prompt, generate_drama3, ensure_drama3_caches,
    Drama3DecodeGraphs,
)
from fish_speech.models.text2semantic.inference import (
    load_codec_model, encode_audio, resolve_codec_path,
)

# Two chunks of ordinary prose: long enough that the per-chunk suffix is a
# realistic fraction of the prompt, short enough that a decode comparison
# finishes in seconds.
CHUNKS = [
    "The migration strategy begins with an inventory of every database on the "
    "legacy cluster, including the ones nobody will admit to owning.",
    "Once the inventory is complete, group the databases by their recovery "
    "model and by how much downtime each owner will tolerate.",
]


# ------------------------------------------------------------------ state

def cache_state(model, rows: int, tower_rows: int) -> dict[str, torch.Tensor]:
    """Every tensor the prefix snapshot is responsible for, cloned.

    Only the live rows. The caches are capacity-sized (10,240 rows) and the
    tail past the current prompt is stale by design -- masked out at
    attention time -- so cloning it would both report differences that cannot
    reach any output and cost 543 MiB of VRAM per snapshot. Holding four of
    those at once is what OOM'd the first version of this test.
    """
    state = {}
    for index, layer in enumerate(model.backbone.model.layers):
        if layer.layer_type == "full_attention":
            attention = layer.self_attn
            state[f"bb{index}.k"] = attention.k_cache[:, :, :rows].clone()
            state[f"bb{index}.v"] = attention.v_cache[:, :, :rows].clone()
        else:
            gdn = layer.linear_attn
            state[f"bb{index}.conv"] = gdn.conv_state.clone()
            state[f"bb{index}.rec"] = gdn.recurrent_state.clone()
    tower = model.audio_decoder
    for index, layer in enumerate((*tower.shared_layers, *tower.residual_layers)):
        attention = layer.transformer_layer.attention
        state[f"tw{index}.k"] = attention.k_cache[:, :, :tower_rows].clone()
        state[f"tw{index}.v"] = attention.v_cache[:, :, :tower_rows].clone()
    return state


def compare(left: dict, right: dict) -> list[tuple]:
    """Max abs difference per tensor."""
    return [(key, (left[key].float() - right[key].float()).abs().max().item())
            for key in left]


def wipe_caches(model) -> None:
    """Zero every cache, so a leftover value cannot stand in for a restored one.

    Without this, A passes for the wrong reason: the previous pass left the
    right bytes in place and a restore that copies nothing looks correct.
    """
    for layer in model.backbone.model.layers:
        if layer.layer_type == "full_attention":
            layer.self_attn.k_cache.zero_()
            layer.self_attn.v_cache.zero_()
        else:
            layer.linear_attn.conv_state.zero_()
            layer.linear_attn.recurrent_state.zero_()
    tower = model.audio_decoder
    for layer in (*tower.shared_layers, *tower.residual_layers):
        layer.transformer_layer.attention.k_cache.zero_()
        layer.transformer_layer.attention.v_cache.zero_()


# ---------------------------------------------------------------- prefills

@torch.inference_mode()
def run_prefill(model, prompt: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Upstream's prefill loop verbatim, for the drift baseline."""
    backbone, tower = model.backbone, model.audio_decoder
    prompt_rows = len(prompt)
    frames = torch.zeros((prompt_rows, prompt.shape[1]), dtype=torch.long,
                         device=tower.device)
    frames.copy_(prompt)
    positions = torch.arange(prompt_rows, device=tower.device)
    hidden = None
    for start in range(0, prompt_rows, chunk_size):
        end = min(start + chunk_size, prompt_rows)
        chunk = frames[start:end]
        embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
        hidden = backbone(inputs_embeds=embeddings[None],
                          input_pos=positions[start:end])[0].to(tower.device)
        paired_end = min(end, prompt_rows - 1)
        if paired_end > start:
            tower.prefill(hidden[: paired_end - start], positions[start:paired_end],
                          frames[start:paired_end], frames[start + 1 : paired_end + 1])
    return hidden[-1:].clone()


@torch.inference_mode()
def resume(model, prompt: torch.Tensor, P: int, carry: torch.Tensor,
           chunk_size: int) -> torch.Tensor:
    """The suffix half: rows [P, N), tower lagging one row behind via `carry`.

    Identical to the loop in `prefix_kv.generate_drama3_prefix`. Kept here as
    a copy on purpose -- if the two drift apart the test stops testing the
    thing it claims to, and that is loud rather than silent because A then
    fails.
    """
    backbone, tower = model.backbone, model.audio_decoder
    prompt_rows = len(prompt)
    frames = torch.zeros((prompt_rows, prompt.shape[1]), dtype=torch.long,
                         device=tower.device)
    frames.copy_(prompt)
    positions = torch.arange(prompt_rows, device=tower.device)
    for start in range(P, prompt_rows, chunk_size):
        end = min(start + chunk_size, prompt_rows)
        chunk = frames[start:end]
        embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
        hidden = backbone(inputs_embeds=embeddings[None],
                          input_pos=positions[start:end])[0].to(tower.device)
        low = start - 1
        paired_end = min(end, prompt_rows - 1)
        if paired_end > low:
            paired = torch.cat((carry, hidden))[: paired_end - low]
            tower.prefill(paired, positions[low:paired_end],
                          frames[low:paired_end], frames[low + 1 : paired_end + 1])
        carry = hidden[-1:]
    return carry.clone()


@torch.inference_mode()
def prefill_prefix(model, rows: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Rows [0, P), tower [0, P-1). The same work `PrefixKV.build` does."""
    backbone, tower = model.backbone, model.audio_decoder
    P = len(rows)
    frames = torch.zeros((P, rows.shape[1]), dtype=torch.long, device=tower.device)
    frames.copy_(rows)
    positions = torch.arange(P, device=tower.device)
    hidden = None
    for start in range(0, P, chunk_size):
        end = min(start + chunk_size, P)
        chunk = frames[start:end]
        embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
        hidden = backbone(inputs_embeds=embeddings[None],
                          input_pos=positions[start:end])[0].to(tower.device)
        paired_end = min(end, P - 1)
        if paired_end > start:
            tower.prefill(hidden[: paired_end - start], positions[start:paired_end],
                          frames[start:paired_end], frames[start + 1 : paired_end + 1])
    return hidden[-1:].clone()


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Drama 3 snapshot dir; found via DRAMA3_CKPT "
                         "or HF_HOME when omitted")
    ap.add_argument("--ref", default="source clips/test audio.mp3")
    ap.add_argument("--ref-seconds", type=float, default=30.0)
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--prefill-chunk", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--copy-mode", default="direct")
    ap.add_argument("--pin-experts", action="store_true")
    ap.add_argument("--dev-cache", action="store_true")
    ap.add_argument("--fetch-backend", choices=("triton", "cpp"), default="cpp")
    ap.add_argument("--vram-reserve", type=float, default=2.0)
    ap.add_argument("--cache-experts", type=int, default=0)
    ap.add_argument("--cuda-graph", choices=("0", "1"), default="0")
    ap.add_argument("--decode-steps", type=int, default=24,
                    help="frames to generate in the end-to-end comparison")
    ap.add_argument("--out-dir", default="out/prefix-kv-test")
    args = ap.parse_args()
    # Falls back to HF_HOME discovery, and raises with an
    # instruction rather than a TypeError deep in from_pretrained.
    args.checkpoint = args.checkpoint or str(paths.checkpoint())

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = args.cuda_graph
    offload.COPY_MODE = args.copy_mode
    offload.CACHE_EXPERTS = args.cache_experts
    offload.TRACE.enabled = False
    offload.install_offload()
    import fish_speech.models.text2semantic.fish_omni.backbone as bb
    bb.FishOmniBackbone.materialize = lambda self, devices: (
        offload.materialize_offloaded(self, device))

    drift = prefix_kv.check_upstream_unchanged()
    print(f"upstream drift check: {'DRIFTED ' + str(drift) if drift else 'clean'}",
          flush=True)

    t0 = time.perf_counter()
    model = FishOmniModel.from_pretrained(
        args.checkpoint, devices=[str(device)], tower_device=str(device),
        dtype=torch.bfloat16)
    if args.pin_experts:
        gib, secs = offload.pin_experts(model, device)
        print(f"pinned {gib:.1f} GiB in {secs:.0f}s", flush=True)
    codec = load_codec_model(resolve_codec_path(Path(args.checkpoint)),
                             device=device, precision=torch.bfloat16)
    if args.dev_cache:
        import devcache
        devcache.FETCH_BACKEND = args.fetch_backend
        cap = (offload.size_cache_to_free_vram(model, device, args.vram_reserve)
               if args.vram_reserve else args.cache_experts)
        for layer in model.backbone.model.layers:
            experts = getattr(layer.mlp, "experts", None)
            if experts is not None and getattr(experts, "_cap", 0):
                experts._cap = 0
                for name in ("_cache_w13", "_cache_w2", "_cache_s13", "_cache_s2"):
                    if hasattr(experts, name):
                        delattr(experts, name)
        torch.cuda.empty_cache()
        cap = devcache.install_dev_cache(model, device, cap)
        print(f"device cache: {cap}/layer", flush=True)
    print(f"loaded in {time.perf_counter()-t0:.0f}s", flush=True)

    wav = out_dir / "reference.wav"
    seconds = snap_to_silence(Path(args.ref), args.ref_seconds)
    to_pcm16(Path(args.ref), wav, seconds)
    ref_text = args.ref_text or transcribe(wav)
    prompt_texts = (ref_text,)
    prompt_tokens = (encode_audio(wav, codec, device),)
    print(f"reference: {seconds:.1f}s, {prompt_tokens[0].shape[1]} codes", flush=True)

    ensure_drama3_caches(model, Drama3DecodeGraphs.graph_capacity(model))
    rows = prefix_kv.prefix_rows(model.tokenizer, model.config,
                                 prompt_texts, prompt_tokens)
    P = len(rows)
    prompts = [build_drama3_prompt(chunk, model.tokenizer, model.config,
                                   prompt_texts, prompt_tokens)
               for chunk in CHUNKS]
    print(f"\nprefix {P} rows; prompts {[len(p) for p in prompts]} rows "
          f"-> prefix is {100*P/len(prompts[0]):.0f}% of the first", flush=True)
    for prompt in prompts:
        assert torch.equal(prompt[:P], rows), "prefix rows are not the prompt's head"
    print("prefix rows match the real prompt head on every chunk", flush=True)

    chunk_size = args.prefill_chunk
    prompt = prompts[0]

    # 0. determinism floor: the same two-phase prefill, run twice.
    live, tower_live = len(prompt), len(prompt) - 1
    wipe_caches(model)
    carry = prefill_prefix(model, rows, chunk_size)
    hidden_a = resume(model, prompt, P, carry, chunk_size)
    state_a = cache_state(model, live, tower_live)
    wipe_caches(model)
    carry = prefill_prefix(model, rows, chunk_size)
    hidden_b = resume(model, prompt, P, carry, chunk_size)
    state_b = cache_state(model, live, tower_live)
    floor_hidden = (hidden_a.float() - hidden_b.float()).abs().max().item()
    floor_state = max(d for _, d in compare(state_b, state_a))
    floor = max(floor_hidden, floor_state)
    del state_a, state_b
    torch.cuda.empty_cache()
    print(f"\n0 determinism   two identical passes differ by {floor:.3e} "
          f"(hidden {floor_hidden:.3e}, cache {floor_state:.3e})", flush=True)

    # Build the snapshot from a clean prefix prefill.
    wipe_caches(model)
    prefix = prefix_kv.PrefixKV(model, rows)
    prefix.build(chunk_size)
    print(f"  snapshot {prefix.bytes/2**20:.0f} MiB, built in "
          f"{prefix.build_seconds:.2f}s", flush=True)

    failures = 0
    for index, prompt in enumerate(prompts):
        print(f"\n--- chunk {index}: {len(prompt)} rows "
              f"({len(prompt)-P} after the prefix) ---", flush=True)

        # A. faithfulness: one pass, versus snapshot -> wipe -> restore -> resume.
        live, tower_live = len(prompt), len(prompt) - 1
        wipe_caches(model)
        t = time.perf_counter()
        carry = prefill_prefix(model, rows, chunk_size)
        hidden_onepass = resume(model, prompt, P, carry, chunk_size)
        torch.cuda.synchronize(device)
        onepass_seconds = time.perf_counter() - t
        state_onepass = cache_state(model, live, tower_live)

        wipe_caches(model)
        t = time.perf_counter()
        prefix.restore()
        hidden_restored = resume(model, prompt, P, prefix.carry, chunk_size)
        torch.cuda.synchronize(device)
        restored_seconds = time.perf_counter() - t
        state_restored = cache_state(model, live, tower_live)

        hidden_difference = (hidden_onepass.float()
                             - hidden_restored.float()).abs().max().item()
        report = compare(state_onepass, state_restored)
        over = [item for item in report if item[1] > floor]
        worst = max(report, key=lambda item: item[1])
        ok = hidden_difference <= floor and not over
        print(f"A faithfulness  {'PASS' if ok else 'FAIL'}  hidden {hidden_difference:.3e}; "
              f"{len(over)}/{len(report)} cache tensors above the floor; "
              f"worst {worst[0]} {worst[1]:.3e}", flush=True)
        if not ok:
            failures += 1
            for key, difference in sorted(over, key=lambda i: -i[1])[:10]:
                print(f"    {key}: {difference:.3e}", flush=True)

        # B. drift against an ordinary single-pass prefill.
        del state_onepass, state_restored
        torch.cuda.empty_cache()
        wipe_caches(model)
        hidden_ordinary = run_prefill(model, prompt, chunk_size)
        scale = hidden_ordinary.float().abs().max().item()
        drift_abs = (hidden_ordinary.float()
                     - hidden_restored.float()).abs().max().item()
        print(f"B drift         vs --prefill-chunk {chunk_size}: max {drift_abs:.3e} "
              f"on a scale of {scale:.3e} "
              f"({100*drift_abs/max(scale,1e-9):.3f}%)", flush=True)

        # C. cost of the prefill half.
        print(f"C cost          prefix+suffix {onepass_seconds*1000:.0f} ms -> "
              f"restore+suffix {restored_seconds*1000:.0f} ms "
              f"({100*(1-restored_seconds/max(onepass_seconds,1e-9)):.0f}% less)",
              flush=True)

    # D. end to end, the same seed through both paths.
    print(f"\n--- end to end, {args.decode_steps} frames, seed 0 ---", flush=True)
    prompt = prompts[0]
    torch.manual_seed(0)
    t = time.perf_counter()
    frames_base, reason_base = generate_drama3(
        model, prompt, max_new_tokens=args.decode_steps, temperature=1.0,
        top_p=0.9, top_k=4097, prefill_chunk_size=chunk_size)
    base_seconds = time.perf_counter() - t
    torch.manual_seed(0)
    stats: dict = {}
    t = time.perf_counter()
    frames_prefix, reason_prefix = prefix_kv.generate_drama3_prefix(
        model, prompt, prefix, max_new_tokens=args.decode_steps, temperature=1.0,
        top_p=0.9, top_k=4097, prefill_chunk_size=chunk_size, stats=stats)
    prefix_seconds = time.perf_counter() - t
    identical = (frames_base.shape == frames_prefix.shape
                 and bool(torch.equal(frames_base, frames_prefix)))
    matching = 0
    for a, b in zip(frames_base, frames_prefix):
        if not torch.equal(a, b):
            break
        matching += 1
    print(f"baseline {len(frames_base)} frames in {base_seconds:.2f}s ({reason_base}); "
          f"prefix {len(frames_prefix)} frames in {prefix_seconds:.2f}s "
          f"({reason_prefix})", flush=True)
    print(f"identical: {identical}" if identical else
          f"identical: False; first divergence at frame {matching}", flush=True)
    print(f"stats: {stats}", flush=True)

    # E. control. The prefix path will not reproduce the baseline frame for
    # frame, and the obvious worry is that something is actually wrong rather
    # than merely different. So move a chunk boundary WITHOUT touching the
    # prefix logic and see whether the same thing happens: if --prefill-chunk
    # 128 diverges from 1024 at a similar frame under the same seed, then a
    # bf16 hidden state one ulp apart flipping a 4,097-way nucleus draw is
    # just what this model does, and the prefix path is not implicated.
    torch.manual_seed(0)
    frames_128, _ = generate_drama3(
        model, prompt, max_new_tokens=args.decode_steps, temperature=1.0,
        top_p=0.9, top_k=4097, prefill_chunk_size=128)
    control = 0
    for a, b in zip(frames_base, frames_128):
        if not torch.equal(a, b):
            break
        control += 1
    same = (frames_base.shape == frames_128.shape
            and bool(torch.equal(frames_base, frames_128)))
    print(f"E control       --prefill-chunk 128 vs {chunk_size}, same seed: "
          f"{'identical' if same else f'first divergence at frame {control}'}",
          flush=True)

    print(f"\n{'FAILED' if failures else 'PASS'}: {failures} chunk(s) above the "
          f"determinism floor", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
