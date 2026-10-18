"""Narrate a long script with Drama 3. Resumable, per-chunk, fault-tolerant.

Built for unattended multi-hour runs: every chunk is written to its own wav as
soon as it is made, so a crash at chunk 140 of 170 costs one chunk, not the
night. Re-running skips whatever is already on disk.

Three splitting modes, because they fail differently and the comparison is the
point:
  paragraph  blank-line separated, the way the user's own tool feeds it
  auto       upstream split_drama3_text (~300 byte chunks, sentence-aware)
  oneshot    the entire script as a single prompt -- expected to fail; the
             interesting part is HOW
"""
from __future__ import annotations

import argparse, json, os, re, socket, subprocess, sys, time, traceback
from pathlib import Path

import paths  # resolves DRAMA3_FISH / DRAMA3_CKPT; see bin/paths.py
FISH = paths.install()

import torch
import numpy as np
import offload

from fish_speech.models.text2semantic.fish_omni.model import FishOmniModel
from fish_speech.models.text2semantic.fish_omni.generate import (
    build_drama3_prompt, generate_drama3, Drama3PromptTooLongError,
)
from fish_speech.models.text2semantic.fish_omni.chunking import (
    split_drama3_text, Drama3History, DRAMA3_CHUNK_MAX_NEW_TOKENS,
)
from fish_speech.models.text2semantic.inference import (
    load_codec_model, encode_audio, decode_to_audio, resolve_codec_path,
)
from fish_speech.models.text2semantic.errors import InvalidInputError

FRAME_RATE = 44100 / 512 / 4          # 21.53 Hz, verified on real output
SR = 44100
# Measured across several generations: ~16-18 UTF-8 bytes of text per second of
# speech. 16 is the conservative end (it over-estimates duration slightly).
BYTES_PER_SECOND = 16.0


def frame_budget(text: str, cap: int, slack: float = 1.6, floor: int = 48) -> int:
    """Frames to allow for this text.

    Why not just use the cap: a 196-byte paragraph given a 600-frame budget
    spoke for 10s and then emitted 17.9s of digital silence, stopping on
    'length' rather than 'eos'. That wasted 64% of the compute and would put
    an 18-second gap in the output. Budgeting from the text length bounds the
    waste; genuine over-runs are caught by the retry below.
    """
    est = len(text.encode()) / BYTES_PER_SECOND * FRAME_RATE
    return int(max(floor, min(cap, est * slack + 32)))


# Marks that earn a longer pause. The hyphen is deliberately conditional:
# "Convenience - Setting up..." is a dash and should breathe; "post-migration
# fixes" is not a pause at all, and a rule that cannot tell them apart inserts
# gaps mid-word. Both forms appear in the SQL script.
PAUSE_MARKS = re.compile(r"[—–…;:]|(?<=\s)-(?=\s)|(?<=\s)--(?=\s)")


def pause_budget(text: str) -> int:
    """How many deliberate pauses this text is entitled to."""
    return len(PAUSE_MARKS.findall(text))


def clamp_internal_silence(d, sr: int, max_gap: float = 0.5,
                           long_gap: float = 1.2, allowance: int = 0,
                           thresh: float = 0.004):
    """Shorten silence inside a chunk, sparing `allowance` deliberate pauses.

    Trailing-silence trimming only touches the end of a chunk. Measured on the
    first full narration: 104 internal gaps of >=0.6s totalling 134.8s -- 7 %
    of a 32-minute file was dead air, the worst 9.8s.

    But a flat cap is wrong: a pause at an em dash or semicolon is
    performance, and flattening it to 0.5s makes the reading worse. The clamp
    operates on AUDIO while the punctuation is in TEXT, and there is no
    alignment between them, so we cannot say "this gap is the one at the
    dash". Instead we count the marks and let that many of the LONGEST gaps
    keep up to `long_gap`, clamping the rest hard.

    ⚠️ The failure case is a stall that happens to be the longest gap in a
    chunk that also contains a dash -- the stall is preserved and a real pause
    is clamped. The duration check in main() is what catches those chunks.
    """
    import numpy as np
    win = max(1, sr // 20)                       # 50 ms resolution
    n = len(d) // win
    if n < 2:
        return d
    rms = np.sqrt((d[: n * win].reshape(n, win) ** 2).mean(axis=1))
    quiet = rms < thresh

    runs = []                                    # (start, length) in frames
    start = None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            runs.append((start, i - start)); start = None
    if start is not None:
        runs.append((start, n - start))
    if not runs:
        return d

    # the `allowance` longest runs are the candidates for a deliberate pause
    spared = {s for s, _ in sorted(runs, key=lambda r: -r[1])[:max(0, allowance)]}
    keep = np.ones(n, dtype=bool)
    for s, length in runs:
        cap = int((long_gap if s in spared else max_gap) * 20)
        if length > cap:
            keep[s + cap : s + length] = False
    if keep.all():
        return d
    out = d[: n * win].reshape(n, win)[keep].reshape(-1)
    return np.concatenate([out, d[n * win :]])


def speech_end(d, sr: int, thresh: float = 0.004) -> float:
    """Seconds at which audible speech stops, by 0.1s RMS buckets."""
    import numpy as np
    win = max(1, sr // 10)
    rms = np.array([np.sqrt((d[i:i + win] ** 2).mean())
                    for i in range(0, len(d), win)])
    loud = np.where(rms > thresh)[0]
    return float((loud[-1] + 1) / 10) if len(loud) else 0.0



# Wall-clock per startup stage. Requested after a 30-minute run was planned
# and "how long does each part take" could only be answered by subtracting
# loguru timestamps out of a log, which only some of the stages emit.
STAGES: list[tuple[str, float]] = []


class stage:
    """with stage("model load"): ... -- appends (name, seconds) to STAGES.

    CUDA work is asynchronous, so a stage that only queues kernels would
    report near-zero and hand its real cost to whoever synchronises next.
    Synchronise on exit.
    """

    def __init__(self, name: str, device=None) -> None:
        self.name, self.device = name, device

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.device is not None and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        STAGES.append((self.name, time.perf_counter() - self.t0))
        return False


def memory_snapshot(device) -> dict:
    """VRAM and host RAM high-water marks for this process.

    VmHWM is read from /proc rather than psutil (not installed) and is the
    peak resident set, which is what matters here: the experts live in host
    RAM, so a run's real cost is VRAM *and* RAM together.
    """
    import torch
    out = {}
    if torch.cuda.is_available():
        out["vram_allocated_gib"] = round(torch.cuda.memory_allocated(device) / 2**30, 2)
        out["vram_peak_gib"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
        out["vram_reserved_gib"] = round(torch.cuda.memory_reserved(device) / 2**30, 2)
        free, total = torch.cuda.mem_get_info(device)
        out["vram_free_gib"] = round(free / 2**30, 2)
        out["vram_total_gib"] = round(total / 2**30, 2)
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                out["ram_peak_gib"] = round(int(line.split()[1]) / 2**20, 2)
            elif line.startswith("VmRSS:"):
                out["ram_rss_gib"] = round(int(line.split()[1]) / 2**20, 2)
    except OSError:
        pass
    return out

def snap_to_silence(src: Path, seconds: float, search: float = 8.0) -> float:
    """Longest cut at or before `seconds` that lands in silence.

    A reference cut mid-word gives the model a chopped-off word as its only
    example of what audio looks like just before end-of-speech, and the ASR
    transcript then describes audio that stops mid-syllable. Measured on this
    project's clip: a hard 20 s slice cut at RMS 0.149 -- full speech level --
    and the voice audibly drifted. Snapping to 28.2 s, which lands in a 0.55 s
    gap, fixed it by the user's ear.

    In that recording every gap >= 0.3 s was also a sentence boundary, so
    snapping to silence snapped to punctuation for free. That is a property of
    clean read speech, not a guarantee.
    """
    import os, tempfile
    import numpy as np, soundfile as sf
    tmpdir = tempfile.mkdtemp(dir=str(Path(src).parent))
    tmp = Path(tmpdir) / "probe.wav"
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(tmp)],
                       check=True)
        d, sr = sf.read(str(tmp), dtype="float32")
    except Exception:
        return seconds
    finally:
        if tmp.exists():
            tmp.unlink()
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
    win = sr // 20
    n = len(d) // win
    if n < 2:
        return seconds
    rms = np.sqrt((d[: n * win].reshape(n, win) ** 2).mean(axis=1))
    quiet = rms < 0.01
    runs, run_start = [], None
    for i, q in enumerate(quiet):
        if q and run_start is None:
            run_start = i
        elif not q and run_start is not None:
            runs.append((run_start, i))
            run_start = None
    # ⚠️ Close a run that reaches the end of the file. Without this, the loop
    # above only ever records a silence that is followed by more speech, so
    # the trailing silence of a cleanly-ended recording is invisible and the
    # snap falls back to the last INTERIOR gap. Measured on this project's
    # 56.0 s clip: speech to 55.8 s, silence to the end, and the snap chose
    # 51.7 s -- throwing away 4.3 s (8 %) of a reference for no reason.
    # `clamp_internal_silence` already handles this case; this function did
    # not, and the asymmetry is the whole bug.
    if run_start is not None:
        runs.append((run_start, n))
    best = None
    for start, end in runs:
        trailing = end == n
        # An interior gap needs 0.3 s to be confidently a pause rather than a
        # stop consonant, because we are cutting in the MIDDLE of it and have
        # to not clip the speech on either side. The end of the file only has
        # to prove the speaker finished, which 0.1 s of silence does. Holding
        # the tail to the interior rule cost 4.3 s of this project's 56.0 s
        # clip: speech to 55.8 s, 0.2 s of silence, and the snap fell back to
        # an interior gap at 51.7 s.
        if (end - start) < (2 if trailing else 6):
            continue
        # A trailing run is a cut at the END of the clip, not at its midpoint
        # -- there is no speech after it to protect, and the midpoint would
        # throw away half the trailing audio for nothing.
        where = (len(d) / sr) if trailing else (start + end) / 2 * 0.05
        if seconds - search <= where <= seconds:
            best = where
    return best if best is not None else seconds


def to_pcm16(src: Path, dst: Path, seconds: float | None) -> None:
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src)]
    if seconds:
        cmd += ["-t", str(seconds)]
    subprocess.run(cmd + ["-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(dst)],
                   check=True)


def transcribe(path: Path) -> str:
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock = paths.stt_socket()
        if sock is None:
            return ("<stt unavailable: set DRAMA3_STT to a socket, "
                    "or pass --ref-text>")
        s.connect(sock)
        s.sendall((str(path.resolve()) + "\n").encode())
        buf = b""
        while True:
            c = s.recv(65536)
            if not c or b"\n" in buf:
                break
            buf += c
        s.close()
        return buf.decode().strip()
    except Exception as exc:                       # STT is a nicety, never fatal
        return f"<stt failed: {exc}>"


def paragraphs(text: str, max_bytes: int, split_lines: bool = True) -> list[str]:
    """Blank-line paragraphs, with overlong ones sub-split.

    The user's tool sends one paragraph per request. Real documents contain
    paragraphs far past what one generation can hold (a markdown table in the
    SQL script is 1692 bytes ~ 100s of speech against a 27.9s budget), so an
    honest emulation still has to sub-split those -- and say that it did.
    """
    out = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        for piece in (_lines(para) if split_lines else [para]):
            if len(piece.encode()) <= max_bytes:
                out.append(piece)
            else:
                out.extend(split_drama3_text(piece, max_bytes))
    return out


def _lines(para: str) -> list[str]:
    """Split a paragraph at newlines that precede a non-sentence line.

    Measured failure: a paragraph holding a bullet list speaks ONLY its first
    line and silently drops the rest --

        'Any potential migration blockers\n Any breaking changes...\n Azure features...'
        -> "any potential migration blockers."

    Four of 192 chunks did this, all the same shape. Isolated, each line
    generates cleanly, so the fix is to stop gluing them together. Lines that
    end in terminal punctuation are left joined, since ordinary multi-sentence
    prose is fine and splitting it would multiply the per-chunk prefill cost.
    """
    lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
    if len(lines) < 2:
        return lines or [para]
    out, buf = [], []
    for ln in lines:
        buf.append(ln)
        if ln[-1] in ".!?;:":          # a finished sentence may keep company
            continue
        out.append(" ".join(buf)); buf = []
    if buf:
        out.append(" ".join(buf))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="Drama 3 snapshot dir; found via DRAMA3_CKPT "
                         "or HF_HOME when omitted")
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--mode", choices=("paragraph", "auto", "oneshot"),
                    default="paragraph")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ref", default="source clips/test audio.mp3")
    # 30 with snapping lands at 28.2 s on this clip, which the user judged
    # clearly better than the old hard 20 s cut. 56 s was marginally better
    # again but costs ~10 % in prefill on every chunk.
    ap.add_argument("--ref-seconds", type=float, default=30.0)
    ap.add_argument("--no-snap", action="store_true",
                    help="cut at exactly --ref-seconds, mid-word or not")
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--para-max-bytes", type=int, default=400,
                    help="sub-split paragraphs longer than this")
    ap.add_argument("--max-new-tokens", type=int, default=DRAMA3_CHUNK_MAX_NEW_TOKENS)
    ap.add_argument("--copy-mode", choices=("gather", "direct", "graphstub"),
                    default="direct")
    ap.add_argument("--codec-cpu", action="store_true",
                    help="run the codec on CPU; frees 1.74 GiB VRAM for cache")
    ap.add_argument("--pin-experts", action="store_true",
                    help="page-lock expert storage: ~1.47x faster DMA, costs "
                         "~45s at load and locks 30 GiB of RAM")
    ap.add_argument("--slice-lm-head", action="store_true",
                    help="keep only the 4,097 sampleable output rows (bit-identical)")
    ap.add_argument("--offload-embed", action="store_true",
                    help="input embedding table to host RAM, decode's 4,097 "
                         "rows resident; frees ~0.94 GiB for more experts")
    ap.add_argument("--prefill-chunk", type=int, default=128,
                    help="rows per prefill batch (upstream default 128)")
    # The system turn + reference block is byte-identical in every chunk and
    # is 84% of a median prompt, so it was being re-read once per chunk.
    # Prefill it once and restore the cached state instead. See bin/prefix_kv.py.
    ap.add_argument("--prefix-kv", action="store_true",
                    help="prefill the system+reference prefix once and reuse it")
    # Opt-in: the probe syncs the host once per layer per prefill batch to
    # read the device cache's slot table, which is a real cost. Diagnostic
    # only -- do not leave it on for a production narration.
    ap.add_argument("--no-hybrid-gather", action="store_true",
                    help="prefill re-fetches experts already resident in the "
                         "device cache (the old behaviour; for A/B only)")
    ap.add_argument("--prefill-probe", default=None, metavar="PATH.json",
                    help="record per-layer prefill gather/compute split and "
                         "device-cache overlap (slows prefill; see task #25)")
    ap.add_argument("--vram-reserve", type=float, default=2.0,
                    help="GiB of VRAM to keep free FOR OTHER PROGRAMS while "
                         "generating. The cache takes what is left after this "
                         "plus ~1.5 GiB of our own activation headroom. "
                         "Set 0 to disable and use --cache-experts instead.")
    ap.add_argument("--cache-experts", type=int, default=0,
                    help="experts held in VRAM per layer (0 = off); each costs 3.001 MiB "
                         "x 40 layers, so 80 ~ 9.4 GiB")
    ap.add_argument("--dev-cache", action="store_true",
                    help="device-resident cache: no host sync, graph-capturable")
    ap.add_argument("--fetch-backend", choices=("triton", "cpp"), default="cpp")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="run exactly max-new-tokens steps regardless of EOS; timing only")
    ap.add_argument("--cuda-graph", choices=("0", "1"), default="0",
                    help="1 only works with a sync-free MoE path (--copy-mode graphstub)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="only first N chunks")
    ap.add_argument("--no-budget", action="store_true",
                    help="always allow max-new-tokens (the old wasteful behaviour)")
    ap.add_argument("--no-split-lines", action="store_true",
                    help="keep bullet lists glued into one chunk (reproduces the bug)")
    ap.add_argument("--max-gap", type=float, default=0.5,
                    help="longest ordinary silence kept inside a chunk, seconds")
    ap.add_argument("--long-gap", type=float, default=1.2,
                    help="cap for pauses earned by punctuation (; : -- em/en dash)")
    ap.add_argument("--no-trim", action="store_true",
                    help="keep trailing silence instead of trimming it")
    ap.add_argument("--temperature", type=float, default=1.0)
    # ⚠️ top_p was never exposed before, and that is why 24 generations of
    # temperature sweeping found nothing: every one ran at top_p 0.7, where
    # the nucleus filter drops the crossing token and EOS becomes
    # unreachable. Measured on this box, EOS holding 25 % of the mass was
    # drawn 0/4000 times at top_p 0.7 for every temperature tested.
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=4097,
                    help="4097 = eos + all semantic codes, i.e. uncapped")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="redraw a chunk whose duration implies dropped or repeated text")
    ap.add_argument("--min-ratio", type=float, default=0.55)
    ap.add_argument("--max-ratio", type=float, default=2.2)
    ap.add_argument("--stt-every", type=int, default=0,
                    help="transcribe every Nth chunk as a drift check (0 = off)")
    args = ap.parse_args()
    # Falls back to HF_HOME discovery, and raises with an
    # instruction rather than a TypeError deep in from_pretrained.
    args.checkpoint = args.checkpoint or str(paths.checkpoint())

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = args.cuda_graph
    offload.COPY_MODE = args.copy_mode
    offload.CACHE_EXPERTS = args.cache_experts   # read during materialize
    offload.HYBRID_GATHER = not args.no_hybrid_gather
    offload.TRACE.enabled = False
    offload.install_offload()
    import fish_speech.models.text2semantic.fish_omni.backbone as bb
    bb.FishOmniBackbone.materialize = lambda self, devices: (
        offload.materialize_offloaded(self, device))

    text = Path(args.text_file).read_text(encoding="utf-8").strip()
    if args.mode == "paragraph":
        chunks = paragraphs(text, args.para_max_bytes, not args.no_split_lines)
    elif args.mode == "auto":
        chunks = split_drama3_text(text)
    else:
        chunks = [text]
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"mode={args.mode}  {len(chunks)} chunk(s)  "
          f"{sum(len(c.encode()) for c in chunks):,} bytes", flush=True)

    t_load = time.perf_counter()
    with stage("model load", device):
        model = FishOmniModel.from_pretrained(
            args.checkpoint, devices=[str(device)], tower_device=str(device),
            dtype=torch.bfloat16)
    if args.copy_mode == "graphstub":
        offload.init_graph_stub(model, device)
        print("!! graphstub: TIMING ONLY, the audio is garbage", flush=True)
    if args.pin_experts:
        with stage("pin experts", device):
            gib, secs = offload.pin_experts(model, device)
        print(f"pinned {gib:.1f} GiB of experts in {secs:.0f}s", flush=True)
    if args.slice_lm_head:
        with stage("slice lm_head", device):
            freed = offload.slice_lm_head(model, device)
        print(f"lm_head sliced to the 4,097 reachable ids; freed {freed:.2f} GiB", flush=True)
    if args.offload_embed:
        with stage("offload embed", device):
            freed = offload.offload_embed(model, device)
        print(f"embed_tokens moved to host RAM; freed {freed:.2f} GiB", flush=True)
    # The codec turns the model's codes into a waveform. On the GPU it holds
    # 1.74 GiB that could instead cache ~14 more experts per layer. Measured
    # on CPU it decodes at 0.64-0.73x realtime, so while the GPU is generating
    # the next chunk at RTF ~1.0 the decode hides completely.
    #
    # ⚠️ The headroom shrinks as generation gets faster. At RTF 0.64 the CPU
    # becomes the bottleneck and caps the pipeline. Worth it while generation
    # is near 1.0; revisit if it drops much below.
    codec_device = "cpu" if args.codec_cpu else device
    with stage("codec load", device):
        codec = load_codec_model(resolve_codec_path(Path(args.checkpoint)),
                                 device=codec_device,
                                 precision=torch.float32 if args.codec_cpu
                                 else torch.bfloat16)
    if args.codec_cpu:
        print(f"codec on CPU (frees ~1.74 GiB VRAM)", flush=True)
    mem_load = memory_snapshot(device)
    print(f"loaded in {time.perf_counter()-t_load:.1f}s; "
          f"VRAM {mem_load.get('vram_allocated_gib')} GiB alloc / "
          f"{mem_load.get('vram_peak_gib')} peak / {mem_load.get('vram_total_gib')} total; "
          f"RAM {mem_load.get('ram_peak_gib')} GiB peak", flush=True)

    # --cache-experts wins if given explicitly; otherwise size from free VRAM.
    # Running both would allocate two caches and waste the difference.
    if args.dev_cache:
        import devcache
        devcache.FETCH_BACKEND = args.fetch_backend
        # size it the same way as the python cache, then hand it the slots
        cap = offload.size_cache_to_free_vram(model, device, args.vram_reserve,
                                              ) if args.vram_reserve else args.cache_experts
        for layer in model.backbone.model.layers:
            e = getattr(layer.mlp, "experts", None)
            if e is not None and getattr(e, "_cap", 0):
                e._cap = 0            # disable the python cache, free its buffers
                for n_ in ("_cache_w13", "_cache_w2", "_cache_s13", "_cache_s2"):
                    if hasattr(e, n_):
                        delattr(e, n_)
        torch.cuda.empty_cache()
        cap = devcache.install_dev_cache(model, device, cap)
        print(f"device cache: {cap} layers, backend={args.fetch_backend}", flush=True)
    elif args.vram_reserve and not args.cache_experts:
        cap = offload.size_cache_to_free_vram(model, device, args.vram_reserve)
        print(f"expert cache sized to free VRAM: {cap}/layer "
              f"({cap*40*3.001/1024:.2f} GiB)", flush=True)

    prompt_texts, prompt_tokens = (), ()
    ref = Path(args.ref)
    if ref.is_file():
      with stage("reference encode", device):
          wav = out_dir / "reference.wav"
          secs = args.ref_seconds or None
          if secs and not args.no_snap:
              snapped = snap_to_silence(ref, secs)
              if abs(snapped - secs) > 0.05:
                  print(f"reference: snapped {secs}s -> {snapped:.1f}s "
                        f"(cuts in silence, not mid-word)", flush=True)
              secs = snapped
          to_pcm16(ref, wav, secs)
          ref_text = args.ref_text or transcribe(wav)
          if ref_text and not ref_text.startswith("<stt failed"):
              prompt_texts = (ref_text,)
              prompt_tokens = (encode_audio(wav, codec, codec_device),)
              print(f"reference: {prompt_tokens[0].shape[1]} codes "
                    f"({prompt_tokens[0].shape[1]/FRAME_RATE:.1f}s)", flush=True)
          else:
              print(f"no usable reference transcript ({ref_text!r})", flush=True)

    # Built here, after the reference is encoded and after the expert cache
    # has been sized: the snapshot is ~90 MiB for a 28 s reference and comes
    # out of the 1.5 GiB of activation headroom offload.py already holds
    # back, NOT out of --vram-reserve.
    prefix = None
    if args.prefix_kv:
        if not prompt_tokens:
            print("--prefix-kv needs a reference; running without it", flush=True)
        else:
            import prefix_kv
            with stage("prefix KV build", device):
                prefix = prefix_kv.make_prefix(model, prompt_texts,
                                               prompt_tokens, args.prefill_chunk)
            print(f"prefix KV: {prefix.P} rows reused per chunk, "
                  f"snapshot {prefix.bytes/2**20:.0f} MiB, built in "
                  f"{prefix.build_seconds:.2f}s", flush=True)

    # Installed after the device cache exists, so `resident` reads a real
    # slot table rather than an empty one.
    probe = None
    if args.prefill_probe:
        import prefill_probe as probe
        probe.install(model)
        print(f"prefill probe on -> {args.prefill_probe} (slows prefill)",
              flush=True)

    history = Drama3History()
    manifest_path = out_dir / "manifest.json"
    records = []
    if manifest_path.is_file():
        records = json.loads(manifest_path.read_text()).get("chunks", [])
    done = {r["index"]: r for r in records if r.get("ok")}
    print(f"resuming: {len(done)} chunk(s) already done", flush=True)

    started = time.perf_counter()
    for index, chunk in enumerate(chunks):
        if probe is not None:
            probe.CHUNK = index
        path = out_dir / f"chunk_{index:04d}.wav"
        if index in done and path.is_file():
            # Rebuild history from disk so the voice stays continuous.
            try:
                import soundfile as sf
                prev, _ = sf.read(str(path), dtype="float32")
                history.push(chunk, encode_audio(path, codec, codec_device).cpu())
            except Exception:
                pass
            continue

        rec = {"index": index, "bytes": len(chunk.encode()),
               "text": chunk[:200], "ok": False}
        t0 = time.perf_counter()
        try:
            prompt = None
            for turns in history.candidates():
                try:
                    prompt = build_drama3_prompt(
                        chunk, model.tokenizer, model.config,
                        prompt_texts, prompt_tokens, history=turns)
                    rec["history_turns"] = len(turns)
                    break
                except Drama3PromptTooLongError as exc:
                    rec["too_long"] = str(exc)
                    continue
            if prompt is None:
                raise Drama3PromptTooLongError(rec.get("too_long", "no fitting prompt"))
            rec["prompt_rows"] = len(prompt)

            # Sampling drops content stochastically: measured ~30% of short
            # multi-sentence chunks either stopped early and stalled into
            # silence, or babbled for 3x the expected duration. Both show up
            # as a speech duration far from what the text implies, so retry on
            # that rather than trusting a single draw.
            expected = len(chunk.encode()) / BYTES_PER_SECOND
            check = expected >= 1.5          # too noisy to judge below this
            attempts = []
            best = None
            for attempt in range(args.max_attempts):
                budget = (args.max_new_tokens if args.no_budget
                          else frame_budget(chunk, args.max_new_tokens))
                if attempt and best is not None:
                    budget = args.max_new_tokens      # be generous on retries
                # prefill_chunk_size defaults to 128 upstream. With experts
                # offloaded, each prefill batch fetches every expert it routes
                # to, and that count saturates near 256 regardless of batch
                # size -- so small batches re-fetch the same experts over and
                # over. Bigger batches amortise it.
                # Timing runs with a stub MoE cannot rely on generation
                # behaviour: zeroed stub weights made graph capture NaN,
                # random ones made the model emit EOS immediately. Disabling
                # the EOS comparison forces exactly max_new_tokens steps so
                # both arms of an A/B do identical work.
                #
                # ⚠️ It MUST be restored: build_drama3_prompt validates
                # tokenizer.token_to_id("<|im_end|>") == config.eos_token_id,
                # so leaving it at -1 fails every subsequent chunk with
                # "tokenizer/config mismatch".
                saved_eos = model.config.eos_token_id
                if args.ignore_eos:
                    model.config.eos_token_id = -1
                try:
                    if prefix is not None:
                        # Falls back to a full prefill on its own if the
                        # prompt does not start with the prefix rows. That
                        # must never happen silently, so it is recorded.
                        pstats: dict = {}
                        f, rsn = prefix_kv.generate_drama3_prefix(
                            model, prompt, prefix,
                            temperature=args.temperature,
                            top_p=args.top_p, top_k=args.top_k,
                            max_new_tokens=budget,
                            prefill_chunk_size=args.prefill_chunk,
                            stats=pstats)
                        rec["prefix"] = pstats
                    else:
                        f, rsn = generate_drama3(
                            model, prompt, temperature=args.temperature,
                            top_p=args.top_p, top_k=args.top_k,
                            max_new_tokens=budget,
                            prefill_chunk_size=args.prefill_chunk)
                finally:
                    model.config.eos_token_id = saved_eos
                if not len(f):
                    attempts.append({"attempt": attempt, "error": "no frames"})
                    continue
                t_codec = time.perf_counter()
                cd = f[:, 1:].T.to(codec_device)
                au = decode_to_audio(cd, codec).float().cpu().numpy()
                # Counted per attempt and summed: a retried chunk decodes
                # more than once, and hiding that would make the codec look
                # cheaper than it is on exactly the chunks that cost most.
                rec["codec_seconds"] = round(
                    rec.get("codec_seconds", 0.0)
                    + time.perf_counter() - t_codec, 3)
                end_s = speech_end(au, SR)
                ratio = end_s / expected if expected else 1.0
                good = (not check) or (args.min_ratio <= ratio <= args.max_ratio)
                attempts.append({"attempt": attempt, "frames": len(f),
                                 "speech": round(end_s, 2),
                                 "expected": round(expected, 2),
                                 "ratio": round(ratio, 2), "stop": rsn,
                                 "accepted": good})
                # Keep whichever draw lands closest to the expected duration,
                # so even an all-bad chunk yields the least-bad audio.
                score = abs(1.0 - ratio)
                if best is None or score < best[0]:
                    best = (score, f, rsn, cd, au, end_s)
                if good:
                    break
            if best is None:
                raise RuntimeError("no frames generated in any attempt")
            _, frames, reason, codes, audio, end = best
            rec["attempts"] = attempts
            rec["accepted"] = attempts[-1].get("accepted", False) if attempts else False
            cut_off = reason == "length" and (len(audio) / SR - end) < 0.5

            rec["raw_seconds"] = round(len(audio) / SR, 2)
            rec["trailing_silence"] = round(len(audio) / SR - end, 2)
            if not args.no_trim and end > 0:
                keep = min(len(audio), int((end + 0.15) * SR))
                audio = audio[:keep]
                before = len(audio)
                audio = clamp_internal_silence(
                    audio, SR, args.max_gap, args.long_gap,
                    pause_budget(chunk))
                rec["gap_trimmed_s"] = round((before - len(audio)) / SR, 2)
            import soundfile as sf
            sf.write(str(path), audio, SR, subtype="PCM_16")
            history.push(chunk, codes.cpu())

            rec.update(ok=True, frames=len(frames), finish_reason=reason,
                       seconds=len(audio) / SR,
                       gen_seconds=time.perf_counter() - t0)
            rec["rtf"] = rec["gen_seconds"] / max(rec["seconds"], 1e-9)
            rec["truncated"] = bool(cut_off)
            if args.stt_every and index % args.stt_every == 0:
                tmp = out_dir / "_stt.wav"
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path),
                                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                                str(tmp)], check=True)
                rec["stt"] = transcribe(tmp)
                tmp.unlink(missing_ok=True)
        except Exception as exc:
            rec.update(ok=False, error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc()[-1500:],
                       gen_seconds=time.perf_counter() - t0)
            print(f"  chunk {index}: FAILED {rec['error']}", flush=True)

        records = [r for r in records if r["index"] != index] + [rec]
        records.sort(key=lambda r: r["index"])
        manifest_path.write_text(json.dumps(
            {"mode": args.mode, "memory_after_load": mem_load,
             "memory_peak": memory_snapshot(device), "chunks": records}, indent=1))
        if rec["ok"]:
            elapsed = time.perf_counter() - started
            did = sum(1 for r in records if r.get("ok"))
            audio_s = sum(r.get("seconds", 0) for r in records if r.get("ok"))
            eta = (len(chunks) - index - 1) * (elapsed / max(did, 1))
            flag = "  !! TRUNCATED" if rec["truncated"] else ""
            if not rec.get("accepted", True):
                flag += f"  !! DURATION OFF ({len(rec.get('attempts', []))} tries)"
            elif len(rec.get("attempts", [])) > 1:
                flag += f"  (retried x{len(rec['attempts'])-1})"
            print(f"  [{index+1}/{len(chunks)}] {rec['frames']:4d}f "
                  f"{rec['seconds']:5.1f}s rtf {rec['rtf']:4.2f} "
                  f"rows {rec['prompt_rows']:4d}  audio {audio_s/60:.1f}min  "
                  f"eta {eta/60:.0f}min{flag}", flush=True)

    if probe is not None:
        probe.save(Path(args.prefill_probe))

    STAGES.append(("generation (all chunks)", time.perf_counter() - started))

    # --- stitch -----------------------------------------------------------
    import soundfile as sf
    pieces = []
    for r in sorted(records, key=lambda r: r["index"]):
        p = out_dir / f"chunk_{r['index']:04d}.wav"
        if r.get("ok") and p.is_file():
            pieces.append(sf.read(str(p), dtype="float32")[0])
    if pieces:
        full = out_dir / "full.wav"
        sf.write(str(full), np.concatenate(pieces), SR, subtype="PCM_16")
        total_audio = sum(len(p) for p in pieces) / SR
        total_gen = sum(r.get("gen_seconds", 0) for r in records)
        ok = sum(1 for r in records if r.get("ok"))
        trunc = sum(1 for r in records if r.get("truncated"))
        fail = sum(1 for r in records if not r.get("ok"))
        print(f"\n=== {args.mode} ===")
        print(f"  {ok} ok, {fail} failed, {trunc} truncated of {len(chunks)}")
        print(f"  audio {total_audio/60:.2f} min, compute {total_gen/60:.2f} min, "
              f"RTF {total_gen/max(total_audio,1e-9):.2f}")
        mem = memory_snapshot(device)
        h, mi = offload.STATS["hits"], offload.STATS["misses"]
        # A cache that was allocated but never consulted costs VRAM and gives
        # nothing, and looks exactly like a correct run that is merely slower.
        # That shipped once: --vram-reserve allocated 8.32 GiB and the
        # dispatch guard skipped it, and the only symptom was a missing line.
        # ⚠️ Must check BOTH caches. `_cap` is the python cache, which
        # --dev-cache deliberately zeroes when it installs `_dc_cap`, so
        # checking only `_cap` printed "a cache was requested but none was
        # allocated" on every single --dev-cache run -- including the ones
        # where a 7.27 GiB cache was working perfectly.
        def caps(attribute):
            return [getattr(l.mlp.experts, attribute, 0)
                    for l in model.backbone.model.layers
                    if getattr(l.mlp, "experts", None) is not None]
        python_cache = any(caps("_cap"))
        # The device cache keeps its counters on the GPU (that is the point
        # of it -- no host sync), so STATS["hits"] stays at zero for a
        # perfectly working --dev-cache run. Report its capacity instead of
        # a hit rate it does not collect, rather than calling it a bug.
        device_cache = max(caps("_dc_cap") or [0])
        if python_cache and h + mi == 0:
            print("  !! EXPERT CACHE WAS ALLOCATED BUT NEVER USED -- VRAM spent "
                  "for nothing. This is a bug, not a slow run.")
        elif h + mi:
            print(f"  expert cache: {100*h/(h+mi):.1f}% hit ({h:,}/{h+mi:,})")
        elif device_cache:
            print(f"  device cache: {device_cache} experts/layer "
                  f"({device_cache*40*3.001/1024:.2f} GiB); decode hit rate is "
                  f"counted on device, use --prefill-probe to read it")
        gh, gm = offload.STATS["gather_hits"], offload.STATS["gather_misses"]
        if gh + gm:
            print(f"  prefill gather: {100*gh/(gh+gm):.1f}% served from VRAM "
                  f"({gh:,}/{gh+gm:,}); "
                  f"{gm*3.001/1024:.1f} GiB over PCIe instead of "
                  f"{(gh+gm)*3.001/1024:.1f}")
        elif args.cache_experts or args.vram_reserve:
            print("  !! a cache was requested but none was allocated")
        print(f"  wrote {full}")
        print(f"  VRAM {mem.get('vram_peak_gib')} GiB peak of "
              f"{mem.get('vram_total_gib')} | RAM {mem.get('ram_peak_gib')} GiB peak")

        # --- where the wall clock went --------------------------------
        done = [r for r in records if r.get("ok")]
        prefill = sum(r.get("prefix", {}).get("prefill_seconds", 0.0) for r in done)
        codec = sum(r.get("codec_seconds", 0.0) for r in done)
        gen = sum(r.get("gen_seconds", 0.0) for r in done)
        wall = sum(secs for _, secs in STAGES)
        print(f"\n  --- stages ({wall/60:.1f} min wall) ---")
        for label, secs in STAGES:
            print(f"  {label:<26s} {secs:8.1f}s {100*secs/max(wall,1e-9):5.1f}%")
        # Inside generation, split out the parts that are not decode. The
        # remainder is decode plus per-chunk python; prefill and codec are
        # measured directly, so the residual is named for what it is rather
        # than asserted to be pure decode.
        if gen:
            print(f"  --- inside generation ({gen/60:.1f} min of chunk time) ---")
            print(f"  {'prefill (prompt)':<26s} {prefill:8.1f}s "
                  f"{100*prefill/gen:5.1f}%")
            print(f"  {'codec (codes -> audio)':<26s} {codec:8.1f}s "
                  f"{100*codec/gen:5.1f}%")
            rest = gen - prefill - codec
            print(f"  {'decode + per-chunk rest':<26s} {rest:8.1f}s "
                  f"{100*rest/gen:5.1f}%")
            frames = sum(r.get("frames", 0) for r in done)
            if frames:
                print(f"  {frames:,} frames, {1000*rest/frames:.2f} ms/frame "
                      f"of decode+rest")


if __name__ == "__main__":
    main()
