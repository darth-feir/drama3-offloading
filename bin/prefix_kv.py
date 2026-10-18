"""Reuse the KV state of the unchanging prompt prefix across chunks.

WHY
---
Measured over 234 single-attempt chunks of the real script, generation time
splits as:

    decode  (producing speech)   25.06 min   80.3%   36.79 ms/frame
    prefill (re-reading prompt)   5.97 min   19.1%    1.91 ms/row
    fixed per chunk               0.19 min    0.6%

Nearly all of that prefill is waste. Every prompt starts with the same
system turn and the same reference block -- 608 rows for a 28.2 s reference
against a 724-row median prompt, **84 %**, byte-identical in all 235 chunks.
We were re-reading the same 28 seconds of audio 235 times.

Prefill the prefix once, snapshot every piece of cache state it produced, and
restore that snapshot before each chunk instead of recomputing it.

The second-order effect is the bigger one. Reference length currently trades
against speed (56 s cost ~10 % on every chunk), so quality was being rationed
to pay for it. Once the prefix is paid once, it stops being a per-chunk cost:

    90 s reference, recomputed:  1938 rows x 1.91 ms x 235 = 14.5 min
    90 s reference, reused:      1938 rows x 1.91 ms x   1 =  3.7 s

WHAT HAS TO BE SNAPSHOT
-----------------------
Three kinds of state, and missing any one of them silently produces fluent
audio in the wrong voice rather than an error:

  * `FullAttention.k_cache` / `v_cache` -- 10 of 40 backbone layers. Indexed
    by absolute position, so rows [0, P) stay valid when the suffix changes.
  * `GatedDeltaNet.conv_state` + `recurrent_state` -- the other 30 layers.
    These are *recurrent*: there is no position index, just one state that
    has absorbed every row so far. Small (48 KiB + 2.0 MiB per layer) but
    mandatory -- a stale recurrent state is the one failure mode that does
    not look like a failure.
  * `TowerAttention.k_cache` / `v_cache` -- 13 tower layers (4 shared +
    9 residual), again positional.

Plus one row of backbone output: `hidden[P-1]`.

THE OFF-BY-ONE THAT MATTERS
---------------------------
`tower.prefill` teacher-forces row p with frame p+1 as its label. Row P-1's
label is frames[P], which is the first row of the *chunk-specific* text, so
row P-1 cannot be towered during the prefix build. The snapshot therefore
holds tower rows [0, P-1) and carries `hidden[P-1]` forward; the resumed
prefill tower-fills row P-1 once it knows frames[P].

Get this wrong by one and the tower sees the reference's last frame paired
with itself. It still generates.

NUMERICS
--------
Resuming at row P changes where the GDN chunk boundaries fall, so output is
not bit-identical to a single-pass prefill -- the same is already true of
`--prefill-chunk 128` versus `1024`. What *is* bit-identical is a full
prefill driven with the same boundaries, and `bin/test_prefix_kv.py` checks
exactly that, which isolates "did restore reproduce the state" from "does
boundary placement move the last bit".
"""

from __future__ import annotations

import hashlib
import inspect
import sys
import time
from pathlib import Path

import paths  # resolves DRAMA3_FISH / DRAMA3_CKPT; see bin/paths.py
FISH = paths.install()

import torch
from loguru import logger

from fish_speech.models.text2semantic.fish_omni import generate as gen_mod
from fish_speech.models.text2semantic.fish_omni.generate import (
    DRAMA3_MESSAGE_END,
    DRAMA3_PREFILL_CHUNK_SIZE,
    DRAMA3_REFERENCE_INSTRUCTION,
    DRAMA3_TTS_INSTRUCTION,
    SPEAKER_TAG_PATTERN,
    UNICODE_WHITE_SPACE,
    Drama3DecodeGraphs,
    build_drama3_prompt,
    ensure_drama3_caches,
    generate_drama3,
    with_speaker_tag,
)
from fish_speech.models.text2semantic.sampling import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
)

# `prefix_rows` below reimplements the first half of `build_drama3_prompt`, and
# `generate_drama3_prefix` is a fork of `generate_drama3` with only the prefill
# loop changed. Both are copies, so both go stale silently if upstream moves.
# These hashes make that loud. Re-read the upstream function, re-derive the
# fork, then update the hash -- do not just bump it.
UPSTREAM_HASHES = {
    "build_drama3_prompt": "8c7cffb118c2b161",
    "generate_drama3": "2e21577ec4f92ddb",
}


def _source_hash(function) -> str:
    source = inspect.getsource(function).encode()
    return hashlib.sha256(source).hexdigest()[:16]


def check_upstream_unchanged(strict: bool = False) -> list[str]:
    """Warn (or raise) if the functions this module forked have changed."""
    drifted = []
    for name, expected in UPSTREAM_HASHES.items():
        actual = _source_hash(getattr(gen_mod, name))
        if actual != expected:
            drifted.append(f"{name}: {expected} -> {actual}")
    if drifted:
        message = (
            "prefix_kv forked upstream code that has since changed: "
            + "; ".join(drifted)
            + ". Re-derive the fork before trusting its output."
        )
        if strict:
            raise RuntimeError(message)
        logger.warning(message)
    return drifted


def prefix_rows(tokenizer, config, prompt_texts, prompt_tokens) -> torch.Tensor:
    """The prompt rows that do not depend on the chunk or the history.

    Mirrors `build_drama3_prompt` lines 139-155: the system turn, the
    instruction, every reference block, and the closing <|im_end|>. History
    turns and the current user turn come after this and vary per chunk, so the
    prefix ends here.

    Chunk 0 is tempting to add: `Drama3History` pins it and emits it as the
    *first* history turn of every later prompt, so the rows would still line
    up. Rejected on the arithmetic -- the reference already covers 734 of a
    ~770-row prompt, leaving the pinned turn worth a few per cent, and it only
    holds while the prompt is short enough that `candidates()` has not dropped
    it. If it is ever added, `PrefixKV.applies_to` is what makes that drop
    safe: a prompt that no longer starts with these rows falls back to a full
    prefill instead of resuming from someone else's state.

    Each literal is tokenized on its own, exactly as upstream does -- tokenizing
    the joined string changes ids at the seams, which would make the prefix
    disagree with the real prompt and trip the equality guard in
    `PrefixKV.applies_to`.
    """
    num_codebooks = config.audio_decoder.num_codebooks
    segments: list[str | torch.Tensor] = [
        "<|im_start|>system\n",
        DRAMA3_REFERENCE_INSTRUCTION if prompt_tokens else DRAMA3_TTS_INSTRUCTION,
    ]
    for index, (transcript, codes) in enumerate(zip(prompt_texts, prompt_tokens), 1):
        tagged = with_speaker_tag(transcript)
        body = tagged.lstrip(UNICODE_WHITE_SPACE)
        tag = SPEAKER_TAG_PATTERN.match(body)
        leading = tagged[: len(tagged) - len(body)]
        transcript = tag.group(0) + leading + body[tag.end() :]
        segments += [f"\n\n[Reference {index}]\nText: {transcript}\nSpeech: ", codes]
    segments.append(DRAMA3_MESSAGE_END)

    rows = []
    for segment in segments:
        if isinstance(segment, str):
            ids = torch.tensor(tokenizer.encode(segment).ids, dtype=torch.long)
            text_rows = torch.zeros((len(ids), 1 + num_codebooks), dtype=torch.long)
            text_rows[:, 0] = ids
            rows.append(text_rows)
        else:
            codes = segment.to(device="cpu", dtype=torch.long)
            main = codes[:1] + config.semantic_start_token_id
            rows.append(torch.cat((main, codes)).T)
    return torch.cat(rows)


class PrefixKV:
    """Prefill the shared prefix once; restore its state before each chunk.

    Held on the GPU: the whole snapshot is ~90 MB for a 28 s reference and
    ~155 MB for 90 s, against the 1.5 GiB of activation headroom
    `size_cache_to_free_vram` already holds back. It does NOT come out of the
    user's --vram-reserve. Restoring is ~53 device-to-device copies, well
    under a millisecond, against the 1.16 s of prefill it replaces.
    """

    def __init__(self, model, rows: torch.Tensor) -> None:
        self.model = model
        self.rows = rows.cpu()
        self.P = len(rows)
        if self.P < 2:
            raise ValueError(f"a prefix needs at least 2 rows; got {self.P}")
        self.carry: torch.Tensor | None = None
        self.backbone_kv: list[tuple] = []
        self.gdn: list[tuple] = []
        self.tower_kv: list[tuple] = []
        self.build_seconds = 0.0
        self.bytes = 0
        self.restores = 0
        # Caches are reallocated when `ensure_drama3_caches` grows them or when
        # graph capture drops and rebuilds them. Either invalidates a snapshot
        # taken against the old buffers, so remember which buffers we filled.
        self._buffer_ids: tuple = ()

    # ---------------------------------------------------------------- state

    def _cache_modules(self):
        backbone, tower = self.model.backbone, self.model.audio_decoder
        full, gdn = [], []
        for layer in backbone.model.layers:
            if layer.layer_type == "full_attention":
                full.append(layer.self_attn)
            else:
                gdn.append(layer.linear_attn)
        towers = [
            layer.transformer_layer.attention
            for layer in (*tower.shared_layers, *tower.residual_layers)
        ]
        return full, gdn, towers

    def _buffer_identity(self) -> tuple:
        full, gdn, towers = self._cache_modules()
        pointers = []
        for module in (*full, *towers):
            pointers.append(
                -1 if module.k_cache is None else module.k_cache.data_ptr()
            )
        for module in gdn:
            pointers.append(
                -1 if module.conv_state is None else module.conv_state.data_ptr()
            )
        return tuple(pointers)

    def is_valid(self) -> bool:
        return bool(self.backbone_kv) and self._buffer_identity() == self._buffer_ids

    def applies_to(self, prompt: torch.Tensor) -> bool:
        """True when prompt really does start with the rows we prefilled.

        Cheap (608 x 11 int64) and worth it: if the prefix ever stops matching
        -- a reference change, an upstream edit to the prompt builder -- the
        failure without this check is not a crash, it is a chunk generated from
        someone else's KV cache.
        """
        return len(prompt) > self.P and torch.equal(
            prompt[: self.P].cpu(), self.rows
        )

    # ---------------------------------------------------------------- build

    @torch.inference_mode()
    def build(self, prefill_chunk_size: int = DRAMA3_PREFILL_CHUNK_SIZE) -> None:
        """Prefill rows [0, P) and snapshot everything it touched.

        Tower rows [0, P-1) only -- row P-1 needs frames[P], which belongs to
        the chunk. See the module docstring.
        """
        backbone, tower = self.model.backbone, self.model.audio_decoder
        P = self.P
        if self.model.cache_capacity < P:
            raise RuntimeError(
                f"caches hold {self.model.cache_capacity} rows, prefix needs {P}; "
                "call ensure_drama3_caches (or capture graphs) first"
            )
        started = time.perf_counter()
        frames = torch.zeros((P, self.rows.shape[1]), dtype=torch.long,
                             device=tower.device)
        frames.copy_(self.rows)
        positions = torch.arange(P, device=tower.device)
        hidden = None
        for start in range(0, P, prefill_chunk_size):
            end = min(start + prefill_chunk_size, P)
            chunk = frames[start:end]
            embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
            hidden = backbone(
                inputs_embeds=embeddings[None], input_pos=positions[start:end]
            )
            hidden = hidden[0].to(tower.device)
            paired_end = min(end, P - 1)
            if paired_end > start:
                tower.prefill(
                    hidden[: paired_end - start],
                    positions[start:paired_end],
                    frames[start:paired_end],
                    frames[start + 1 : paired_end + 1],
                )
        self.carry = hidden[-1:].clone()

        full, gdn, towers = self._cache_modules()
        self.backbone_kv = [
            (m, m.k_cache[:, :, :P].clone(), m.v_cache[:, :, :P].clone())
            for m in full
        ]
        self.gdn = [
            (m, m.conv_state.clone(), m.recurrent_state.clone()) for m in gdn
        ]
        self.tower_kv = [
            (m, m.k_cache[:, :, : P - 1].clone(), m.v_cache[:, :, : P - 1].clone())
            for m in towers
        ]
        self._buffer_ids = self._buffer_identity()
        self.bytes = sum(
            t.numel() * t.element_size()
            for group in (self.backbone_kv, self.gdn, self.tower_kv)
            for _, a, b in group
            for t in (a, b)
        ) + self.carry.numel() * self.carry.element_size()
        self.build_seconds = time.perf_counter() - started
        logger.info(
            f"prefix KV: {P} rows prefilled in {self.build_seconds:.2f}s, "
            f"snapshot {self.bytes / 2**20:.0f} MiB "
            f"({len(self.backbone_kv)} full-attn + {len(self.gdn)} gdn + "
            f"{len(self.tower_kv)} tower)"
        )

    # -------------------------------------------------------------- restore

    @torch.inference_mode()
    def restore(self) -> None:
        P = self.P
        for module, k, v in self.backbone_kv:
            module.k_cache[:, :, :P].copy_(k)
            module.v_cache[:, :, :P].copy_(v)
        for module, conv, recurrent in self.gdn:
            module.conv_state.copy_(conv)
            module.recurrent_state.copy_(recurrent)
        for module, k, v in self.tower_kv:
            module.k_cache[:, :, : P - 1].copy_(k)
            module.v_cache[:, :, : P - 1].copy_(v)
        self.restores += 1


@torch.inference_mode()
def generate_drama3_prefix(
    model,
    prompt: torch.Tensor,
    prefix: PrefixKV,
    *,
    max_new_tokens: int,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    top_p: float = DEFAULT_TOP_P,
    prefill_chunk_size: int = DRAMA3_PREFILL_CHUNK_SIZE,
    stats: dict | None = None,
):
    """`generate_drama3`, resuming from a prefilled prefix.

    A fork, not a wrapper: upstream's prefill loop has no seam to hook. Only
    the loop differs -- it starts at row P with the prefix's state restored,
    and carries hidden[P-1] in so the tower's teacher-forcing lag still lines
    up. Everything from `hidden = hidden[-1:]` onwards is upstream's, verbatim.

    Falls back to `generate_drama3` whenever the prefix does not apply, so a
    caller can pass it unconditionally.
    """
    from fish_speech.models.text2semantic.errors import InvalidInputError

    if top_k < 1:
        raise InvalidInputError(f"top_k must be at least 1; got {top_k}")
    if not prefix.applies_to(prompt):
        if stats is not None:
            stats["prefix_used"] = False
        return generate_drama3(
            model, prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, prefill_chunk_size=prefill_chunk_size,
        )

    backbone, tower = model.backbone, model.audio_decoder
    prompt_rows = len(prompt)
    capacity = prompt_rows + max_new_tokens
    if max_new_tokens < 1 or capacity > model.config.text.max_position_embeddings:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} does not fit after {prompt_rows} "
            "prompt rows"
        )
    graphs = Drama3DecodeGraphs.for_model(model, capacity)
    if graphs is None:
        ensure_drama3_caches(model, capacity)
    # Capturing graphs reallocates the caches and then runs three decode steps
    # through them, so a snapshot taken earlier is both stale and pointing at
    # freed buffers. Rebuild rather than silently restoring garbage.
    if not prefix.is_valid():
        logger.info("prefix KV: caches were reallocated, rebuilding the snapshot")
        prefix.build(prefill_chunk_size)

    frames = torch.zeros(
        (capacity, prompt.shape[1]), dtype=torch.long, device=tower.device
    )
    frames[:prompt_rows] = prompt.to(tower.device)
    positions = torch.arange(capacity, device=tower.device)

    started = time.perf_counter()
    prefix.restore()
    P = prefix.P
    # hidden[P-1], computed during the prefix build. The tower has not seen
    # row P-1 yet -- its label frames[P] only exists now -- so this row is
    # prepended to the first resumed batch rather than dropped.
    carry = prefix.carry
    for start in range(P, prompt_rows, prefill_chunk_size):
        end = min(start + prefill_chunk_size, prompt_rows)
        chunk = frames[start:end]
        embeddings = tower.embed_backbone(chunk, backbone.embed(chunk[:, 0]))
        hidden = backbone(
            inputs_embeds=embeddings[None], input_pos=positions[start:end]
        )
        hidden = hidden[0].to(tower.device)
        low = start - 1
        paired_end = min(end, prompt_rows - 1)
        if paired_end > low:
            paired = torch.cat((carry, hidden))[: paired_end - low]
            tower.prefill(
                paired,
                positions[low:paired_end],
                frames[low:paired_end],
                frames[low + 1 : paired_end + 1],
            )
        carry = hidden[-1:]
    elapsed = time.perf_counter() - started
    logger.info(
        f"Drama 3 prefill: {prompt_rows - P} of {prompt_rows} rows in "
        f"{elapsed:.2f}s ({P} reused)"
    )
    if stats is not None:
        stats.update(prefix_used=True, prefix_rows=P,
                     prefill_rows=prompt_rows - P,
                     prefill_seconds=round(elapsed, 4))

    hidden = carry
    produced, finish_reason = 0, "length"
    if graphs is not None:
        graphs.start(
            hidden, prompt_rows - 1, frames[:prompt_rows], temperature, top_k, top_p
        )
    for step in range(max_new_tokens):
        position = prompt_rows - 1 + step
        if graphs is not None:
            sampled, main = graphs.step(position)
        else:
            logits = backbone.compute_logits(hidden).to(tower.device)
            sampled = tower.decode_step(
                logits,
                hidden,
                positions[position : position + 1],
                frames[position : position + 1],
                frames[: position + 1, 0],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            main = sampled[0, 0].item()
        if main == model.config.eos_token_id:
            finish_reason = "eos"
            break
        frames[position + 1] = sampled[0]
        produced += 1
        if graphs is None and step + 1 < max_new_tokens:
            embeddings = tower.embed_backbone(sampled, backbone.embed(sampled[:, 0]))
            hidden = backbone(
                inputs_embeds=embeddings[None],
                input_pos=positions[position + 1 : position + 2],
            )[0].to(tower.device)
    return frames[prompt_rows : prompt_rows + produced].cpu(), finish_reason


def make_prefix(model, prompt_texts, prompt_tokens,
                prefill_chunk_size: int = DRAMA3_PREFILL_CHUNK_SIZE) -> PrefixKV:
    """Build a PrefixKV for the system + reference turn, caches allocated first."""
    check_upstream_unchanged()
    rows = prefix_rows(model.tokenizer, model.config, prompt_texts, prompt_tokens)
    # The snapshot has to be taken against the buffers decode will actually
    # use. Graph capture allocates its own at DRAMA3_GRAPH_CAPACITY, so get
    # that out of the way first; `generate_drama3_prefix` rebuilds anyway if
    # this turns out to have been the wrong guess.
    ensure_drama3_caches(model, Drama3DecodeGraphs.graph_capacity(model))
    prefix = PrefixKV(model, rows)
    prefix.build(prefill_chunk_size)
    return prefix
