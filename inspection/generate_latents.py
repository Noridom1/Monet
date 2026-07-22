"""Phase A: configurable HF generation that captures Monet latent hidden states.

This reimplements, in plain HF Transformers, the latent state machine that the
vLLM runner performs in ``inference/vllm/monet_gpu_model_runner.py`` (override:
lines 1434-1455; boundary update: 1641-1666). The point is to obtain, for a single
example, the *actual latent hidden-state vectors* (not the cosmetic ``<latent>``
placeholder string) together with their positions, so Phase B can replay the
sequence teacher-forced and extract logit-lens + attention.

Why we compute logits ourselves: with the default ``loss_type=[]`` the patched
``Qwen2_5_VLForConditionalGeneration.forward`` returns ``logits=None`` (logits are
only built when ``"ce" in loss_type``; see modeling lines 2270-2275). So we call the
inner ``model.model`` (which returns ``last_hidden_state``) and apply ``model.lm_head``
ourselves. That same final hidden state is what we (a) feed back as a latent embedding
and (b) later decode with the logit lens — one consistent vector.

Greedy decoding remains the default. Positive temperatures enable seeded top-k/top-p
sampling while retaining the same latent state machine.

============================ VALIDATION REQUIRED (A100) ============================
Do NOT trust the captured latents until the greedy-parity gate passes. The latent
*boundary* semantics (how many hidden states are injected and where the closing
</abs_vis_token> lands) are subtle in the runner. This file implements a clean,
documented protocol (see LATENT BOUNDARY PROTOCOL below). After running on the A100,
compare the produced ``answer text`` against a greedy vLLM generation on the SAME
example. If they diverge, the boundary off-by-one is the prime suspect — adjust the
clearly-marked block. Instructions are printed by this script on completion.
====================================================================================
"""
import os
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import torch
import PIL.Image

from inspection.load_model import load_monet
# Reuse the exact input-construction path used at real inference time.
from inference.load_and_gen_vllm import (
    vllm_mllm_process_batch_from_messages,  # builds the prompt string + image
)
from qwen_vl_utils import process_vision_info


@dataclass
class Trace:
    """Everything Phase B needs to replay the sequence and inspect it."""
    input_ids: torch.Tensor                  # [S] long; latent_token_id at latent positions
    inputs_embeds: torch.Tensor              # [S, H] float; latent vectors substituted in
    latent_positions: List[int]              # absolute indices of latent tokens in [0, S)
    latent_blocks: List[List[int]]           # latent positions grouped per <abs_vis_token> block
    latent_hidden_states: torch.Tensor       # [N_latent, H] the captured latent vectors
    image_positions: List[int]               # absolute indices of <|image_pad|> tokens
    image_grid_thw: Optional[torch.Tensor]   # [num_img, 3]
    spatial_merge_size: int
    prompt_len: int                          # number of prompt tokens (before generation)
    answer_positions: List[int]              # generated text positions after answer_start_pattern
    generated_text: str                      # raw decoded generation (incl. special tokens)
    latent_start_candidates: List[dict]       # steps where latent_start survived filtering
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SamplingConfig:
    """Next-token sampling controls shared by single and batch capture."""
    temperature: float = 0.0
    top_k: int = 50
    top_p: float = 0.8
    repetition_penalty: float = 1.0
    seed: int = 0

    def validate(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < -1:
            raise ValueError("top_k must be -1, 0, or a positive integer")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")


def _apply_repetition_penalty(logits: torch.Tensor, token_ids: List[int], penalty: float):
    """Apply the standard Transformers/vLLM sign-aware repetition penalty in place."""
    if penalty == 1.0 or not token_ids:
        return logits
    ids = torch.tensor(sorted(set(token_ids)), device=logits.device, dtype=torch.long)
    scores = logits.index_select(-1, ids)
    scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    logits.scatter_(-1, ids, scores)
    return logits


def sample_next_token(
    logits: torch.Tensor,
    history: List[int],
    config: SamplingConfig,
    generator: torch.Generator,
    tracked_token_id: int,
):
    """Return ``(token_id, diagnostics)`` without retaining full-vocabulary tensors."""
    logits = logits.float().reshape(-1)
    raw_probability = float(torch.softmax(logits, dim=-1)[tracked_token_id])
    processed = _apply_repetition_penalty(logits.clone(), history, config.repetition_penalty)
    rank = int((processed > processed[tracked_token_id]).sum()) + 1

    if config.temperature == 0:
        sampled_id = int(processed.argmax())
        in_pool = sampled_id == tracked_token_id
        return sampled_id, {
            "rank": rank,
            "raw_probability": raw_probability,
            "sampling_probability": 1.0 if in_pool else 0.0,
            "in_top_k": in_pool,
            "in_top_p": in_pool,
            "in_sampling_pool": in_pool,
        }

    filtered = processed / config.temperature
    vocab_size = filtered.numel()
    top_k = config.top_k
    if top_k not in (-1, 0) and top_k < vocab_size:
        cutoff = torch.topk(filtered, top_k).values[-1]
        top_k_mask = filtered >= cutoff
    else:
        top_k_mask = torch.ones_like(filtered, dtype=torch.bool)
    in_top_k = bool(top_k_mask[tracked_token_id])
    filtered = filtered.masked_fill(~top_k_mask, float("-inf"))

    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(sorted_probs, dim=-1) > config.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        top_p_mask = torch.ones_like(top_k_mask)
        top_p_mask.scatter_(0, sorted_indices, ~remove)
        top_p_mask &= top_k_mask
    else:
        top_p_mask = top_k_mask
    in_top_p = bool(top_p_mask[tracked_token_id])
    filtered = filtered.masked_fill(~top_p_mask, float("-inf"))
    probs = torch.softmax(filtered, dim=-1)
    sampled_id = int(torch.multinomial(probs, 1, generator=generator))
    return sampled_id, {
        "rank": rank,
        "raw_probability": raw_probability,
        "sampling_probability": float(probs[tracked_token_id]),
        "in_top_k": in_top_k,
        "in_top_p": in_top_p,
        "in_sampling_pool": in_top_p,
    }


@torch.no_grad()
def generate_with_latents(
    model,
    processor,
    special_ids: dict,
    conversation: list,
    latent_size: int,
    max_new_tokens: int = 1024,
    sampling: Optional[SamplingConfig] = None,
    device: str = "cuda",
) -> Trace:
    sampling = sampling or SamplingConfig()
    sampling.validate()
    tokenizer = processor.tokenizer
    H = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
    merge = model.config.vision_config.spatial_merge_size

    start_id = special_ids["latent_start"]
    end_id = special_ids["latent_end"]
    latent_pad_id = special_ids["latent_pad"]
    image_pad_id = special_ids["image_pad"]

    # ---- build inputs exactly like inference (load_and_gen_vllm.py:75-94) ----
    vllm_inputs = vllm_mllm_process_batch_from_messages([conversation], processor)
    prompt = vllm_inputs[0]["prompt"]
    image_inputs, _ = process_vision_info(conversation, return_video_kwargs=False)
    proc = processor(text=[prompt], images=image_inputs, return_tensors="pt")
    proc = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in proc.items()}

    input_ids = proc["input_ids"]                 # [1, P]
    pixel_values = proc.get("pixel_values")
    image_grid_thw = proc.get("image_grid_thw")
    prompt_len = input_ids.shape[1]

    inner = model.model  # Qwen2_5_VLModel — returns last_hidden_state
    embed = model.get_input_embeddings()

    # Running record of the full sequence.
    seq_ids: List[int] = input_ids[0].tolist()
    seq_embeds: List[torch.Tensor] = []  # filled lazily below for prompt; see note
    latent_positions: List[int] = []
    latent_blocks: List[List[int]] = []
    latent_vectors: List[torch.Tensor] = []
    latent_start_candidates: List[dict] = []
    sampling_history: List[int] = list(seq_ids)
    generator = torch.Generator(device=device)
    generator.manual_seed(sampling.seed)

    # ---- prefill the prompt ----
    out = inner(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        use_cache=True,
        return_dict=True,
    )
    past = out.past_key_values
    # final hidden state of the last prompt token -> first generation logits
    final_h = out.last_hidden_state[:, -1, :]          # [1, H]
    # Capture the prompt's *input* embeddings exactly as the prefill saw them: token
    # embeddings with the vision features scattered into the <|image_pad|> rows. This
    # mirrors the inner model's own scatter (modeling lines 1602-1622) so the Trace is
    # self-contained and Phase B can replay from inputs_embeds without pixel_values.
    prompt_embeds = embed(input_ids)[0]                # [P, H]
    if pixel_values is not None:
        image_embeds = model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(prompt_embeds.dtype)
        img_rows = (input_ids[0] == image_pad_id)
        assert int(img_rows.sum()) == image_embeds.shape[0], (
            f"image token/feature mismatch: {int(img_rows.sum())} vs {image_embeds.shape[0]}"
        )
        prompt_embeds = prompt_embeds.clone()
        prompt_embeds[img_rows] = image_embeds

    pos = prompt_len            # cache_position of the *next* token to be written
    # ---- latent state machine (mirrors runner 1641-1666) ----
    active = False
    pending: Optional[torch.Tensor] = None
    current_len = 0
    cur_block: List[int] = []

    def boundary_update(tok: int, fh: torch.Tensor):
        """Replicates monet_gpu_model_runner.py:1654-1666 ordering."""
        nonlocal active, pending, current_len, cur_block
        forced = tok
        if (not active) and tok == start_id:
            active = True
        elif active and (tok == end_id or current_len >= latent_size):
            active = False
            pending = None
            current_len = 0
            forced = end_id            # runner forces the closing token (line 1661)
            if cur_block:
                latent_blocks.append(cur_block)
                cur_block = []
        if active:
            pending = fh.detach()       # store [1,H]; consumed as next input (runner 1663-1666)
            current_len += 1
        return forced

    num_decode_steps = 0
    latent_start_sampled_count = 0

    def choose_next(fh: torch.Tensor, generated_step: int, sequence_position: int):
        nonlocal num_decode_steps, latent_start_sampled_count
        logits = model.lm_head(fh)[0]
        sampled_id, diag = sample_next_token(
            logits, sampling_history, sampling, generator, start_id)
        num_decode_steps += 1
        if sampled_id == start_id:
            latent_start_sampled_count += 1
        if diag["in_sampling_pool"]:
            latent_start_candidates.append({
                "generated_step": generated_step,
                "sequence_position": sequence_position,
                "context_position": sequence_position - 1,
                "inside_latent_block": bool(active),
                **diag,
                "sampled": sampled_id == start_id,
            })
        return sampled_id

    next_id = choose_next(final_h, generated_step=0, sequence_position=prompt_len)
    next_id = boundary_update(next_id, final_h)
    sampling_history.append(next_id)
    gen_embeds: List[torch.Tensor] = []  # input embeds of generated positions

    # ---- decode loop ----
    # LATENT BOUNDARY PROTOCOL (the block to revisit if parity fails):
    #   At each step we decide THIS position's input embedding:
    #     - if `active and pending is not None`  -> input = pending (a hidden state);
    #       this position is a LATENT (record latent_pad_id + store the vector).
    #     - else                                  -> input = token embedding of next_id.
    #   Then forward one step, sample the next token, and run boundary_update.
    for step in range(max_new_tokens):
        if active and pending is not None:
            step_embed = pending.to(device=device, dtype=prompt_embeds.dtype)  # [1,H]
            recorded_id = latent_pad_id
            is_latent = True
        else:
            step_embed = embed(torch.tensor([[next_id]], device=device))[0]    # [1,H]
            recorded_id = next_id
            is_latent = False

        seq_ids.append(recorded_id)
        gen_embeds.append(step_embed[0].detach().to("cpu"))
        if is_latent:
            latent_positions.append(pos)
            cur_block.append(pos)
            latent_vectors.append(step_embed[0].detach().to("cpu"))

        out = inner(
            inputs_embeds=step_embed.unsqueeze(0),     # [1,1,H]
            past_key_values=past,
            use_cache=True,
            cache_position=torch.tensor([pos], device=device),
            return_dict=True,
        )
        past = out.past_key_values
        final_h = out.last_hidden_state[:, -1, :]
        pos += 1

        sampled_id = choose_next(final_h, generated_step=step + 1, sequence_position=pos)
        next_id = boundary_update(sampled_id, final_h)
        sampling_history.append(next_id)

        # stop on EOS only when not inside a latent block
        if (not active) and next_id == tokenizer.eos_token_id:
            break

    if cur_block:
        latent_blocks.append(cur_block)

    # ---- assemble full embedding sequence + trace ----
    full_embeds = torch.cat([prompt_embeds.to("cpu"), torch.stack(gen_embeds, 0)], dim=0)  # [S,H]
    seq_ids_t = torch.tensor(seq_ids, dtype=torch.long)

    image_positions = (seq_ids_t == image_pad_id).nonzero(as_tuple=False).flatten().tolist()
    answer_positions = _answer_positions(seq_ids, special_ids["answer_start_pattern"])
    generated_text = tokenizer.decode(seq_ids[prompt_len:], skip_special_tokens=False)

    latent_hidden = (
        torch.stack(latent_vectors, 0) if latent_vectors else torch.empty(0, full_embeds.shape[-1])
    )

    return Trace(
        input_ids=seq_ids_t,
        inputs_embeds=full_embeds,
        latent_positions=latent_positions,
        latent_blocks=latent_blocks,
        latent_hidden_states=latent_hidden,
        image_positions=image_positions,
        image_grid_thw=(image_grid_thw.to("cpu") if image_grid_thw is not None else None),
        spatial_merge_size=int(merge),
        prompt_len=prompt_len,
        answer_positions=answer_positions,
        generated_text=generated_text,
        latent_start_candidates=latent_start_candidates,
        meta={
            "latent_size": latent_size,
            "num_latent": len(latent_positions),
            "num_latent_blocks": len(latent_blocks),
            "seq_len": len(seq_ids),
            "num_decode_steps": num_decode_steps,
            "latent_activated": bool(latent_blocks),
            "latent_start_pool_count": len(latent_start_candidates),
            "latent_start_sampled_count": latent_start_sampled_count,
            "sampling": asdict(sampling),
        },
    )


def _answer_positions(seq_ids: List[int], pattern: List[int]) -> List[int]:
    """Positions after the last occurrence of the answer_start_pattern."""
    n, m = len(seq_ids), len(pattern)
    start = None
    for i in range(n - m + 1):
        if seq_ids[i : i + m] == pattern:
            start = i + m
    return list(range(start, n)) if start is not None else []


def conversation_from_sample(sample: dict, system_prompt: str, data_root: str) -> list:
    """Build a (system + user) conversation from a manifest sample, mirroring eval-time input.

    ``data_root`` is the directory holding ``samples.json`` so ``sample['image']`` (a relative
    path) resolves. The system prompt matches VLMEvalKit/run_monet.py so the captured run
    reproduces the eval prediction.
    """
    img_path = os.path.join(data_root, sample["image"])
    conv = []
    if system_prompt:
        conv.append({"role": "system",
                     "content": [{"type": "text", "text": system_prompt}]})
    conv.append({
        "role": "user",
        "content": [
            {"type": "text", "text": sample["question_text"]},
            {"type": "image", "image": PIL.Image.open(img_path).convert("RGB")},
        ],
    })
    return conv


def _demo_conversation():
    """The same example as inference/vllm_inference_example.py."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Question:  Which car has the longest rental period? The choices are listed below:\n"
                    "(A)DB11 COUPE.\n(B) V12 VANTAGES COUPES.\n(C) VANQUISH VOLANTE.\n(D) V12 VOLANTE.\n"
                    "(E) The image does not feature the time. Put your final answer in \\boxed{}.",
                },
                {"type": "image", "image": PIL.Image.open("images/example_question.png").convert("RGB")},
            ],
        }
    ]


def _save_trace(trace, out_path, label=""):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(asdict(trace), out_path)
    print(f"[Phase A]{(' ' + label) if label else ''} saved trace -> {out_path}  "
          f"(seq_len={trace.meta['seq_len']}, "
          f"num_latent={trace.meta['num_latent']} in {trace.meta['num_latent_blocks']} block(s), "
          f"latent_start pool/sample={trace.meta['latent_start_pool_count']}/"
          f"{trace.meta['latent_start_sampled_count']})")


def _sampling_from_args(args):
    config = SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    config.validate()
    return config


def _run_batch(args, model, processor, special_ids):
    """Capture a trace per sample in a manifest, keeping the model loaded throughout."""
    with open(args.manifest) as f:
        manifest = json.load(f)
    data_root = os.path.dirname(os.path.abspath(args.manifest))
    system_prompt = manifest.get("system_prompt", "")
    samples = manifest["samples"]
    sampling = _sampling_from_args(args)
    print(f"[Phase A] manifest {args.manifest}: {len(samples)} samples -> {args.out_dir}")
    print(f"[Phase A] sampling={asdict(sampling)} max_new_tokens={args.max_new_tokens}")

    for i, sample in enumerate(samples):
        sid = sample["id"]
        print(f"[Phase A] ({i + 1}/{len(samples)}) sample {sid} (idx={sample.get('index')})")
        conv = conversation_from_sample(sample, system_prompt, data_root)
        trace = generate_with_latents(
            model, processor, special_ids,
            conversation=conv,
            latent_size=args.latent_size,
            max_new_tokens=args.max_new_tokens,
            sampling=sampling,
        )
        # stash eval metadata so Phase B can show correctness alongside the latents
        trace.meta.update({
            "sample_id": sid, "bucket": sample.get("bucket"),
            "index": sample.get("index"), "gold": sample.get("gold"),
            "pred_letter": sample.get("pred_letter"), "hit": sample.get("hit"),
            "category": sample.get("category"), "image": sample.get("image"),
        })
        _save_trace(trace, os.path.join(args.out_dir, sid, "trace.pt"), label=sid)
    print("=" * 78)
    print(f"[Phase A] batch done: {len(samples)} traces under {args.out_dir}")
    print("  Next: python -m inspection.inspect --manifest "
          f"{args.manifest} --out_dir {args.out_dir}")
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=os.environ.get("MODEL_PATH", "models/Monet-7B"))
    ap.add_argument("--latent_size", type=int, default=int(os.environ.get("LATENT_SIZE", "10")))
    ap.add_argument("--temperature", type=float,
                    default=float(os.environ.get("TEMPERATURE", "0.0")))
    ap.add_argument("--top_k", type=int, default=int(os.environ.get("TOP_K", "50")))
    ap.add_argument("--top_p", type=float, default=float(os.environ.get("TOP_P", "0.8")))
    ap.add_argument("--repetition_penalty", type=float,
                    default=float(os.environ.get("REPETITION_PENALTY", "1.0")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    ap.add_argument("--max_new_tokens", type=int,
                    default=int(os.environ.get("MAX_NEW_TOKENS", "1024")))
    ap.add_argument("--out", default="inspection/outputs/demo/trace.pt",
                    help="single-sample output trace path (ignored when --manifest is set)")
    ap.add_argument("--manifest", default=None,
                    help="data/inspect_samples/samples.json — batch mode over multiple samples")
    ap.add_argument("--out_dir", default="inspection/outputs/eval_samples",
                    help="batch mode: one <out_dir>/<id>/trace.pt per sample")
    args = ap.parse_args()

    try:
        sampling = _sampling_from_args(args)
    except ValueError as e:
        ap.error(str(e))
    if args.max_new_tokens <= 0:
        ap.error("max_new_tokens must be > 0")

    model, processor, special_ids = load_monet(args.model_path)

    if args.manifest:
        _run_batch(args, model, processor, special_ids)
        return

    trace = generate_with_latents(
        model, processor, special_ids,
        conversation=_demo_conversation(),
        latent_size=args.latent_size,
        max_new_tokens=args.max_new_tokens,
        sampling=sampling,
    )
    _save_trace(trace, args.out)

    print("=" * 78)
    print(f"  latent_positions   = {trace.latent_positions}")
    print("-" * 78)
    print("RAW GENERATION (special tokens kept):")
    print(trace.generated_text)
    print("=" * 78)
    print("VALIDATION GATE — run BEFORE building Phase B:")
    print("  1. Generate the SAME example with greedy vLLM and compare the answer text")
    print("     outside the <abs_vis_token>...</abs_vis_token> block. They must match.")
    print("  2. To make vLLM greedy, set temperature=0 in inference/load_and_gen_vllm.py")
    print("     (temperature, line ~21) and run run_scripts/inference/run_example.sh.")
    print("  3. If the answers diverge, the latent BOUNDARY PROTOCOL in this file")
    print("     (decode loop) is the prime suspect — see the marked comment block.")
    print("=" * 78)


if __name__ == "__main__":
    main()
