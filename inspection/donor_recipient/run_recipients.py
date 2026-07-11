"""Stage 2: inject donor states into vanilla Qwen2.5-VL and run all controls."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from inspection.donor_recipient import PROTOCOL_VERSION
from inspection.donor_recipient.common import (
    CONDITIONS,
    SCORING_PROTOCOL,
    atomic_json_dump,
    cyclic_wrong_sample_ids,
    donor_artifact_error,
    load_manifest,
    make_messages,
    non_identity_permutation,
    norm_matched_random,
    parse_seeds,
    qwen_decode_position_ids,
    resolve_image,
    result_path,
    score_response,
    solid_gray_image,
    stable_seed,
)


def _prepare_inputs(processor, image: Image.Image, question_text: str, device: str) -> dict:
    messages = make_messages(image, question_text)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}


def image_grid(processor, image: Image.Image, question_text: str) -> torch.Tensor:
    messages = make_messages(image, question_text)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    grid = inputs.get("image_grid_thw")
    if grid is None:
        raise RuntimeError("processor did not produce image_grid_thw")
    return grid.cpu()


@torch.inference_mode()
def generate_with_intervention(
    model,
    processor,
    image: Image.Image,
    question_text: str,
    latents: torch.Tensor | None,
    max_new_tokens: int,
    device: str,
) -> str:
    inputs = _prepare_inputs(processor, image, question_text, device)
    output = model(
        **inputs,
        use_cache=True,
        return_dict=True,
        logits_to_keep=1,
    )
    past = output.past_key_values
    logits = output.logits[:, -1, :]
    rope_deltas = output.rope_deltas
    if rope_deltas is None or rope_deltas.numel() != 1:
        shape = None if rope_deltas is None else tuple(rope_deltas.shape)
        raise RuntimeError(f"expected one Qwen RoPE delta after prefill, got {shape}")
    rope_delta = int(rope_deltas.reshape(-1)[0].item())
    prompt_len = int(inputs["input_ids"].shape[1])
    latent_count = 0 if latents is None else int(latents.shape[0])
    model_dtype = model.get_input_embeddings().weight.dtype

    if latents is not None:
        for index, vector in enumerate(latents):
            step = vector.to(device=device, dtype=model_dtype).view(1, 1, -1)
            output = model(
                inputs_embeds=step,
                past_key_values=past,
                use_cache=True,
                cache_position=torch.tensor([prompt_len + index], device=device),
                position_ids=qwen_decode_position_ids(prompt_len + index, rope_delta, device),
                return_dict=True,
                logits_to_keep=1,
            )
            past = output.past_key_values
            logits = output.logits[:, -1, :]

    generated = []
    eos_id = processor.tokenizer.eos_token_id
    for index in range(max_new_tokens):
        token = int(logits.argmax(dim=-1).item())
        generated.append(token)
        if token == eos_id:
            break
        token_ids = torch.tensor([[token]], device=device)
        step = model.get_input_embeddings()(token_ids)
        output = model(
            inputs_embeds=step,
            past_key_values=past,
            use_cache=True,
            cache_position=torch.tensor([prompt_len + latent_count + index], device=device),
            position_ids=qwen_decode_position_ids(
                prompt_len + latent_count + index, rope_delta, device
            ),
            return_dict=True,
            logits_to_keep=1,
        )
        past = output.past_key_values
        logits = output.logits[:, -1, :]
    return processor.tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _load_donors(samples: list[dict], donor_dir: Path, latent_size: int) -> dict[str, dict]:
    artifacts = {}
    hidden_size = None
    for sample in samples:
        path = donor_dir / f"{sample['id']}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing donor artifact: {path}")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        error = donor_artifact_error(artifact, sample["id"], latent_size)
        if error:
            raise RuntimeError(f"invalid donor artifact {path}: {error}")
        current_hidden = int(artifact["latents"].shape[1])
        if hidden_size is None:
            hidden_size = current_hidden
        elif hidden_size != current_hidden:
            raise RuntimeError("donor artifacts have inconsistent hidden dimensions")
        artifacts[sample["id"]] = artifact
    return artifacts


def _existing_result_error(
    path: Path,
    sample_id: str,
    condition: str,
    seed: int,
    model_path: str,
    latent_size: int,
    max_new_tokens: int,
):
    try:
        with open(path, encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, ValueError) as error:
        return f"unreadable result: {error}"
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "sample_id": sample_id,
        "condition": condition,
        "seed": seed,
        "recipient_model_path": model_path,
        "latent_size": latent_size,
        "max_new_tokens": max_new_tokens,
    }
    mismatches = [key for key, value in expected.items() if result.get(key) != value]
    return f"metadata mismatch: {', '.join(mismatches)}" if mismatches else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--donor_dir", required=True, help="directory containing <sample_id>.pt donor artifacts")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model_path",
        default=os.environ.get("RECIPIENT_MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"),
    )
    parser.add_argument("--latent_size", type=int, default=int(os.environ.get("LATENT_SIZE", "10")))
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seeds", default="0", help="comma-separated intervention seeds")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn_implementation", default="sdpa", choices=("sdpa", "flash_attention_2", "eager"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.latent_size < 2 or args.max_new_tokens <= 0:
        parser.error("--latent_size must be >= 2 and --max_new_tokens must be positive")
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as error:
        parser.error(str(error))

    manifest = load_manifest(args.manifest)
    all_samples = manifest["samples"]
    samples = all_samples[: args.limit]
    all_ids = [sample["id"] for sample in all_samples]
    wrong_mappings = {seed: cyclic_wrong_sample_ids(all_ids, seed) for seed in seeds}
    needed_ids = {sample["id"] for sample in samples}
    needed_ids.update(wrong_mappings[seed][sample["id"]] for seed in seeds for sample in samples)
    needed_samples = [sample for sample in all_samples if sample["id"] in needed_ids]
    donors = _load_donors(needed_samples, Path(args.donor_dir), args.latent_size)
    donor_hidden = int(next(iter(donors.values()))["latents"].shape[1])

    pending = []
    for seed in seeds:
        for sample in samples:
            for condition in CONDITIONS:
                path = result_path(args.output_dir, condition, seed, sample["id"])
                if path.is_file() and not args.overwrite:
                    error = _existing_result_error(
                        path,
                        sample["id"],
                        condition,
                        seed,
                        args.model_path,
                        args.latent_size,
                        args.max_new_tokens,
                    )
                    if error:
                        raise RuntimeError(f"incompatible existing result {path}: {error}; use --overwrite")
                else:
                    pending.append((seed, sample, condition, path))
    if not pending:
        print("[recipient] all requested results are complete")
        return

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    recipient_hidden = int(model.config.text_config.hidden_size)
    if donor_hidden != recipient_hidden:
        raise RuntimeError(f"hidden-size mismatch: donor={donor_hidden}, recipient={recipient_hidden}")
    print(f"[recipient] loaded {args.model_path}; hidden={recipient_hidden}; pending={len(pending)}")

    image_cache: dict[str, tuple[Image.Image, Image.Image]] = {}
    for position, (seed, sample, condition, path) in enumerate(pending, 1):
        sid = sample["id"]
        if sid not in image_cache:
            with Image.open(resolve_image(args.manifest, sample)) as opened:
                real = opened.convert("RGB")
            masked = solid_gray_image(real)
            real_grid = image_grid(processor, real, sample["question_text"])
            masked_grid = image_grid(processor, masked, sample["question_text"])
            if not torch.equal(real_grid, masked_grid):
                raise RuntimeError(f"mask changed visual grid for {sid}: {real_grid.tolist()} vs {masked_grid.tolist()}")
            image_cache[sid] = (real, masked)
        real, masked = image_cache[sid]
        source_id = sid
        latents = None
        intervention = {}
        image = real

        if condition == "same_sample":
            latents = donors[sid]["latents"]
        elif condition == "order_shuffled":
            latents = donors[sid]["latents"]
            permutation = non_identity_permutation(
                len(latents), stable_seed(seed, condition, sid)
            )
            latents = latents[permutation]
            intervention["permutation"] = permutation.tolist()
        elif condition == "norm_matched_random":
            source = donors[sid]["latents"]
            latents = norm_matched_random(source, stable_seed(seed, condition, sid))
            source_norms = torch.linalg.vector_norm(source.float(), dim=-1)
            random_norms = torch.linalg.vector_norm(latents.float(), dim=-1)
            max_relative_error = float(((random_norms - source_norms).abs() / source_norms.clamp_min(1e-12)).max())
            intervention["max_norm_relative_error"] = max_relative_error
            if max_relative_error > 0.01:
                raise RuntimeError(f"norm matching failed for {sid}: relative error {max_relative_error:.4g}")
        elif condition == "recipient_image_masked":
            latents = donors[sid]["latents"]
            image = masked
            intervention["mask"] = "uniform_rgb_127_same_dimensions"
        elif condition == "wrong_sample":
            source_id = wrong_mappings[seed][sid]
            if source_id == sid:
                raise AssertionError("wrong-sample mapping contains a fixed point")
            latents = donors[source_id]["latents"]
        elif condition != "vanilla_baseline":
            raise AssertionError(f"unknown condition: {condition}")

        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        response = generate_with_intervention(
            model,
            processor,
            image,
            sample["question_text"],
            latents,
            args.max_new_tokens,
            args.device,
        )
        elapsed = time.perf_counter() - started
        parsed, correct = score_response(response, sample["gold"])
        result = {
            "protocol_version": PROTOCOL_VERSION,
            "sample_id": sid,
            "index": sample["index"],
            "condition": condition,
            "seed": seed,
            "recipient_model_path": args.model_path,
            "latent_size": args.latent_size,
            "max_new_tokens": args.max_new_tokens,
            "latent_source_id": source_id if latents is not None else None,
            "num_latents": int(len(latents)) if latents is not None else 0,
            "image_masked": condition == "recipient_image_masked",
            "response": response,
            "parsed": parsed,
            "gold": sample["gold"],
            "correct": correct,
            "scoring_protocol": SCORING_PROTOCOL,
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else None
            ),
            "intervention": intervention,
        }
        atomic_json_dump(result, path)
        print(f"[recipient] {position}/{len(pending)} {condition} seed={seed} {sid}: {parsed}/{sample['gold']}")
        del latents
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
