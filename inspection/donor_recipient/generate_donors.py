"""Stage 1: force and capture a fixed Monet latent block for each MMVP item."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from PIL import Image

from inspection.donor_recipient import PROTOCOL_VERSION
from inspection.donor_recipient.common import (
    atomic_torch_save,
    donor_artifact_error,
    load_manifest,
    make_messages,
    resolve_image,
)


@torch.inference_mode()
def force_latent_block(model, processor, image: Image.Image, question_text: str, latent_size: int, device: str):
    messages = make_messages(image, question_text)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
    inputs = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in inputs.items()}

    inner = model.model
    output = inner(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        pixel_values=inputs.get("pixel_values"),
        image_grid_thw=inputs.get("image_grid_thw"),
        use_cache=True,
        return_dict=True,
    )
    past = output.past_key_values
    hidden = output.last_hidden_state[:, -1, :]
    prompt_len = int(inputs["input_ids"].shape[1])
    vectors = []

    for index in range(latent_size):
        vectors.append(hidden[0].detach().cpu().to(torch.bfloat16))
        if index == latent_size - 1:
            break
        output = inner(
            inputs_embeds=hidden.unsqueeze(1),
            past_key_values=past,
            use_cache=True,
            cache_position=torch.tensor([prompt_len + index], device=device),
            return_dict=True,
        )
        past = output.past_key_values
        hidden = output.last_hidden_state[:, -1, :]

    latents = torch.stack(vectors)
    grid = inputs.get("image_grid_thw")
    return latents, (grid.detach().cpu() if grid is not None else None), prompt_len


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model_path", default=os.environ.get("DONOR_MODEL_PATH", "models/Monet-7B"))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--latent_size", type=int, default=int(os.environ.get("LATENT_SIZE", "10")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N samples")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.latent_size < 2:
        parser.error("--latent_size must be at least 2 for the shuffle control")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    manifest = load_manifest(args.manifest)
    samples = manifest["samples"][: args.limit]
    donor_dir = Path(args.output_dir) / "donors"
    pending = []
    for sample in samples:
        path = donor_dir / f"{sample['id']}.pt"
        if path.is_file() and not args.overwrite:
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            error = donor_artifact_error(artifact, sample["id"], args.latent_size)
            if error is None and artifact.get("model_path") != args.model_path:
                error = "donor model path mismatch"
            if error:
                raise RuntimeError(f"incompatible existing artifact {path}: {error}; use --overwrite")
            print(f"[donor] skip {sample['id']} (complete)")
        else:
            pending.append((sample, path))
    if not pending:
        print("[donor] all requested artifacts are complete")
        return

    # Importing this loader intentionally applies the Monet patch in this process only.
    from inspection.load_model import load_monet

    model, processor, _ = load_monet(args.model_path, device=args.device, dtype=torch.bfloat16)
    hidden_size = int(model.config.text_config.hidden_size)
    print(f"[donor] loaded {args.model_path}; hidden={hidden_size}; pending={len(pending)}")

    for position, (sample, path) in enumerate(pending, 1):
        image_path = resolve_image(args.manifest, sample)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        latents, grid, prompt_len = force_latent_block(
            model, processor, image, sample["question_text"], args.latent_size, args.device
        )
        artifact = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol": "forced_immediate_recurrent_block",
            "sample_id": sample["id"],
            "index": sample["index"],
            "model_path": args.model_path,
            "latent_size": args.latent_size,
            "hidden_size": hidden_size,
            "latents": latents,
            "norms": torch.linalg.vector_norm(latents.float(), dim=-1),
            "image_grid_thw": grid,
            "prompt_len": prompt_len,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if args.device.startswith("cuda") else None
            ),
        }
        error = donor_artifact_error(artifact, sample["id"], args.latent_size)
        if error:
            raise RuntimeError(f"invalid generated artifact for {sample['id']}: {error}")
        atomic_torch_save(artifact, path)
        print(f"[donor] {position}/{len(pending)} {sample['id']} -> {path}")


if __name__ == "__main__":
    main()
