"""Shared, model-independent utilities for the donor-recipient experiment."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image

from inspection.donor_recipient import PROTOCOL_VERSION


CONDITIONS = (
    "vanilla_baseline",
    "same_sample",
    "order_shuffled",
    "norm_matched_random",
    "recipient_image_masked",
    "wrong_sample",
)
TASK_INSTRUCTION = "\nAnswer with the option's letter from the given choices directly."
SCORING_PROTOCOL = "mmvp_robust_option_v2"


def build_question(question: str, options: str) -> str:
    """Reproduce the MMVP prompt construction used by the LVR evaluation."""
    text = f"{question}\nOptions:\n{options}"
    return text.replace("(a)", "A.").replace("(b)", "B.") + TASK_INSTRUCTION


def normalize_label(label: object) -> str:
    value = str(label).strip().upper()
    if value in {"(A)", "(B)"}:
        value = value[1]
    if value not in {"A", "B"}:
        raise ValueError(f"MMVP label must be A or B, got {label!r}")
    return value


def parse_option(response: str | None) -> str | None:
    """Extract an MMVP option while tolerating reasoning and Qwen control tokens."""
    if not response:
        return None

    answer_tags = re.findall(r"<answer>\s*([AB])\s*(?:</answer>)?", response, flags=re.IGNORECASE)
    if answer_tags:
        return answer_tags[-1].upper()

    boxed = re.findall(r"\\boxed\s*\{\s*([AB])\s*\}", response, flags=re.IGNORECASE)
    if boxed:
        return boxed[-1].upper()

    cleaned = re.sub(r"<\|[^|>]+\|>", " ", response)
    standalone = re.findall(r"(?<![A-Za-z0-9])([AB])(?![A-Za-z0-9])", cleaned, flags=re.IGNORECASE)
    return standalone[-1].upper() if standalone else None


def score_response(response: str | None, gold: object) -> tuple[str | None, bool]:
    parsed = parse_option(response)
    return parsed, parsed == normalize_label(gold)


def make_messages(image: Image.Image, question_text: str) -> list[dict]:
    """Image-first message ordering matching the LVR MMVP evaluator."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question_text},
        ],
    }]


def load_manifest(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"manifest has no samples: {path}")
    ids = [sample.get("id") for sample in samples]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("manifest sample ids must be present and unique")
    return manifest


def resolve_image(manifest_path: str | os.PathLike, sample: dict) -> Path:
    path = Path(sample["image"])
    if not path.is_absolute():
        path = Path(manifest_path).resolve().parent / path
    if not path.is_file():
        raise FileNotFoundError(f"image not found for {sample['id']}: {path}")
    return path


def solid_gray_image(image: Image.Image) -> Image.Image:
    return Image.new("RGB", image.size, color=(127, 127, 127))


def stable_seed(seed: int, *parts: object) -> int:
    payload = "\0".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def non_identity_permutation(length: int, seed: int) -> torch.Tensor:
    if length < 2:
        raise ValueError("order shuffling requires at least two latent vectors")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(length, generator=generator)
    identity = torch.arange(length)
    if torch.equal(permutation, identity):
        permutation = identity.roll(1)
    return permutation


def norm_matched_random(latents: torch.Tensor, seed: int) -> torch.Tensor:
    if latents.ndim != 2 or latents.shape[0] == 0:
        raise ValueError("latents must be a non-empty [K,H] tensor")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    source = latents.detach().cpu().float()
    random = torch.randn(source.shape, generator=generator, dtype=torch.float32)
    target_norms = torch.linalg.vector_norm(source, dim=-1, keepdim=True)
    random_norms = torch.linalg.vector_norm(random, dim=-1, keepdim=True).clamp_min(1e-12)
    return (random * target_norms / random_norms).to(latents.dtype)


def qwen_decode_position_ids(cache_position: int, rope_delta: int, device: str) -> torch.Tensor:
    """Build Qwen2.5-VL's [3,B,1] mRoPE positions without an expanded writable view.

    Qwen uses the same scalar position for all three mRoPE axes after the multimodal
    prefill. ``rope_delta`` is converted from the batch-size-one prefill output once,
    before the decode loop, to avoid a device synchronization on every token.
    """
    return torch.full(
        (3, 1, 1),
        cache_position + rope_delta,
        dtype=torch.long,
        device=device,
    )


def cyclic_wrong_sample_ids(sample_ids: Sequence[str], seed: int) -> dict[str, str]:
    if len(sample_ids) < 2 or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("wrong-sample assignment requires at least two unique samples")
    offset = seed % (len(sample_ids) - 1) + 1
    mapping = {sid: sample_ids[(index + offset) % len(sample_ids)] for index, sid in enumerate(sample_ids)}
    assert all(source != target for source, target in mapping.items())
    return mapping


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one intervention seed is required")
    return list(dict.fromkeys(seeds))


def atomic_json_dump(value: object, path: str | os.PathLike) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def atomic_torch_save(value: object, path: str | os.PathLike) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, destination)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def donor_artifact_error(artifact: dict, sample_id: str, latent_size: int) -> str | None:
    if artifact.get("protocol_version") != PROTOCOL_VERSION:
        return "protocol version mismatch"
    if artifact.get("sample_id") != sample_id:
        return "sample id mismatch"
    latents = artifact.get("latents")
    if not torch.is_tensor(latents) or latents.ndim != 2 or latents.shape[0] != latent_size:
        return f"expected latent shape [{latent_size},H]"
    if not torch.isfinite(latents).all():
        return "latents contain non-finite values"
    norms = artifact.get("norms")
    if not torch.is_tensor(norms) or norms.shape != (latent_size,):
        return f"expected norms shape [{latent_size}]"
    return None


def result_path(output_dir: str | os.PathLike, condition: str, seed: int, sample_id: str) -> Path:
    return Path(output_dir) / "recipients" / condition / f"seed_{seed:03d}" / f"{sample_id}.json"


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
