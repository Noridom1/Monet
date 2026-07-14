#!/usr/bin/env python
"""GPU smoke test for natural, forced, and suppressed Monet generation."""

from __future__ import annotations

import os

import PIL.Image
from transformers import AutoProcessor

import inference.apply_vllm_monet  # noqa: F401 - installs the runner before vLLM import
from inference.load_and_gen_vllm import (
    vllm_generate,
    vllm_mllm_init,
    vllm_mllm_process_batch_from_messages,
)
from inference.vllm.latent_policy_logits import (
    FORCE_FIRST_POLICY,
    POLICY_EXTRA_ARG,
    SUPPRESS_LATENT_START_POLICY,
)


LATENT_START_ID = 151666


def main() -> None:
    model_path = os.environ.get("MODEL_PATH", "models/Monet-7B")
    gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))
    engine, base_params = vllm_mllm_init(
        model_path,
        tp=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    conversations = [[{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Question: Which car has the longest rental period? "
                    "Answer with the option letter in \\boxed{}."
                ),
            },
            {
                "type": "image",
                "image": PIL.Image.open("images/example_question.png").convert("RGB"),
            },
        ],
    }]]
    inputs = vllm_mllm_process_batch_from_messages(conversations, processor)

    natural_params = base_params.clone()
    natural_params.max_tokens = 128
    natural = vllm_generate(inputs, natural_params, engine)[0].outputs[0]

    forced_params = base_params.clone()
    forced_params.max_tokens = 128
    forced_params.extra_args = {POLICY_EXTRA_ARG: FORCE_FIRST_POLICY}
    forced = vllm_generate(inputs, forced_params, engine)[0].outputs[0]

    suppressed_params = base_params.clone()
    suppressed_params.max_tokens = 128
    suppressed_params.extra_args = {POLICY_EXTRA_ARG: SUPPRESS_LATENT_START_POLICY}
    suppressed = vllm_generate(inputs, suppressed_params, engine)[0].outputs[0]

    natural_repeat_params = base_params.clone()
    natural_repeat_params.max_tokens = 128
    natural_repeat = vllm_generate(inputs, natural_repeat_params, engine)[0].outputs[0]

    natural_blocks = list(natural.token_ids).count(LATENT_START_ID)
    forced_blocks = list(forced.token_ids).count(LATENT_START_ID)
    suppressed_blocks = list(suppressed.token_ids).count(LATENT_START_ID)
    if not forced.token_ids or forced.token_ids[0] != LATENT_START_ID:
        raise RuntimeError("force_first did not emit the latent start token first")
    if forced_blocks < 1:
        raise RuntimeError("force_first did not activate latent mode")
    if suppressed_blocks != 0:
        raise RuntimeError("suppress_latent_start emitted the latent start token")
    if natural.token_ids != natural_repeat.token_ids:
        raise RuntimeError("natural greedy generation changed after a forced request")

    print(
        "[latent-policy-smoke] passed "
        f"natural_blocks={natural_blocks} forced_blocks={forced_blocks} "
        f"suppressed_blocks={suppressed_blocks} natural_tokens={len(natural.token_ids)} "
        f"forced_tokens={len(forced.token_ids)} suppressed_tokens={len(suppressed.token_ids)}"
    )


if __name__ == "__main__":
    main()
