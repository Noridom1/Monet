#!/usr/bin/env python
"""Register Monet and its latent-policy hooks with VLMEvalKit."""

import os
import re
import runpy
import sys
from functools import partial

from latent_activation import annotate_latent_response
from latent_policy import (
    LatentPolicyManifest,
    attach_policy_to_sampling_params,
    validate_policy_block_count,
)
from secret_redaction import install_log_secret_redaction, redact_cli_secret


_JUDGE_KEY = os.environ.get("JUDGE_API_KEY", "")
install_log_secret_redaction(_JUDGE_KEY)

if os.environ.get("MONET_RATE_LIMIT_JUDGE") == "1":
    from vlmeval_rate_limit import install as _install_judge_rate_limit

    _install_judge_rate_limit()

# Honor local, possibly subsetted TSVs and keep secrets out of run metadata.
from vlmeval import smp as _smp
from vlmeval.dataset import image_base as _ib
from vlmeval.dataset import image_mcq as _image_mcq


_orig_upsert_run_status = _smp.upsert_run_status


def _upsert_run_status_redacted(run_dir, **fields):
    if "argv" in fields:
        fields["argv"] = redact_cli_secret(fields["argv"])
    return _orig_upsert_run_status(run_dir, **fields)


_smp.upsert_run_status = _upsert_run_status_redacted

_orig_prepare_tsv = _ib.ImageBaseDataset.prepare_tsv


def _prepare_tsv_no_md5(self, url, file_md5=None):
    return _orig_prepare_tsv(self, url, None)


_ib.ImageBaseDataset.prepare_tsv = _prepare_tsv_no_md5

_orig_get_intermediate_file_path = _image_mcq.get_intermediate_file_path


def _get_intermediate_file_path_safe(eval_file, suffix, target_format=None):
    safe_suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", suffix)
    return _orig_get_intermediate_file_path(eval_file, safe_suffix, target_format)


_image_mcq.get_intermediate_file_path = _get_intermediate_file_path_safe

from vlmeval import config as _cfg
from vlmeval import vlm as _vlm


class MonetQwen2VLChat(_vlm.Qwen2VLChat):
    """Qwen2.5-VL adapter with activation capture and per-sample policy control."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tokenizer = self.processor.tokenizer
        self._latent_start_id = int(tokenizer.convert_tokens_to_ids("<abs_vis_token>"))
        self._latent_end_id = int(tokenizer.convert_tokens_to_ids("</abs_vis_token>"))
        if (self._latent_start_id, self._latent_end_id) != (151666, 151667):
            raise RuntimeError(
                "unexpected Monet latent token IDs: "
                f"{self._latent_start_id}/{self._latent_end_id}"
            )

        manifest_path = os.environ.get("MONET_LATENT_POLICY_MANIFEST")
        self._latent_policy = (
            LatentPolicyManifest.load(manifest_path) if manifest_path else None
        )
        self._sample_context = None
        if self._latent_policy is not None:
            self._latent_policy.validate_runtime(
                model_path=self.model_path,
                latent_size=int(os.environ.get("LATENT_SIZE", "0")),
                max_new_tokens=self.max_new_tokens,
                max_pixels=self.max_pixels,
                system_prompt=self.system_prompt or "",
                latent_start_id=self._latent_start_id,
                latent_end_id=self._latent_end_id,
            )
            print(
                "[latent-policy] loaded "
                f"{self._latent_policy.source} sha256={self._latent_policy.digest}"
            )

    def set_monet_dataset_indices(self, dataset, indices):
        if self._latent_policy is not None:
            self._latent_policy.validate_dataset_indices(dataset, indices)

    def set_monet_sample_context(self, dataset, index):
        policy = self._latent_policy.policy_for(dataset, index) if self._latent_policy else None
        self._sample_context = (dataset, index, policy)

    def generate_inner_vllm(self, message, dataset=None):
        if self._latent_policy is not None and self._sample_context is None:
            raise RuntimeError("latent-policy generation is missing sample context")
        policy = self._sample_context[2] if self._sample_context else None
        block_count = 0
        generate = self.llm.generate

        def generate_and_capture(*args, **kwargs):
            nonlocal block_count
            sampling_params = kwargs.get("sampling_params")
            if sampling_params is None and len(args) > 1:
                sampling_params = args[1]
            if sampling_params is None:
                raise RuntimeError("Monet generation did not provide SamplingParams")
            attach_policy_to_sampling_params(sampling_params, policy)
            outputs = generate(*args, **kwargs)
            for request_output in outputs:
                for completion in request_output.outputs:
                    block_count += list(completion.token_ids).count(self._latent_start_id)
            return outputs

        self.llm.generate = generate_and_capture
        try:
            response = super().generate_inner_vllm(message, dataset=dataset)
            validate_policy_block_count(policy, block_count)
            return annotate_latent_response(response, block_count)
        finally:
            self.llm.generate = generate
            self._sample_context = None


_MODEL_PATH = os.environ.get("MODEL_PATH")
assert _MODEL_PATH, "MODEL_PATH env var must point to the Monet model directory"
_SYSTEM_PROMPT = os.environ.get(
    "MONET_SYSTEM_PROMPT",
    "You are a helpful multimodal assistant. You are required to answer the "
    "question based on the image provided. Put your final answer in \\boxed{}.",
)
_cfg.supported_VLM["Monet"] = partial(
    MonetQwen2VLChat,
    model_path=_MODEL_PATH,
    use_vllm=True,
    system_prompt=_SYSTEM_PROMPT,
    post_process=False,
    max_new_tokens=int(os.environ.get("MONET_MAX_NEW_TOKENS", "2048")),
    max_pixels=int(os.environ.get("MONET_MAX_PIXELS", str(1280 * 28 * 28))),
)

_here = os.environ.get("VLMEVAL_ROOT", os.path.dirname(os.path.abspath(__file__)))
if "--judge-key" not in sys.argv and _JUDGE_KEY:
    sys.argv.extend(["--judge-key", _JUDGE_KEY])
sys.argv[0] = os.path.join(_here, "run.py")
runpy.run_path(sys.argv[0], run_name="__main__")
