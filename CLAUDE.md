# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Monet (CVPR 2026) is a training/inference framework that lets Qwen2.5-VL-7B reason in a **latent visual space** — generating continuous embeddings ("visual thoughts") interleaved with text instead of decoding every step into language. The core trick across the whole repo is **replacing the stock Qwen2.5-VL `forward` with a Monet variant** so that special latent tokens are expanded into hidden-state embeddings at runtime.

There are three independent codebases here, each with its own model patch:
- **SFT** (`src/`, `monet_qwen_model/`) — supervised fine-tuning, HF Transformers + DeepSpeed.
- **RL** (`RL/`) — VLPO reinforcement learning, a vendored fork of [EasyR1](https://github.com/hiyouga/EasyR1)/verl + vLLM.
- **Inference** (`inference/`) — vLLM serving with a patched GPU model runner.

## Two conda environments

These are not interchangeable — they pin different Python/vLLM versions.

```bash
# SFT + inference
conda create -n monet python=3.10 && conda activate monet
pip install -r requirements.txt          # vllm==0.10.0, transformers==4.54.0, trl==0.15.2

# RL only
cd RL && conda create -n easyr1 python=3.11 && conda activate easyr1
pip install -r requirements.txt          # vllm==0.8.5
```

## The model-patching mechanism (read this first)

The official Qwen2.5-VL code in `transformers`/`vllm` is overridden at import time. Each context does it differently — match the one you're touching:

| Context | Patch file | How it's applied |
|---|---|---|
| SFT | `monet_qwen_model/modeling_qwen2_5_vl_monet.py` | `monet_qwen_model/apply_qwen2_5_monet.py` swaps `sys.modules["transformers.models.qwen2_5_vl.modeling_qwen2_5_vl"]`; imported at top of `src/main.py`. |
| RL | `RL/monet_models/transformers/`, `RL/monet_models/vllm/` | `RL/monet_rl_patch.py` monkey-patches `Qwen2_5_VLForConditionalGeneration.forward` in-place; imported at top of `RL/verl/trainer/main.py`. Gated by `MONET_RL_PATCH=1`. |
| Inference | `inference/vllm/monet_gpu_model_runner.py` | `inference/apply_vllm_monet.py` (or a `sitecustomize.py`) replaces `vllm.v1.worker.gpu_model_runner`. |

The main forward logic lives in `Qwen2_5_VLModel.forward` and `Qwen2_5_VLForConditionalGeneration.forward` of `modeling_qwen2_5_vl_monet.py` (~130KB, the heart of the project).

**Latent reasoning protocol:** the model emits `<abs_vis_token>` (ID 151666) to enter latent mode; the runner then replaces the next `LATENT_SIZE` decoded tokens with last-layer hidden states until `</abs_vis_token>` (ID 151667). `LATENT_SIZE` is set via env var (`export LATENT_SIZE=10`) at both training and inference. Added special tokens: `<abs_vis_token>`, `</abs_vis_token>`, `<abs_vis_token_pad>`, `<observation>`, `</observation>`.

## SFT: the 3-stage distillation pipeline

Entry point is `python -m src.main` under `torchrun`. Stage scripts are in `script_examples/`. Each stage builds on the previous checkpoint, and stages 2/3 require a **precompute step first** that dumps teacher tensors to disk.

1. **Stage 1** (`--stage sft_stage1`): warm-up text-CoT SFT. `script_examples/sft_stage1.sh`.
2. **Stage 2** (`--stage sft_stage2`): first run `python -m src.precompute_teacher_reps` (with `--output_hidden_states`) to cache observation-token representations from the stage-1 model, then train with an **alignment loss** to those teacher reps. `script_examples/sft_stage2.sh`.
3. **Stage 3** (note: script passes `--stage "avt_v5_stage2"` but `get_args` only accepts `sft_stage{1,2,3}` — verify before running): first run `python -m src.precompute_teacher_latents` to cache target latent embeddings from the stage-2 model, then distill. `script_examples/sft_stage3.sh`.

Key `src/main.py` args (see `get_args` in `src/utils.py` for the full list): `--stage`, `--latent_size`, `--ce_emphasize_factor`, `--alignment_weight`, `--alignment_layer {all_layers,last_layer}`, `--emphasize_latent_weight`, `--teacher_reps_dir`, `--teacher_latent_dir`, `--data_path` (space-separated JSONs), `--dataset_root`, `--deepspeed`. Per-device batch size and grad-accum come from the DeepSpeed config (`deepspeed/ds_zero2_gpu.json`), not the CLI.

`src/` layout: `main.py` (orchestration + collate fns per stage), `task.py` (dataset preprocessing, `Monet_single_input_images_preprocess_function`), `trainer.py` (custom `SFTTrainer` subclasses computing CE + alignment loss; `compute_latents_only_loss` does latent-only backprop), `utils.py` (args, attention-mask/4D-bias builders, image resize helpers).

## RL: VLPO via EasyR1/verl

Entry point: `python -m verl.trainer.main config=examples/config_monet.yaml ...`. Driver script `RL/examples/vlpo_train.sh` (run with `MONET_RL_PATCH=1`, sets many `RAY_*`/`NCCL_*` env vars and `LATENT_SIZE`).

Key Monet-specific config knobs (override on the CLI as `worker.rollout....=`):
- `worker.rollout.sampling_strategy=monet` enables latent reasoning during rollout (this *is* VLPO); `=greedy` is plain text reasoning.
- `worker.rollout.monet.select_acc_threshold=0.6` + `online_difficulty_sampling=true` — select hard samples (accuracy in (0, threshold)).
- `worker.actor.monet_rl_sigma` / `worker.ref.monet_rl_sigma` — the VLPO σ.
- `worker.reward.repetition_penalty=true` — penalize repetitive output.

**External LLM judge:** RL reward uses Gemini or DeepSeek as a rule-based judge via `RL/tools/custom_api.py`. Set `GOOGLE_API_KEY` (for `api_name="gemini-2.5-pro"`) or `DEEPSEEK_API_KEY` (for `api_name="deepseek-chat"`).

After training, FSDP shards must be merged: `bash RL/examples/merge_model.sh` (`python3 -m scripts.model_merger --local_dir=.../actor`).

## Inference

```bash
conda activate monet
export LATENT_SIZE=10
python -m inference.vllm_inference_example   # see run_scripts/inference/run_example.sh
```

Output post-processing: detect `<abs_vis_token>` and replace the enclosed (non-human-readable) latent tokens with a placeholder like `<latent>`.

**VLMEvalKit evaluation:** requires `vllm==0.10.0`. Copy `inference/vllm/monet_gpu_model_runner.py` into the VLMEvalKit repo and create a `sitecustomize.py` that sets `LATENT_START_ID=151666`/`LATENT_END_ID=151667` and remaps `vllm.v1.worker.gpu_model_runner` to it (exact snippet in README.md). Use the system prompt: `You are a helpful multimodal assistant. ... Put your final answer in \boxed{}.` and an API judge rather than exact match.

## Data

SFT: [Monet-SFT-125K](https://huggingface.co/datasets/NOVAglow646/Monet-SFT-125K) — subsets `Visual_CoT`, `CogCoM`, `ReFocus`, `Zebra_CoT_count`, `Zebra_CoT_visual_search`, `Zebra_CoT_geometry`, each with a `train.json`. RL: [Thyme-RL](https://huggingface.co/datasets/Kwai-Keye/Thyme-RL). Models on HF: `NOVAglow646/Monet-7B`, `Monet-SFT-7B`.
