# Monet — Inference & Evaluation Quickstart

Scripts to go from nothing to a running Monet-7B latent-reasoning inference, plus
VLMEvalKit benchmark evaluation. Run from the repo root.

```bash
# --- inference ---
bash run_scripts/00_setup_env.sh        # create conda env `monet` + install deps (once)
bash run_scripts/01_download_model.sh   # download NOVAglow646/Monet-7B -> models/Monet-7B
bash run_scripts/02_run_inference.sh    # run the example (images/example_question.png)

# --- evaluation (VLMEvalKit) ---
bash run_scripts/03_setup_eval.sh                       # clone+install VLMEvalKit, wire in Monet (once)
DATASETS="MMBench_DEV_EN" bash run_scripts/04_run_eval.sh                       # full eval
DATASETS="MMBench_DEV_EN" SUBSET=head   FRAC=0.1 bash run_scripts/04_run_eval.sh   # first 10%
DATASETS="MMBench_DEV_EN" SUBSET=random N=200    bash run_scripts/04_run_eval.sh   # random 200
```

## What each step does
- **00_setup_env.sh** — creates the `monet` conda env (python=3.10) and installs
  `requirements.txt` (vllm==0.10.0, transformers==4.54.0, ...) plus the HF downloader CLI.
- **01_download_model.sh** — downloads the model to `models/Monet-7B`.
- **02_run_inference.sh** — sets `LATENT_SIZE`, points the example at the model, and runs it.
- **03_setup_eval.sh** — clones VLMEvalKit into `./VLMEvalKit`, installs it into the `monet`
  env (then restores the `vllm==0.10.0` / `transformers==4.54.0` pins the Monet runner needs),
  copies `monet_gpu_model_runner.py` into `VLMEvalKit/Monet_models/`, and writes two helper files:
  `sitecustomize.py` (swaps in the Monet vLLM runner in every process) and `run_monet.py`
  (registers the `Monet` model + the required system prompt, and disables md5 TSV re-download).
- **04_run_eval.sh** — restores each dataset's TSV to full, optionally subsets it, then runs
  `run_monet.py --model Monet --data $DATASETS`.
- **eval_subset.py** — helper used by 04 to build a reproducible row-subset of a dataset TSV.

## Evaluation knobs (env vars for 04_run_eval.sh)
| Var | Default | Notes |
|---|---|---|
| `DATASETS` | `MMBench_DEV_EN` | space-separated official VLMEvalKit dataset names |
| `SUBSET` | `full` | `full` \| `head` (first k%/N) \| `random` (seeded) |
| `FRAC` | — | fraction in (0,1], e.g. `0.1`. Set **one** of FRAC/N for subsets |
| `N` | — | absolute sample count, e.g. `200` |
| `SEED` | `0` | seed for `SUBSET=random` |
| `JUDGE` | — | API judge model (e.g. `gpt-4o-mini`); README recommends one. With `JUDGE_BASE_URL`+`JUDGE_KEY` for DeepSeek/Gemini-compatible endpoints |
| `MONET_MAX_NEW_TOKENS` | `2048` | generation budget per question |
| `WORK_DIR` | `eval_outputs/<subset>` | output dir (auto-separated per subset) |

## How the subset works (and its one caveat)
VLMEvalKit has **no built-in subset flag**. Each dataset is a single self-contained TSV in
`$LMUData` (`~/LMUData` by default) with images inline as base64 — so a subset is just a
selection of rows. `eval_subset.py` keeps a one-time full backup (`<name>.full.tsv`) and
overwrites `<name>.tsv` with the chosen rows; `run_monet.py` disables VLMEvalKit's
md5-triggered re-download so the subset is honored. `04_run_eval.sh` always restores to full
first, so runs are reproducible and never accumulate.

**Why in place instead of a renamed dataset:** VLMEvalKit picks the evaluator from the
dataset *name*. A renamed subset becomes a "Custom" dataset — fine for MCQ, but
`CustomVQADataset.evaluate` raises `NotImplementedError`, so VQA subsets would get no score.
Keeping the official name preserves the official scorer for every dataset type. The only
limitation: datasets that don't load from a plain `$LMUData` TSV (a minority — some
HF-streamed / video sets) can't be subsetted this way and should be run with `SUBSET=full`.

## Data
No dataset download is needed for inference — the example uses the bundled image
`images/example_question.png`. To run your own input, edit the `conversations` list in
[inference/vllm_inference_example.py](../inference/vllm_inference_example.py).

## Configurable env vars
| Var | Default | Notes |
|---|---|---|
| `CONDA_BASE` | `~/miniconda3` | conda install location |
| `ENV_NAME` | `monet` | conda env name |
| `MODEL_REPO` | `NOVAglow646/Monet-7B` | use `NOVAglow646/Monet-SFT-7B` for the SFT model |
| `MODEL_DIR` / `MODEL_PATH` | `models/Monet-7B` | where the model lives |
| `LATENT_SIZE` | `10` | latent embeddings per latent block (must match training) |
| `GPU_MEMORY_UTILIZATION` | `0.9` | lower if sharing the GPU |

## Hardware notes
- **RTX 4090 (24GB):** 7B bf16 weights are ~15GB; `0.9` utilization fits with room for
  KV cache. If you hit OOM, lower `GPU_MEMORY_UTILIZATION` or `max_model_len` in
  [inference/load_and_gen_vllm.py](../inference/load_and_gen_vllm.py).
- **A100 80GB:** runs comfortably; same scripts, no changes needed.

## How latent reasoning shows up in output
The model emits `<abs_vis_token>` to enter latent mode; the patched vLLM runner replaces
the next `LATENT_SIZE` tokens with last-layer hidden states until `</abs_vis_token>`.
The example post-processes the output, replacing the (non-readable) latent span with
`<latent>`.
