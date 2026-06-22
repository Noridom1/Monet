# Monet — Inference Quickstart

Three scripts to go from nothing to a running Monet-7B latent-reasoning inference.
Run them in order from the repo root.

```bash
bash run_scripts/00_setup_env.sh        # create conda env `monet` + install deps (once)
bash run_scripts/01_download_model.sh   # download NOVAglow646/Monet-7B -> models/Monet-7B
bash run_scripts/02_run_inference.sh    # run the example (images/example_question.png)
```

## What each step does
- **00_setup_env.sh** — creates the `monet` conda env (python=3.10) and installs
  `requirements.txt` (vllm==0.10.0, transformers==4.54.0, ...) plus the HF downloader CLI.
- **01_download_model.sh** — downloads the model to `models/Monet-7B`.
- **02_run_inference.sh** — sets `LATENT_SIZE`, points the example at the model, and runs it.

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
