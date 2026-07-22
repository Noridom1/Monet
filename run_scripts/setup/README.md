# Setup

Prepare the shared SFT/inference environment and Monet model. Run commands from the repository root.

## Local machine

Create or refresh the `monet` Conda environment and download the model when a complete local checkpoint is not already present:

```bash
bash run_scripts/setup/local.sh
```

Common overrides:

```bash
ENV_NAME=monet-dev \
MODEL_REPO=NOVAglow646/Monet-SFT-7B \
MODEL_DIR="$PWD/models/Monet-SFT-7B" \
bash run_scripts/setup/local.sh
```

Set `FORCE_MODEL_DOWNLOAD=1` to run the Hugging Face download again. The lower-level commands are also available when only one setup step is needed:

```bash
bash run_scripts/setup/environment.sh
bash run_scripts/setup/download_model.sh
```

`CONDA_BASE` defaults to `~/miniconda3`; `ENV_NAME` defaults to `monet`.

## Google Colab

Install and authenticate the Google Colab CLI first, then run the end-to-end setup:

```bash
bash run_scripts/setup/colab.sh --session monet-a100 --gpu A100
```

This creates the session, uploads the current source tree to `/content/Monet`, prepares `/root/miniconda3/envs/monet`, reports the requested and allocated accelerator, and always downloads the model into the runtime. Reusing a session still replaces `/content/Monet` and downloads the model again:

```bash
bash run_scripts/setup/colab.sh --session monet-a100 --gpu A100 --reuse
```

Select another checkpoint or destination with `--model-repo` and `--model-dir`. Monet's vLLM V1 runner requires compute capability 8.0 or newer; verify the allocated GPU in the command output before starting inference or evaluation.

Open the prepared runtime with:

```bash
colab console -s monet-a100
```

Before a Colab runtime is stopped, copy a remote output directory back to its matching local path:

```bash
bash run_scripts/setup/colab/download_outputs.sh \
  --session monet-a100 \
  --remote-dir /content/Monet/eval_outputs/table3
```

Evaluation kit and dataset setup are intentionally separate; see [../evaluation/README.md](../evaluation/README.md).
