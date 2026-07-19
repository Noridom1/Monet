# Running Monet with the Colab CLI

The scripts in this directory package the local Monet source, create or reuse a
Google Colab GPU session, unpack the source under `/content/Monet`, prepare the
conda-based inference environment, and restore evaluation data from Drive.

The dependency setup has been smoke-tested on a Colab T4 runtime. Monet's vLLM
V1 inference runner requires compute capability 8.0 or newer, so the session
launcher now requests an A100 by default. It uses Colab's standard system RAM;
the launcher does not request a high-RAM runtime.

## What is and is not provisioned

The automated setup includes:

- the current local Monet source, including tracked and untracked changes;
- Miniconda under `/root/miniconda3`;
- the `monet` conda environment from `run_scripts/00_setup_env.sh`; and
- a CUDA/import validation after dependency installation.

The source archive intentionally excludes `.git`, `models`, `data`,
`eval_outputs`, `VLMEvalKit`, checkpoints, inspection outputs, and common
caches. Model weights, VLMEvalKit, judge credentials, and output persistence
must be provisioned separately. The four Table 3 evaluation datasets have a
separate, Conda-free Drive workflow described below.

Do not upload a local conda environment or `site-packages`. Recreate the
environment in Colab with the dependency script so its binaries match the
remote CUDA runtime.

## Complete manual flow

Run the following commands locally from the Monet repository root.

Create an A100 session with standard system RAM and upload the source:

```bash
bash run_scripts/colab/prepare_session.sh \
  --session monet-a100
```

`--gpu A100` is optional because A100 is the script default. There is no
high-RAM flag in this workflow.

Install Miniconda and the inference dependencies remotely:

```bash
bash run_scripts/colab/prepare_dependencies.sh \
  --session monet-a100
```

Check the actual accelerator allocated by Colab and open a console:

```bash
colab status -s monet-a100
colab console -s monet-a100
```

Restore the four evaluation datasets from the persistent Drive archive:

```bash
bash run_scripts/colab/prepare_eval_data.sh --session monet-a100
```

This mounts Drive, verifies
`MyDrive/Monet/monet_eval_datasets.zip`, and extracts the TSVs to
`/content/LMUData`. If the archive does not exist, the same command downloads
VStarBench, HRBench4K, HRBench8K, and MME-RealWorld-Lite directly into a new
archive before restoring them. It does not install Conda or download a model.
Set `LMUData` before running VLMEvalKit:

```bash
export LMUData=/content/LMUData
```

Inside the Colab console, activate and verify the environment:

```bash
cd /content/Monet
source /root/miniconda3/etc/profile.d/conda.sh
conda activate monet

nvidia-smi
python -c "import torch, transformers, vllm; print(torch.cuda.get_device_name()); print(torch.__version__, transformers.__version__, vllm.__version__)"
```

Download the default Monet-7B model:

```bash
bash run_scripts/01_download_model.sh
```

For a gated or private Hugging Face repository, authenticate first:

```bash
hf auth login
bash run_scripts/01_download_model.sh
```

Run the bundled inference example:

```bash
GPU_MEMORY_UTILIZATION=0.98 \
MODEL_PATH=/content/Monet/models/Monet-7B \
bash run_scripts/02_run_inference.sh
```

## A100 and T4 expectations

The Monet inference patch forces vLLM V1. A T4 has compute capability 7.5, but
this vLLM version requires compute capability 8.0 or newer for V1, so the
bundled Monet inference example cannot run on T4 merely by setting
`VLLM_USE_V1=0`. An A100 has compute capability 8.0 and addresses that specific
compatibility requirement.

For full Monet-7B inference or evaluation, use the default A100 session:

```bash
bash run_scripts/colab/prepare_session.sh \
  --session monet-a100

bash run_scripts/colab/prepare_dependencies.sh \
  --session monet-a100
```

The requested accelerator is not guaranteed. Always confirm it with
`colab status` and `nvidia-smi` before preparing dependencies, downloading
large assets, or starting an evaluation. If the allocated GPU is not an A100,
stop the session immediately with `colab stop -s monet-a100` to avoid consuming
compute unnecessarily.

## Refreshing local source

After editing the local checkout, refresh `/content/Monet` in an existing
session without allocating another runtime:

```bash
bash run_scripts/colab/prepare_session.sh \
  --session monet-a100 \
  --reuse
```

This replaces `/content/Monet`. Models or datasets stored inside that directory
are not preserved because they are excluded from the uploaded archive. Keep
large reusable assets outside `/content/Monet` or restore them after a refresh.

The dependency setup is idempotent and can be rerun when the environment setup
changes:

```bash
bash run_scripts/colab/prepare_dependencies.sh --session monet-a100
```

## Ephemeral storage and cleanup

Colab runtime storage is ephemeral. Copy evaluation outputs, logs, and other
artifacts to persistent storage or download them locally before the session is
stopped or expires.

Stop the session when finished:

```bash
colab stop -s monet-a100
```

Starting a new runtime requires source transfer and dependency preparation
again. Model and dataset provisioning may also be required unless those assets
are restored from persistent storage.
