---
name: monet-colab-session
description: Create or reuse a Google Colab CLI GPU session, package and transfer the current Monet source, and provision its conda-based inference dependencies remotely. Use when preparing this Monet checkout for Colab evaluation work, refreshing source or dependencies in an existing session, or explaining the repository's Colab lifecycle.
---

# Prepare a Monet Colab session

Use `run_scripts/setup/colab.sh` from the repository root for end-to-end provisioning. Use the lower-level scripts under `run_scripts/setup/colab/` only when refreshing an individual step; do not recreate their archive or transfer logic manually.

## Execution flow

1. Confirm `colab` is installed and authentication is available.
2. Review `git status --short`. The archive intentionally includes tracked and untracked source changes.
3. Run the setup script. It packages source, creates the named GPU session, uploads the archive, unpacks it into `/content/Monet`, and verifies required files.
4. Report the requested accelerator and the actual accelerator from `colab status`; allocation is not guaranteed merely because it was requested.
5. The end-to-end setup bootstraps Miniconda through `run_scripts/setup/environment.sh` and downloads the requested model on every invocation. For dependency-only refreshes, run `run_scripts/setup/colab/prepare_dependencies.sh --session <session>`.
6. Enter the runtime only when interactive work is requested: `colab console -s <session>`.
7. Provision VLMEvalKit and evaluation datasets as later, separate operations. Never upload a local conda environment or `site-packages`.
8. Download outputs and logs before stopping a session. Stop it only when the user requests cleanup or the task explicitly includes cleanup.

## Commands

Create the default A100 session with standard system RAM:

```bash
bash run_scripts/setup/colab.sh
```

Create a named session or request another GPU:

```bash
bash run_scripts/setup/colab.sh --session monet-a100 --gpu A100
bash run_scripts/setup/colab.sh --session monet-t4 --gpu T4
```

Refresh source in an existing session without allocating another runtime:

```bash
bash run_scripts/setup/colab.sh --session monet-a100 --gpu A100 --reuse
```

Install or refresh the inference conda environment in an existing session:

```bash
bash run_scripts/setup/colab/prepare_dependencies.sh --session monet-a100
```

Inspect and connect:

```bash
colab status -s monet-a100
colab console -s monet-a100
```

## Boundaries

The archive excludes `.git`, `models`, `data`, `eval_outputs`, `VLMEvalKit`, inspection outputs, checkpoints, and common caches. Keep those artifacts separate because they are large, generated, remotely installed, or provisioned by a later workflow.

Creating a session changes external state and consumes Colab resources. The script defaults to A100 and does not request high RAM. If the user merely asks for an explanation or review, do not create one. If creating a session was explicitly requested, run the script directly. Treat authentication prompts, unavailable accelerators, and quota failures as blockers; do not silently select a different accelerator.

The unpack step replaces `/content/Monet` in the selected remote session. Use `--reuse` only when refreshing that directory is intended. It does not delete models or datasets stored elsewhere under `/content`.

Dependency preparation installs Miniconda and packages into the ephemeral Colab runtime. It is idempotent, but a newly allocated runtime must be provisioned again. Do not claim the environment is ready unless the final Python import/version probe succeeds.
