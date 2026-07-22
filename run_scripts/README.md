# Monet workflows

User-facing commands are grouped by purpose. Run every command from the repository root.

| Workflow | Guide | Primary command |
|---|---|---|
| Setup | [setup/README.md](setup/README.md) | `bash run_scripts/setup/local.sh` |
| Evaluation | [evaluation/README.md](evaluation/README.md) | `bash run_scripts/evaluation/run.sh` |
| Latent inspection | [latent_inspection/README.md](latent_inspection/README.md) | `bash run_scripts/latent_inspection/run_batch.sh` |
| Inference | [inference/README.md](inference/README.md) | `bash run_scripts/inference/run_example.sh` |

The SFT and RL training launchers remain in `script_examples/` and `RL/` because they use separate environments and patch stacks.
