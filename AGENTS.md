# Repository Guidelines

## Project Structure & Module Organization

Monet contains three Python stacks. `src/` and `monet_qwen_model/` implement supervised fine-tuning and the Hugging Face Qwen2.5-VL patch. `RL/` is a vendored EasyR1/verl stack with its own Transformers and vLLM patches. `inference/` contains vLLM serving code, while `inspection/` provides latent-capture and visualization tools. Training launchers live in `script_examples/`; setup, evaluation, and dataset workflows live in `run_scripts/`. Store documentation images in `images/`. Treat `eval_outputs/`, `inspection/outputs/`, checkpoints, and prepared datasets as generated artifacts.

## Build, Test, and Development Commands

Use separate environments because SFT/inference and RL require incompatible vLLM versions.

```bash
bash run_scripts/00_setup_env.sh        # create the SFT/inference environment
bash run_scripts/02_run_inference.sh    # run the bundled image example
python -m inspection.generate_latents --help
bash inspection/run_phase_a.sh          # capture a demo trace
bash inspection/run_phase_b.sh          # analyze and visualize that trace
cd RL && make quality                    # run Ruff lint and format checks
```

Launch SFT through `torchrun ... -m src.main`; use the stage templates in `script_examples/`. Launch RL from `RL/` with `python -m verl.trainer.main config=examples/config_monet.yaml ...` and `MONET_RL_PATCH=1`.

## Coding Style & Naming Conventions

Write Python with four-space indentation, `snake_case` functions and modules, `PascalCase` classes, and uppercase environment variables such as `LATENT_SIZE`. Follow nearby typing and docstring patterns; avoid broad refactors in vendored `RL/verl/` code. RL uses Ruff with a 119-character line limit and double-quoted strings (`RL/pyproject.toml`). Keep shell scripts executable and name ordered workflows with numeric prefixes, as in `run_scripts/02_run_inference.sh`.

## Testing Guidelines

There is currently no dedicated automated test suite or coverage threshold. Validate focused changes with CLI `--help`, the smallest relevant inspection or inference smoke run, and `cd RL && make quality` for RL edits. GPU-dependent changes should document the model, GPU count, environment variables, and command used. Never claim heavyweight training or evaluation passed unless it was actually run.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative, area-prefixed subjects such as `Latent Inspection: Add more visual sketches`. Keep each commit scoped and avoid committing large generated traces or checkpoints. Pull requests should explain the affected stack, behavioral change, verification commands, required environment/configuration, and linked issue. Include before/after images for visualization changes and call out any modification to latent token IDs, patch ordering, or checkpoint compatibility.

## Configuration & Safety

Do not commit API keys or local paths. Supply judge credentials through `GOOGLE_API_KEY` or `DEEPSEEK_API_KEY`. Preserve the context-specific model-patching mechanism described in `CLAUDE.md`; SFT, RL, and inference patches are not interchangeable.
