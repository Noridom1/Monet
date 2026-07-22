# Evaluation

Prepare VLMEvalKit and benchmark data, run evaluations, and summarize scored results. Run commands from the repository root after completing [setup](../setup/README.md).

## Prepare VLMEvalKit

Clone the pinned external checkout, install it into the `monet` environment, apply the tracked Monet patches, and install the runtime adapter:

```bash
bash run_scripts/evaluation/setup_kit.sh
```

The default checkout is `./VLMEvalKit`. Override it with `EVAL_DIR`; override the tested revision with `VLMEVALKIT_REF`.

## Prepare data

In Colab, restore the four Table 3 TSVs from the persistent Drive archive, creating that archive if needed:

```bash
bash run_scripts/evaluation/colab/prepare_data.sh --session monet-a100
```

Then set `LMUData=/content/LMUData` in the Colab console.

To normalize a Hugging Face dataset into `data/<name>/{images,samples.json}` for downstream evaluation or inspection:

```bash
bash run_scripts/evaluation/prepare_data.sh VisualPuzzles
bash run_scripts/evaluation/prepare_data.sh MathVision --split testmini
bash run_scripts/evaluation/prepare_data.sh MyBench --repo org/MyBench --split test
```

Use `--limit N` for a smoke dataset. Image columns are auto-detected; `--image-fields a,b` overrides detection.

## Run VLMEvalKit

Configure judge credentials through exported variables or an uncommitted `.env` file. Policy launchers load `.env`; the general runner expects variables to be exported:

```bash
export MODEL_PATH="$PWD/models/Monet-7B"
export LATENT_SIZE=16
export LMUData="$HOME/LMUData"
export JUDGE="${JUDGE_MODEL:-}"
DATASETS=VStarBench WORK_DIR="$PWD/eval_outputs/vstar" \
bash run_scripts/evaluation/run.sh
```

Subset examples:

```bash
DATASETS=MMBench_DEV_EN SUBSET=head FRAC=0.1 bash run_scripts/evaluation/run.sh
DATASETS=MMBench_DEV_EN SUBSET=random N=200 SEED=0 bash run_scripts/evaluation/run.sh
DATASETS=VStarBench SUBSET=indices INDICES_FILE=targets.csv bash run_scripts/evaluation/run.sh
```

The runner restores the full official TSV before applying a subset so official evaluators remain available and repeated runs do not accumulate subsets. Set `REUSE=0` when previous predictions must not be reused.

## Latent-policy evaluations

Run deterministic VStarBench interventions from an existing judged baseline:

```bash
BASELINE_RESULT=/path/to/Monet_VStarBench_result.xlsx \
bash run_scripts/evaluation/run_latent_policy.sh

JUDGE="<same judge as baseline>" \
BASELINE_RESULT=/path/to/Monet_VStarBench_result.xlsx \
bash run_scripts/evaluation/run_targeted_rescue.sh
```

The first command evaluates forced-latent percentages from `X_VALUES` (default `15 25`). The second reruns only natural-inactive baseline errors. Both write immutable policy manifests and analyses under `eval_outputs/`.

## Summarize Table 3

```bash
python run_scripts/evaluation/summarize_table3.py \
  --work-dir eval_outputs/table3 \
  --latent-sizes 8 10 12 16 \
  --datasets VStarBench HRBench4K HRBench8K MME-RealWorld-Lite
```

This writes Markdown, CSV, activation-detail, and metadata artifacts into the selected work directory.
