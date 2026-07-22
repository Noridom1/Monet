# Latent inspection

Capture Monet latent states and generate logit-lens, attention, and scoring reports. Run commands from the repository root after completing [setup](../setup/README.md). GPU capture requires an A100-class device; analysis reuses the `monet` environment.

## Single bundled example

```bash
bash run_scripts/latent_inspection/capture_demo.sh
bash run_scripts/latent_inspection/analyze_demo.sh
```

The default trace is `inspection/outputs/demo/trace.pt`; reports are written beside it. Override `MODEL_PATH`, `LATENT_SIZE`, or `TRACE` as needed.

## Batch inspection from evaluation results

Prepare a balanced manifest from a VLMEvalKit result, then capture and analyze every selected sample:

```bash
RESULTS=/path/to/Monet_MMBench_DEV_EN_result.xlsx \
DATASET=MMBench_DEV_EN \
bash run_scripts/latent_inspection/prepare_eval_samples.sh

bash run_scripts/latent_inspection/run_batch.sh
```

Common batch overrides are `MANIFEST`, `OUT_DIR`, `TEMPERATURE`, `TOP_K`, `TOP_P`, `REPETITION_PENALTY`, `SEED`, and `MAX_NEW_TOKENS`.

## Batch inspection from normalized Hugging Face data

First prepare the dataset through the [evaluation data workflow](../evaluation/README.md), then create an inspection manifest:

```bash
bash run_scripts/evaluation/prepare_data.sh VisualPuzzles
DATA_DIR=data/VisualPuzzles N=10 MODE=random SEED=0 \
bash run_scripts/latent_inspection/prepare_dataset_samples.sh

MANIFEST=data/VisualPuzzles/inspect_manifest.json \
OUT_DIR=inspection/outputs/VisualPuzzles \
bash run_scripts/latent_inspection/run_batch.sh
```

## Temperature sweep

```bash
TEMPERATURES="0.1 0.3 0.5 0.7" \
MANIFEST=data/inspect_samples/samples.json \
OUT_ROOT=inspection/outputs/temp_sweep \
bash run_scripts/latent_inspection/run_temperature_sweep.sh
```

## Donor-recipient experiment

```bash
GPU_ID=0 \
DONOR_MODEL_PATH=models/Monet-7B \
RECIPIENT_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
bash run_scripts/latent_inspection/run_donor_recipient.sh
```

The implementation modules and detailed artifact schema remain documented in [../../inspection/README.md](../../inspection/README.md).
