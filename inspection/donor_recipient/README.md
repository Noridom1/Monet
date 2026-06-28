# Donor–recipient latent intervention

This experiment asks whether hidden states produced by the released Monet checkpoint help
an unmodified Qwen2.5-VL recipient answer MMVP questions. It is intentionally separate from
the existing latent-inspection traces.

## Protocol

The donor is forced into one immediate recurrent block of 10 hidden states. Only those BF16
vectors are saved. The recipient receives them directly after its assistant prompt, without
Monet delimiters or donor text. Recipient generation is greedy.

The six conditions are `vanilla_baseline`, `same_sample`, `order_shuffled`,
`norm_matched_random`, `recipient_image_masked`, and `wrong_sample`. The masked condition uses
a uniform RGB-127 image with the original dimensions and verifies that Qwen's visual token
grid is unchanged. Wrong-sample assignments are cyclic derangements.

This differs from the LVR paper: Monet has a different training objective, only its released
checkpoint is used, and the fixed block is forced so every item has a paired intervention.

## Run

The end-to-end launcher defaults to physical GPU 3, BF16, SDPA, seed 0, and the `monet`
environment. Only one 7B model is loaded at a time.

```bash
bash inspection/donor_recipient/run_experiment.sh
```

Important overrides:

```bash
GPU_ID=3 \
DONOR_MODEL_PATH=models/Monet-7B \
RECIPIENT_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
OUTPUT_DIR=inspection/donor_recipient/outputs/mmvp_seed0 \
bash inspection/donor_recipient/run_experiment.sh
```

For a one-target smoke test, the launcher automatically captures the second donor required
by the wrong-sample condition:

```bash
GPU_ID=3 TARGET_LIMIT=1 MAX_NEW_TOKENS=32 \
OUTPUT_DIR=inspection/donor_recipient/outputs/smoke \
bash inspection/donor_recipient/run_experiment.sh
```

Use a separate output directory when changing checkpoint, latent size, prompt protocol, or
seed. Completed compatible artifacts are skipped. Use the Python entry points' `--overwrite`
flags only when intentionally replacing a run.

## Individual stages

```bash
python -m inspection.donor_recipient.prepare_mmvp \
  --data_dir inspection/donor_recipient/data/mmvp

python -m inspection.donor_recipient.generate_donors \
  --manifest inspection/donor_recipient/data/mmvp/manifest.json \
  --model_path models/Monet-7B \
  --output_dir inspection/donor_recipient/outputs/mmvp_seed0

python -m inspection.donor_recipient.run_recipients \
  --manifest inspection/donor_recipient/data/mmvp/manifest.json \
  --donor_dir inspection/donor_recipient/outputs/mmvp_seed0/donors \
  --output_dir inspection/donor_recipient/outputs/mmvp_seed0

python -m inspection.donor_recipient.analyze_results \
  --manifest inspection/donor_recipient/data/mmvp/manifest.json \
  --output_dir inspection/donor_recipient/outputs/mmvp_seed0
```

## Outputs

```text
outputs/<run>/
  donors/<sample_id>.pt
  recipients/<condition>/seed_000/<sample_id>.json
  summary.json
  summary.csv
  report.md
```

The primary metric is per-question A/B accuracy. Scoring prefers `<answer>` and `\\boxed{}`
answers, then uses the last standalone A/B after removing Qwen control tokens. This handles
recipient reasoning introduced by latent injection while applying one parser to every
condition. Analysis refreshes stale derived `parsed`/`correct` fields without rerunning GPU
inference. The report also includes MMVP pair accuracy, paired changes from vanilla,
wrong-to-right/right-to-wrong counts, and paired-bootstrap confidence intervals. The paper's
68.67% vanilla result is recorded as a reference, not enforced as a pass criterion.

## Verification

CPU-only tests and CLI checks:

```bash
python -m unittest inspection.donor_recipient.test_experiment
python -m inspection.donor_recipient.prepare_mmvp --help
python -m inspection.donor_recipient.generate_donors --help
python -m inspection.donor_recipient.run_recipients --help
python -m inspection.donor_recipient.analyze_results --help
```

Peak allocated CUDA memory is recorded in every donor and recipient artifact. The 24 GB RTX
4090 target requires sequential stages; do not run donor and recipient processes concurrently.
