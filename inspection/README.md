# Latent inspection

Tooling to inspect Monet's **latent visual tokens** — the last-layer hidden-state vectors
(`[hidden=3584]`) the runner feeds back as input embeddings inside
`<abs_vis_token> … </abs_vis_token>`. A latent is *not* a token id; the placeholder ids
printed during generation are meaningless. So we capture the actual vectors and ask two
questions:

- **Objective A — logit lens:** what vocabulary tokens does each latent most resemble?
  (final-layer + depth-resolved.)
- **Objective B — attention:** how much do generated tokens attend *back* to the latents
  (text→latent), and where in the image does each latent look (latent→image), without ever
  materialising the full `S×S` matrix.

> **Runs on the A100 / `monet` env.** The 7B model and the localized eval dataset
> (`~/LMUData`) are required. A laptop can only re-analyse the produced `.npz`/`trace.pt`
> with numpy after download.

## Two phases

| Phase | File | What it does |
|---|---|---|
| **A** capture | `generate_latents.py` | Greedy HF generation that mirrors the vLLM latent state machine and records each latent vector + its position into a `trace.pt`. |
| **B** analyse | `inspect.py` | One teacher-forced replay of the trace → logit lens (`logit_lens.py`), sliced attention (`attn_hook.py`), figures (`visualize.py`), and a `report.md`. A numerical gate checks the replay reproduces generation. |

## Quick start — single demo example

```bash
bash inspection/run_phase_a.sh     # -> inspection/outputs/demo/trace.pt
bash inspection/run_phase_b.sh     # -> inspection/outputs/demo/{report.md, logit_lens.md, *.npz, heatmaps/}
```

## Batch: inspect real eval samples (correct vs. incorrect)

Pick a handful of samples straight from a VLMEvalKit eval run and inspect each one, so you
can compare latent behaviour on questions the model got right vs. wrong.

### 1. Prepare samples into `data/`

```bash
bash inspection/run_prepare.sh
# env knobs: RESULTS=<*_result.xlsx>  DATASET=MMBench_DEV_EN
#            N_CORRECT=4  N_INCORRECT=5  SEED=0  OUT=data/inspect_samples
```

`prepare_eval_samples.py` reads the scored xlsx, **re-derives correctness** from the boxed
answer in `prediction` vs. the gold `answer` (the `hit` column is unreliable — the judge fell
back to `exact_matching`), then seeded-random picks **4 correct + 5 genuine-wrong** samples
(genuine-wrong = a valid option letter that differs from gold, not a parse/format failure).

For each pick it reconstructs the **exact** eval input via VLMEvalKit's
`build_dataset(...).build_prompt(...)` and writes:

```
data/inspect_samples/
  images/<id>.png          # localized image the model saw
  samples.json             # manifest: question_text, gold, pred_letter, hit, category, …
```

> The downloaded result files carry **no images** — they live only in the source dataset TSV
> under `~/LMUData`. If a prior eval left a *subset* TSV there, restore full first:
> `python run_scripts/eval_subset.py --dataset MMBench_DEV_EN --mode restore`.

### 2. Capture + analyse all samples

```bash
bash inspection/run_batch.sh
# env knobs: MANIFEST=data/inspect_samples/samples.json  OUT_DIR=inspection/outputs/eval_samples
```

This loads the model **once** for each phase and loops the manifest:

```
inspection/outputs/eval_samples/
  index.md                 # overview table: verdict, replay gate, effective-rank, latent→image mass
  <id>/report.md           # per-sample report; header shows gold→pred and correctness
  <id>/{logit_lens.md, *.npz, heatmaps/, trace.pt}
```

Start at `index.md` to compare buckets, then open a sample's `report.md` for detail.

### Reproducing the eval prediction

Phase A prepends the same system prompt VLMEvalKit used
(`run_monet.py`: *"You are a helpful multimodal assistant … Put your final answer in
\boxed{}."*) and reuses the official `build_prompt` text, so the captured generation should
match the original eval prediction. Each `report.md` and `index.md` shows the gold answer vs.
the model's predicted letter so you can confirm.

## Batch: inspect a prepared dataset (VisualPuzzles, MathVision, …)

To inspect arbitrary benchmarks downloaded with `run_scripts/05_prepare_data.sh` (which
writes `data/<name>/{images/, samples.json}`), first adapt that into an inspection manifest.
The two formats differ — the prepared `samples.json` is a flat list with `question`+`options`
separate and `answer`, while Phase A/B want a `{system_prompt, samples:[…]}` manifest with
`question_text` (options inlined) and `gold`. `prepare_dataset_samples.py` bridges them and,
since these sets have thousands of rows (inspection is expensive), picks **N** samples.

```bash
# 1. download + normalize the dataset (writes data/<name>/)
bash run_scripts/05_prepare_data.sh VisualPuzzles
bash run_scripts/05_prepare_data.sh MathVision --split testmini

# 2. adapt N samples into data/<name>/inspect_manifest.json (reuses the extracted images)
DATA_DIR=data/VisualPuzzles N=10 MODE=head bash inspection/run_prepare_dataset.sh
DATA_DIR=data/MathVision   N=10 MODE=random SEED=0 bash inspection/run_prepare_dataset.sh

# 3. capture + analyse — run_batch.sh is unchanged, just point MANIFEST/OUT_DIR at it
MANIFEST=data/VisualPuzzles/inspect_manifest.json OUT_DIR=inspection/outputs/VisualPuzzles bash inspection/run_batch.sh
MANIFEST=data/MathVision/inspect_manifest.json   OUT_DIR=inspection/outputs/MathVision   bash inspection/run_batch.sh
```

You get the same artifacts as the eval-sample flow — per-sample `report.md` (header shows
gold answer; the model's freshly-captured answer is embedded), `logit_lens.md`, `*.npz`,
`heatmaps/`, `trace.pt`, and a top-level `index.md`. These datasets carry **no prior model
prediction**, so `pred_letter` is `None` and `index.md`'s verdict column stays blank — read
each `report.md`'s generation block for the model's actual answer. Open-ended rows (e.g. some
MathVision questions without options) keep their raw `gold` string. Field names are
overridable on `prepare_dataset_samples.py` (`--question_field`/`--options_field`/etc.) if a
future dataset names things differently.

### `eval_summary.json` — one evaluation-style summary per run

`run_batch.sh` also writes `<OUT_DIR>/eval_summary.json` (via `summarize_eval.py`), scoring
the run as if it were an eval. It extracts the **last** `\boxed{...}` from each model output
(brace-balanced, so LaTeX like `\boxed{\frac{1}{2}}` works; `null` if no box) and matches it
to the gold answer by **case-insensitive exact string match** (no box ⇒ wrong):

```jsonc
{
  "metadata": {
    "dataset": "VisualPuzzles", "model_path": "…/Monet-7B",
    "num_samples": 10, "indices": [3, 17, …],
    "num_correct": 5, "num_boxed": 9, "accuracy": 0.5, "correctness": "5/10"
  },
  "results": [
    {"id": "VisualPuzzles_000003", "index": 3,
     "model_output": "…\\boxed{A}", "gold": "A", "extracted": "A", "correct": true},
    …
  ]
}
```

It only reads `trace.pt` (no GPU), so you can also produce it straight after Phase A:
`python -m inspection.summarize_eval --manifest <manifest> --out_dir <out_dir>`.

### `eval_summary.json` — one evaluation-style summary per run

`run_batch.sh` also writes `<OUT_DIR>/eval_summary.json` (via `summarize_eval.py`), scoring
the run as if it were an eval. It extracts the **last** `\boxed{...}` from each model output
(brace-balanced, so LaTeX like `\boxed{\frac{1}{2}}` works; `null` if no box) and **exact-
string-matches** it to the gold answer (no box ⇒ wrong):

```jsonc
{
  "metadata": {
    "dataset": "VisualPuzzles", "model_path": "…/Monet-7B",
    "num_samples": 10, "indices": [3, 17, …],
    "num_correct": 5, "num_boxed": 9, "accuracy": 0.5, "correctness": "5/10"
  },
  "results": [
    {"id": "VisualPuzzles_000003", "index": 3,
     "model_output": "…\\boxed{A}", "gold": "A", "extracted": "A", "correct": true},
    …
  ]
}
```

It only reads `trace.pt` (no GPU), so you can also produce it straight after Phase A:
`python -m inspection.summarize_eval --manifest <manifest> --out_dir <out_dir>`.

## CLI (advanced / single-sample)

```bash
# Phase A, one sample
python -m inspection.generate_latents --model_path $MODEL_PATH --out inspection/outputs/demo/trace.pt
# Phase A, batch
python -m inspection.generate_latents --manifest data/inspect_samples/samples.json --out_dir inspection/outputs/eval_samples

# Phase B, one trace
python -m inspection.inspect --trace inspection/outputs/demo/trace.pt --image images/example_question.png
# Phase B, batch
python -m inspection.inspect --manifest data/inspect_samples/samples.json --out_dir inspection/outputs/eval_samples
```

### Sampling configurations

Phase A is greedy by default (`temperature=0.0`). Both the single-sample and batch
launchers accept `TEMPERATURE`, `TOP_K`, `TOP_P`, `REPETITION_PENALTY`, `SEED`, and
`MAX_NEW_TOKENS`. For example:

```bash
TEMPERATURE=0.7 TOP_K=50 TOP_P=0.8 REPETITION_PENALTY=1.0 SEED=42 \
MAX_NEW_TOKENS=2048 OUT_DIR=inspection/outputs/eval-t07-k50-p08-s42 \
bash inspection/run_batch.sh
```

Use a distinct `TRACE` (single sample) or `OUT_DIR` (batch) for each configuration.
Every trace stores the effective values in `meta.sampling`.

`latent_start_candidates` contains each decoding step where `<abs_vis_token>` survives
top-k and top-p filtering. Its `rank` is one-based after repetition penalty and before
truncation. `raw_probability` is the unmodified model softmax probability, while
`sampling_probability` is the final renormalized probability after all sampling controls.

For batch runs, `eval_summary.json` reports `latent_activation_rate`: the fraction of all
manifest samples containing at least one latent block. Missing traces remain in the
denominator, consistent with answer accuracy.

## Files

- `generate_latents.py` — Phase A capture (single `--out` or batch `--manifest`/`--out_dir`).
- `inspect.py` — Phase B driver; `run_for_trace` (preloaded model) / `run_batch` / `run`.
- `prepare_eval_samples.py` — build `data/inspect_samples/` from a VLMEvalKit result xlsx.
- `prepare_dataset_samples.py` — adapt a `run_scripts/05_prepare_data.sh` dataset
  (`data/<name>/samples.json`) into an inspection manifest (`inspect_manifest.json`).
- `summarize_eval.py` — score a batch run into `<out_dir>/eval_summary.json` (boxed-answer
  extraction + correctness); reads traces only, no GPU.
- `logit_lens.py`, `attn_hook.py`, `visualize.py`, `load_model.py` — analysis helpers.
- `run_phase_a.sh`, `run_phase_b.sh` — single-example launchers.
- `run_prepare.sh`, `run_prepare_dataset.sh`, `run_batch.sh` — batch launchers.

Design notes and the validation rationale live in `__plans__/latent_inspection_plan.md`.
