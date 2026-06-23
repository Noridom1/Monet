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

## Files

- `generate_latents.py` — Phase A capture (single `--out` or batch `--manifest`/`--out_dir`).
- `inspect.py` — Phase B driver; `run_for_trace` (preloaded model) / `run_batch` / `run`.
- `prepare_eval_samples.py` — build `data/inspect_samples/` from a VLMEvalKit result xlsx.
- `logit_lens.py`, `attn_hook.py`, `visualize.py`, `load_model.py` — analysis helpers.
- `run_phase_a.sh`, `run_phase_b.sh` — single-example launchers.
- `run_prepare.sh`, `run_batch.sh` — batch launchers.

Design notes and the validation rationale live in `__plans__/latent_inspection_plan.md`.
