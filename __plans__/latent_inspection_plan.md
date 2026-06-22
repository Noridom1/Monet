# Plan: Inspecting Monet Latent Tokens (Logit-Lens + Attention)

Status: **draft / pre-implementation**
Owner: research (LVR interpretability)
Scope: read-only analysis tooling. No change to training/inference behaviour.

---

## 0. Background facts this plan depends on (verified in code)

- A latent "token" in Monet is **not a token id** — it is the **last-layer hidden state**
  (`[hidden=3584]`) that the runner feeds back as the next input embedding. During vLLM
  generation the recorded token ids inside `<abs_vis_token> … </abs_vis_token>` are
  meaningless placeholders whose embeddings were overwritten
  (`inference/vllm/monet_gpu_model_runner.py:1434-1455`, `1641-1666`). The `<latent>`
  string printed by `inference/vllm_inference_example.py:9-11` is a **cosmetic regex
  substitution**, not model output.
- Therefore the objects we must capture and inspect are the **latent hidden-state vectors**,
  plus their position in the sequence.
- Special token ids / config attributes are set the same way everywhere
  (`src/main.py:100-125`): `<abs_vis_token>`=start, `</abs_vis_token>`=end,
  `<abs_vis_token_pad>`=`config.latent_token_id` (the placeholder we will record at latent
  positions), `<|image_pad|>`=`config.image_token_id`. `LATENT_SIZE` comes from env.
- The HF model (`monet_qwen_model/modeling_qwen2_5_vl_monet.py`) uses **eager/sdpa attention
  and returns `attn_weights`** and already contains the memory-safe per-query attention
  recipe in the `collect_emphasize_attn` block (`Qwen2_5_VLAttention.forward`, lines
  948-1063). vLLM uses fused kernels and **cannot** yield attention — all inspection happens
  in the HF model.

---

## 1. Objectives of the inspection

### Objective A — Logit-lens decoding of latent hidden states
For every latent hidden-state vector produced during a generation, answer:
**"what vocabulary tokens does this latent most resemble?"**

- A.1 (primary) **Final-layer logit lens**: project the latent hidden state through the
  model's final norm + `lm_head` → softmax → **top-k token ids + probabilities**.
- A.2 (secondary) **Depth-resolved logit lens**: do the same at each transformer layer's
  residual stream at the latent position, to watch the latent's "meaning" evolve across
  depth (uses `output_hidden_states`).

### Objective B — Attention maps
For a chosen generation, extract two attention relationships **without materialising the
full `S×S` matrix**:

- B.1 **text → latent**: how much each generated-answer token attends to each latent token.
- B.2 **latent → image**: how much each latent token attends to each image patch, folded
  into the 2D image-patch grid → a heatmap overlay per latent token.

Both broken out **per layer and per head** (averaging is a post-processing choice, not
forced at capture time).

---

## 2. Execution flow (two phases, one driver)

```
            ┌─────────────────────────── Phase A: GENERATE + CAPTURE ───────────────────────────┐
 input ───► HF latent-generation loop (greedy, deterministic)                                    │
 (image+   │   • normal decode until <abs_vis_token>                                              │
  prompt)  │   • for LATENT_SIZE steps: feed back last hidden state, record placeholder id        │
            │   • close with </abs_vis_token>, resume normal decode                                │
            └───► writes a TRACE: input_ids, inputs_embeds, latent_positions, image_positions,    │
                  image_grid_thw, generated_text, per-latent hidden states                        │
                                            │
            ┌──────────────────────────── Phase B: INSPECT ─────────────────────────────────────┐
 trace ───► single teacher-forced forward (no_grad, plain causal mask)                            │
            │   • output_hidden_states=True  → Objective A (logit lens, all layers)               │
            │   • attention-capture hook     → Objective B (sliced [H,Q,S] per layer)             │
            └───► writes ARTIFACTS: logit-lens tables, attention tensors, heatmap PNGs, report    │
```

**Why teacher-forced (not autoregressive) for Phase B:** in one forward over the already-known
full sequence, every query position attends to all earlier positions simultaneously, giving
clean `[query, key]` maps. Autoregressive decode would fragment this across hundreds of
length-1 steps. Phase A *is* autoregressive (it has to be — that's generation); Phase B replays
its output once.

**Determinism contract:** Phase A uses **greedy** decode (`argmax`, `temperature=0`) so the run
is reproducible and so it can be validated against a greedy vLLM run on the same example.

---

## 3. Changes to the codebase

Guiding principle: **add new files under `inspection/`; do not edit training/inference code.**
The one mechanism that needs runtime instrumentation (attention slicing) is installed by
**monkey-patching from the inspection process**, leaving source files untouched.

### 3.1 New files

| File | Responsibility |
|---|---|
| `inspection/__init__.py` | package marker |
| `inspection/load_model.py` | apply the SFT patch (`monet_qwen_model.apply_qwen2_5_monet`), load `Qwen2_5_VLForConditionalGeneration` + processor, set the `config.*` special-token ids exactly as `src/main.py:100-125`. Returns `(model, processor, special_ids)`. |
| `inspection/generate_latents.py` | **Phase A.** The HF latent-generation loop (see 3.2). Produces and saves a `Trace`. |
| `inspection/attn_hook.py` | **Phase B instrument.** Installs/removes the attention-capture monkey-patch (see 3.3). Holds a module-level capture buffer keyed by `layer_idx`. |
| `inspection/logit_lens.py` | **Objective A.** Given hidden states `[N_latent, H]` (+ optional per-layer), apply final norm + `lm_head`, return top-k token ids/strings/probs. |
| `inspection/inspect.py` | **Phase B driver.** Loads a trace, runs the single teacher-forced forward with hooks + `output_hidden_states`, calls logit_lens + attention extraction, writes artifacts. |
| `inspection/visualize.py` | render latent→image heatmaps over the (resized) input image; render text→latent and per-layer/head summaries. |
| `inspection/run_inspection.sh` | env (`LATENT_SIZE`, `MODEL_PATH`) + `python -m inspection.generate_latents …` then `python -m inspection.inspect …`. Mirrors `run_scripts/02_run_inference.sh`. |

### 3.2 Phase A — the HF latent-generation loop (`generate_latents.py`)

Re-implements the runner's latent state machine in HF (~30 lines), mirroring
`monet_gpu_model_runner.py:1641-1666`:

1. Build inputs with the processor (same path as `inference/load_and_gen_vllm.py:75-94`):
   prompt via `apply_chat_template`, image via `process_vision_info`. Get `input_ids`,
   `pixel_values`, `image_grid_thw`.
2. Decode autoregressively with `use_cache=True`, **greedy**, requesting
   `output_hidden_states=True` so each step exposes the last-layer hidden state.
3. State machine per step:
   - normal: append `argmax` token.
   - on `<abs_vis_token>` (start id): set `active`, `count=0`.
   - while `active`: **do not embed the sampled token**; instead set next-step
     `inputs_embeds` row = this step's last hidden state (`hidden_states[-1][:, -1, :]`).
     Record placeholder id = `config.latent_token_id` (`<abs_vis_token_pad>`) at this
     position and store the hidden-state vector. `count += 1`.
   - when `count == LATENT_SIZE` (or end id): append `</abs_vis_token>`, clear `active`.
4. Stop at EOS / `max_tokens`.
5. Assemble and save the **Trace** (see §4.1). Recording `latent_token_id` at latent
   positions means Phase B finds them with the same
   `(input_ids == config.latent_token_id).nonzero()` used in
   `modeling_qwen2_5_vl_monet.py:1736`.

**Validation step (gate before trusting anything):** run the same example through vLLM
(`02_run_inference.sh`) in greedy mode and assert the decoded answer text matches Phase A.
If mismatch → masking/sampling differs and must be reconciled before proceeding.

### 3.3 Phase B attention capture — the monkey-patch (`attn_hook.py`)

`output_attentions=True` would build `[layers × heads × S × S]` (tens of GB for VL seq lengths
— the explosion risk discussed earlier). Instead we install an **instrumented copy** of
`Qwen2_5_VLAttention.forward` at runtime that reuses the existing memory-safe recipe
(`modeling_qwen2_5_vl_monet.py:1008-1024`):

```
q_b    = query_states[b, :, QUERY_POSS, :]                 # [H, Q, D]  post-RoPE
k_rep  = repeat_kv(key_states, num_key_value_groups)       # [H, S, D]  GQA-expanded
logits = einsum('hqd,hsd->hqs', q_b, k_rep) * scaling      # [H, Q, S]  — never S×S
probs  = softmax(logits + causal_mask_rows, dim=-1)        # [H, Q, S]
store probs[:, :, KEY_COLS].to('cpu', float16)             # slice keys, offload, free
```

- `QUERY_POSS` and `KEY_COLS` are set globally before the forward (latent/text/image index
  sets from the trace). Capturing only selected query rows + key cols keeps memory at
  `O(H·Q·S)` (megabytes), not `O(layers·H·S²)`.
- Patch is installed by `attn_hook.install(model)` and reverted by `attn_hook.remove()` so the
  process leaves the class clean. **Zero edits to the source file.**
- Fallback for short sequences / sanity check: a flag to use stock `output_attentions=True`
  and slice afterwards, to cross-check the patched path returns identical numbers.

### 3.4 No changes required to

`src/*`, `monet_qwen_model/*` (only imported + monkey-patched at runtime), `inference/*`,
`run_scripts/*`, `RL/*`.

---

## 4. Outputs / artifacts

All written under `inspection/outputs/<run_id>/`.

### 4.1 Trace (Phase A) — `trace.pt`
A single torch save with:
- `input_ids` `[S]` (full prompt+generation, placeholder id at latent positions)
- `inputs_embeds` `[S, H]` (image embeds scattered in, latent hidden states substituted)
- `latent_positions` `List[int]`, grouped per latent block
- `image_positions` `List[int]`, plus `image_grid_thw` `[num_img, 3]`, `spatial_merge_size`
- `answer_positions` `List[int]` (generated text after `answer_start_pattern`)
- `generated_text` (raw) and `cleaned_text` (with `<latent>` substitution, for reading)
- `latent_hidden_states` `[N_latent, H]` (the captured vectors)
- meta: `model_path`, `LATENT_SIZE`, token-id map, greedy flag

### 4.2 Logit-lens artifacts (Objective A)
- `logit_lens_final.json` — per latent token: `{latent_idx, block_idx, step_in_block,
  topk:[{token_id, token_str, prob}]}` (default k=20).
- `logit_lens_by_layer.npz` *(optional, A.2)* — `[N_latent, num_layers, k]` token ids + probs.
- `logit_lens.md` — human-readable table, one row per latent token showing its top tokens,
  and (A.2) a per-layer trajectory of the top-1 token.

### 4.3 Attention artifacts (Objective B)
- `attn_text2latent.npz` — `[num_layers, H, Q_text, N_latent]` (float16).
- `attn_latent2image.npz` — `[num_layers, H, N_latent, N_image]` (float16) + the grid shape
  `(h/merge, w/merge)` so it can be reshaped to heatmaps.
- `heatmaps/latent{li}_layer{l}_head{h}.png` — latent→image attention overlaid on the resized
  input image (plus head-averaged and layer-averaged summary PNGs).
- `attn_text2latent.png` — heatmap of answer tokens (rows) × latent tokens (cols).

### 4.4 Report — `report.md`
Per run: the question + image, the cleaned generated answer, the logit-lens table for each
latent token, the strongest text→latent links, and thumbnails of the latent→image heatmaps.
The single artifact to skim per example.

---

## 5. Open questions / risks to confirm during implementation

1. **Greedy parity** between HF Phase A and vLLM (the §3.2 gate). If it fails, inspect whether
   generation applies any non-causal masking to latent tokens that the plain-causal replay
   omits.
2. **Final-norm choice for logit lens**: confirm the exact module (`language_model.norm` /
   `model.norm`) so A.1 matches how the model actually decodes; an un-normed logit lens is a
   known pitfall.
3. **Per-latent vs per-block**: `LATENT_SIZE` (=10) hidden states per `<abs_vis_token>` block.
   Decide whether logit-lens/attention are reported per individual latent or aggregated per
   block (plan keeps per-latent; aggregation is a post-step).
4. **Memory ceiling**: even sliced, keeping all `num_layers` for `attn_latent2image` is fine
   (`Q=N_latent` is tiny); `attn_text2latent` `Q_text` can be large — cap or stream answer
   tokens if needed, and `log` if truncated.

---

## 6. Suggested implementation order

1. `load_model.py` + confirm model loads and special ids resolve.
2. `generate_latents.py` + **greedy parity gate** vs vLLM.
3. `logit_lens.py` + `inspect.py` (Objective A only) → produce `logit_lens.md`. Cheapest,
   highest insight per line.
4. `attn_hook.py` + extend `inspect.py` (Objective B) → attention tensors.
5. `visualize.py` + `report.md`.
