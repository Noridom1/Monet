"""Summarize a batch latent-inspection run into a single evaluation-style JSON.

Reads the inspection manifest (which samples, their gold answer + dataset index) and each
sample's ``<out_dir>/<id>/trace.pt`` (the model output captured in Phase A), then scores
every sample by extracting the final ``\\boxed{...}`` answer and exact-matching it to gold.

Output: ``<out_dir>/eval_summary.json``::

    {
      "metadata": {
        "dataset", "model_path", "num_samples",
        "indices": [...],            # dataset indices of the samples run
        "num_correct", "num_boxed",  # how many matched / how many emitted a box
        "accuracy", "correctness": "5/10"
      },
      "results": [
        {"id", "index", "model_output", "gold",
         "extracted",   # content of the last \\boxed{...}, or null if none
         "correct"}     # extracted == gold (exact); false if no box
      ]
    }

No GPU/model needed — it only reads traces, so it can run straight after Phase A.

Usage:
    python -m inspection.summarize_eval \
        --manifest data/VisualPuzzles/inspect_manifest.json \
        --out_dir inspection/outputs/VisualPuzzles
"""
import os
import json
import argparse


def extract_boxed(text):
    r"""Return the content of the LAST ``\boxed{...}`` in ``text`` (brace-balanced so LaTeX
    like ``\boxed{\frac{1}{2}}`` works), or None if there is no well-formed box."""
    if not text:
        return None
    key = "\\boxed"
    found, i = [], 0
    while True:
        j = text.find(key, i)
        if j == -1:
            break
        k = j + len(key)
        while k < len(text) and text[k] == " ":
            k += 1
        if k >= len(text) or text[k] != "{":
            i = j + len(key)
            continue
        depth, m, start = 0, k, k + 1
        while m < len(text):
            if text[m] == "{":
                depth += 1
            elif text[m] == "}":
                depth -= 1
                if depth == 0:
                    break
            m += 1
        if depth == 0:
            found.append(text[start:m].strip())
            i = m + 1
        else:
            break  # unterminated box -> stop
    return found[-1] if found else None


def score(extracted, gold):
    """Exact string match, case-insensitive + whitespace-stripped (so 'a' matches 'A').
    No box (extracted is None) -> wrong."""
    if extracted is None or gold is None:
        return False
    return extracted.strip().lower() == str(gold).strip().lower()


def summarize(manifest_path, out_dir, model_path=None):
    import torch

    with open(manifest_path) as f:
        manifest = json.load(f)
    samples = manifest["samples"]

    results, indices, num_correct, num_boxed = [], [], 0, 0
    for sample in samples:
        sid = sample["id"]
        gold = sample.get("gold")
        index = sample.get("index")
        indices.append(index)

        trace_path = os.path.join(out_dir, sid, "trace.pt")
        model_output = None
        if os.path.exists(trace_path):
            trace = torch.load(trace_path, map_location="cpu", weights_only=False)
            model_output = trace.get("generated_text")
        else:
            print(f"[summarize] WARNING: trace missing for {sid} ({trace_path}); output=None.")

        extracted = extract_boxed(model_output)
        correct = score(extracted, gold)
        num_boxed += extracted is not None
        num_correct += correct
        results.append({
            "id": sid,
            "index": index,
            "model_output": model_output,
            "gold": gold,
            "extracted": extracted,
            "correct": correct,
        })

    n = len(results)
    summary = {
        "metadata": {
            "dataset": manifest.get("dataset"),
            "model_path": model_path or os.environ.get("MODEL_PATH"),
            "num_samples": n,
            "indices": indices,
            "num_correct": num_correct,
            "num_boxed": num_boxed,
            "accuracy": (num_correct / n) if n else 0.0,
            "correctness": f"{num_correct}/{n}",
        },
        "results": results,
    }

    out_path = os.path.join(out_dir, "eval_summary.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[summarize] {summary['metadata']['correctness']} correct "
          f"({num_boxed}/{n} emitted a box) -> {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="the inspection manifest used for the run")
    ap.add_argument("--out_dir", required=True, help="where Phase A wrote <id>/trace.pt")
    ap.add_argument("--model_path", default=None, help="recorded in metadata (default $MODEL_PATH)")
    args = ap.parse_args()
    summarize(args.manifest, args.out_dir, model_path=args.model_path)


if __name__ == "__main__":
    main()
