"""Post-process unresolved MMVP answers with a batched OpenAI-compatible parser."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

import requests

from inspection.donor_recipient.common import (
    CONDITIONS,
    HYBRID_SCORING_PROTOCOL,
    atomic_json_dump,
    load_manifest,
    normalize_label,
    parse_option,
    parse_seeds,
    response_digest,
    result_path,
    stored_hybrid_score,
)


DEFAULT_ENDPOINT = "https://maas-llm-aiplatform-hcm.api.vngcloud.vn/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it"
PROMPT_VERSION = 1
SYSTEM_MESSAGE = (
    "You extract multiple-choice answers. Follow the requested JSON schema exactly and do not "
    "solve the questions yourself when the model answer does not imply a choice."
)
PROMPT_PREFIX = """Convert each model answer into its intended option choice using the question and options.

Rules:
- Return A or B only when the model answer clearly implies that choice.
- Return null when the answer is empty, irrelevant, ambiguous, or does not imply either option.
- Do not judge whether the model answer is factually correct.
- Return exactly one result for every id.
- Output JSON only, with this schema: {"results":[{"id":"...","choice":"A"},{"id":"...","choice":null}]}

Items:
"""


def load_dotenv(path: str | os.PathLike) -> None:
    """Load simple KEY=VALUE entries without adding a python-dotenv dependency."""
    dotenv = Path(path)
    if not dotenv.is_file():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ[key] = value


def estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens for batching without a server tokenizer."""
    return max(1, math.ceil(len(text.encode("utf-8")) / 3))


def make_item(record_id: str, question: str, model_answer: str) -> dict[str, str]:
    return {"id": record_id, "question": question, "model_answer": model_answer}


def render_user_prompt(items: list[dict[str, str]]) -> str:
    return PROMPT_PREFIX + json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def estimate_request_tokens(items: list[dict[str, str]]) -> int:
    # Include a conservative allowance for chat-message framing tokens.
    return estimate_tokens(SYSTEM_MESSAGE) + estimate_tokens(render_user_prompt(items)) + 12


def _truncated_answer(answer: str, keep: int) -> str:
    head = (keep + 1) // 2
    tail = keep - head
    suffix = answer[-tail:] if tail else ""
    return answer[:head] + "\n...[truncated]...\n" + suffix


def _clip_item(item: dict[str, str], max_prompt_tokens: int) -> dict[str, str]:
    if estimate_request_tokens([item]) <= max_prompt_tokens:
        return item
    answer = item["model_answer"]
    low, high = 0, len(answer)
    while low < high:
        keep = (low + high + 1) // 2
        clipped = dict(item)
        clipped["model_answer"] = _truncated_answer(answer, keep)
        if estimate_request_tokens([clipped]) <= max_prompt_tokens:
            low = keep
        else:
            high = keep - 1
    clipped = dict(item)
    clipped["model_answer"] = _truncated_answer(answer, low)
    if estimate_request_tokens([clipped]) > max_prompt_tokens:
        raise ValueError(
            f"question for {item['id']} exceeds --max_prompt_tokens={max_prompt_tokens} without its answer"
        )
    return clipped


def make_batches(items: list[dict[str, str]], max_prompt_tokens: int) -> list[list[dict[str, str]]]:
    if max_prompt_tokens <= estimate_request_tokens([]):
        raise ValueError("--max_prompt_tokens is too small for the parser instructions")
    batches: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for original in items:
        item = _clip_item(original, max_prompt_tokens)
        candidate = [*current, item]
        if current and estimate_request_tokens(candidate) > max_prompt_tokens:
            batches.append(current)
            current = [item]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def parse_llm_response(content: str, expected_ids: set[str]) -> dict[str, str | None]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM parser response does not contain a JSON object")
        payload = json.loads(cleaned[start:end + 1])
    rows = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("LLM parser response must contain a results array")
    parsed: dict[str, str | None] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("id") not in expected_ids:
            raise ValueError("LLM parser returned an unknown or malformed result id")
        record_id = row["id"]
        if record_id in parsed:
            raise ValueError(f"LLM parser returned duplicate id {record_id}")
        choice = row.get("choice")
        if isinstance(choice, str):
            choice = choice.strip().upper()
        if choice not in {None, "A", "B"}:
            raise ValueError(f"LLM parser returned invalid choice for {record_id}: {choice!r}")
        parsed[record_id] = choice
    missing = expected_ids - parsed.keys()
    if missing:
        raise ValueError(f"LLM parser omitted {len(missing)} result(s)")
    return parsed


def request_batch(
    session: requests.Session,
    endpoint: str,
    api_key: str,
    model: str,
    items: list[dict[str, str]],
    timeout: float,
    max_retries: int,
    max_tokens: int,
) -> dict[str, str | None]:
    payload = {
        "model": model,
        "messages": [
            {"role": "assistant", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": render_user_prompt(items)},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 0.7,
        "presence_penalty": 0,
    }
    for attempt in range(max_retries + 1):
        try:
            response = session.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            return parse_llm_response(content, {item["id"] for item in items})
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            if attempt == max_retries:
                raise
            time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


def _apply_score(path: Path, result: dict, parsed: str | None, gold: object, parsing: dict) -> None:
    result["parsed"] = parsed
    result["correct"] = parsed == normalize_label(gold)
    result["scoring_protocol"] = HYBRID_SCORING_PROTOCOL
    parsing["response_sha256"] = response_digest(result.get("response"))
    result["parsing"] = parsing
    atomic_json_dump(result, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api_key_env", default="AI_PLATFORM_API_KEY")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)
    parser.add_argument("--max_tokens", type=int, default=512, help="maximum response tokens per batch")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="rerun completed LLM fallback parses")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    requested_conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    unknown = sorted(set(requested_conditions) - set(CONDITIONS))
    if unknown:
        parser.error(f"unknown condition(s): {', '.join(unknown)}")
    if args.max_prompt_tokens <= 0 or args.max_tokens <= 0 or args.timeout <= 0 or args.max_retries < 0:
        parser.error("token limits and timeout must be positive; retries must be non-negative")
    try:
        seeds = parse_seeds(args.seeds)
    except ValueError as error:
        parser.error(str(error))

    manifest = load_manifest(args.manifest)
    samples = {sample["id"]: sample for sample in manifest["samples"]}
    deterministic: list[tuple[Path, dict, dict]] = []
    pending: list[tuple[Path, dict, dict, dict[str, str]]] = []
    missing = []
    for condition in requested_conditions:
        for seed in seeds:
            for sample_id, sample in samples.items():
                path = result_path(args.output_dir, condition, seed, sample_id)
                if not path.is_file():
                    missing.append(path)
                    continue
                with open(path, encoding="utf-8") as handle:
                    result = json.load(handle)
                parsed = parse_option(result.get("response"))
                if parsed is not None:
                    deterministic.append((path, result, sample))
                    continue
                if not args.overwrite and stored_hybrid_score(result, sample["gold"]) is not None:
                    continue
                record_id = f"{condition}/seed_{seed:03d}/{sample_id}"
                item = make_item(record_id, sample["question_text"], result.get("response") or "")
                pending.append((path, result, sample, item))

    batches = make_batches([entry[3] for entry in pending], args.max_prompt_tokens)
    print(
        f"[postprocess] deterministic={len(deterministic)} llm_pending={len(pending)} "
        f"batches={len(batches)} missing={len(missing)}"
    )
    if args.dry_run:
        return
    if missing:
        raise FileNotFoundError(f"{len(missing)} result file(s) are missing; first: {missing[0]}")

    for path, result, sample in deterministic:
        parsed = parse_option(result.get("response"))
        _apply_score(
            path,
            result,
            parsed,
            sample["gold"],
            {"method": "deterministic", "prompt_version": PROMPT_VERSION},
        )
    if not pending:
        print("[postprocess] no unresolved answers require LLM parsing")
        return

    load_dotenv(args.dotenv)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"set {args.api_key_env} in the environment or {args.dotenv}")
    by_id = {entry[3]["id"]: entry[:3] for entry in pending}
    session = requests.Session()
    for index, batch in enumerate(batches, 1):
        choices = request_batch(
            session,
            args.endpoint,
            api_key,
            args.model,
            batch,
            args.timeout,
            args.max_retries,
            args.max_tokens,
        )
        for record_id, parsed in choices.items():
            path, result, sample = by_id[record_id]
            _apply_score(
                path,
                result,
                parsed,
                sample["gold"],
                {
                    "method": "llm_fallback",
                    "model": args.model,
                    "prompt_version": PROMPT_VERSION,
                },
            )
        print(f"[postprocess] batch {index}/{len(batches)} parsed {len(batch)} answer(s)")


if __name__ == "__main__":
    main()
