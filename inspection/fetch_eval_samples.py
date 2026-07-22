"""Materialize selected VLMEvalKit samples without downloading a full base64 TSV.

The evaluation spreadsheets contain the questions and predictions but not their images.
The upstream benchmark TSVs can be multiple gigabytes, so this utility locates requested
records with HTTP byte-range requests and writes only those images and their metadata.

Example:
    python -m inspection.fetch_eval_samples \
      --dataset MME-RealWorld-Lite \
      --predictions eval_outputs/table3_realworld/latent_10/Monet/Monet_MME-RealWorld-Lite.xlsx \
      --source-url https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Base64/resolve/main/mme_realworld_lite.tsv \
      --indices 3993,4864,4958 \
      --out data/benchmark_inspection/realworld
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree
from zipfile import ZipFile


SHEET_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CELL_REFERENCE = re.compile(r"([A-Z]+)")
# Some upstream TSVs quote every field (including the numeric index), while
# VLMEvalKit-normalized TSVs leave fields unquoted.
ROW_START = re.compile(rb'(?m)^"?(\d+)"?\t')
LATENT_MARKER = re.compile(r"<ltnt:(\d+)>")
MEBIBYTE = 1024 * 1024

# Base64-encoded image fields are routinely several megabytes long.
csv.field_size_limit(2**31 - 1)


def column_index(reference: str) -> int:
    value = 0
    for character in CELL_REFERENCE.match(reference).group(1):
        value = value * 26 + ord(character) - ord("A") + 1
    return value - 1


def shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    tag = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
    return ["".join(node.text or "" for node in item.iter(tag)) for item in root]


def read_xlsx(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as archive:
        strings = shared_strings(archive)
        root = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[dict[int, str]] = []
    for row in root.findall(".//main:sheetData/main:row", SHEET_NS):
        values: dict[int, str] = {}
        for cell in row.findall("main:c", SHEET_NS):
            raw_value = cell.find("main:v", SHEET_NS)
            raw = "" if raw_value is None else raw_value.text or ""
            if cell.attrib.get("t") == "s" and raw:
                values[column_index(cell.attrib["r"])] = strings[int(raw)]
            else:
                values[column_index(cell.attrib["r"])] = raw
        rows.append(values)

    if not rows:
        raise ValueError(f"{path} has no rows")
    header = rows[0]
    return [{header[index]: row.get(index, "") for index in header} for row in rows[1:]]


def remote_size(url: str) -> int:
    with urlopen(Request(url, method="HEAD"), timeout=60) as response:
        value = response.headers.get("Content-Length")
    if value is None:
        raise RuntimeError(f"Could not determine remote file size for {url}")
    return int(value)


def get_range(url: str, start: int, end: int) -> bytes:
    request = Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=120) as response:
        data = response.read()
    expected = end - start + 1
    if len(data) != expected:
        raise RuntimeError(f"Range request returned {len(data)} bytes, expected {expected}")
    return data


def locate_record(
    url: str,
    total_size: int,
    ordinal_by_index: dict[str, int],
    target_ordinal: int,
    probe_size: int,
) -> int:
    """Find the byte offset of a TSV record, knowing its source-row ordinal."""
    lower, upper = 0, total_size - 1
    for attempt in range(24):
        midpoint = (lower + upper) // 2
        start = max(0, midpoint - probe_size // 2)
        end = min(total_size - 1, start + probe_size - 1)
        start = max(0, end - probe_size + 1)
        data = get_range(url, start, end)
        starts = [
            (start + match.start(), ordinal_by_index.get(match.group(1).decode("ascii")))
            for match in ROW_START.finditer(data)
        ]
        starts = [(offset, ordinal) for offset, ordinal in starts if ordinal is not None]
        if not starts:
            if probe_size >= 64 * MEBIBYTE:
                raise RuntimeError(
                    f"No TSV record boundaries found near byte {midpoint}; "
                    "the upstream file may have changed."
                )
            probe_size *= 2
            continue

        ordinals = [ordinal for _, ordinal in starts]
        for offset, ordinal in starts:
            if ordinal == target_ordinal:
                return offset
        minimum, maximum = min(ordinals), max(ordinals)
        if target_ordinal < minimum:
            upper = min(upper, min(offset for offset, ordinal in starts if ordinal == minimum) - 1)
        elif target_ordinal > maximum:
            lower = max(lower, max(offset for offset, ordinal in starts if ordinal == maximum) + 1)
        else:
            raise RuntimeError(
                f"Target ordinal {target_ordinal} lies inside a probe but was not found; "
                "the source order may not match the evaluation spreadsheet."
            )
        if lower >= upper:
            break
    raise RuntimeError(f"Could not locate source row {target_ordinal} after {attempt + 1} probes")


def record_at(url: str, total_size: int, offset: int) -> list[str]:
    """Fetch and parse one TSV row beginning at ``offset``."""
    size = 16 * MEBIBYTE
    while size <= 128 * MEBIBYTE:
        end = min(total_size - 1, offset + size - 1)
        data = get_range(url, offset, end)
        boundary = re.search(rb'\n(?="?\d+"?\t)', data)
        if boundary:
            line = data[: boundary.start()].decode("utf-8")
            return next(csv.reader([line], delimiter="\t"))
        if end == total_size - 1:
            return next(csv.reader([data.decode("utf-8")], delimiter="\t"))
        size *= 2
    raise RuntimeError(f"Could not find the end of record at byte {offset}")


def source_header(url: str, total_size: int) -> list[str]:
    data = get_range(url, 0, min(total_size - 1, MEBIBYTE - 1))
    line, _, _ = data.partition(b"\n")
    return next(csv.reader([line.decode("utf-8")], delimiter="\t"))


def latent_blocks(prediction: str) -> int | None:
    matches = LATENT_MARKER.findall(prediction)
    return int(matches[-1]) if matches else None


def materialize(args: argparse.Namespace) -> None:
    rows = read_xlsx(args.predictions)
    rows_by_index = {str(row["index"]): row for row in rows}
    requested = [value.strip() for value in args.indices.split(",") if value.strip()]
    missing = [index for index in requested if index not in rows_by_index]
    if missing:
        raise ValueError(f"Indices not in {args.predictions}: {', '.join(missing)}")

    ordinal_by_index = {str(row["index"]): ordinal for ordinal, row in enumerate(rows)}
    total_size = remote_size(args.source_url)
    header = source_header(args.source_url, total_size)
    try:
        image_column = header.index("image")
    except ValueError as error:
        raise RuntimeError(f"Source TSV has no `image` column: {header}") from error
    output = args.out
    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    samples = []

    for index in requested:
        row = rows_by_index[index]
        ordinal = ordinal_by_index[index]
        print(f"[fetch] {args.dataset} index={index} source-row={ordinal}", flush=True)
        offset = locate_record(args.source_url, total_size, ordinal_by_index, ordinal, args.probe_size)
        source_row = record_at(args.source_url, total_size, offset)
        if len(source_row) <= image_column or source_row[0] != index:
            raise RuntimeError(f"Source record mismatch at offset {offset}: expected index {index}")
        source_values = dict(zip(header, source_row))
        for column in ("question", "answer"):
            if source_values.get(column) != row.get(column):
                raise RuntimeError(
                    f"Source {column} mismatch for index {index}: "
                    "the evaluation output and source TSV do not describe the same sample"
                )
        image = base64.b64decode(source_row[image_column])
        extension = ".png" if image.startswith(b"\x89PNG") else ".jpg"
        image_name = f"{args.dataset}_{index}{extension}"
        (images / image_name).write_bytes(image)

        sample = {
            "dataset": args.dataset,
            "index": index,
            "image": str(Path("images") / image_name),
            "source_offset": offset,
            "source_verified": True,
            "category": row.get("category"),
            "l2_category": row.get("l2-category"),
            "question": row.get("question"),
            "options": {letter: row[letter] for letter in "ABCDE" if row.get(letter)},
            "gold": row.get("answer"),
            "hit": row.get("hit") or None,
            "latent_blocks": latent_blocks(row.get("prediction", "")),
            "prediction": row.get("prediction"),
        }
        samples.append(sample)

    manifest = {
        "dataset": args.dataset,
        "source_url": args.source_url,
        "prediction_file": str(args.predictions),
        "samples": samples,
    }
    (output / "samples.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"[fetch] wrote {len(samples)} sample(s) to {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--indices", required=True, help="Comma-separated source `index` values")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe-size", type=int, default=16 * MEBIBYTE)
    materialize(parser.parse_args())


if __name__ == "__main__":
    main()
