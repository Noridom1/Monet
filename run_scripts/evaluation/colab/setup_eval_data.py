"""Create or restore Monet's canonical VLMEvalKit dataset bundle in Colab."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path


ARCHIVE = Path("/content/drive/MyDrive/Monet/monet_eval_datasets.zip")
PARTIAL_ARCHIVE = ARCHIVE.with_suffix(".zip.partial")
DATA_DIR = Path("/content/LMUData")
TEMP_DATA_DIR = Path("/content/LMUData.tmp")
CHUNK_SIZE = 8 * 1024 * 1024
DATASETS = {
    "VStarBench.tsv": "https://huggingface.co/datasets/xjtupanda/VStar_Bench/resolve/main/VStarBench.tsv",
    "HRBench4K.tsv": "https://huggingface.co/datasets/mm-eval/VLMEvalKit/resolve/main/HRBench4K.tsv",
    "HRBench8K.tsv": "https://huggingface.co/datasets/mm-eval/VLMEvalKit/resolve/main/HRBench8K.tsv",
    "MME-RealWorld-Lite.tsv": (
        "https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Base64/resolve/main/"
        "mme_realworld_lite.tsv"
    ),
}


def copy_and_hash(source, destination) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while chunk := source.read(CHUNK_SIZE):
        destination.write(chunk)
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest()


def create_archive() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    PARTIAL_ARCHIVE.unlink(missing_ok=True)
    records = []

    print(f"[monet-colab] archive missing; creating {ARCHIVE}", flush=True)
    try:
        with zipfile.ZipFile(PARTIAL_ARCHIVE, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as bundle:
            for name, url in DATASETS.items():
                print(f"[monet-colab] downloading {name}", flush=True)
                request = urllib.request.Request(url, headers={"User-Agent": "Monet-Colab/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response:
                    with bundle.open(f"LMUData/{name}", "w", force_zip64=True) as destination:
                        size, sha256 = copy_and_hash(response, destination)
                records.append({"name": name, "url": url, "size": size, "sha256": sha256})
                print(f"[monet-colab] saved {name}: {size} bytes sha256={sha256}", flush=True)

            manifest = {"format_version": 1, "datasets": records}
            bundle.writestr("MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        with zipfile.ZipFile(PARTIAL_ARCHIVE) as bundle:
            bad_entry = bundle.testzip()
            if bad_entry is not None:
                raise RuntimeError(f"ZIP integrity check failed for {bad_entry}")
        os.replace(PARTIAL_ARCHIVE, ARCHIVE)
    except BaseException:
        PARTIAL_ARCHIVE.unlink(missing_ok=True)
        raise


def restore_archive() -> None:
    if TEMP_DATA_DIR.exists():
        shutil.rmtree(TEMP_DATA_DIR)
    TEMP_DATA_DIR.mkdir(parents=True)

    with zipfile.ZipFile(ARCHIVE) as bundle:
        manifest = json.loads(bundle.read("MANIFEST.json"))
        raw_records = manifest.get("datasets", [])
        if isinstance(raw_records, dict):
            raw_records = [dict(value, name=key) for key, value in raw_records.items()]
        records = {}
        for record in raw_records:
            raw_name = record.get("filename") or record.get("name") or record.get("path")
            name = Path(raw_name).name if raw_name else None
            records[name] = record
        if set(records) != set(DATASETS):
            raise RuntimeError("Dataset archive manifest does not contain the expected four datasets")

        for name, record in records.items():
            print(f"[monet-colab] restoring and verifying {name}", flush=True)
            with bundle.open(f"LMUData/{name}") as source:
                with (TEMP_DATA_DIR / name).open("wb") as destination:
                    size, sha256 = copy_and_hash(source, destination)
            expected_size = record.get("size", record.get("size_bytes"))
            expected_sha256 = record.get("sha256")
            if expected_sha256 is None:
                raise RuntimeError(f"Manifest has no SHA-256 for {name}")
            if (expected_size is not None and size != expected_size) or sha256 != expected_sha256:
                raise RuntimeError(f"Manifest verification failed for {name}")

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    TEMP_DATA_DIR.replace(DATA_DIR)


def main() -> None:
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Google Drive is not mounted at /content/drive")
    if not ARCHIVE.exists():
        create_archive()
    else:
        print(f"[monet-colab] reusing {ARCHIVE}", flush=True)
    restore_archive()
    print(f"[monet-colab] EVAL_DATA_READY path={DATA_DIR} archive={ARCHIVE}", flush=True)


if __name__ == "__main__":
    main()
