#!/usr/bin/env python3
"""Package a configured remote Monet output directory for download."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


SOURCE_CONFIG = Path("/tmp/monet-download-source.txt")
ARCHIVE = Path("/tmp/monet-download.zip")


def main() -> None:
    if not SOURCE_CONFIG.is_file():
        raise SystemExit(f"Download source configuration not found: {SOURCE_CONFIG}")

    source = Path(SOURCE_CONFIG.read_text(encoding="utf-8").strip())
    if not source.is_dir():
        raise SystemExit(f"Remote output directory not found: {source}")

    files = [path for path in sorted(source.rglob("*")) if path.is_file()]
    with ZipFile(ARCHIVE, "w", ZIP_DEFLATED) as zip_file:
        for path in files:
            zip_file.write(path, path.relative_to(source.parent))

    print(f"[monet-colab] OUTPUT_ARCHIVE_READY files={len(files)} bytes={ARCHIVE.stat().st_size}")


if __name__ == "__main__":
    main()
