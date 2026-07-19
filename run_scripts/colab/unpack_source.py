"""Unpack the uploaded Monet source archive inside a Colab runtime."""

from pathlib import Path
import shutil
import tarfile


ARCHIVE = Path("/content/monet-source.tar.gz")
DESTINATION = Path("/content/Monet")
REQUIRED_FILES = ("AGENTS.md", "requirements.txt", "run_scripts/04_run_eval.sh")


def safe_members(archive: tarfile.TarFile):
    destination = DESTINATION.resolve()
    for member in archive.getmembers():
        target = (DESTINATION / member.name).resolve()
        if target != destination and destination not in target.parents:
            raise RuntimeError(f"Unsafe archive member: {member.name}")
        yield member


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Uploaded source archive not found: {ARCHIVE}")

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    DESTINATION.mkdir(parents=True)

    with tarfile.open(ARCHIVE, "r:gz") as archive:
        archive.extractall(DESTINATION, members=safe_members(archive), filter="data")

    missing = [name for name in REQUIRED_FILES if not (DESTINATION / name).is_file()]
    if missing:
        raise RuntimeError(f"Source verification failed; missing: {', '.join(missing)}")

    print(f"[colab] unpacked Monet source to {DESTINATION}")
    print("[colab] verified: " + ", ".join(REQUIRED_FILES))


if __name__ == "__main__":
    main()
