"""Download the configured Hugging Face model inside a prepared Colab runtime."""

from pathlib import Path
import os
import subprocess


CONFIG = Path("/content/monet-model-download.conf")
HF = Path("/root/miniconda3/envs/monet/bin/hf")


def main() -> None:
    if not CONFIG.is_file():
        raise FileNotFoundError("Model configuration was not uploaded to the Colab runtime")
    values = CONFIG.read_text(encoding="utf-8").splitlines()
    if len(values) != 2 or not all(values):
        raise ValueError("Model configuration must contain a repository and destination")
    model_repo, raw_model_dir = values
    model_dir = Path(raw_model_dir)
    if model_dir != Path("/content") and Path("/content") not in model_dir.parents:
        raise ValueError("Model destination must be under /content")
    if not HF.is_file():
        raise FileNotFoundError("Monet environment is missing; prepare dependencies first")

    model_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    subprocess.run(
        [str(HF), "download", model_repo, "--local-dir", str(model_dir)],
        check=True,
        env=env,
    )

    weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    if not (model_dir / "config.json").is_file() or not weight_files:
        raise RuntimeError(f"Downloaded model is incomplete: {model_dir}")
    print(f"[monet-colab] MODEL_READY repo={model_repo} path={model_dir}", flush=True)


if __name__ == "__main__":
    main()
