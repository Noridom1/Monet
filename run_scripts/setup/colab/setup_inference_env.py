"""Bootstrap the repository's inference conda environment in a Colab runtime."""

from pathlib import Path
import os
import platform
import subprocess
import urllib.request


REPO_DIR = Path("/content/Monet")
CONDA_BASE = Path("/root/miniconda3")
INSTALLER = Path("/tmp/miniconda.sh")
MINICONDA_URL = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"


def run(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(args), flush=True)
    result = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="", flush=True)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, args, output=result.stdout)


def main() -> None:
    if platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError(f"Unsupported Colab architecture: {platform.machine()}")
    setup_script = REPO_DIR / "run_scripts/setup/environment.sh"
    requirements = REPO_DIR / "requirements.txt"
    if not setup_script.is_file() or not requirements.is_file():
        raise FileNotFoundError(
            "Monet source is missing from /content/Monet; run prepare_session.sh first."
        )

    conda = CONDA_BASE / "bin/conda"
    if not conda.is_file():
        print(f"[setup] downloading Miniconda from {MINICONDA_URL}", flush=True)
        urllib.request.urlretrieve(MINICONDA_URL, INSTALLER)
        run("bash", str(INSTALLER), "-b", "-p", str(CONDA_BASE))
        INSTALLER.unlink(missing_ok=True)
    else:
        print(f"[setup] reusing Miniconda at {CONDA_BASE}", flush=True)

    env = os.environ.copy()
    env["CONDA_BASE"] = str(CONDA_BASE)
    env["ENV_NAME"] = "monet"
    # Avoid requiring acceptance of Anaconda's default-channel terms in an
    # unattended Colab session. Local setup keeps its existing channel policy.
    env["CONDA_CHANNEL"] = "conda-forge"
    run("bash", str(setup_script), cwd=REPO_DIR, env=env)

    python = CONDA_BASE / "envs/monet/bin/python"
    run(
        str(python),
        "-c",
        (
            "import torch, transformers, vllm; "
            "print('python environment OK'); "
            "print('torch', torch.__version__); "
            "print('transformers', transformers.__version__); "
            "print('vllm', vllm.__version__); "
            "print('cuda_available', torch.cuda.is_available())"
        ),
        cwd=REPO_DIR,
        env=env,
    )
    print("[monet-colab] ENVIRONMENT_READY", flush=True)


if __name__ == "__main__":
    main()
