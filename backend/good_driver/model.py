"""Model path resolution and download logic shared by CLI and API."""

import sys
import tarfile
from pathlib import Path

MODEL_NAME = "yolopv2_384x640.onnx"
RESOURCES_URL = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/326_YOLOPv2/resources.tar.gz"


def get_model_path() -> Path:
    """Return the expected ONNX model path.

    - Frozen binary (PyInstaller): next to the executable in a models/ subdirectory.
    - Dev / run-desktop: backend/models/ relative to this file.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(sys.executable).parent / "models" / MODEL_NAME
    return Path(__file__).resolve().parents[1] / "models" / MODEL_NAME


def extract_model(archive: Path, target: Path) -> None:
    """Extract MODEL_NAME from a tar.gz archive to *target*."""
    with tarfile.open(archive, "r:gz") as tf:
        member = tf.getmember(MODEL_NAME)
        member.name = target.name  # strip any subdirectory prefix
        tf.extract(member, target.parent)


def download_cli() -> None:
    """Download the model using wget (CLI / server-installation entry point)."""
    import subprocess

    target = get_model_path()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        print(f"Model already exists at {target}")
        return

    archive = target.parent / "resources.tar.gz"
    print(f"Downloading {RESOURCES_URL} ...")
    subprocess.run(
        ["wget", "-c", "--progress=bar:force", "-O", str(archive), RESOURCES_URL],
        check=True,
    )
    print("Extracting ...")
    extract_model(archive, target)
    archive.unlink()
    print(f"Model saved to {target}")
