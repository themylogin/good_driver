#!/usr/bin/env python3
"""Download yolopv2_384x640.onnx from PINTO model zoo."""

import subprocess
import tarfile
from pathlib import Path

RESOURCES_URL = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/326_YOLOPv2/resources.tar.gz"
MODELS_DIR = Path(__file__).parent / "models"
TARGET = MODELS_DIR / "yolopv2_384x640.onnx"
TARGET_IN_ARCHIVE = "yolopv2_384x640.onnx"


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    if TARGET.exists():
        print(f"Model already exists at {TARGET}")
        return

    archive = MODELS_DIR / "resources.tar.gz"
    print(f"Downloading {RESOURCES_URL} ...")
    subprocess.run(["wget", "-c", "--progress=bar:force", "-O", str(archive), RESOURCES_URL], check=True)
    print("Extracting ...")

    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        # Find the target onnx file inside the archive
        candidates = [m for m in members if m.name.endswith(TARGET_IN_ARCHIVE)]
        if not candidates:
            candidates = [m for m in members if "384" in m.name and "640" in m.name and m.name.endswith(".onnx")]
        if not candidates:
            candidates = [m for m in members if m.name.endswith(".onnx")]
        if not candidates:
            raise RuntimeError(f"No ONNX file found in archive. Contents: {[m.name for m in members]}")
        member = candidates[0]
        member.name = TARGET.name  # extract flat, no subdirs
        tf.extract(member, MODELS_DIR)

    archive.unlink()
    print(f"Model saved to {TARGET}")


if __name__ == "__main__":
    main()
