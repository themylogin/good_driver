import os
import sys
from enum import Enum
from pathlib import Path


class Mode(Enum):
    DEV = "dev"
    DESKTOP = "desktop"
    PRODUCTION = "production"


def get_mode() -> Mode:
    if getattr(sys, "_MEIPASS", None):
        return Mode.DESKTOP
    env = os.environ.get("GOOD_DRIVER_MODE", "").lower()
    if env == "desktop":
        return Mode.DESKTOP
    if env == "production":
        return Mode.PRODUCTION
    return Mode.DEV


def get_frontend_dist_path() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "frontend_dist"
    return Path(__file__).resolve().parent.parent / "frontend_dist"


def get_data_dir() -> Path:
    """Return the user data directory (videos, calibration images, frame JSONs).

    - Frozen binary: sibling ``data/`` directory next to the exe.
    - Dev / run-desktop: ``<project_root>/data/``.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(sys.executable).parent / "data"
    # config.py lives at backend/good_driver/config.py → parents[2] = project root
    return Path(__file__).resolve().parents[2] / "data"
