# -*- mode: python ; coding: utf-8 -*-
import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

ROOT = Path(SPECPATH).parent
BACKEND = ROOT / "backend"
FRONTEND_DIST = BACKEND / "frontend_dist"

IS_WINDOWS = sys.platform == "win32"

webview_origin = importlib.util.find_spec("webview").origin
WEBVIEW_LIB = Path(webview_origin).parent / "lib"

ort_origin = importlib.util.find_spec("onnxruntime").origin
ORT_CAPI = Path(ort_origin).parent / "capi"

# NOTE: The ONNX model (yolopv2_384x640.onnx) is intentionally NOT bundled here.
# It must be downloaded separately after installation via:  download-model
# The launcher will check for the model on startup and prompt the user to download it.

# --- Platform-specific binaries, datas, and hidden imports ---

if IS_WINDOWS:
    platform_binaries = [
        (str(ORT_CAPI / "DirectML.dll"), "onnxruntime/capi"),
        (str(ORT_CAPI / "onnxruntime.dll"), "onnxruntime/capi"),
        (str(ORT_CAPI / "onnxruntime_providers_shared.dll"), "onnxruntime/capi"),
    ]
    platform_datas = [
        (str(ROOT / "desktop" / "MicrosoftEdgeWebview2Setup.exe"), "."),
    ]
    platform_hiddenimports = [
        "webview.platforms.winforms",
        "webview.platforms.edgechromium",
        "clr",
    ]
else:
    ort_libs = list(ORT_CAPI.glob("libonnxruntime*"))
    platform_binaries = [(str(lib), "onnxruntime/capi") for lib in ort_libs]
    platform_datas = []
    platform_hiddenimports = [
        "webview.platforms.gtk",
    ]

a = Analysis(
    [str(ROOT / "desktop" / "launcher.py")],
    pathex=[str(BACKEND)],
    binaries=platform_binaries,
    datas=[
        (str(FRONTEND_DIST), "frontend_dist"),
        (str(WEBVIEW_LIB), "webview/lib"),
    ] + platform_datas,
    hiddenimports=[
        "pydantic_core._pydantic_core",
    ] + collect_submodules("good_driver") + platform_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="good_driver",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
)
