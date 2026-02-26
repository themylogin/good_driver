# -*- mode: python ; coding: utf-8 -*-
import importlib.util
from pathlib import Path

block_cipher = None

ROOT = Path(SPECPATH).parent
BACKEND = ROOT / "backend"
FRONTEND_DIST = BACKEND / "frontend_dist"

webview_origin = importlib.util.find_spec("webview").origin
WEBVIEW_LIB = Path(webview_origin).parent / "lib"

ort_origin = importlib.util.find_spec("onnxruntime").origin
ORT_CAPI = Path(ort_origin).parent / "capi"

# NOTE: The ONNX model (yolopv2_384x640.onnx) is intentionally NOT bundled here.
# It must be downloaded separately after installation via:  download-model
# The launcher will check for the model on startup and prompt the user to download it.

a = Analysis(
    [str(ROOT / "desktop" / "launcher.py")],
    pathex=[str(BACKEND)],
    binaries=[
        (str(ORT_CAPI / "DirectML.dll"), "onnxruntime/capi"),
        (str(ORT_CAPI / "onnxruntime.dll"), "onnxruntime/capi"),
        (str(ORT_CAPI / "onnxruntime_providers_shared.dll"), "onnxruntime/capi"),
    ],
    datas=[
        (str(FRONTEND_DIST), "frontend_dist"),
        (str(WEBVIEW_LIB), "webview/lib"),
        (str(ROOT / "desktop" / "MicrosoftEdgeWebview2Setup.exe"), "."),
    ],
    hiddenimports=[
        "pydantic_core._pydantic_core",
        "webview.platforms.winforms",
        "webview.platforms.edgechromium",
        "clr",
        "good_driver",
        "good_driver.app",
        "good_driver.config",
        "good_driver.model",
        "good_driver.static_files",
        "good_driver.api",
        "good_driver.api.health",
        "good_driver.api.model",
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
    ],
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
