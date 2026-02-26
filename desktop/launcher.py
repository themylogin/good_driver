import socket
import sys
import threading
import time
from pathlib import Path

import urllib.request
import urllib.error


def get_log_stream():
    if sys.platform == "win32":
        if getattr(sys, "frozen", False):
            log_path = Path(sys.executable).parent / "good_driver.log"
        else:
            log_path = Path(__file__).parent / "good_driver.log"
        return open(log_path, "w", buffering=1, encoding="utf-8")
    return sys.stderr


def _ensure_webview2():
    import subprocess
    import winreg

    key_path = r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path):
            return
    except FileNotFoundError:
        pass

    if getattr(sys, "frozen", False):
        bootstrapper = Path(sys._MEIPASS) / "MicrosoftEdgeWebview2Setup.exe"
    else:
        bootstrapper = Path(__file__).parent / "MicrosoftEdgeWebview2Setup.exe"

    if bootstrapper.exists():
        subprocess.run([str(bootstrapper), "/silent", "/install"], check=False)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int) -> None:
    try:
        import uvicorn

        # Ensure the backend package is importable
        backend_dir = Path(__file__).resolve().parent.parent / "backend"
        if backend_dir.is_dir() and str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))

        from good_driver.app import create_app

        app = create_app()
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()


def wait_for_server(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health")
            return
        except urllib.error.URLError:
            time.sleep(0.05)
    raise TimeoutError("Server did not start in time")


def main() -> None:
    import os
    os.environ["GOOD_DRIVER_MODE"] = "desktop"

    log_stream = get_log_stream()
    if log_stream is not sys.stderr:
        sys.stdout = log_stream
        sys.stderr = log_stream

    port = find_free_port()

    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()

    try:
        wait_for_server(port)
    except TimeoutError as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.stderr.flush()
        sys.exit(1)

    if sys.platform == "win32":
        _ensure_webview2()

    import webview

    webview.settings["WEBVIEW2_RUNTIME_PATH"] = "."

    window = webview.create_window(
        "Good Driver",
        f"http://127.0.0.1:{port}",
        width=1200,
        height=800,
    )
    webview.start(func=window.restore, debug=True)


if __name__ == "__main__":
    main()
