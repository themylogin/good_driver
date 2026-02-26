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


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int, log_stream) -> None:
    import uvicorn
    import logging

    # Ensure the backend package is importable
    backend_dir = Path(__file__).resolve().parent.parent / "backend"
    if backend_dir.is_dir() and str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from good_driver.app import create_app

    logging.basicConfig(stream=log_stream, level=logging.INFO)

    app = create_app()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


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

    port = find_free_port()

    server_thread = threading.Thread(target=start_server, args=(port, log_stream), daemon=True)
    server_thread.start()

    wait_for_server(port)

    import webview

    window = webview.create_window(
        "Good Driver",
        f"http://127.0.0.1:{port}",
        width=1200,
        height=800,
    )
    webview.start()


if __name__ == "__main__":
    main()
