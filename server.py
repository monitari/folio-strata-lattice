"""실험 노트 서버. 표준 라이브러리만 쓴다.

    py server.py            → http://localhost:8765 이 브라우저에서 열린다
    py server.py --port 9000 --no-browser

GET /api/results : candidates.site.payload() (실행 결과가 바뀌면 다음 요청에서 다시 읽는다)
"""
import argparse
import json
import threading
import traceback
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from candidates.benchmark import OUT
from candidates.site import payload

WEB_DIR = Path(__file__).parent / "web"
_cache = {"stamp": None, "body": None}
_lock = threading.Lock()


def _stamp():
    """결과 파일들의 최종 수정 시각 묶음. 바뀌었을 때만 다시 만든다."""
    files = list((OUT / "runs").glob("*/*.json")) + [OUT / n for n in ("measure.json", "checks.json", "samples.json")]
    return tuple(sorted((str(p), p.stat().st_mtime) for p in files if p.exists()))


def results_body() -> bytes:
    with _lock:
        stamp = _stamp()
        if stamp != _cache["stamp"]:
            _cache["body"] = json.dumps(payload(), ensure_ascii=False).encode("utf-8")
            _cache["stamp"] = stamp
        return _cache["body"]


class Server(ThreadingHTTPServer):
    # Windows에서는 주소 재사용을 켜면 같은 포트에 서버가 둘 뜰 수 있다 → 이미 떠 있으면 바로 실패하게 한다
    allow_reuse_address = False


class Handler(SimpleHTTPRequestHandler):
    def _send(self, body: bytes, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/results":
            try:
                return self._send(results_body())
            except Exception as exc:  # 실패 원인을 화면에 그대로 보여준다
                traceback.print_exc()
                return self._send(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False).encode(), 500)
        return super().do_GET()

    def end_headers(self):
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        if self.path.startswith("/api/"):
            super().log_message(fmt, *args)


def serve(port: int = 8765, open_browser: bool = True):
    server = Server(("127.0.0.1", port), partial(Handler, directory=str(WEB_DIR)))
    url = f"http://localhost:{port}"
    print(f"실험 노트: {url}  (종료: Ctrl+C)")
    if open_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="실험 노트 서버")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    serve(args.port, not args.no_browser)


if __name__ == "__main__":
    main()
