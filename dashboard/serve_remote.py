#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在训练服务器上直接起大屏（本地那份的备份方案）。

本地那份是「本机起服务 + ssh 拉数据」；这份跑在训练机上，直接读日志，
连 ssh 都不需要，因此可以关掉自己的电脑继续看。

用法（在服务器上）：setsid python3 serve_remote.py 8000 > /tmp/dash.log 2>&1 < /dev/null &
"""
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
TTL = 4.0

_cache = {"t": 0.0, "d": None}
_lock = threading.Lock()


def metrics():
    with _lock:
        if _cache["d"] is not None and time.time() - _cache["t"] < TTL:
            return _cache["d"]
        try:
            p = subprocess.run([sys.executable, os.path.join(ROOT, "remote_metrics.py")],
                               capture_output=True, text=True, timeout=30)
            out = (p.stdout or "").strip()
            d = json.loads(out.splitlines()[-1]) if out else {"error": "采集器无输出"}
        except Exception as e:
            d = {"error": str(e)}
        _cache.update(t=time.time(), d=d)
        return d


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            try:
                with open(os.path.join(ROOT, "index.html"), "r", encoding="utf-8") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, "index.html 不存在", "text/plain; charset=utf-8")
        elif p in ("/api/status", "/api/refresh"):
            if p == "/api/refresh":
                with _lock:
                    _cache["d"] = None
            self._send(200, json.dumps(metrics(), ensure_ascii=False),
                       "application/json; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("训练机大屏已启动： http://0.0.0.0:%d" % PORT, flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
