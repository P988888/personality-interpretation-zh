#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练监控大屏 · 本地服务端

浏览器 → http://127.0.0.1:8000 → 本进程 --ssh--> 训练服务器 remote_metrics.py

为什么把服务放在本地、而不是直接起在训练服务器上：
  1. 打开即用，不依赖云平台给哪个端口做了外网映射
     （趣算云的对外 URL 是随机端口，我们控制不了 8000 是否被映射）；
  2. SSH 密钥留在本机，服务器上不留常驻进程；
  3. 大屏挂掉不影响正在跑的训练——它只是读日志的旁观者。

用法：python3 dashboard/server.py [端口]
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
SSH_HOST = os.environ.get("TRAIN_SSH_HOST", "personality-gpu")
# 和 serve_remote.py 用同一份采集器，只留一个真源。
# 曾经这里指向 /data/personality/remote_metrics.py，而部署只更新 dashboard/ 下那份，
# 结果旧副本不返回 log_lines，大屏日志面板整个空了。
REMOTE_SCRIPT = os.environ.get(
    "TRAIN_METRICS_SCRIPT", "/data/personality/dashboard/remote_metrics.py")
CACHE_TTL = 4.0          # 秒；比前端的轮询间隔略短

_cache = {"t": 0.0, "data": None}
_lock = threading.Lock()


def fetch_metrics():
    """带缓存地取一次远端指标。缓存是为了让多个浏览器标签页共用一个 SSH 连接。"""
    with _lock:
        if _cache["data"] is not None and time.time() - _cache["t"] < CACHE_TTL:
            return _cache["data"]
        try:
            p = subprocess.run(
                ["ssh", "-T", "-o", "ConnectTimeout=8", SSH_HOST, f"python3 {REMOTE_SCRIPT}"],
                capture_output=True, text=True, timeout=30,
            )
            out = (p.stdout or "").strip()
            if not out:
                data = {"error": "ssh 无输出", "stderr": (p.stderr or "")[-400:]}
            else:
                # 登录 banner 可能混在前面，取最后一个能解析成 JSON 的行
                data = None
                for line in reversed(out.splitlines()):
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            data = json.loads(line)
                            break
                        except Exception:
                            continue
                if data is None:
                    data = {"error": "无法解析远端输出", "raw": out[-400:]}
        except subprocess.TimeoutExpired:
            data = {"error": "ssh 超时（30s）"}
        except Exception as e:
            data = {"error": str(e)}
        _cache.update(t=time.time(), data=data)
        return data


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            try:
                with open(os.path.join(ROOT, "index.html"), "r", encoding="utf-8") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, "index.html 不存在", "text/plain; charset=utf-8")
        elif path == "/api/status":
            self._send(200, json.dumps(fetch_metrics(), ensure_ascii=False),
                       "application/json; charset=utf-8")
        elif path == "/api/refresh":
            with _lock:
                _cache["data"] = None
            self._send(200, json.dumps(fetch_metrics(), ensure_ascii=False),
                       "application/json; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def log_message(self, *a):
        pass  # 别把访问日志刷到终端


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("=" * 58)
    print("  本地：http://127.0.0.1:%d" % PORT)
    print("  局域网：http://%s:%d  （手机/平板也能看）" % (lan_ip(), PORT))
    print("  数据源：%s:%s  （每 %.0fs 刷新一次）" % (SSH_HOST, REMOTE_SCRIPT, CACHE_TTL))
    print("=" * 58)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
