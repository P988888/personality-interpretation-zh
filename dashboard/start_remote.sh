#!/bin/bash
# 在训练机上重启大屏。
# 注意：必须放在脚本里跑，不能把 pkill 和启动命令写在同一条 ssh 命令行里——
# 那样 pkill -f 会匹配到同一条命令行中后半句的 "serve_remote.py"，把自己杀掉。
cd "$(dirname "$0")" || exit 1
pkill -f "serve_remote.py" >/dev/null 2>&1
sleep 1
setsid python3 serve_remote.py "${1:-8000}" >/tmp/dash.log 2>&1 </dev/null &
sleep 2
if pgrep -f "serve_remote.py" >/dev/null; then
  echo "大屏已启动  pid=$(pgrep -f 'serve_remote.py' | head -1)  端口=${1:-8000}"
  curl -s -m 6 "http://127.0.0.1:${1:-8000}/api/status" | head -c 140
  echo
else
  echo "启动失败："
  cat /tmp/dash.log
fi
