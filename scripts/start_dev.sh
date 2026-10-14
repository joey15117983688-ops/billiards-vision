#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "未找到项目虚拟环境：$PYTHON" >&2
    echo "请按照 README.md 创建 Python 3.10.20 虚拟环境并安装依赖。" >&2
    exit 1
fi

cd "$PROJECT_DIR"
echo "启动本地台球球桌视频（关闭窗口即可退出）"
exec "$PYTHON" "$PROJECT_DIR/main.py" "$@"
