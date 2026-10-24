#!/usr/bin/env bash
set -euo pipefail

# 从脚本位置定位仓库，避免依赖调用者的工作目录或写死某台机器的路径。
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"

# 只使用项目虚拟环境；缺失时提示准备步骤，不自动安装或升级系统依赖。
if [[ ! -x "$PYTHON" ]]; then
    echo "未找到项目虚拟环境：$PYTHON" >&2
    echo "请按照 README.md 创建 Python 3.10.20 虚拟环境并安装依赖。" >&2
    exit 1
fi

cd "$PROJECT_DIR"
echo "启动台球视觉识别（默认样本图片，可切换视频；关闭窗口即可退出）"
# 用 exec 替换 shell，让关闭/中断信号直接交给 Python 窗口进程。
exec "$PYTHON" "$PROJECT_DIR/main.py" "$@"
