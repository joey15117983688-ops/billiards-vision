# 台球视觉识别项目

当前阶段：第一阶段，开发环境初始化。仅包含环境验证，不包含业务功能。

## 固定环境

| 项目 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0（包版本 4.10.0.84） |
| PyQt5 | 5.15.11 |
| PyTorch | 2.6.0 |
| Ultralytics YOLO | 8.3.203 |
| ONNX | 1.17.0 |
| ONNX Runtime | 1.20.1 |
| C++ 标准 | C++17 |

## 创建虚拟环境

如果已有 Python 3.10.20，直接使用，不重装、不修改系统 Python：

```bash
cd /Users/joey/公司产品/视觉识别项目/billiards-vision
python3.10 --version
python3.10 -m venv .venv
source .venv/bin/activate
python --version
python -m pip --version
```

本机没有现成的 Python 3.10.20，已通过 uv 在本项目 `.runtime/` 中安装独立运行时，`.venv/` 已创建。日常使用只需激活环境：

```bash
source .venv/bin/activate
```

若需要用该独立运行时重建虚拟环境：

```bash
.runtime/cpython-3.10.20-macos-aarch64-none/bin/python3.10 -m venv .venv
```

Windows 激活命令为 `.venv\Scripts\activate`。
不要使用 Python 3.11+、OpenCV 5.x、PyQt6 或 PySide 替代固定环境。

## 安装依赖

```bash
python -m pip install -r requirements.txt
python -m pip check
```

所有直接依赖都固定版本。安装其他包时使用 `-c requirements.txt` 保持固定约束。
如果出现依赖冲突，停止并报告，不通过升级固定环境解决。

复制本次完整环境：

```bash
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` 是本机 macOS ARM64 的 `pip freeze` 快照；跨操作系统、CPU 架构或旧系统时需先检查这些版本是否提供对应 wheel。此阶段只验收本机环境。

## 运行环境检查

```bash
python scripts/check_environment.py
python scripts/test_opencv.py
python scripts/test_pyqt.py
python scripts/test_yolo.py
python -m pytest tests/test_imports.py -q
```

PyQt5 测试窗口打开后手动关闭即可；自动验证使用：

```bash
python scripts/test_pyqt.py --auto-close-ms 3000
```

YOLO 使用官方 `yolov8n.pt`，首次运行需要联网下载至 `models/`。
测试图为 `data/opencv_test.jpg`，仅验证普通图片推理流程，无台球识别或训练。
YOLO 固定使用 CPU 验证，不要求 CUDA。运行配置及验证日志位于 `logs/`。

检查 C++ 编译器：

```bash
g++ --version
```

验收结果见 `ENVIRONMENT_REPORT.md`。
