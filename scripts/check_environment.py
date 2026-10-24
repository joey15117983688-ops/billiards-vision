"""Print environment versions and enforce the fixed compatibility requirements."""
# 环境自检入口；记录已安装版本及运行条件，不自动升级依赖。

import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
config_dir = ROOT / "logs" / "ultralytics"
config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


# 检查现有运行环境并输出诊断，导入成功不等价视觉算法或跨平台已验收。
def main():
    import cv2
    import numpy
    import onnx
    import onnxruntime as ort
    import torch
    import ultralytics
    from PyQt5.QtCore import PYQT_VERSION_STR, QT_VERSION_STR

    print("Python:", platform.python_version())
    print("Platform:", platform.platform())
    print("NumPy:", numpy.__version__)
    print("OpenCV:", cv2.__version__)
    print("PyQt5:", PYQT_VERSION_STR)
    print("Qt:", QT_VERSION_STR)
    print("PyTorch:", torch.__version__)
    print("Ultralytics:", ultralytics.__version__)
    print("ONNX:", onnx.__version__)
    print("ONNX Runtime:", ort.__version__)
    print("CUDA Available:", torch.cuda.is_available())
    print("ONNX Providers:", ort.get_available_providers())
    assert sys.version_info[:3] == (3, 10, 20), "Python must be 3.10.20"
    assert cv2.__version__.startswith("4.10."), "OpenCV must be 4.10.x"
    assert PYQT_VERSION_STR.startswith("5."), "PyQt5 required"
    print("Environment Check Complete")


if __name__ == "__main__":
    main()
