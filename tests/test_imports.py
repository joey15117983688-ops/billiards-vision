"""Only basic dependency imports and the specified fixed versions."""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
config_dir = ROOT / "logs" / "ultralytics"
config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


@pytest.mark.parametrize("module", [
    "cv2", "numpy", "torch", "onnx", "onnxruntime", "PyQt5", "ultralytics",
])
def test_import(module):
    importlib.import_module(module)


def test_fixed_versions():
    import cv2
    from PyQt5.QtCore import PYQT_VERSION_STR

    assert sys.version_info[:3] == (3, 10, 20)
    assert cv2.__version__.startswith("4.10.")
    assert PYQT_VERSION_STR.startswith("5.")
