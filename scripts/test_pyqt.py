"""Minimal native Qt window. Optional timed close for environment verification."""
# PyQt5 窗口冒烟测试；可用自动关闭参数验证正常启动/释放，不代表完整业务回归。

import argparse
import sys
from pathlib import Path

import PyQt5

from PyQt5.QtCore import QTimer, QT_VERSION_STR, PYQT_VERSION_STR
from PyQt5.QtWidgets import QApplication, QLabel


# 准备 Qt 插件位置并显示简单窗口，指定超时只控制本测试窗口的自动关闭。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-close-ms", type=int, default=0)
    args = parser.parse_args()
    if args.auto_close_ms < 0:
        parser.error("--auto-close-ms must be nonnegative")
    # Explicit Unicode path avoids Qt 5 relocation errors under Chinese paths.
    plugins = Path(PyQt5.__path__[0]) / "Qt5" / "plugins"
    QApplication.addLibraryPath(str(plugins))
    app = QApplication([sys.argv[0]])
    window = QLabel("PyQt5 environment test OK")
    window.setWindowTitle("PyQt5 Environment Test")
    window.resize(360, 120)
    window.show()
    app.processEvents()
    assert window.isVisible(), "Window did not open"
    print("PyQt5:", PYQT_VERSION_STR, "Qt:", QT_VERSION_STR, flush=True)
    print("Platform:", app.platformName(), "Window opened", flush=True)
    if args.auto_close_ms:
        QTimer.singleShot(args.auto_close_ms, window.close)
    code = app.exec_()
    assert code == 0 and not window.isVisible(), "Window did not close normally"
    print("PyQt test PASS")
    return code


if __name__ == "__main__":
    sys.exit(main())
