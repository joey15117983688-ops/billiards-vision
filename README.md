# 台球视觉识别项目

当前阶段：基础环境已初始化，新增本地台球球桌俯拍视频播放。已训练并接入台球桌／台球两类专用模型点选识别；尚未接入真实摄像头或开发台球业务。

## 启动项目窗口

本机环境已创建，可直接执行，无需手动激活虚拟环境：

```bash
cd /Users/joey/projects/billiards-vision
./scripts/start_dev.sh
```

也可以从任意目录执行：

```bash
/Users/joey/projects/billiards-vision/scripts/start_dev.sh
```

当前入口为 `main.py`，启动后立即循环播放本地台球球桌俯拍视频，画面右上角显示白底黑字圆圈“1”，下方提供播放和暂停按钮。
视频按比例缩放，关闭窗口即可退出。目前为无声视频播放；真实摄像头尚未接入；台球桌与球采用专用模型进行点选识别。

素材说明见 `DEMO_ASSETS.md`，SDK 官方来源及平台限制见 `HIKVISION_SDK.md`。
视频已保存在 `data/billiards_table.mp4`，本机启动无需联网。
新机器拉取代码后，先运行 `.venv/bin/python scripts/download_demo_assets.py` 下载视频。

播放其他本地视频：

```bash
./scripts/start_dev.sh --video /absolute/path/to/video.mp4
```

## 点击物体调用 YOLO

1. 启动后，点击下方 `YOLO：关闭` 开关，将其切换为开启。
2. 等待右上角显示“请点击物体”，模型已就绪。
3. 点击视频中的物体；程序对点击当时的画面进行一次推理，在右上角白底圆圈显示点击位置命中的 YOLO 中文类别，并用亮黄色框及中文标签标出实际预测目标，状态栏显示置信度。
4. 未检测到包含点击位置的目标框时显示“未识别”。点击识别会自动暂停并保留该帧，便于核对目标；点击“播放”继续时清除旧框。
5. 关闭开关后圆圈恢复“1”，清除目标框，不再提交点击识别。

使用本地 `models/billiards_v1.pt` 和 CPU，无需联网。该权重由 `yolov8n.pt` 微调得到，只包含 `billiards_table` 和 `billiard_ball` 两类，不回退到通用模型。
选择坐标按视频实际缩放比例映射，黑边和右上角圆圈不参与点选。
YOLO 在独立后台进程中运行，避免本机 PyTorch 在 Qt 工作线程中推理时崩溃，也避免阻塞播放；关闭窗口会释放模型进程及临时文件。

本地测试录像的 6 个抽样画面各检出 1 张球桌和 12 个红黄球。验证集规模有限，同一录像后段仍与训练画面相似，不能代表新球桌或新摄像头的准确率；当前补充标注未覆盖录像中的白球，白球效果尚未验收。训练与验证详情见 `TRAINING_REPORT.md`。

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
cd /Users/joey/projects/billiards-vision
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
