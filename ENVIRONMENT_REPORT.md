# Environment Report

验证日期：2026-10-13（Asia/Shanghai）  
项目路径：`/Users/joey/projects/billiards-vision`
Platform: macOS 27.0.1 / ARM64  
阶段：第一阶段，开发环境初始化。

目录迁移（2026-10-13）：项目已完整迁移到 `/Users/joey/projects/billiards-vision`，包含 Git 历史、独立 Python 运行时、虚拟环境、模型、数据、日志及本地新增文件。已修复虚拟环境中的绝对路径，迁移后 Python 3.10.20、pip、环境检查、OpenCV、PyQt5 原生窗口、YOLO、8 项 Import 测试和 pip check 全部通过。原中文项目目录已移除。

## Python

Version: 3.10.20  
Status: PASS

原默认 Python 为 3.14.7，未修改。未找到现成的 3.10.20，使用 uv 0.13.0 在项目 `.runtime/` 内安装独立 CPython 3.10.20，并创建 `.venv/`。pip 26.2.1 指向项目虚拟环境。

运行时来源：[uv 管理 Python 的官方说明](https://docs.astral.sh/uv/concepts/python-versions/)，使用 Astral python-build-standalone 发行版。

## C++

Compiler: Apple clang version 21.0.0 (clang-2100.3.34.2), `/usr/bin/g++`  
Target: arm64-apple-darwin27.0.0  
C++17 Supported: YES / PASS

以 `g++ -std=c++17` 编译临时测试，使用 `std::optional`、`std::string_view` 和 `static_assert(__cplusplus >= 201703L)`；运行输出 `C++17 PASS 17`。未创建项目 C++ 模块。

## NumPy

Version: 1.26.4  
Status: PASS

## OpenCV

Version: 4.10.0（opencv-python 4.10.0.84）  
Status: PASS

成功写入并读回 `data/opencv_test.jpg`（800 × 500，3 通道）。

## PyQt5

Version: 5.15.11  
Qt Runtime Version: 5.15.14  
PyQt5-Qt5 Package Version: 5.15.19  
Status: PASS

使用原生 `cocoa` 平台打开测试窗口，3 秒后关闭，事件循环正常退出（退出码 0）。

## PyTorch

Version: 2.6.0  
CUDA Available: False  
Status: PASS

导入成功；YOLO 使用 CPU 完成推理。不要求本机具备 CUDA。

## Ultralytics YOLO

Version: 8.3.203  
Status: PASS

加载官方 nano 模型 `models/yolov8n.pt`，对 `data/opencv_test.jpg` 完成单张图片 CPU 推理，正常退出。
图片内容为 OpenCV 测试文字，无检测目标属于正常结果。

Model Source: [Ultralytics 官方模型](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt)  
Model SHA256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`  
Inference: 18.4 ms（本次单张测试结果，非性能基准）

## ONNX Runtime

Version: 1.20.1  
Providers: CoreMLExecutionProvider, AzureExecutionProvider, CPUExecutionProvider  
Status: PASS

ONNX 1.17.0 也已安装并成功导入。此阶段 ONNX Runtime 仅验证导入及可用 Providers，未执行模型导出或 ONNX 推理。

## Tests

OpenCV Test: PASS  
PyQt Test: PASS（原生窗口打开、关闭，无异常退出）  
YOLO Test: PASS（官方模型加载、单张图片推理、正常退出）  
Import Test: PASS（8 passed，包含固定版本校验）  
Dependency Check: PASS（pip check: No broken requirements found）

每项直接依赖安装后均单独验证导入；安装期间通过 `-c requirements.txt` 约束固定版本。

验证记录：

- `logs/install.log`
- `logs/dependency_resolution.log`
- `logs/environment_check.log`
- `logs/opencv_test.log`
- `logs/pyqt_test.log`
- `logs/yolo_test.log`
- `logs/import_test.log`
- `logs/pip_check.log`

## Known Issues

- 当前验收无未解决问题。
- 已解决：Qt 5 默认插件路径在本机中文目录下出现问号，首次窗口启动找不到 Cocoa 插件。`scripts/test_pyqt.py` 通过 PyQt5 安装位置显式添加 Unicode 插件目录后，原生窗口测试通过；未替换 PyQt5 或移动项目。
- 已解决：Ultralytics 首次导入时配置父目录未创建，回退到临时目录。检查脚本及测试现已在导入前创建项目内配置目录 `logs/ultralytics/`。
- `requirements-lock.txt` 为本机 macOS ARM64 的完整 pip freeze 快照。旧设备尚未提供操作系统、架构和硬件信息，跨设备安装及兼容性需在目标设备另行验收，不能以本机结果代替。
- `.venv/` 依赖项目 `.runtime/` 中的运行时，迁移时应在目标设备用 Python 3.10.20 重新创建虚拟环境并安装锁定依赖。

## Result

Environment Ready: YES

Python 3.10.20、C++17、NumPy、OpenCV 4.10.x、PyQt5、PyTorch、YOLO、ONNX Runtime 均 PASS。
最小项目目录、直接依赖清单、完整版本快照、README 和验证脚本均已创建。
仅完成环境初始化；未接入 SDK、摄像头、视频流，未开发台球识别、训练或业务系统。第一阶段完成后停止。


## 2026-10-14 接续更新

CPU 台球专用模型 40 轮训练完成，最佳权重已接入现有点选窗口。SciPy 1.15.3 使用官方 macOS 12 ARM64 wheel 修复当前系统的动态库加载错误，版本保持不变。完整模型验证、实际窗口点选、11 项 pytest 和 pip check 全部通过。详见 `TRAINING_REPORT.md`。此更新覆盖上文初始化阶段“未训练／未开发台球识别”的历史描述；真实摄像头和 SDK 尚未接入。
