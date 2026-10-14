# 台球专用模型训练与接续验收

验收日期：2026-10-14（北京时间）。实际项目目录：`/Users/joey/projects/billiards-vision`；原任务已将项目迁移至此。

## 结果

已完成 CPU 40 轮训练，将最佳权重部署至 `models/billiards_v1.pt`，接入现有点选界面。模型类别为 `billiards_table`（台球桌）和 `billiard_ball`（台球）；不回退到通用 COCO 模型。

SHA256：`c8551cc7be1d9de4bfb09af38a1b8ec9038dc5400f4e7fb0af63052a66089146`

## 数据与训练

- 76 张训练图、20 张验证图；共 96 个球桌框、960 个球框。
- 原始八类球标注合并为“台球”；球桌框由绿色区域提议并检查，补充录像红黄球标注。
- 录像补充图：0 至 24 秒用于训练，30／33／36 秒用于验证。
- 同一录像前后段、原始图的相似场景使验证存在相关性。这是本地演示验证，未完成独立摄像头或新球桌泛化验收。
- 预训练模型：YOLOv8n；CPU，40 epochs，imgsz=768，batch=4，AdamW，seed=42。
- 训练参数及逐轮指标：`runs/train/billiards_cpu_v1/args.yaml`、`results.csv`。

复现训练：

```bash
.venv/bin/python scripts/prepare_billiards_dataset.py
.venv/bin/python scripts/train_billiards.py --device cpu --epochs 40 --name billiards_cpu_retrain
.venv/bin/python scripts/evaluate_billiards.py --model runs/train/billiards_cpu_retrain/weights/best.pt
```

新训练结果通过验证后再复制到界面使用的 `models/billiards_v1.pt`。

## 验证

- 验证集 Precision：0.9614；Recall：0.9484。
- mAP50：0.9723；mAP50-95：0.8432。
- 按界面参数 CPU、imgsz=960、conf=0.25 检查录像 0／5／27／30／33／36 秒：每帧 1 张球桌、12 个红黄球。
- 实际 PyQt 窗口：加载专用模型、点击球显示“台球”、点击桌面显示“台球桌”、黄色框、关闭开关清框、恢复播放、关闭窗口释放模型进程全部 PASS。
- pytest：11 passed；pip check：PASS。
- 预测数据：`logs/billiards_evaluation.json`；验证日志：`logs/billiards_validation.log`。
- 窗口截图：`logs/trained_ui_ball.png`、`logs/trained_ui_table.png`。

## 中断原因及修复

原 MPS 训练出现内存不足和验证异常，保留 CPU 训练路径。CPU 40 轮已经完成，最后验证因 SciPy 1.15.3 的 macOS 14 ARM64 wheel 在当前系统加载失败而中断。

同版本重装 macOS 14 wheel 仍复现错误；改装官方 PyPI 同版本 macOS 12 ARM64 wheel 后 `scipy.sparse.linalg` 导入及完整验证通过。Python、OpenCV、PyQt5、PyTorch、Ultralytics 等固定版本没有升级。此问题与 [SciPy 官方问题记录](https://github.com/scipy/scipy/issues/25635) 一致。

`requirements-lock.txt` 已更新，记录实际安装的 SciPy wheel URL；它仍仅适合本机 macOS ARM64，其他平台需另行生成环境快照。

## 当前限制

录像补充标注没有覆盖白球，本次确认的 12 个球均为红黄球；白球及其他设备画面仍需补标和独立验收。球桌检测为矩形框，不是精确桌面轮廓或透视校正。当前是暂停后点选识别，没有实时全帧识别、跟踪或评分功能。
