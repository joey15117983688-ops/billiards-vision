# 台球与白球训练验收报告

验收日期：2026-10-14（北京时间）。文中路径均相对于仓库根目录。

## 最终结果

原 CPU 40 轮训练已完成。本次为补充独立白球类别，在原模型基础上完成 20 轮全图微调和 15 轮白球局部图强化，均正常结束。新最佳权重已部署至 `models/billiards_v1.pt`，窗口显示“台球桌／台球／白球”三个类别。

模型 SHA256：`6f9f9a0cce4e54c952993a63a485a59505b4ab1a8bc77e67bc2ade28d88a94e0`。旧两类权重保留在 Git 提交 `e942237`，本机备份为 `logs/billiards_before_white.pt`。

## 内存控制及实际结果

- 本机物理内存：24 GiB；使用 CPU，batch=2、imgsz=768、workers=0、cache=False、2 个计算线程，关闭训练绘图。
- 进程 RSS 超过 6 GiB，或系统可用内存低于 4 GiB，于批次边界中止；不使用 MPS、不关闭系统内存保护。
- 全图微调采样 RSS 峰值：1.59 GiB；采样系统可用内存最低：9.75 GiB。
- 局部图强化采样 RSS 峰值：1.41 GiB；采样系统可用内存最低：9.93 GiB。
- 本次没有触发内存保护，没有出现内存不足。
- 阈值检查为软件保护，不是操作系统硬性内存上限；采样不能覆盖所有瞬时峰值。
- 文件锁拒绝同时训练；每轮保留完整 `weights/resume_last.pt`（含优化器），另存周期检查点。
- 已验证：并行训练被拒绝、低内存预算启动被中止、最终 15/15 检查点保留优化器、已完成检查点被拒绝重复续训。

完整配置、运行和续训命令见 `LOW_MEMORY_TRAINING.md`；[Ultralytics 官方训练和续训说明](https://docs.ultralytics.com/modes/train/)。

## 白球数据

原始标注已有 84 个 `whiteball` 框，旧模型将其合并成普通台球；本次核对全部白球裁剪预览，单列为 `white_ball`。训练 67 个、验证 17 个，原始划分保持不变。

第一轮全图微调白球效果不足，随后从训练图增加 67 张白球局部图，保留局部图中其他可见目标标注。最终训练 143 张图：143 个球桌、977 个普通球、134 个白球框（其中 67 个白球框来自增强图）。验证 20 张原图：20 个球桌、107 个普通球、17 个白球。局部图并非新增独立采集样本。

裁剪来源及边界校验通过；验证原图、验证框未修改，验证来源未参与局部图生成。

## 最终验证

通用验证参数：CPU、imgsz=768、batch=2。三个类别平均指标：

- Precision：0.9636；Recall：0.9604。
- mAP50：0.9764；mAP50-95：0.8008。
- 白球 mAP50：0.9950。

实际界面参数：CPU、imgsz=960，台球／白球置信度门槛 0.25、球桌门槛 0.60。球桌未经门槛筛选时，在已知录像边缘有 0.32–0.48 置信度的额外框；0.60 门槛排除了这些框。验证脚本和界面共用 `src/billiards_inference.py`，原始检测计数也保留在报告 JSON 中。

- 对全部 20 张验证图执行白球检查，以 IoU>=0.5 一对一匹配：17 个标注、17 个预测、17 个正确匹配；Precision=100%、Recall=100%。该结果仅代表本次有限验证集。
- 录像 0／5／27／30／33／36 秒：门槛筛选后均为 1 张球桌、12 个红黄球、0 个白球预测。该录像没有用于白球验收。
- 实际 Qt 窗口：点击验证图白球显示“白球”，点击录像红黄球显示“台球”，点击桌面显示“台球桌”；框、开关、播放与关闭释放进程全部 PASS。
- pytest：12 passed；pip check：PASS。

## 产物

- 最终最佳权重：`runs/train/billiards_white_focused_v1/weights/best.pt`。
- 完整续训状态：`runs/train/billiards_white_focused_v1/weights/resume_last.pt`（15/15 已完成，无剩余轮数）。
- 训练指标：各训练目录的 `results.csv`、`args.yaml`。
- 内存记录：`logs/billiards_white_v1_memory.jsonl`、`logs/billiards_white_focused_v1_memory.jsonl`。
- 验证结果：`logs/billiards_white_evaluation.json`、`logs/white_ball_check.json`。
- 窗口截图：`logs/trained_ui_white.png`。

## 已知限制

同一录像前后段、原始相似场景会使验证相关，不能代表新摄像头、新球桌或各种光照的准确率。球桌仍是矩形目标框，不是精确轮廓。当前是暂停点选识别，没有实时全帧跟踪或评分。新模型增加独立白球类别；与旧两类模型的平均指标不能直接等同比较。

此前 SciPy 1.15.3 macOS 14 wheel 的动态库加载问题，已通过同版本官方 macOS 12 ARM64 wheel 解决。固定 Python、OpenCV、PyQt5、PyTorch、Ultralytics 版本未升级。
