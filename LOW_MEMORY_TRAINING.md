# 本机低内存训练

本机为 24 GiB Apple Silicon 统一内存。此前 MPS 出现内存不足及验证异常，当前训练只允许 CPU，不关闭系统内存保护。

默认：batch=2、imgsz=768、workers=0、cache=False、2 个计算线程、关闭训练绘图。基于已有台球权重微调，不从头重训。降低 batch 主要降低单步内存；保留 768 分辨率以照顾小球。内存监控采用已安装的 psutil。

训练进程 RSS 超过 6 GiB，或系统可用内存低于 4 GiB，在批次边界抛出错误并退出。此检查是软件阈值，不是操作系统硬性内存上限；单个算子仍可能出现瞬时峰值，其他应用也可能消耗系统内存。

每轮保存权重及优化器，并在 `weights/resume_last.pt` 保留完整状态，避免训练结束时 Ultralytics 清除优化器后无法续训。中途停止只重做尚未完成的当前轮。训练日志记录采样到的 RSS 峰值和系统可用内存。脚本通过文件锁拒绝同时启动第二个训练任务（本机 macOS）。

## 白球微调

原始 `whiteball` 共 84 个框，之前合并为“台球”；现在单列为 `white_ball`，中文显示“白球”，其余球保留“台球”。保持原始训练／验证划分：训练 67 个白球，验证 17 个白球。不将验证图加入训练。

```bash
.venv/bin/python scripts/prepare_billiards_dataset.py --white-class
.venv/bin/python scripts/train_billiards.py --data data/dataset_white/data.yaml --model models/billiards_v1.pt --epochs 20 --name billiards_white_v1
```

训练期间不要覆盖输入模型或修改训练数据。

## 低内存续训

```bash
.venv/bin/python scripts/train_billiards.py --resume runs/train/billiards_white_v1/weights/resume_last.pt --batch 1 --imgsz 768 --name billiards_white_resume
```

`--resume` 恢复检查点的训练目标、优化器和轮数；这里的 `--name` 用于内存日志，训练目录仍沿用原目录。如果系统可用内存不足，先关闭占用较大的应用再运行。已完成且清除了优化器的旧 `best.pt`／`last.pt` 用作新微调的 `--model`，不能假装是完整断点续训。

## 验证

```bash
.venv/bin/python scripts/evaluate_billiards.py --model runs/train/billiards_white_v1/weights/best.pt --data data/dataset_white/data.yaml --report-name billiards_white_evaluation
.venv/bin/python scripts/check_white_ball.py --model runs/train/billiards_white_v1/weights/best.pt
```

白球单独检查采用界面参数 imgsz=960、conf=0.25，以 IoU>=0.5 一对一匹配验证框，统计 precision／recall，并在所有验证图上统计白球误检。还需检查原录像球桌和红黄球是否退化，通过后才替换界面模型。

## 白球局部图强化

若全图微调后白球仍不足，增加 67 张训练集白球局部图，让小球在训练尺寸下更清晰。局部图也保留其他可见目标标注，验证图及验证框不变。局部图不是新的独立采集样本。

```bash
.venv/bin/python scripts/prepare_white_crops.py
.venv/bin/python scripts/train_billiards.py --data data/dataset_white_focused/data.yaml --model runs/train/billiards_white_v1/weights/last.pt --epochs 15 --name billiards_white_focused_v1
```

当前脚本使用 nbs=16 的梯度累积目标，提高小数据集每轮的优化器更新次数；实际每批仍是 2 张。新微调学习率为 0.0005。续训使用检查点自身的优化器及训练参数。
