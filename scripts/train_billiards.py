"""CPU fine-tuning with conservative batches, memory checks and resumable snapshots."""
# 已有显式训练入口，本轮只添加注释，不运行或更换权重。
# CPU 小批次训练带内存采样、并发锁和完整优化器检查点；软件门槛不是操作系统硬性内存上限。

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'logs' / 'ultralytics'))


# 校验新训练/续训输入，设置既有 CPU 与内存策略，再注册内存/检查点回调。
# 完整 resume 与用 best.pt 发起新微调不同；已完成或无优化器的检查点不能冒充可续训状态。
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu'], default='cpu')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--imgsz', type=int, default=768)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--name', default='billiards_white_v1')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/dataset/data.yaml')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/billiards_v1.pt')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--max-rss-gb', type=float, default=6)
    parser.add_argument('--min-available-gb', type=float, default=4)
    args = parser.parse_args()
    if not (1 <= args.batch <= 2 and 32 <= args.imgsz <= 768 and args.threads > 0
            and args.epochs > 0 and args.max_rss_gb > 0 and args.min_available_gb > 0):
        parser.error('Use batch 1–2, imgsz <=768, and positive epochs/threads/memory budgets')
    (ROOT / 'logs').mkdir(exist_ok=True)
    lock_stream = (ROOT / 'logs' / 'training.lock').open('a+')
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another training process already holds logs/training.lock')
    import psutil
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    peak = 0.0
    log = ROOT / 'logs' / (args.name + '_memory.jsonl')
    log.parent.mkdir(parents=True, exist_ok=True)

    # 按训练回调采样进程 RSS 和系统可用内存，超出预算中断；无法覆盖所有瞬时算子峰值。
    def check_memory(trainer=None, record=False):
        nonlocal peak
        rss = process.memory_info().rss / 2**30
        available = psutil.virtual_memory().available / 2**30
        peak = max(peak, rss)
        if record:
            with log.open('a') as f:
                f.write(json.dumps({'epoch': getattr(trainer, 'epoch', -1) + 1,
                                    'rss_gb': rss, 'peak_rss_gb': peak,
                                    'available_gb': available}) + '\n')
        if rss > args.max_rss_gb or available < args.min_available_gb:
            raise RuntimeError(f'Memory budget reached: RSS={rss:.2f} GiB, available={available:.2f} GiB; '
                               'training stopped, resume from weights/resume_last.pt of the last completed epoch')

    # 保留含优化器的续训状态，避免只留下被剥离训练状态的部署权重。
    def preserve_checkpoint(trainer):
        # Ultralytics strips optimizer/epoch from last.pt at completion; keep a resumable copy.
        dest = trainer.last.parent / 'resume_last.pt'
        temporary = dest.with_suffix('.tmp')
        shutil.copy2(trainer.last, temporary)
        temporary.replace(dest)
        check_memory(trainer, record=True)

    check_memory(record=True)
    source = args.resume or args.model
    if not source.is_file():
        parser.error(f'Model does not exist: {source}')
    if args.resume:
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
        if checkpoint.get('optimizer') is None or checkpoint.get('epoch', -1) < 0:
            parser.error('Checkpoint has no optimizer state; use --model for a new fine-tuning run')
        completed = checkpoint['epoch'] + 1
        target = checkpoint['train_args']['epochs']
        if completed >= target:
            parser.error(f'Training is already complete ({completed}/{target}); nothing remains to resume')
        del checkpoint
    model = YOLO(str(source))
    model.add_callback('on_train_batch_start', check_memory)
    model.add_callback('on_train_batch_end', check_memory)
    model.add_callback('on_val_batch_start', lambda validator: check_memory())
    model.add_callback('on_model_save', preserve_checkpoint)
    options = dict(device='cpu', batch=args.batch, imgsz=args.imgsz, workers=0,
                   cache=False, amp=False, plots=False, save=True, save_period=1,
                   close_mosaic=0)
    if args.resume:
        model.train(resume=True, **options)
    else:
        model.train(data=str(args.data), epochs=args.epochs,
                    optimizer='AdamW', lr0=.0005, nbs=16, patience=10, seed=42,
                    mosaic=.2, scale=.2, translate=.1, fliplr=.5, flipud=.2,
                    hsv_h=.01, hsv_s=.2, hsv_v=.2, warmup_epochs=1,
                    project=str(ROOT / 'runs/train'), name=args.name,
                    exist_ok=False, **options)
    check_memory(record=True)
    print(f'Training complete; peak sampled process RSS: {peak:.2f} GiB')


if __name__ == '__main__':
    main()
