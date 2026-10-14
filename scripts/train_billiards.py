"""Reproducible two-class fine-tuning using the project's fixed environment."""
import argparse
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('YOLO_AUTOINSTALL','false')
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'logs'/'ultralytics'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--imgsz',type=int,default=768)
    parser.add_argument('--name',default='billiards_v2')
    args=parser.parse_args()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model=YOLO(str(ROOT/'models'/'yolov8n.pt'))
    if args.device == 'mps':
        model.add_callback('on_train_batch_end', lambda trainer: torch.mps.empty_cache())
        model.add_callback('on_val_end', lambda validator: torch.mps.empty_cache())
    model.train(data=str(ROOT/'data/dataset/data.yaml'),epochs=args.epochs,
                imgsz=args.imgsz,batch=args.batch,device=args.device,workers=0,amp=False,
                optimizer='AdamW',lr0=.001,patience=15,seed=42,
                mosaic=.5,close_mosaic=10,scale=.25,translate=.1,
                fliplr=.5,flipud=.2,hsv_h=.01,hsv_s=.3,hsv_v=.3,
                project=str(ROOT/'runs/train'),name=args.name,
                plots=True,save=True,exist_ok=False)


if __name__=='__main__':main()
