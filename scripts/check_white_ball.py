"""Check white-ball predictions against held-out boxes at the UI's confidence threshold."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.billiards_inference import filter_detections
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'logs/ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
os.environ.setdefault('MPLBACKEND', 'Agg')


def overlap(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2-x1) * max(0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    import cv2
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    model = YOLO(str(args.model))
    white_id = next(c for c, name in model.names.items() if name == 'white_ball')
    dataset = ROOT / 'data/dataset_white'
    output = ROOT / 'logs/white_predictions'
    output.mkdir(exist_ok=True)
    totals = {'ground_truth': 0, 'predictions': 0, 'matched': 0}
    frames = []
    for path in sorted((dataset/'images/val').glob('*.jpg')):
        image = cv2.imread(str(path))
        height, width = image.shape[:2]
        labels = (dataset/'labels/val'/path.with_suffix('.txt').name).read_text().splitlines()
        boxes = []
        for line in labels:
            cls, x, y, w, h = map(float, line.split())
            if int(cls) == white_id:
                boxes.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
        result = model.predict(image, device='cpu', imgsz=960, conf=.25, verbose=False)[0]
        result = filter_detections(result)
        predictions = [b.xyxy[0].tolist() for b in result.boxes if int(b.cls[0]) == white_id]
        pairs = sorted(((overlap(p,g), i,j) for i,p in enumerate(predictions) for j,g in enumerate(boxes)), reverse=True)
        used_predictions, used_labels = set(), set()
        for iou,i,j in pairs:
            if iou >= .5 and i not in used_predictions and j not in used_labels:
                used_predictions.add(i); used_labels.add(j)
        totals['ground_truth'] += len(boxes)
        totals['predictions'] += len(predictions)
        totals['matched'] += len(used_labels)
        frames.append({'file': str(path), 'white_boxes': boxes, 'predictions': predictions,
                       'matched': len(used_labels)})
        if boxes:
            cv2.imwrite(str(output/path.name), result.plot(font_size=12, line_width=2))
    report = {**totals, 'precision': totals['matched']/max(totals['predictions'],1),
              'recall': totals['matched']/max(totals['ground_truth'],1), 'frames': frames}
    (ROOT/'logs/white_ball_check.json').write_text(json.dumps(report, indent=2))
    print({k:v for k,v in report.items() if k != 'frames'})


if __name__ == '__main__':
    main()
