"""JSON-line worker: local YOLO inference outside the Qt application's process."""
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def reply(value):
    print(json.dumps(value), flush=True)


def main():
    try:
        model_path = ROOT / "models" / "billiards_v1.pt"
        if not model_path.is_file():
            raise FileNotFoundError("未找到台球专用模型 models/billiards_v1.pt，请先完成训练")
        config = ROOT / "logs" / "ultralytics"
        config.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        with redirect_stdout(sys.stderr):
            import numpy as np
            import torch
            from ultralytics import YOLO
            torch.set_num_threads(1)
            model = YOLO(str(model_path))
            if set(model.names.values()) != {"billiards_table", "billiard_ball"}:
                raise ValueError("模型类别不符合台球桌/台球专用模型要求")
        reply({"ready": True, "model": model_path.name})
    except Exception as exc:
        reply({"error": str(exc)})
        return 1
    for line in sys.stdin:
        try:
            request = json.loads(line)
            frame = np.load(request["frame"], allow_pickle=False)
            with redirect_stdout(sys.stderr):
                prediction = model.predict(frame, device="cpu", imgsz=960,
                                           conf=0.25, verbose=False)[0]
            matches = []
            for box in prediction.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                if x1 <= request["x"] <= x2 and y1 <= request["y"] <= y2:
                    confidence = float(box.conf[0])
                    name = prediction.names[int(box.cls[0])]
                    matches.append(((x2 - x1) * (y2 - y1), -confidence, name,
                                    confidence, [x1, y1, x2, y2]))
            if matches:
                _, _, name, confidence, box = min(matches)
                reply({"name": name, "confidence": confidence, "box": box})
            else:
                reply({"name": "", "confidence": 0.0, "box": None})
        except Exception as exc:
            reply({"error": str(exc)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
