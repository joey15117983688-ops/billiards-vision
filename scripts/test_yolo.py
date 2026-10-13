"""Load an official nano model and infer on one ordinary test image."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
config_dir = ROOT / "logs" / "ultralytics"
config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from ultralytics import YOLO


def main():
    image = ROOT / "data" / "opencv_test.jpg"
    if not image.exists():
        raise FileNotFoundError("Run scripts/test_opencv.py first")
    model = YOLO(str(ROOT / "models" / "yolov8n.pt"))
    results = model.predict(source=str(image), imgsz=320, device="cpu",
                            save=False, verbose=True)
    assert len(results) == 1 and results[0].orig_shape == (500, 800)
    print("Model: official yolov8n.pt")
    print("YOLO test PASS")


if __name__ == "__main__":
    main()
