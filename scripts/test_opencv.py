"""Generate and read back a basic OpenCV test image; no business logic."""
# OpenCV 最小图像读写验证，不是台球定位或性能基准。

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# 生成并回读本地测试图片，校验安装与基础图像 IO，不启动模型训练。
def main():
    print("OpenCV:", cv2.__version__)
    print("NumPy:", np.__version__)
    assert cv2.__version__.startswith("4.10."), "OpenCV must remain 4.10.x"
    image = np.zeros((500, 800, 3), dtype=np.uint8)
    cv2.putText(image, "OpenCV OK", (200, 250), cv2.FONT_HERSHEY_SIMPLEX,
                2, (255, 255, 255), 3)
    output = ROOT / "data" / "opencv_test.jpg"
    output.parent.mkdir(exist_ok=True)
    assert cv2.imwrite(str(output), image), "Image write failed"
    decoded = cv2.imread(str(output))
    assert decoded is not None and decoded.shape == image.shape
    print("Image:", output)
    print("OpenCV test PASS")


if __name__ == "__main__":
    main()
