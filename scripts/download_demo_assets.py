"""Download the pinned, publicly provided overhead billiards test recording."""
from pathlib import Path
from urllib.request import urlopen

import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "b1dff416c06f9a311b77aac4a73715219b8fd31b"
SOURCE_URL = (
    "https://raw.githubusercontent.com/louisgeisler/Billard/"
    + SOURCE_COMMIT + "/NOdebug.mp4"
)
OUTPUT = ROOT / "data" / "billiards_table.mp4"
EXPECTED_BYTES = 86174221


def main():
    if OUTPUT.exists():
        print("Video already exists:", OUTPUT)
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".mp4.part")
    try:
        with urlopen(SOURCE_URL, timeout=60) as response, temporary.open("wb") as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        if temporary.stat().st_size != EXPECTED_BYTES:
            raise RuntimeError("Downloaded video is incomplete")
        capture = cv2.VideoCapture(str(temporary))
        try:
            ok, _ = capture.read()
            if not ok:
                raise RuntimeError("Cannot decode downloaded video")
        finally:
            capture.release()
        temporary.replace(OUTPUT)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print("Billiards video ready:", OUTPUT)


if __name__ == "__main__":
    main()
