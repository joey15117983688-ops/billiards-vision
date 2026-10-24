"""Download and verify the default overhead pool clip."""
# 公开演示视频下载/验证工具；用户现场录像不会经此脚本上传或分发。
# 按固定来源、大小、摘要和解码检查接受文件，流式下载避免整段视频占内存。

import hashlib
from pathlib import Path
from urllib.request import Request, urlopen

import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = 'https://raw.githubusercontent.com/FedericoPivotto/billiard-video-analysis/13fa4f15b41614c894e4cdd49a7557461680efaf/dataset/game1_clip1/game1_clip1.mp4'
OUTPUT = ROOT / 'data' / 'overhead_pool.mp4'
EXPECTED_BYTES = 951516
EXPECTED_SHA1 = '3fe9c16245b82fd987f237ba869d4838bf7137cd'


# 分块求文件摘要，用于固定素材身份核验。
def digest(path):
    checksum = hashlib.sha1()
    with path.open('rb') as source:
        while block := source.read(1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


# 核验体积/校验和及首中末帧能解码，不能仅因文件存在就认为下载完整。
def validate(path):
    if path.stat().st_size != EXPECTED_BYTES or digest(path) != EXPECTED_SHA1:
        raise RuntimeError('Video size/checksum does not match the source')
    capture = cv2.VideoCapture(str(path))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or frames <= 0:
            raise RuntimeError('Cannot open downloaded overhead pool video')
        for index in [0, frames // 2, frames - 1]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, _ = capture.read()
            if not ok:
                raise RuntimeError(f'Cannot decode video frame {index}')
    finally:
        capture.release()


# 按已有下载策略获取公开演示片并验证；失败保留诊断，不将错误网页当作视频。
def main():
    if OUTPUT.exists():
        validate(OUTPUT)
        print('Overhead pool video already verified:', OUTPUT)
        return
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix('.mp4.part')
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > EXPECTED_BYTES:
        raise RuntimeError('Partial file is larger than expected; inspect it before retrying')
    if offset < EXPECTED_BYTES:
        request = Request(SOURCE_URL, headers={'Range': f'bytes={offset}-'} if offset else {})
        with urlopen(request, timeout=60) as response:
            resume = response.status == 206 and offset > 0
            if response.status == 206:
                start = response.headers.get('Content-Range', '').split('-')[0]
                if start != f'bytes {offset}':
                    raise RuntimeError('Unexpected server resume offset')
            downloaded = offset if resume else 0
            with temporary.open('ab' if resume else 'wb') as target:
                while block := response.read(1024 * 1024):
                    target.write(block)
                    downloaded += len(block)
                    if downloaded > EXPECTED_BYTES:
                        raise RuntimeError('Server returned more data than expected')
                    if downloaded // (16 * 1024 * 1024) != (downloaded-len(block)) // (16 * 1024 * 1024):
                        print(f'Downloaded {downloaded / 1024**2:.0f}/{EXPECTED_BYTES / 1024**2:.0f} MiB', flush=True)
    validate(temporary)
    temporary.replace(OUTPUT)
    print('Overhead pool video ready:', OUTPUT)


if __name__ == '__main__':
    main()
