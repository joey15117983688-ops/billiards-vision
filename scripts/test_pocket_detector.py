"""Run one pocket localization on an image or a video frame and save diagnostic artifacts."""
# 单帧袋口检测的命令行诊断；输入本地图片或指定视频时间点，调试产物不等于现场准确率。

__test__ = False
import argparse
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from src.vision.pocket import PocketConfig, PocketDetector
from src.vision.pocket.debug import save_debug_result


# 读取指定原帧，运行已有 PocketDetector 并保存结构化结果与本地调试图；不更新用户持久化校准。
def main():
    parser=argparse.ArgumentParser(description='独立检测球桌／袋口并保存 JSON 和调试图，不加载 YOLO')
    inputs=parser.add_mutually_exclusive_group()
    inputs.add_argument('--image',type=Path)
    inputs.add_argument('--video',type=Path)
    parser.add_argument('--at',type=float,default=5,help='video timestamp in seconds')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--config',type=Path,help='JSON configuration for PocketConfig, including cloth HSV ranges')
    parser.add_argument('--long-axis',choices=['auto','edge_01','edge_12'])
    parser.add_argument('--require-valid',action='store_true')
    args=parser.parse_args()
    if args.at < 0:parser.error("--at must not be negative")
    options=json.loads(args.config.read_text()) if args.config else {}
    if args.long_axis:options['long_axis']=args.long_axis
    config=PocketConfig(**options)
    source=(args.image or args.video or ROOT/'data/billiards_table.mp4').expanduser().resolve()
    if not source.is_file():parser.error(f'Input does not exist: {source}')
    if args.image:
        frame=cv2.imdecode(np.fromfile(str(source),np.uint8),cv2.IMREAD_COLOR)
        origin={'kind':'image','path':str(source)}
    else:
        capture=cv2.VideoCapture(str(source))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC,args.at*1000)
            ok,frame=capture.read()
            if not ok:frame=None
            origin={'kind':'video','path':str(source),'requested_seconds':args.at,
                    'frame_index':max(0,int(capture.get(cv2.CAP_PROP_POS_FRAMES))-1)}
        finally:capture.release()
    if frame is None:parser.error('Could not decode requested image/video frame')
    result=PocketDetector(config).detect(frame);result.source=origin
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    output=args.output or ROOT/'data/debug/pocket_detection'/stamp
    path=save_debug_result(frame,result,output)
    print('Table detected:',result.table.detected,'Complete table:',result.table.complete)
    print('Playable polygon:',result.table.playable_polygon)
    print('Inner cushion boundaries:',result.table.boundary_lines)
    for pocket in result.pockets:
        print(pocket.id,'candidate=',pocket.candidate_center,'center=',pocket.center,
              'status=',pocket.status,'quality=',round(pocket.confidence,3),'reasons=',pocket.reasons)
    print('Valid:',result.valid,'Elapsed ms:',round(result.elapsed_ms,2),'Reasons:',result.reasons)
    print('JSON:',path)
    return 2 if args.require_valid and not result.valid else 0


if __name__=='__main__':raise SystemExit(main())
