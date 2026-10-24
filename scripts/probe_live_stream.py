#!/usr/bin/env python3
"""Read-only live-input preflight; does not count pots or save calibrations."""
# 独立流诊断入口，尚不接入主窗口；本地 paced replay 可在无硬件时测试消费/连续性流程。
# 配置与日志可能含私有连接地址，运行产物只保留本地；不把连接成功当作识别达标。

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.live_source import LiveSourceConfig,LatestFrameStream
from src.live_readiness import LiveReadiness
from src.camera_alignment import CameraAlignment,map_points
from src.calibration_state import load_automatic


# 给本地视频读取加节奏，模拟持续生产者，属于测试回放而非现场摄像头。
class PacedReplay:
    """A local file acting like a paced camera; only for preflight tests."""
    def __init__(self,path):
        self.capture=cv2.VideoCapture(str(path));self.deadline=time.monotonic()
        fps=self.capture.get(cv2.CAP_PROP_FPS);self.period=1/(fps if 1<=fps<=120 else 30)
    def isOpened(self):return self.capture.isOpened()
    # 按既定时间节奏读取原片，保留真实帧内容，不生成检测/入袋结果。
    def read(self):
        remaining=self.deadline-time.monotonic()
        if remaining>0:time.sleep(remaining)
        result=self.capture.read();self.deadline=max(self.deadline+self.period,time.monotonic())
        return result
    def release(self):self.capture.release()


# 读取明确的来源、源标识与诊断参数，不自动发现或连接未知设备。
def arguments():
    p=argparse.ArgumentParser(description='实时输入前期诊断：不改主程序、不保存校准、不判进球')
    p.add_argument('--camera-id',required=True,help='稳定的球房／球桌／摄像机标识，与拉流地址无关')
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--stream-env',metavar='ENV_NAME',help='从环境变量读取 RTSP／HTTP 地址，地址不写入诊断日志')
    source.add_argument('--video',type=Path,help='本地视频模拟实时输入，用于提前测试')
    p.add_argument('--duration',type=float,default=30)
    p.add_argument('--consumer-ms',type=float,default=0,help='模拟识别处理耗时，验证慢消费者丢旧帧')
    p.add_argument('--calibration',type=Path,help='已有 .auto.json，只读；无参考图的旧记录不认定为已完成对齐')
    p.add_argument('--output',type=Path,help='可选：保存逐帧诊断 JSONL，不保存视频')
    args=p.parse_args()
    if args.duration<=0 or args.consumer_ms<0:p.error('duration 必须大于 0，consumer-ms 不得为负')
    if args.stream_env and not os.environ.get(args.stream_env):p.error('指定环境变量为空，请先配置视频流地址')
    if args.video and not args.video.is_file():p.error('找不到模拟输入视频')
    if args.calibration and not args.calibration.is_file():p.error('找不到校准记录')
    return args


# 独立启动最新帧流并评估预检状态，最后回收读取器，输出仅作为前期准备诊断。
def run(args):
    source=os.environ[args.stream_env] if args.stream_env else str(args.video)
    config=LiveSourceConfig(args.camera_id,source)
    stream=LatestFrameStream(config) if args.stream_env else LatestFrameStream(config,lambda _:PacedReplay(args.video))
    reference_data=None;manual=None
    if args.calibration:
        reference_data=json.loads(args.calibration.read_text())
        manual_path=args.calibration.with_name(args.calibration.name.replace('.auto.json','.json'))
        if manual_path!=args.calibration and manual_path.exists():manual=json.loads(manual_path.read_text())
    output=None
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);output=args.output.open('w',encoding='utf-8')
    readiness=LiveReadiness();alignment=None;base_scope=None;geometry_shape=None
    received_any=False;last_report=0.;last_row=None;start=time.monotonic();last_session=None
    stream.start()
    try:
        while time.monotonic()-start<args.duration:
            packet=stream.read_latest(timeout=.1)
            if packet is not None:
                received_any=True;geometry_ready=False;scope=None
                # A reconnect/resolution change invalidates the working registration.
                if (packet.session,packet.image.shape[:2])!=geometry_shape:
                    alignment=None;base_scope=None
                    geometry_shape=(packet.session,packet.image.shape[:2])
                    if reference_data:
                        restored=load_automatic(args.calibration,packet.image,manual,verify_scene=False)
                        if restored:
                            base_scope=restored[1]['table']['scope_polygon']
                            reference=reference_data.get('alignment_reference')
                            if reference:alignment=CameraAlignment(packet.image,base_scope,reference)
                if alignment is not None:
                    alignment.update(packet.image,force=True)
                    geometry_ready=not alignment.lost
                    scope=map_points(base_scope,alignment.matrix)
                assessment=readiness.assess(packet,geometry_ready,scope)
                last_row={'kind':'frame','sequence':packet.sequence,'session':packet.session,
                          'received_at_monotonic':packet.received_at,'frame_size':list(packet.image.shape[:2]),
                          **asdict(assessment)}
                if output:output.write(json.dumps(last_row,ensure_ascii=False)+'\n')
                if args.consumer_ms:time.sleep(args.consumer_ms/1000)
                last_session=packet.session
            now=time.monotonic()
            if now-last_report>=1:
                print(json.dumps({'kind':'status','elapsed_seconds':round(now-start,2),**stream.snapshot(),
                                  'last_assessment':last_row},ensure_ascii=False),flush=True)
                last_report=now
    except KeyboardInterrupt:pass
    finally:
        stopped=stream.stop()
        if output:output.close()
        print(json.dumps({'kind':'summary','duration_seconds':round(time.monotonic()-start,2),
                          'reader_stopped':stopped,'received_any':received_any,**stream.snapshot()},ensure_ascii=False),flush=True)
    return 0 if received_any and stopped else 2


if __name__=='__main__':
    try:sys.exit(run(arguments()))
    except (OSError,ValueError,cv2.error):
        # Avoid printing decoder/config exception messages that may embed the URL.
        print('诊断未完成：请检查输入地址、FFmpeg 支持及校准文件格式。',file=sys.stderr)
        sys.exit(2)
