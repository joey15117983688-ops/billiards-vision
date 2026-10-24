"""Replay a complete recording through the same engine used by the desktop UI."""
# 离线进球诊断入口，串行读取本地帧并运行现有观察/事件模块，不改变 GUI 主循环。
# 输出事件是算法判定，仍需原图证据复核；本地结果可能含私人视频信息。

import argparse
import json
from pathlib import Path
import sys
import cv2

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.vision.pocket import PocketDetector
from src.vision.pocket_event.adapter import PocketEventAdapter


# 打开指定来源，建立几何和观测链路，按帧时间推进事件状态并保存诊断结果。
def run(video,output,pockets=None,detections=None,black_only=False,hz=10):
    capture=cv2.VideoCapture(str(video));runner=None
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    try:
        fps=capture.get(cv2.CAP_PROP_FPS)
        if not fps>0:raise ValueError('Invalid video frame rate')
        ok,frame=capture.read()
        if not ok:raise ValueError('Cannot decode recording')
        geometry=json.loads(Path(pockets).read_text()) if pockets else PocketDetector().detect(frame).to_dict()
        (output/'pockets.json').write_text(json.dumps(geometry,ensure_ascii=False,indent=2))
        adapter=PocketEventAdapter(geometry,balls=('black_ball',) if black_only else ('black_ball','white_ball'),track_target_balls=not black_only)
        replay=None
        if detections:
            replay={round(float(row['timestamp'])*fps):row for row in map(json.loads,Path(detections).read_text().splitlines())}
        else:
            from main import YoloWorker
            runner=YoloWorker();errors=[];runner.failed.connect(lambda *args:errors.append(args))
            runner.load()
            if errors:raise RuntimeError(str(errors))
            runner.calibration_polygon=geometry['table']['playable_polygon']
        stride=max(1,round(fps/hz));index=0;previous=None;events=[];count=0
        with (output/'timeline.jsonl').open('w') as timeline:
            while ok:
                row=replay.get(index) if replay is not None else None
                if row is not None or replay is None and index%stride==0:
                    timestamp=index/fps
                    raw=row['detections'] if row else runner.request_prediction(frame,{'mode':'all'})['detections']
                    results=adapter.update(timestamp,raw,frame,scene_clear=row.get('scene_clear') if row else None)
                    snapshot=adapter.engine.snapshot();count+=1
                    signature=[(b,s['state'],s['candidate_pocket']) for b,s in snapshot['balls'].items()]
                    key_frame=None
                    if signature!=previous or results:
                        key_frame=f'frame_{index:06d}_{timestamp:.3f}.jpg'
                        cv2.imwrite(str(output/key_frame),frame);previous=signature
                    timeline.write(json.dumps({'timestamp':timestamp,'snapshot':snapshot,'results':results,'key_frame':key_frame,'detections':raw,'scene_clear':adapter.last_scene_clear},ensure_ascii=False)+'\n')
                    events.extend(results)
                index+=1;ok,frame=capture.read()
            if not adapter.engine.completed and adapter.engine.timestamp is not None:
                events.extend(adapter.engine.finish(force_unknown='recording_ended_before_final_review'))
        summary={'video':str(Path(video).resolve()),'input_kind':'annotated_detection_replay' if replay is not None else 'existing_yolo_inference',
                 'frame_count':index,'inference_samples':count,'results':list({r['shot_id']:r for r in events}.values()),'coordinate_units':'fraction_of_table_short_side',
                 'confidence_kind':'evidence_rules_not_calibrated_probability'}
        (output/'result.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        return summary
    finally:
        capture.release()
        if runner:runner.shutdown()


# 解析离线来源与输出参数，显式调用 run，不默认启动真实摄像头或训练。
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pockets',type=Path)
    parser.add_argument('--detections',type=Path,help='optional timestamped annotated detections JSONL for deterministic replay')
    parser.add_argument('--black-only',action='store_true')
    parser.add_argument('--hz',type=float,default=10)
    args=parser.parse_args()
    if not 1<=args.hz<=30:parser.error('--hz must be 1..30')
    result=run(args.video,args.output,args.pockets,args.detections,args.black_only,args.hz)
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
