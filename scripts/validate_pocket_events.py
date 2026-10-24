"""Generate complete synthetic shot videos and annotated inputs, then replay.
These are logic regressions, NOT measurements of real-camera/YOLO accuracy.
"""
# 进球模块的受控验证工具；合成轨迹/已有素材分别用于规则回归，不宣称现场准确率。

import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from detect_pocket_events import run

CENTERS={'TOP_LEFT':(0,0),'TOP_MIDDLE':(1,0),'TOP_RIGHT':(2,0),
         'BOTTOM_LEFT':(0,1),'BOTTOM_MIDDLE':(1,1),'BOTTOM_RIGHT':(2,1)}


# 生成完整合成视频及配套观测 JSONL，区分正常入袋、重新出现/遮挡等状态机条件。
def generate(root,kind,pocket='TOP_RIGHT'):
    root.mkdir(parents=True,exist_ok=True)
    def pixels(point):return (np.asarray(point)*250+[70,70]).tolist()
    pockets=[]
    for name,c in CENTERS.items():
        p=pixels(c);r=9
        pockets.append({'id':name,'center':dict(zip(('x','y'),p)),'status':'estimated' if kind=='unreliable' else 'observed',
                        'entry_polygon':[[p[0]-r,p[1]-r],[p[0]+r,p[1]-r],[p[0]+r,p[1]+r],[p[0]-r,p[1]+r]]})
    geometry={'table':{'playable_polygon':[[70,70],[570,70],[570,320],[70,320]],'quality':{'geometry_reliable':True}},'pockets':pockets}
    (root/'pockets.json').write_text(json.dumps(geometry))
    video=root/'shot.mp4';writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(640,400))
    if not writer.isOpened():raise RuntimeError('Video writer unavailable')
    c=np.asarray(CENTERS[pocket]);direction=np.array([1,.5])-c;direction/=np.linalg.norm(direction)
    with (root/'detections.jsonl').open('w') as output:
        for i in range(80):
            t=i/20;frame=np.full((400,640,3),(50,65,85),np.uint8)
            cv2.rectangle(frame,(70,70),(570,320),(35,150,35),-1)
            for p in pockets:cv2.circle(frame,tuple(np.int32(list(p['center'].values()))),11,(4,4,4),-1)
            target='white_ball' if kind=='white_pot' else 'black_ball'
            points={'black_ball':(.7,.5),'white_ball':(1.25,.6)}
            if kind!='both_visible':
                if t<.8:points[target]=c+direction*max(.008,.25-.25*t/.65)
                elif kind in ('bounce','stop'):points[target]=c+direction*(.12 if kind=='bounce' else .035)
                elif kind in ('occlusion','miss') and t>=1.1:points[target]=c+direction*.07
                else:points.pop(target)
            if kind=='elsewhere' and t<.8:points[target]=(.7+t*.15,.5)
            detections=[]
            for name,point in points.items():
                x,y=pixels(point);cv2.circle(frame,(round(x),round(y)),4,(240,240,240) if name=='white_ball' else (8,8,8),-1)
                detections.append({'name':name,'confidence':.95,'box':[x-4,y-4,x+4,y+4],'track_id':1 if t<.4 else 99})
            occluded=kind=='occlusion' and .7<t<1.2
            if occluded:cv2.rectangle(frame,(400,60),(600,130),(100,150,210),-1)
            cv2.putText(frame,f'{kind} {pocket} {t:.2f}s',(15,380),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            writer.write(frame)
            if i%2==0:output.write(json.dumps({'timestamp':t,'detections':detections,'scene_clear':not occluded})+'\n')
    writer.release()
    return video


# 组织验证场景并保存实际裁决，规则测试成功不等于新球房视频已验收。
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'data/pocket_events_validation')
    args=parser.parse_args();report=[]
    cases=[('pot',p,'POTTED') for p in CENTERS]+[(kind,'TOP_MIDDLE',expected) for kind,expected in [
        ('bounce','NOT_POTTED'),('stop','NOT_POTTED'),('occlusion','NOT_POTTED'),('miss','NOT_POTTED'),
        ('elsewhere','UNKNOWN'),('white_pot','POTTED'),('both_visible','NOT_POTTED'),('unreliable','UNKNOWN')]]
    for kind,pocket,expected in cases:
        folder=args.output/f'{kind}_{pocket}';video=generate(folder,kind,pocket)
        # Stage one is exercised separately with the exact same inputs/engine.
        if kind=='pot' and pocket=='TOP_RIGHT':run(video,folder/'black_only',folder/'pockets.json',folder/'detections.jsonl',True)
        result=run(video,folder/'replay',folder/'pockets.json',folder/'detections.jsonl')
        ball='white_ball' if kind=='white_pot' else 'black_ball'
        event=next(e for e in result['results'][0]['events'] if e['ball']==ball)
        passed=event['event']==expected and (expected!='POTTED' or event['pocket']==pocket)
        report.append({'case':folder.name,'expected':expected,'actual':event,'passed':passed,'artifacts':str(folder.resolve())})
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'validation.json').write_text(json.dumps({'kind':'synthetic_video_annotated_detection_replay','cases':report},ensure_ascii=False,indent=2))
    print(f'{sum(c["passed"] for c in report)}/{len(report)} synthetic recorded cases passed')
    if not all(c['passed'] for c in report):raise SystemExit(1)

if __name__=='__main__':main()
