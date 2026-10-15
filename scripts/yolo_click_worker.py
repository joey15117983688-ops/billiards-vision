"""JSON-line worker: local YOLO inference outside the Qt application's process."""
from contextlib import redirect_stdout
import json
import cv2
import numpy as np
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.billiards_inference import filter_detections
from src.target_balls import scene_detections
from src.table_geometry import prediction_detections, contains_point


def reply(value):
    print(json.dumps(value), flush=True)


def detect_selected(model,frame,target):
    if target['name']=='billiards_table':
        from src.vision.pocket import TableDetector
        table=TableDetector().detect(frame)
        if not table.detected:return []
        polygon=table.scope_polygon or table.playable_polygon
        points=np.array(polygon)
        return [{'name':'billiards_table','box':[float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())],
                 'polygon':polygon,'playable_polygon':table.playable_polygon,'confidence':table.confidence}]
    x1,y1,x2,y2=target['box']
    cx,cy=(x1+x2)/2,(y1+y2)/2
    radius=max(40.,2.5*max(x2-x1,y2-y1))
    ax,ay=max(0,int(cx-radius)),max(0,int(cy-radius))
    bx,by=min(frame.shape[1],int(cx+radius)),min(frame.shape[0],int(cy+radius))
    candidates=[]
    if bx>ax and by>ay:
        with redirect_stdout(sys.stderr):
            result=model.predict(frame[ay:by,ax:bx],device='cpu',imgsz=320,conf=.25,verbose=False)[0]
        result=filter_detections(result)
        for box in result.boxes:
            if result.names[int(box.cls[0])]=='billiards_table':continue
            a,b,c,d=box.xyxy[0].cpu().tolist();bounds=[a+ax,b+ay,c+ax,d+ay]
            if min(c-a,d-b)<.45*min(x2-x1,y2-y1) or max(c-a,d-b)>2.5*max(x2-x1,y2-y1):continue
            distance=((a+c)/2+ax-cx)**2+((b+d)/2+ay-cy)**2
            if distance>max(25.,1.5*max(x2-x1,y2-y1))**2:continue
            candidates.append((distance,{'name':target['name'],'box':bounds,'confidence':float(box.conf[0])}))
        if target['name'] in ('black_ball','white_ball'):
            hsv=cv2.cvtColor(frame[ay:by,ax:bx],cv2.COLOR_BGR2HSV)
            selected=((hsv[:,:,2]<100)&(hsv[:,:,1]<180)) if target['name']=='black_ball' else ((hsv[:,:,2]>155)&(hsv[:,:,1]<90))
            mask=cv2.morphologyEx(selected.astype(np.uint8)*255,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                a,b,c,d=cv2.boundingRect(contour)
                area=cv2.contourArea(contour)
                if not .25*(x2-x1)*(y2-y1)<area<3*(x2-x1)*(y2-y1):continue
                if min(c,d)<.5*min(x2-x1,y2-y1) or max(c,d)/min(c,d)>2.2:continue
                distance=(a+c/2+ax-cx)**2+(b+d/2+ay-cy)**2
                if distance>max(25.,1.5*max(x2-x1,y2-y1))**2:continue
                candidates.append((distance,{'name':target['name'],'box':[a+ax,b+ay,a+c+ax,b+d+ay],'confidence':.8}))
    return [min(candidates,key=lambda pair:pair[0])[1]] if candidates else []


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
            if set(model.names.values()) not in ({"billiards_table", "billiard_ball"},
                                                  {"billiards_table", "billiard_ball", "white_ball"}):
                raise ValueError("模型类别不符合台球桌/台球专用模型要求")
        reply({"ready": True, "model": model_path.name})
    except Exception as exc:
        reply({"error": str(exc)})
        return 1
    for line in sys.stdin:
        try:
            request = json.loads(line)
            frame = np.load(request["frame"], allow_pickle=False)
            if request.get("mode") in ("single","selected"):
                targets=request.get('targets') or [request['target']]
                selected=[]
                for target in targets:
                    for item in detect_selected(model,frame,target):
                        item['selection_id']=target.get('selection_id',target.get('track_id'))
                        selected.append(item)
                reply({'detections':selected})
                continue
            with redirect_stdout(sys.stderr):
                prediction = model.predict(frame, device="cpu", imgsz=960,
                                           conf=0.25, verbose=False)[0]
            prediction = filter_detections(prediction)
            detections = scene_detections(frame, prediction_detections(prediction, frame))
            if request.get("mode") == "all":
                reply({"detections": detections})
                continue
            matches = [item for item in detections if contains_point(item, request["x"], request["y"])]
            if matches:
                item = min(matches, key=lambda d: ((d["box"][2]-d["box"][0]) *
                                                  (d["box"][3]-d["box"][1]), -d["confidence"]))
                reply(item)
            else:
                reply({"name": "", "confidence": 0.0, "box": None, "polygon": None})
        except Exception as exc:
            reply({"error": str(exc)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
