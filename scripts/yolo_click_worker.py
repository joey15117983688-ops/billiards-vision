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
from src.ball_observation import BALL_NAMES,local_balls
from src.ball_tracks import appearance,BallTracks
from src.table_geometry import prediction_detections, contains_point


def reply(value):
    print(json.dumps(value), flush=True)


def detect_selected(model,frame,target,calibration_polygon=None,scope_polygon=None):
    if target['name']=='billiards_table':
        from src.vision.pocket import TableDetector
        table=TableDetector().detect(frame) if calibration_polygon is None else None
        if table is not None and not table.detected:return []
        inner=calibration_polygon if calibration_polygon is not None else table.playable_polygon
        from src.cloth_region import cloth_region
        region=None if scope_polygon is not None else cloth_region(frame,inner)
        polygon=scope_polygon if scope_polygon is not None else region['polygon'] if region else inner
        points=np.array(polygon)
        return [{'name':'billiards_table','box':[float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())],
                 'polygon':polygon,'playable_polygon':inner,'confidence':target.get('confidence',0.) if table is None else table.confidence}]
    if target['name'] not in BALL_NAMES:return []
    # The same rounded observation is used for coloured, black and white balls.
    # YOLO still runs only on the selected local crop; cue-shaped predictions
    # cannot become ball observations without a rounded image component.
    x1,y1,x2,y2=target['box'];cx,cy=(x1+x2)/2,(y1+y2)/2
    radius=max(40.,3*max(x2-x1,y2-y1))
    ax,ay=max(0,int(cx-radius)),max(0,int(cy-radius))
    bx,by=min(frame.shape[1],int(cx+radius)),min(frame.shape[0],int(cy+radius))
    if bx<=ax or by<=ay:return []
    with redirect_stdout(sys.stderr):
        prediction=model.predict(frame[ay:by,ax:bx],device='cpu',imgsz=320,conf=.25,verbose=False)[0]
    candidates=local_balls(frame,target,3.)
    prediction=filter_detections(prediction)
    supported=[]
    for box in prediction.boxes:
        if prediction.names[int(box.cls[0])] not in BALL_NAMES:continue
        a,b,c,d=box.xyxy[0].cpu().tolist()
        if min(c-a,d-b)<=0 or max(c-a,d-b)/min(c-a,d-b)>1.8:continue
        supported.append(((a+c)/2+ax,(b+d)/2+ay,float(box.conf[0])))
    reference=target.get('_appearance')
    reference=np.array(reference,np.float32) if reference is not None else None
    ranked=[]
    for item in candidates:
        colour=BallTracks.distance(reference,appearance(frame,item))
        if colour>.65:continue
        a,b,c,d=item['box'];distance=np.hypot((a+c)/2-cx,(b+d)/2-cy)
        size=max(x2-x1,y2-y1)
        if distance>max(25.,2.5*size):continue
        item.pop('_appearance',None)
        evidence=[confidence for px,py,confidence in supported if np.hypot(px-(a+c)/2,py-(b+d)/2)<.6*size]
        if evidence:item['confidence']=max(evidence)
        ranked.append((distance/size+2*colour,item))
    return [min(ranked,key=lambda pair:pair[0])[1]] if ranked else []


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
                    for item in detect_selected(model,frame,target,request.get("calibration_polygon"),request.get("scope_polygon")):
                        item['selection_id']=target.get('selection_id',target.get('track_id'))
                        selected.append(item)
                reply({'detections':selected})
                continue
            with redirect_stdout(sys.stderr):
                prediction = model.predict(frame, device="cpu", imgsz=960,
                                           conf=0.25, verbose=False)[0]
            prediction = filter_detections(prediction)
            detections = scene_detections(frame, prediction_detections(prediction, frame), request.get("calibration_polygon"),request.get("scope_polygon"))
            if request.get("mode") == "all":
                reply({"detections": detections})
                continue
            scene_geometry=next((item for item in detections if item.get("calibration_evidence")),None)
            matches = [item for item in detections if contains_point(item, request["x"], request["y"])]
            if matches:
                item = min(matches, key=lambda d: ((d["box"][2]-d["box"][0]) *
                                                  (d["box"][3]-d["box"][1]), -d["confidence"]))
                reply({**item,"scene_geometry":scene_geometry})
            else:
                reply({"name": "", "confidence": 0.0, "box": None, "polygon": None,"scene_geometry":scene_geometry})
        except Exception as exc:
            reply({"error": str(exc)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
