"""Stream position records to JSONL; never retain an entire video in memory."""
from datetime import datetime
from pathlib import Path
import json
import time


class TrackRecorder:
    def __init__(self,root):
        self.root=Path(root);self.stream=None;self.identity=None;self.path=None;self.last_frame=None;self.last_flush=0

    def close(self):
        if self.stream:self.stream.close()
        self.stream=None;self.identity=None;self.last_frame=None

    def record(self,identity,frame_id,video_frame,fps,coordinates,source,dimensions):
        if not identity.states:return
        if self.identity is not identity or self.stream is None:
            self.close();self.root.mkdir(parents=True,exist_ok=True)
            self.path=self.root/(datetime.now().astimezone().strftime('%Y%m%d_%H%M%S_%f')+'.jsonl')
            self.stream=self.path.open('w',encoding='utf-8');self.identity=identity
            self.write({'type':'session','source':source,'fps':fps,'dimensions':dimensions,
                        'units':{'image':'px','table':'mm'},'center_kind':'detected_box_center_projected_to_cloth',
                        'identity_policy':'motion_and_appearance; ambiguous observations are rejected'})
        if self.last_frame==frame_id:return
        self.last_frame=frame_id;tracks=[]
        for key,state in identity.states.items():
            point=state['center'].tolist() if state['status']=='tracking' else None
            mm=coordinates.millimetres(*point) if point and coordinates else None
            tracks.append({'track_id':key,'ball_id':state['label_id'],'name':state['name'],
                           'status':state['status'],'center_px':point,'center_mm':list(mm) if mm else None})
        self.write({'type':'positions','frame_id':frame_id,'video_frame':video_frame,
                    'video_seconds':video_frame/fps if fps else None,'tracks':tracks})
        if time.monotonic()-self.last_flush>1:
            self.stream.flush();self.last_flush=time.monotonic()

    def write(self,value):self.stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
