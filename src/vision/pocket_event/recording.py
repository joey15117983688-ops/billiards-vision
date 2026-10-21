"""Background JSONL and key-frame capture; bounded queue keeps the UI responsive."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
import json
import cv2


class EventRecording:
    def __init__(self,root,source):
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.path=Path(root)/stamp
        self.source=str(source);self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pocket-events')
        self.pending=[];self.error=None;self.sequence=0

    def write(self,snapshot,results,frame=None):
        self.pending=[f for f in self.pending if not f.done()]
        if len(self.pending)>=8:
            self.error='进球记录队列已满，部分关键帧未保存';return
        self.sequence+=1
        payload={'source':self.source,'snapshot':deepcopy(snapshot),'results':deepcopy(results),'sequence':self.sequence}
        self.pending.append(self.executor.submit(self._save,payload,None if frame is None else frame.copy()))

    def _save(self,payload,frame):
        try:
            self.path.mkdir(parents=True,exist_ok=True)
            image=f"{payload['sequence']:05d}_{payload['snapshot']['timestamp'] or 0:.3f}.jpg"
            if frame is not None:
                if not cv2.imwrite(str(self.path/image),frame):raise OSError('关键帧写入失败')
                payload['key_frame']=image
            with (self.path/'events.jsonl').open('a',encoding='utf-8') as output:
                output.write(json.dumps(payload,ensure_ascii=False,allow_nan=False)+'\n')
        except Exception as exc:self.error=str(exc)

    def close(self):self.executor.shutdown(wait=True)
