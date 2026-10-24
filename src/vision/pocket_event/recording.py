"""Background JSONL and key-frame capture; bounded queue keeps the UI responsive."""
# 单后台线程保存事件 JSONL 与真实关键帧；待写任务最多 8 个，避免图片队列无限占内存。
# 队列有界不代表磁盘容量受限；文件中可含私人来源与图像，不随源码提交到公开仓库。

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime
import json
import cv2


class EventRecording:
    # 建立会话目录名和单线程执行器，实际目录在首次写入时创建。
    def __init__(self,root,source):
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.path=Path(root)/stamp
        self.source=str(source);self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pocket-events')
        self.pending=[];self.error=None;self.sequence=0

    # 移除已完成任务，满队列时记录错误；提交前深拷贝数据和帧，防止 UI 后续改动它们。
    def write(self,snapshot,results,frame=None):
        self.pending=[f for f in self.pending if not f.done()]
        if len(self.pending)>=8:
            self.error='进球记录队列已满，部分关键帧未保存';return
        self.sequence+=1
        payload={'source':self.source,'snapshot':deepcopy(snapshot),'results':deepcopy(results),'sequence':self.sequence}
        self.pending.append(self.executor.submit(self._save,payload,None if frame is None else frame.copy()))

    # 先保存关键帧再追加对应 JSONL，捕获 IO 错误供界面提示而非让后台线程静默失败。
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

    # 等待已提交的记录任务结束；当前实现没有磁盘阻塞超时，不能承诺强制及时退出。
    def close(self):self.executor.shutdown(wait=True)
