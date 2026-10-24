"""Save calibration data and independent debug images; does not alter a detector/model."""
# 本地调试图与 JSON 导出；只表现检测结果，不参与候选接受或进球判定。
# 图片含现场画面，运行产物保存在被忽略的 data/debug 目录，不默认公开上传。

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import cv2
import numpy as np

COLORS={'observed':(40,210,40),'estimated':(0,210,255),'missing':(40,40,240)}


# 用 OpenCV Hershey 字体绘制调试标签（非 Qt 中文排版），不把文字转换为识别输入。
def _label(image,text,point,color):
    x,y=map(int,point)
    (width,height),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.45,1)
    x=min(max(0,x+7),max(0,image.shape[1]-width-6))
    y=min(max(height+4,y-8),image.shape[0]-4)
    cv2.rectangle(image,(x,y-height-3),(x+width+4,y+3),(15,15,15),-1)
    cv2.putText(image,text,(x+2,y),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)


# 在原图副本上按状态绘制台面、袋口和入口，保留源图不被涂改。
def visualize(frame,result,mode='final'):
    image=frame.copy()
    if result.table.corners:
        points=np.rint(result.table.corners).astype(np.int32)
        cv2.polylines(image,[points],True,(255,200,0),2,cv2.LINE_AA)
        for i,edge in enumerate(result.table.boundary_lines):
            point=(np.array(edge['start'])+edge['end'])/2
            _label(image,'EDGE '+str(i),point,(255,200,0))
    if mode!='boundaries':
        for pocket in result.pockets:
            point=pocket.candidate_center if mode=='candidates' else pocket.center
            if point is None:point=pocket.candidate_center
            if point is None:continue
            center=tuple(np.rint(point).astype(int))
            color=(235,235,235) if mode=='candidates' else COLORS[pocket.status]
            if mode=='candidates':
                cv2.drawMarker(image,center,color,cv2.MARKER_TILTED_CROSS,12,1)
            else:
                if pocket.entry_polygon:
                    cv2.polylines(image,[np.rint(pocket.entry_polygon).astype(np.int32)],True,color,1,cv2.LINE_AA)
                cv2.circle(image,center,7,color,2,cv2.LINE_AA)
                if pocket.status=='missing':cv2.drawMarker(image,center,color,cv2.MARKER_TILTED_CROSS,14,2)
            _label(image,pocket.id+(' candidate' if mode=='candidates' else ' '+pocket.status),center,color)
    counts={state:sum(p.status==state for p in result.pockets) for state in COLORS}
    title=f"complete={result.valid} observed={counts['observed']} estimated={counts['estimated']} missing={counts['missing']}"
    cv2.rectangle(image,(0,0),(min(image.shape[1],700),25),(15,15,15),-1)
    cv2.putText(image,title,(7,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1,cv2.LINE_AA)
    return image


# 保存原图、掩模、展开图、叠加图和结构化 JSON，便于人工追踪失败原因。
def save_debug_result(frame,result,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    outputs={'original':frame,'table_mask':result.table_mask,'pocket_regions':result.pocket_mask,
             'inner_cushions':visualize(frame,result,'boundaries'),
             'pocket_candidates':visualize(frame,result,'candidates'),
             'pockets':visualize(frame,result)}
    if result.warped is not None:outputs['warped']=result.warped
    for name,image in outputs.items():
        path=folder/(name+'.png')
        if not cv2.imwrite(str(path),image):raise OSError(f'Cannot save debug image: {path}')
        result.artifacts[name]=str(path.resolve())
    path=folder/'result.json'
    result.artifacts['json']=str(path.resolve())
    result.source['frame_sha256']=hashlib.sha256(np.ascontiguousarray(frame)).hexdigest()
    result.source['saved_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result.to_dict(),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)
    return path
