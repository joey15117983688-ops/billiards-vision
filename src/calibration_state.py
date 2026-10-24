"""Persist accepted automatic corrections separately from human ground truth."""
# 自动校准记录、人工配置摘要及显式纠错反馈的持久化。
# 保存使用同目录临时文件替换；归一化四角支持同纵横比分辨率恢复，人工摘要隔离不同基准。

import hashlib
from copy import deepcopy
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import cv2
from src.table_coordinates import TableCoordinates


# 对排序后的人工配置 JSON 求摘要，防止自动记录被错误应用到另一个人工基准。
def profile_digest(profile):
    return hashlib.sha256(json.dumps(profile,sort_keys=True).encode()).hexdigest()


# 先写同目录临时文件再 replace，减少中途退出留下半个 JSON 的风险；不是磁盘 fsync 保证。
def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    temporary.replace(path)


# 仅显式纠错时记录本次否定边界和尝试次数，最多保留 12 个拒绝范围供搜索剔除。
def begin_feedback(path,frame,previous,manual_profile):
    path=Path(path);scale=np.array([frame.shape[1],frame.shape[0]])
    digest=profile_digest(manual_profile)
    try:
        history=json.loads(path.read_text()) if path.exists() else {}
        if not isinstance(history,dict) or history.get('manual_digest')!=digest:history={}
    except (OSError,ValueError):history={}
    rejected=[]
    for entry in history.get('rejected',[]) if isinstance(history.get('rejected',[]),list) else []:
        try:
            q=np.asarray(entry,float)
            if q.shape==(4,2) and np.isfinite(q).all() and np.all(q>=0) and np.all(q<=1):rejected.append(q.tolist())
        except (TypeError,ValueError):continue
    try:attempt=max(0,int(history.get('attempt',0)))+1
    except (TypeError,ValueError):attempt=1
    if previous is not None:
        normalized=(TableCoordinates(previous).corners/scale).tolist()
        if normalized not in rejected:rejected.append(normalized)
    history.update({'version':1,'manual_digest':digest,'attempt':attempt,
                    'rejected':rejected[-12:],'frame_shape':list(frame.shape[:2]),
                    'frame_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),
                    'requested_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
    write_json(path,history)
    return {'attempt':history['attempt'],'previous_polygon':previous,
            'rejected_polygons':[(np.asarray(q)*scale).tolist() for q in history['rejected']],
            'manual':manual_profile is not None}


# 保存内库归一化坐标、袋口和固定参考图；内容相同时不重复刷新保存时间。
# 真正更新前保留 previous 备份；保存的是接受的几何，不是人工标注真值。
def save_automatic(path,frame,result,manual_profile):
    h,w=frame.shape[:2]
    pockets=deepcopy(result['pockets'])
    if result.get('scope_polygon') is not None:pockets['table']['scope_polygon']=np.asarray(result['scope_polygon']).tolist()
    path=Path(path)
    if path.exists():
        previous=json.loads(path.read_text())
        normalized=(np.asarray(result['polygon'])/[w,h]).tolist()
        same=(previous.get('scope_frozen') is True and previous.get('shape')==[h,w] and
              previous.get('polygon_normalized')==normalized and
              previous.get('pockets')==pockets and
              previous.get('manual_digest')==profile_digest(manual_profile))
        if same:return
        write_json(path.with_suffix('.previous.json'),previous)
    from src.camera_alignment import encode_reference
    write_json(path,{'alignment_reference':encode_reference(frame),'saved_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                     'version':1,'shape':[h,w],'manual_digest':profile_digest(manual_profile),
                     'polygon_normalized':(np.asarray(result['polygon'])/[w,h]).tolist(),
                     'scope_frozen':True,
                     'recognition_scope_semantics':'whole_connected_cloth_including_matching_cushions',
                     'pockets':pockets,'report':result['calibration_report']})


# 验证版本、人工摘要、纵横比并按当前分辨率恢复四角/六袋及变换矩阵。
# verify_scene 可做离线图像复核；显式固定机位恢复不应因开局有人遮挡就强行重检测。
def _load_automatic(path,frame,manual_profile,verify_scene=True):
    path=Path(path)
    if not path.exists():return None
    data=json.loads(path.read_text())
    h,w=frame.shape[:2];old_h,old_w=data['shape']
    if old_h<=0 or old_w<=0:return None
    if data.get('version')!=1 or data.get('manual_digest')!=profile_digest(manual_profile):return None
    if abs(w/h-old_w/old_h)>.01:return None
    polygon=TableCoordinates(np.asarray(data['polygon_normalized'])*[w,h]).corners.tolist()
    # Restoring an explicitly saved camera setup must not fit a new table to
    # an occluded opening frame. Scene verification remains available offline.
    evidence={}
    if verify_scene:
        from src.adaptive_calibration import ImageEvidence
        evidence=ImageEvidence(frame).measure(polygon)
        if evidence['cloth_fill']<.7 or evidence['black_band']>.22 or evidence['edge_mean']<.3:return None
    pockets=data['pockets'];sx,sy=w/old_w,h/old_h
    table=pockets['table']
    had_scope=bool(table.get('scope_polygon'))
    raw_scope=table.get('scope_polygon') or table.get('playable_polygon')
    scope=TableCoordinates(np.asarray(raw_scope)*[sx,sy]).corners.tolist() if raw_scope is not None else polygon
    table.update({'corners':polygon,'playable_polygon':polygon,'scope_polygon':scope,'boundary_lines':[]})
    if not data.get('scope_frozen') and not had_scope:
        from src.cloth_region import cloth_region
        region=cloth_region(frame,polygon)
        if region:table['scope_polygon']=region['polygon']
    table['quality'].update(evidence)
    geometry=pockets.get('geometry',{})
    if geometry:
        geometry['image_to_normalized']=(np.asarray(geometry['image_to_normalized'])@np.diag([1/sx,1/sy,1])).tolist()
        geometry['normalized_to_image']=(np.diag([sx,sy,1])@np.asarray(geometry['normalized_to_image'])).tolist()
        geometry['ordered_corners']=(np.asarray(geometry['ordered_corners'])*[sx,sy]).tolist()
    for pocket in pockets['pockets']:
        for key in ['center','candidate_center']:
            if pocket.get(key):pocket[key]={'x':pocket[key]['x']*sx,'y':pocket[key]['y']*sy}
        pocket['entry_polygon']=[[x*sx,y*sy] for x,y in pocket.get('entry_polygon',[])]
    pockets['image_width']=w;pockets['image_height']=h
    return polygon,pockets


# 把损坏、缺失或不兼容记录转成 None，由调用方提示/回退，不自动覆盖文件。
def load_automatic(path,frame,manual_profile,verify_scene=True):
    try:return _load_automatic(path,frame,manual_profile,verify_scene)
    except (OSError,ValueError,KeyError,TypeError,IndexError,cv2.error):return None
