"""One-shot calibration and six local pocket validations; no tracking or scoring."""
# 一次性台面与六个局部袋口验证，不做跟踪或进球计数。
# 可以复用调用方同帧拟合的 table；observed/estimated/missing 的来源语义必须保留。

import time
import cv2
import numpy as np
from .models import PocketConfig, PocketResult, PocketDetectionResult
from .table_detector import TableDetector
from .table_geometry import CANDIDATES, TableGeometry
from .pocket_validator import PocketValidator


class PocketDetector:
    # 台面检测器和局部验证器共享同一配置，避免候选方向/尺度不一致。
    def __init__(self, config=None):
        self.config=config or PocketConfig()
        self.table_detector=TableDetector(self.config)
        self.validator=PocketValidator(self.config)

    # 可靠内库确定后展开台面，对标准六袋位置逐一验证；长轴不明时用中袋证据选择方向。
    # 袋口几何冲突会降为 estimated；只有六袋均 observed 才 valid，不以“数量为六”代表完整实测。
    def detect(self, frame, table=None):
        """Validate pockets, optionally reusing a freshly fitted table from this frame."""
        start=time.perf_counter()
        table=self.table_detector.detect(frame) if table is None else table
        height,width=frame.shape[:2] if isinstance(frame,np.ndarray) and frame.ndim>=2 else (0,0)
        result=PocketDetectionResult(width,height,table,[PocketResult(name,reasons=['table_geometry_unavailable']) for name in CANDIDATES])
        result.table_mask=np.zeros((height,width),np.uint8)
        result.pocket_mask=np.zeros((height,width),np.uint8)
        if table.detected:
            cv2.fillConvexPoly(result.table_mask,np.rint(table.playable_polygon).astype(np.int32),255)
        if not table.detected or not table.quality.get("geometry_reliable",False):
            result.reasons=table.reasons or ['table_not_detected']
            result.elapsed_ms=(time.perf_counter()-start)*1000
            return result
        try:
            options=[]
            for alternate in ([False,True] if self.config.long_axis=='auto' else [False]):
                geometry=TableGeometry(table,self.config,alternate=alternate)
                warped,valid=geometry.warp(frame)
                pockets=[self.validator.validate(name,point,geometry,warped,valid) for name,point in CANDIDATES.items()]
                middle=[p for p in pockets if 'MIDDLE' in p.id]
                evidence=sum(p.status=='observed' for p in middle)
                strength=sum(p.confidence for p in middle if p.status=='observed')
                options.append((evidence,strength,geometry,warped,pockets))
            chosen=options[0]
            if len(options)>1 and (options[1][0]>chosen[0] or
                                  (options[1][0]==chosen[0] and options[1][1]>chosen[1]+.2)):
                chosen=options[1];chosen[2].orientation_method='side_pocket_visual_validation'
            _,_,geometry,warped,pockets=chosen
            result.geometry=geometry.to_dict();result.warped=warped;result.pockets=pockets
            observed=[p for p in pockets if p.status=='observed']
            for i,p in enumerate(observed):
                for other in observed[i+1:]:
                    if np.linalg.norm(np.array(p.normalized_center)-other.normalized_center)<.1:
                        p.status=other.status='estimated'
                        p.reasons.append('pocket_geometry_conflict');other.reasons.append('pocket_geometry_conflict')
            for p in pockets:
                if p.entry_polygon:
                    cv2.fillPoly(result.pocket_mask,[np.rint(p.entry_polygon).astype(np.int32)],255)
                if p.status!='observed':result.reasons.append(f'{p.id}:{p.status}')
            result.reasons=table.reasons+result.reasons
            result.valid=all(p.status=='observed' for p in pockets)
        except (ValueError,cv2.error,np.linalg.LinAlgError) as exc:
            result.reasons=['geometry_failed:'+str(exc)]
        result.elapsed_ms=(time.perf_counter()-start)*1000
        return result
