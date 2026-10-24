"""Physical coordinates on the inner-cushion plane (millimetres)."""
# 将可靠内库四角映射到毫米平面；单位是 mm，不能把包含库边的范围框当成比赛尺寸。
# 默认尺寸是参考值；球心是检测框中心在台面平面的投影，未补偿镜头畸变和球心高度视差。

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class TableDimensions:
    x_mm: float = 3569.0
    y_mm: float = 1778.0
    reference: bool = True

    # 拒绝非有限或明显越界的毫米尺寸，避免构造退化透视映射。
    def __post_init__(self):
        if not all(np.isfinite(v) and 100<=v<=20000 for v in (self.x_mm,self.y_mm)):
            raise ValueError('X/Y dimensions must be finite values between 100 and 20000 mm')


class TableCoordinates:
    # 验证凸四边形并按屏幕左上角整理邻边顺序，建立 image_to_mm 与逆变换。
    # 映射条件数不可靠时抛错；不能仅因凑齐四个点就接受一个奇异平面。
    def __init__(self, polygon, dimensions=None):
        self.dimensions=dimensions or TableDimensions()
        q=np.asarray(polygon,np.float32)
        if q.shape!=(4,2) or not np.isfinite(q).all() or not cv2.isContourConvex(q) or abs(cv2.contourArea(q))<100:
            raise ValueError('Four reliable inner-cushion corners are required')
        # Start at the screen's top-left corner, preserving adjacent edges.
        q=np.roll(q,-int(np.argmin(q.sum(axis=1))),axis=0)
        if q[1,0]<q[3,0]:q=q[[0,3,2,1]]
        self.corners=q
        d=self.dimensions
        target=np.float32([[0,0],[d.x_mm,0],[d.x_mm,d.y_mm],[0,d.y_mm]])
        self.image_to_mm=cv2.getPerspectiveTransform(q,target)
        condition,self.mm_to_image=cv2.invert(self.image_to_mm,flags=cv2.DECOMP_SVD)
        if condition<1e-10 or not np.isfinite(self.image_to_mm).all():
            raise ValueError('Unstable table perspective')

    # 点必须有限且在内库多边形内；台外返回 None，不把桌外位置夹成边界坐标。
    def millimetres(self,x,y):
        if not np.isfinite([x,y]).all() or cv2.pointPolygonTest(self.corners,(float(x),float(y)),False)<0:
            return None
        point=cv2.perspectiveTransform(np.float32([[[x,y]]]),self.image_to_mm)[0,0]
        if not np.isfinite(point).all():return None
        d=self.dimensions
        return (float(np.clip(point[0],0,d.x_mm)),float(np.clip(point[1],0,d.y_mm)))

    # 将毫米平面上的点批量映射回原图像素，供轴线/标注绘制使用。
    def pixels(self,points):
        return cv2.perspectiveTransform(np.float32([points]),self.mm_to_image)[0].tolist()

    # 取检测框中心再换算毫米值，不把用户点击位置当球心。
    def ball_center(self,box):
        if box is None:return None
        x1,y1,x2,y2=box
        return self.millimetres((x1+x2)/2,(y1+y2)/2)
