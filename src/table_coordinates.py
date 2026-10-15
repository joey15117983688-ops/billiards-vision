"""Physical coordinates on the inner-cushion plane (millimetres)."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class TableDimensions:
    x_mm: float = 3569.0
    y_mm: float = 1778.0
    reference: bool = True

    def __post_init__(self):
        if not all(np.isfinite(v) and 100<=v<=20000 for v in (self.x_mm,self.y_mm)):
            raise ValueError('X/Y dimensions must be finite values between 100 and 20000 mm')


class TableCoordinates:
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

    def millimetres(self,x,y):
        if not np.isfinite([x,y]).all() or cv2.pointPolygonTest(self.corners,(float(x),float(y)),False)<0:
            return None
        point=cv2.perspectiveTransform(np.float32([[[x,y]]]),self.image_to_mm)[0,0]
        if not np.isfinite(point).all():return None
        d=self.dimensions
        return (float(np.clip(point[0],0,d.x_mm)),float(np.clip(point[1],0,d.y_mm)))

    def pixels(self,points):
        return cv2.perspectiveTransform(np.float32([points]),self.mm_to_image)[0].tolist()

    def ball_center(self,box):
        if box is None:return None
        x1,y1,x2,y2=box
        return self.millimetres((x1+x2)/2,(y1+y2)/2)
