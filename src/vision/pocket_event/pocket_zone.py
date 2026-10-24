"""Entry polygons and outward directions in a shared table plane."""
# 事件平面内的袋口入口几何；距离、半径和入口方向均已按物理短边归一化。
# 入口形状可信与否由观测来源决定，几何估算的四边形不能自动升级为真实袋口。

from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class PocketZone:
    id: str
    center: tuple
    entry_polygon: list
    entry_direction: tuple
    status: str='estimated'
    approach_radius: float=.16
    trusted_shape: bool=False

    # 求点到袋心的欧氏距离，单位为事件平面短边比例。
    def distance(self,point):return float(np.linalg.norm(np.asarray(point)-self.center))

    # 以接近半径加球半径判断是否进入候选邻域，不等于已经穿过袋口。
    def near(self,point,radius):return self.distance(point)<=self.approach_radius+radius

    # 用带符号多边形距离检查球体是否与入口区域相交。
    def touches_entry(self,point,radius):
        contour=np.asarray(self.entry_polygon,np.float32)
        return len(contour)>=3 and cv2.pointPolygonTest(contour,tuple(map(float,point)),True)>=-radius

    # 要求运动与入袋方向一致且确实靠近袋心，静止/远离不积累入袋方向证据。
    def toward(self,start,end):
        velocity=np.asarray(end)-start
        speed=np.linalg.norm(velocity)
        if speed<.002:return False
        direction=np.asarray(self.entry_direction)
        to_entry=np.asarray(self.center)-start
        return (np.dot(velocity,direction)/speed>.25 and
                np.dot(velocity,to_entry)>0 and self.distance(end)<self.distance(start)-.002)

    # 沿两次实测中心连线按球径尺度采样，可覆盖快速运动跳过入口的情况。
    # 连线仅用于验证两端真实观测之间的入口交集，不在无观测区间制造新球心。
    def segment_enters(self,start,end,radius):
        # Sample at a ball-radius scale, including the whole observed segment.
        count=min(128,max(2,int(np.linalg.norm(np.asarray(end)-start)/max(.002,radius*.5))+1))
        return any(self.touches_entry(p,radius) for p in np.linspace(start,end,count))


# 按观测状态生成可用袋口；缺入口时补几何候选供接近判断，但不把它标为可信袋形。
def make_zones(pockets,table_center,config):
    zones=[]
    for p in pockets:
        center=p.get('center') or p.get('candidate_center')
        if center is None or p.get('status')=='missing':continue
        center=tuple(center)
        direction=np.asarray(center)-table_center
        length=np.linalg.norm(direction)
        if length<1e-8:continue
        polygon=p.get('entry_polygon') or []
        measured=len(polygon)>=3 and p.get('status') in ('observed','manual')
        if len(polygon)<3:
            r=config.fallback_entry_radius;x,y=center
            polygon=[[x-r,y-r],[x+r,y-r],[x+r,y+r],[x-r,y+r]]
        zones.append(PocketZone(p['id'],center,polygon,tuple(direction/length),p.get('status','estimated'),config.approach_radius,measured))
    return zones
