"""Entry polygons and outward directions in a shared table plane."""
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

    def distance(self,point):return float(np.linalg.norm(np.asarray(point)-self.center))

    def near(self,point,radius):return self.distance(point)<=self.approach_radius+radius

    def touches_entry(self,point,radius):
        contour=np.asarray(self.entry_polygon,np.float32)
        return len(contour)>=3 and cv2.pointPolygonTest(contour,tuple(map(float,point)),True)>=-radius

    def toward(self,start,end):
        velocity=np.asarray(end)-start
        speed=np.linalg.norm(velocity)
        if speed<.002:return False
        direction=np.asarray(self.entry_direction)
        to_entry=np.asarray(self.center)-start
        return (np.dot(velocity,direction)/speed>.25 and
                np.dot(velocity,to_entry)>0 and self.distance(end)<self.distance(start)-.002)

    def segment_enters(self,start,end,radius):
        # Sample at a ball-radius scale, including the whole observed segment.
        count=min(128,max(2,int(np.linalg.norm(np.asarray(end)-start)/max(.002,radius*.5))+1))
        return any(self.touches_entry(p,radius) for p in np.linspace(start,end,count))


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
