"""Recover the whole connected cloth colour, including matching cushion bands."""
import cv2
import numpy as np
from src.table_coordinates import TableCoordinates
from src.vision.pocket.table_detector import _line,_intersection


def cloth_region(frame,seed_polygon):
    scale=min(1.,1200/max(frame.shape[:2]))
    image=cv2.resize(frame,None,fx=scale,fy=scale) if scale<1 else frame
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);h,w=hsv.shape[:2]
    q=TableCoordinates(seed_polygon).corners.astype(float)*scale
    seed=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(seed,np.rint(q).astype(np.int32),255)
    short=min(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1))
    radius=max(2,int(short*.08))
    center=cv2.erode(seed,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,)*2))>0
    colorful=center&(hsv[:,:,1]>45)&(hsv[:,:,2]>20)
    if np.count_nonzero(colorful)<100:return None
    hist=np.bincount(hsv[:,:,0][colorful],minlength=180)
    hue=int(np.argmax(np.convolve(np.r_[hist[-5:],hist,hist[:5]],np.ones(11),'valid')))
    distance=np.abs(hsv[:,:,0].astype(float)-hue);distance=np.minimum(distance,180-distance)
    saturation=float(np.median(hsv[:,:,1][colorful]));value=float(np.median(hsv[:,:,2][colorful]))
    color=((distance<=12)&(hsv[:,:,1]>=max(35,saturation*.4))&
           (hsv[:,:,2]>=max(15,value*.18))).astype(np.uint8)*255
    kernel=max(3,min(9,int(short*.012)//2*2+1))
    joined=cv2.morphologyEx(color,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel)))
    count,labels,stats,_=cv2.connectedComponentsWithStats(joined,8)
    if count<2:return None
    overlaps=np.bincount(labels[center].ravel(),minlength=count);overlaps[0]=0
    chosen=int(np.argmax(overlaps))
    if overlaps[chosen]<.45*np.count_nonzero(center):return None
    component=(labels==chosen).astype(np.uint8)*255
    contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour=max(contours,key=cv2.contourArea)
    hull=cv2.convexHull(contour);perimeter=cv2.arcLength(hull,True)
    points=contour.reshape(-1,2).astype(float)
    lines=[]
    for i in range(4):
        a,b=q[i],q[(i+1)%4];vector=b-a;length=np.linalg.norm(vector)
        unit=vector/length;normal=np.array([-unit[1],unit[0]])
        along=(points-a)@unit/length;depth=(points-a)@normal
        middle=(along>.12)&(along<.88)&(depth>-short*.25)&(depth<short*.2)
        if np.count_nonzero(middle)<12:break
        # The densest straight boundary survives pockets, arms and isolated
        # same-colour protrusions; an extreme percentile follows those outliers.
        bins=np.rint(depth[middle]).astype(int)
        low=int(bins.min());hist=np.bincount(bins-low)
        smoothed=np.convolve(hist,np.ones(5),'same')
        outer=float(np.argmax(smoothed)+low)
        selected=points[middle&(np.abs(depth-outer)<max(2.5,short*.012))]
        if len(selected)<12:break
        vx,vy,cx,cy=cv2.fitLine(selected.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel()
        lines.append(np.array([vy,-vx,vx*cy-vy*cx],float))
    candidates=[]
    if len(lines)==4:
        fitted=[_intersection(lines[i-1],lines[i]) for i in range(4)]
        if all(point is not None for point in fitted):candidates.append(np.asarray(fitted))
    for fraction in (.003,.005,.008,.012,.016,.02,.025,.03,.04):
        approx=cv2.approxPolyDP(hull,fraction*perimeter,True)
        if len(approx)==4 and cv2.isContourConvex(approx):candidates.append(approx.reshape(4,2))
    valid=[]
    for candidate in candidates:
        try:points=TableCoordinates(candidate).corners.astype(float)
        except (ValueError,cv2.error):continue
        mask=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(mask,np.rint(points).astype(np.int32),255)
        fill=float(np.mean(color[mask>0]>0))
        coverage=float(np.count_nonzero((component>0)&(mask>0))/max(1,cv2.countNonZero(component)))
        if fill<.8 or coverage<.9:continue
        points/=scale
        points[:,0]=np.clip(points[:,0],0,frame.shape[1]-1)
        points[:,1]=np.clip(points[:,1],0,frame.shape[0]-1)
        valid.append({'polygon':points.tolist(),'hue':hue,'color_fill':fill,'connected_color_coverage':coverage})
    if valid:
        # When cloth continues all the way to an image boundary there is no
        # visible outer rail to fit. Retain full coverage of that cropped table.
        touches=int(np.any(component[:,0]))+int(np.any(component[:,-1]))+int(np.any(component[0,:]))+int(np.any(component[-1,:]))
        if touches>=3:return max(valid,key=lambda item:item['connected_color_coverage']+.15*item['color_fill'])
        return max(valid,key=lambda item:min(.97,item['connected_color_coverage'])+.3*item['color_fill'])
    return None


def scope_improvement(frame,old,new,hue):
    """Require measurable evidence before changing an explicitly saved scope."""
    scale=min(1.,1200/max(frame.shape[:2]))
    image=cv2.resize(frame,None,fx=scale,fy=scale) if scale<1 else frame
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    delta=np.abs(hsv[:,:,0].astype(float)-hue);delta=np.minimum(delta,180-delta)
    color=(delta<=12)&(hsv[:,:,1]>=45)&(hsv[:,:,2]>=30)
    def measure(polygon):
        mask=np.zeros(color.shape,np.uint8)
        cv2.fillConvexPoly(mask,np.rint(np.asarray(polygon)*scale).astype(np.int32),1)
        area=int(mask.sum());cloth=int(np.count_nonzero(color&(mask>0)))
        return cloth/max(1,area),cloth
    old_fill,old_cloth=measure(old);new_fill,new_cloth=measure(new)
    return bool((new_fill>=old_fill+.012 and new_cloth>=old_cloth*.95) or
                (new_cloth>=old_cloth*1.025 and new_fill>=old_fill-.005))
