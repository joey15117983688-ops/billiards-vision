"""Local, colour-independent ball observations; reject elongated cue components."""
# 以已有目标框为搜索提示，在当前 BGR 图中寻找局部球体；这里只返回像素观测，不运行 YOLO。
# HSV 主色、轮廓和外观阈值是启发式，彩灯鲁棒性尚待后续阶段改进。

import cv2
import numpy as np

BALL_NAMES = frozenset(('billiard_ball', 'white_ball', 'black_ball'))


# 以目标球径限定裁剪和轮廓尺寸，排除台呢、细长杆体及接触裁剪边界的组件。
# 运动模糊标记只放宽形状筛选；返回 _ball_observed 表示中心来自当前轮廓矩，而非预测中心。
def local_balls(frame, target, radius_factor=2.5):
    x1,y1,x2,y2=target['box'];w,h=x2-x1,y2-y1
    if min(w,h)<=2 or max(w,h)/min(w,h)>2.2:return []
    cx,cy=(x1+x2)/2,(y1+y2)/2;size=max(w,h)
    radius=max(16.,radius_factor*size)
    ax,ay=max(0,int(cx-radius)),max(0,int(cy-radius))
    bx,by=min(frame.shape[1],int(cx+radius+1)),min(frame.shape[0],int(cy+radius+1))
    if bx<=ax or by<=ay:return []
    hsv=cv2.cvtColor(frame[ay:by,ax:bx],cv2.COLOR_BGR2HSV)
    # Cloth dominates the neighbourhood, including on blue and green tables.
    saturated=(hsv[:,:,1]>65)&(hsv[:,:,2]>45)
    if not saturated.any():return []
    hues=np.bincount(hsv[:,:,0][saturated],minlength=180)
    hue=int(hues.argmax());delta=np.abs(hsv[:,:,0].astype(float)-hue);delta=np.minimum(delta,180-delta)
    cloth=(delta<10)&saturated
    background=float(np.median(hsv[:,:,2][cloth])) if cloth.any() else 140.
    selected=(delta>12)|(hsv[:,:,1]<65)|(hsv[:,:,2]<background-38)
    binary=cv2.morphologyEx(selected.astype(np.uint8)*255,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    found=[]
    blurred=target.get('_allow_motion_blur',False)
    for contour in contours:
        x,y,cw,ch=cv2.boundingRect(contour);area=cv2.contourArea(contour)
        if min(cw,ch)<.45*min(w,h) or max(cw,ch)>(2.8 if blurred else 1.9)*size:continue
        if max(cw,ch)/min(cw,ch)>(2.8 if blurred else 1.65) or area/max(cw*ch,1)<.52:continue
        perimeter=cv2.arcLength(contour,True)
        if 4*np.pi*area/max(perimeter**2,1)<(.35 if blurred else .53):continue
        if x==0 or y==0 or x+cw>=bx-ax or y+ch>=by-ay:continue
        moments=cv2.moments(contour)
        if not moments['m00']:continue
        px,py=ax+moments['m10']/moments['m00'],ay+moments['m01']/moments['m00']
        # Use the measured centre, keeping a stable box size for association.
        found.append({**target,'box':[px-w/2,py-h/2,px+w/2,py+h/2],
                      'confidence':min(float(target.get('confidence',.8)),.9),'_ball_observed':True})
    return found


# 结合像素候选到提示中心的距离与外观差异排序；歧义拒绝标记开启时不强行选一个。
def closest_ball(frame,target,radius_factor=2.5):
    candidates=local_balls(frame,target,radius_factor)
    center=lambda d:np.mean(np.array(d['box']).reshape(2,2),axis=0)
    if not candidates:return None
    origin=center(target)
    from .ball_tracks import appearance,BallTracks
    reference=target.get('_appearance')
    if reference is not None:reference=np.asarray(reference,np.float32)
    ranked=[]
    for candidate in candidates:
        colour=BallTracks.distance(reference,appearance(frame,candidate))
        if colour>.65:continue
        ranked.append((np.linalg.norm(center(candidate)-origin)/max(8,target['box'][2]-target['box'][0])+2*colour,candidate))
    ranked.sort(key=lambda pair:pair[0])
    if target.get('_reject_ambiguous') and len(ranked)>1 and ranked[1][0]-ranked[0][0]<.15:return None
    return ranked[0][1] if ranked else None
