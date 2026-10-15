"""Bounded optical-flow bridge between asynchronous YOLO results and displayed frames."""
from collections import deque
from copy import deepcopy
import cv2
import numpy as np
from .ball_tracks import BallTracks


class LiveTracking:
    def __init__(self, history_size=16):
        self.frames=deque(maxlen=history_size)
        self.detections=[]
        self.next_id=1
        self.identities=BallTracks()
        self.current_color=None

    def clear(self,keep_tracks=False):
        self.frames.clear();self.detections=[];self.current_color=None
        if not keep_tracks:self.identities=BallTracks()

    @staticmethod
    def gray(frame):
        scale=min(1.,640/max(frame.shape[:2]))
        small=cv2.resize(frame,None,fx=scale,fy=scale) if scale<1 else frame
        return cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),scale,frame.shape[:2],small

    def push(self,frame_id,frame):
        gray,scale,shape,self.current_color=self.gray(frame)
        if self.frames:
            _,previous,old_scale,old_shape=self.frames[-1]
            if old_shape==shape:
                self.detections=self.flow(previous,gray,self.detections,scale,shape)
            else:self.detections=[];self.frames.clear()
        self.detections=self.identities.update(frame_id,self.current_color,self.detections,scale)
        self.frames.append((frame_id,gray,scale,shape))
        return deepcopy(self.detections)

    @staticmethod
    def flow(previous,current,items,scale,shape):
        output=deepcopy(items);points=[];owners=[]
        for i,item in enumerate(output):
            if item['name']=='billiards_table':continue
            x1,y1,x2,y2=np.array(item['box'])*scale
            ax,ay=max(0,int(x1)),max(0,int(y1));bx,by=min(current.shape[1],int(x2)+1),min(current.shape[0],int(y2)+1)
            if bx<=ax or by<=ay:continue
            features=cv2.goodFeaturesToTrack(previous[ay:by,ax:bx],maxCorners=6,qualityLevel=.02,minDistance=2)
            if features is None:continue
            for point in features[:,0]+[ax,ay]:points.append(point);owners.append(i)
        valid_offsets={}
        if points:
            origin=np.float32(points).reshape(-1,1,2)
            destination,status,error=cv2.calcOpticalFlowPyrLK(previous,current,origin,None,winSize=(21,21),maxLevel=3)
            if destination is not None:
                for owner,a,b,ok,e in zip(owners,origin[:,0],destination[:,0],status[:,0],error[:,0]):
                    offset=(b-a)/scale
                    if ok and np.isfinite(offset).all() and e<35 and np.linalg.norm(offset)<.15*min(shape):
                        valid_offsets.setdefault(owner,[]).append(offset)
        keep=[]
        for i,item in enumerate(output):
            if item['name']=='billiards_table':keep.append(item);continue
            if i in valid_offsets:
                dx,dy=np.median(valid_offsets[i],axis=0)
                x1,y1,x2,y2=item['box']
                item['box']=[float(x1+dx),float(y1+dy),float(x2+dx),float(y2+dy)]
                item['_flow_misses']=0
            else:item['_flow_misses']=item.get('_flow_misses',0)+1
            x1,y1,x2,y2=item['box']
            if item['_flow_misses']<=3 and 0<=(x1+x2)/2<shape[1] and 0<=(y1+y2)/2<shape[0]:keep.append(item)
        table=next((d for d in keep if d['name']=='billiards_table' and d.get('polygon')),None)
        if table:
            polygon=np.array(table['polygon'],np.float32)
            keep=[d for d in keep if d['name']=='billiards_table' or
                  cv2.pointPolygonTest(polygon,((d['box'][0]+d['box'][2])/2,(d['box'][1]+d['box'][3])/2),False)>=0]
        return keep

    def replace(self,frame_id,items):
        history=list(self.frames)
        index=next((i for i,f in enumerate(history) if f[0]==frame_id),None)
        if index is None:return False
        aligned=deepcopy(items)
        for before,after in zip(history[index:],history[index+1:]):
            aligned=self.flow(before[1],after[1],aligned,after[2],after[3])
        used=set()
        for item in aligned:
            if item.get("selection_id") is not None:
                item["track_id"]=item["selection_id"];used.add(item["track_id"]);continue
            center=np.mean(np.array(item['box']).reshape(2,2),axis=0)
            choices=[]
            for old in self.detections:
                if old['name']!=item['name'] or old.get('track_id') in used:continue
                old_center=np.mean(np.array(old['box']).reshape(2,2),axis=0)
                distance=np.linalg.norm(center-old_center)
                threshold=max(20.,(item['box'][2]-item['box'][0])*2)
                if distance<threshold:choices.append((distance,old))
            if choices:item['track_id']=min(choices,key=lambda pair:pair[0])[1]['track_id']
            else:item['track_id']=self.next_id;self.next_id+=1
            used.add(item['track_id'])
        self.detections=self.identities.update(history[-1][0],self.current_color,aligned,history[-1][2])
        return True
