"""Bounded optical-flow bridge between asynchronous YOLO results and displayed frames."""
# YOLO 异步结果与显示帧之间的桥接；帧历史是有界低分辨率灰度/彩色图，不缓存整段视频。
# 光流与模板用于寻找候选位置，彩色当前帧局部球体像素负责复核；失败就丢弃，不无限外推。
# 延迟响应只可在仍保留的历史窗口内回放，历史已过期时返回失败。

from collections import deque
from copy import deepcopy
import cv2
import numpy as np
from .ball_tracks import BallTracks
from .ball_observation import BALL_NAMES, closest_ball


class LiveTracking:
    # 建立有限历史和身份关联器；monitored_ids 用于短期丢失后的局部重新观察。
    def __init__(self, history_size=16):
        self.frames=deque(maxlen=history_size)
        self.detections=[]
        self.next_id=1
        self.identities=BallTracks()
        self.current_color=None
        self.monitored_ids=set()

    # 清空帧和当前检测，按 keep_tracks 决定是否保留身份容器，便于调用方区分暂停与换源。
    def clear(self,keep_tracks=False):
        self.frames.clear();self.detections=[];self.current_color=None;self.monitored_ids.clear()
        if not keep_tracks:self.identities=BallTracks()

    # 最长边缩至 640，返回灰度、缩放比、原图尺寸和彩色小图；像素框需要按比例换算。
    @staticmethod
    def gray(frame):
        scale=min(1.,640/max(frame.shape[:2]))
        small=cv2.resize(frame,None,fx=scale,fy=scale) if scale<1 else frame
        return cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),scale,frame.shape[:2],small

    # 把上一帧目标传播到当前帧，并对失踪的监控球在预测邻域尝试局部像素复核。
    # 分辨率变化直接清除旧历史；最后更新身份并追加有限历史，避免跨不同尺寸计算光流。
    def push(self,frame_id,frame):
        gray,scale,shape,small=self.gray(frame)
        self.current_color=frame
        for item in self.detections:
            state=self.identities.states.get(item.get('track_id'))
            if state is not None:
                item['_velocity']=state['velocity'].tolist()
                if state['appearance'] is not None:item['_appearance']=state['appearance'].tolist()
        if self.frames:
            _,previous,old_scale,old_shape,*_=self.frames[-1]
            if old_shape==shape:
                self.detections=self.flow(previous,gray,self.detections,scale,shape,frame)
            else:self.detections=[];self.frames.clear()
        seen={d.get('track_id') for d in self.detections}
        for key in self.monitored_ids-seen:
            state=self.identities.states.get(key)
            if state is None:continue
            dt=frame_id-state['frame']
            if not 0<dt<=self.identities.max_age:continue
            position=state['center']+state['velocity']*min(dt,12);half=state['size']/2
            proposal={'name':state['name'],'box':[*(position-half),*(position+half)],
                      'selection_id':key,'confidence':.7,'_appearance':None if state['appearance'] is None else state['appearance'].tolist()}
            proposal['_allow_motion_blur']=np.linalg.norm(state['velocity'])>state['size']*.35
            observed=closest_ball(frame,proposal,3.5)
            if observed is None:
                position=state['center']
                proposal['box']=[*(position-half),*(position+half)]
                observed=closest_ball(frame,proposal,3.5)
            if observed:self.detections.append(observed)
        for item in self.detections:
            if item.get('_ball_observed') and item.get('track_id') in self.identities.states:
                item['selection_id']=item['track_id']
        self.detections=self.identities.update(frame_id,self.current_color,self.detections)
        self.frames.append((frame_id,gray,scale,shape,small.copy()))
        return deepcopy(self.detections)

    # 在比原框更大的邻域匹配图像模板，补充快速运动时的小窗口光流。
    # 峰值不足或两个分离候选同样相似时拒绝，避免跳到相邻同色球。
    @staticmethod
    def patch_offset(previous,current,item,scale):
        x1,y1,x2,y2=np.asarray(item['box'])*scale
        ax,ay=max(0,int(x1)-2),max(0,int(y1)-2)
        bx,by=min(previous.shape[1],int(x2)+3),min(previous.shape[0],int(y2)+3)
        template=previous[ay:by,ax:bx]
        if min(template.shape,default=0)<5 or float(template.std())<5:return None
        radius=max(12,int(max(x2-x1,y2-y1)*6))
        sx,sy=max(0,ax-radius),max(0,ay-radius)
        ex,ey=min(current.shape[1],bx+radius),min(current.shape[0],by+radius)
        scores=cv2.matchTemplate(current[sy:ey,sx:ex],template,cv2.TM_CCOEFF_NORMED)
        _,peak,_,location=cv2.minMaxLoc(scores)
        if peak<.65:return None
        # Two equally good, separate balls are not enough identity evidence.
        x,y=location;radius=max(3,min(template.shape)//2)
        scores[max(0,y-radius):y+radius+1,max(0,x-radius):x+radius+1]=-1
        if float(scores.max())>peak-.04:return None
        return (np.asarray(location)+[sx-ax,sy-ay])/scale

    # 球心区域提取角点求光流，结合模板与局部实测候选修正位置。
    # 所有偏移还原到传入坐标尺度；当前彩色像素未复核成功的球不作为有效输出保留。
    @staticmethod
    def flow(previous,current,items,scale,shape,color=None):
        output=deepcopy(items);points=[];owners=[]
        for i,item in enumerate(output):
            if item['name'] not in BALL_NAMES:continue
            x1,y1,x2,y2=np.array(item['box'])*scale
            ax,ay=max(0,int(x1)),max(0,int(y1));bx,by=min(current.shape[1],int(x2)+1),min(current.shape[0],int(y2)+1)
            if bx<=ax or by<=ay:continue
            yy,xx=np.indices((by-ay,bx-ax));cx,cy=(x1+x2)/2,(y1+y2)/2
            mask=(((xx+ax-cx)**2+(yy+ay-cy)**2)<(.45*min(x2-x1,y2-y1))**2).astype(np.uint8)*255
            features=cv2.goodFeaturesToTrack(previous[ay:by,ax:bx],mask=mask,maxCorners=8,qualityLevel=.02,minDistance=2)
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
            if item['name'] not in BALL_NAMES:continue
            original=deepcopy(item)
            original['_reject_ambiguous']=True
            # A small feature window can follow the cloth when a ball moves
            # several diameters. Match its image patch over a wider local area.
            near=closest_ball(color,original,1.2) if color is not None else None
            stationary=(near is not None and np.linalg.norm(np.mean(np.asarray(near['box']).reshape(2,2),axis=0)-np.mean(np.asarray(original['box']).reshape(2,2),axis=0))<max(original['box'][2]-original['box'][0],original['box'][3]-original['box'][1])*.2)
            patch=None if stationary else LiveTracking.patch_offset(previous,current,item,scale)
            if patch is not None:valid_offsets[i]=[patch]
            if i in valid_offsets:
                dx,dy=np.median(valid_offsets[i],axis=0)
                x1,y1,x2,y2=item['box']
                item['box']=[float(x1+dx),float(y1+dy),float(x2+dx),float(y2+dy)]
                item['_flow_misses']=0
            else:item['_flow_misses']=item.get('_flow_misses',0)+1
            if color is not None:
                item['_allow_motion_blur']=np.linalg.norm(np.mean(np.asarray(item['box']).reshape(2,2),axis=0)-np.mean(np.asarray(original['box']).reshape(2,2),axis=0))>max(original['box'][2]-original['box'][0],original['box'][3]-original['box'][1])*.35
                observed=closest_ball(color,item,1.8)
                if observed is None:
                    # Measure pixels near the motion prediction, not a line
                    # extrapolated through an occluder.
                    velocity=np.asarray(original.get('_velocity',[0.,0.]))
                    proposal={**original,'box':(np.asarray(original['box'])+np.tile(velocity,2)).tolist(),
                              '_allow_motion_blur':True}
                    observed=closest_ball(color,proposal,4.)
                if observed is None:continue
                item=observed;item['_flow_misses']=0
            x1,y1,x2,y2=item['box']
            if item['_flow_misses']<=3 and 0<=(x1+x2)/2<shape[1] and 0<=(y1+y2)/2<shape[0]:keep.append(item)
        table=next((d for d in keep if d['name']=='billiards_table' and d.get('polygon')),None)
        if table:
            polygon=np.array(table['polygon'],np.float32)
            keep=[d for d in keep if d['name']=='billiards_table' or
                  cv2.pointPolygonTest(polygon,((d['box'][0]+d['box'][2])/2,(d['box'][1]+d['box'][3])/2),False)>=0]
        return keep

    # 从模型请求帧沿保留历史逐帧对齐到最新帧，再将实测位置关联到身份。
    # 只回填尚未被实时处理覆盖的路径区间；不重写已有观测，也不在历史窗口之外补猜轨迹。
    def replace(self,frame_id,items):
        history=list(self.frames)
        index=next((i for i,f in enumerate(history) if f[0]==frame_id),None)
        if index is None:return False
        aligned=[{**deepcopy(d),'_alignment_index':i} for i,d in enumerate(items)]
        measured={i:[(frame_id,np.mean(np.asarray(d['box']).reshape(2,2),axis=0))] for i,d in enumerate(items) if d['name'] in BALL_NAMES}
        for before,after in zip(history[index:],history[index+1:]):
            small_items=[{**d,'box':(np.asarray(d['box'])*after[2]).tolist()} for d in aligned]
            small_items=self.flow(before[1],after[1],small_items,1.,after[1].shape,after[4])
            aligned=[{**d,'box':(np.asarray(d['box'])/after[2]).tolist()} for d in small_items]
            for d in aligned:
                if d['name'] in BALL_NAMES and d.get('_ball_observed'):
                    measured[d['_alignment_index']].append((after[0],np.mean(np.asarray(d['box']).reshape(2,2),axis=0)))
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
        end_frame=history[-1][0]
        before_paths={k:(state['frame'],list(state['path'])) for k,state in self.identities.states.items()}
        self.detections=self.identities.update(end_frame,self.current_color,aligned)
        for d in self.detections:
            if d['name'] not in BALL_NAMES:continue
            key=d['track_id'];points=measured.get(d.get('_alignment_index'),[])
            if key in self.identities.states and len(points)>1:
                last,prior=before_paths.get(key,(frame_id-1,[]))
                # Never rewrite observations that were already tracked live.
                if last>=end_frame:continue
                recovered=[(f,p) for f,p in points if f>last]
                if len(recovered)<2:continue
                for f,p in recovered:
                    gap=bool(prior) and f-last>1
                    if not prior or np.linalg.norm(p-prior[-1][:2])>=1 or gap:
                        prior.append((float(p[0]),float(p[1]),gap))
                    last=f
                self.identities.states[key]['path']=deque(prior,maxlen=self.identities.max_points)
        self.monitored_ids.update(d['track_id'] for d in self.detections if d['name'] in BALL_NAMES)
        return True
