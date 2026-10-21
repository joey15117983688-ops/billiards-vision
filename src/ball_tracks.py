"""Motion/appearance association and bounded trails, independent of YOLO classes."""
from collections import deque
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from .ball_observation import BALL_NAMES


def center(item):return np.mean(np.array(item['box'],float).reshape(2,2),axis=0)


def appearance(frame,item):
    x1,y1,x2,y2=np.rint(item['box']).astype(int)
    cx,cy=(x1+x2)/2,(y1+y2)/2
    radius=max(2,.42*min(x2-x1,y2-y1))
    ax,ay=max(0,int(cx-radius)),max(0,int(cy-radius))
    bx,by=min(frame.shape[1],int(cx+radius+1)),min(frame.shape[0],int(cy+radius+1))
    if bx<=ax or by<=ay:return None
    hsv=cv2.cvtColor(frame[ay:by,ax:bx],cv2.COLOR_BGR2HSV)
    yy,xx=np.indices(hsv.shape[:2]);mask=(((xx+ax-cx)**2+(yy+ay-cy)**2)<=radius**2).astype(np.uint8)*255
    hist=cv2.calcHist([hsv],[0,1],mask,[18,4],[0,180,0,256]).ravel()
    hist/=max(hist.sum(),1)
    return hist


class BallTracks:
    def __init__(self,max_points=600,max_age=60):
        self.states={};self.next_id=1;self.max_points=max_points;self.max_age=max_age
        self.retired=set();self.next_label=1

    @staticmethod
    def distance(a,b):
        if a is None or b is None:return 0.
        return float(cv2.compareHist(a.astype(np.float32),b.astype(np.float32),cv2.HISTCMP_BHATTACHARYYA))

    def update(self,frame_id,frame,items,appearance_scale=1.):
        balls=[dict(d) for d in items if d['name'] in BALL_NAMES]
        # Suppress duplicate local-crop reports of the same physical object.
        unique=[]
        for item in balls:
            if not any(np.linalg.norm(center(item)-center(other))<.3*min(item['box'][2]-item['box'][0],item['box'][3]-item['box'][1]) for other in unique):unique.append(item)
        balls=unique;descriptors=[appearance(frame,{**d,"box":(np.array(d["box"])*appearance_scale).tolist()}) for d in balls]
        active=[(key,state) for key,state in self.states.items() if frame_id-state['frame']<=self.max_age]
        costs=np.full((len(active),len(balls)),1e6,float)
        for i,(key,state) in enumerate(active):
            dt=max(0,frame_id-state['frame']);predicted=state['center']+state['velocity']*dt
            for j,item in enumerate(balls):
                hint=item.get('selection_id')
                if hint is not None and hint!=key:continue
                if item['name']!=state['name']:continue
                size=max(8.,state['size']);distance=float(np.linalg.norm(center(item)-predicted))
                colour=self.distance(state['appearance'],descriptors[j])
                observed_hint=hint==key and item.get('_ball_observed') and dt<=3
                if distance>size*((6 if observed_hint else 2)+min(dt,15)*.5):continue
                if colour>.65 and not (item.get("_pocket_continuity") and hint==key):continue
                costs[i,j]=distance/size+.9*colour
        matches={}
        if costs.size:
            rows,cols=linear_sum_assignment(costs)
            for i,j in zip(rows,cols):
                if costs[i,j]>=1e5:continue
                # Visually indistinguishable, overlapping candidates are ambiguous.
                alternatives=sorted(costs[:,j])
                if len(alternatives)>1 and alternatives[1]<1e5 and alternatives[1]-alternatives[0]<.12:continue
                matches[j]=active[i][0]
        output=[];seen=set()
        for j,item in enumerate(balls):
            key=matches.get(j)
            hint=item.get('selection_id')
            if key is None:
                # Do not rename a failed match to a selected ID or silently reconnect it.
                if hint in self.states or hint in self.retired:continue
                if hint is None and any(costs[i,j]<1e5 for i in range(len(active))):continue
                key=hint if hint is not None else self.next_id
                self.next_id=max(self.next_id,key+1)
                self.states[key]={'name':item['name'],'center':center(item),'velocity':np.zeros(2),
                                  'frame':frame_id,'appearance':descriptors[j],
                                  'size':max(item['box'][2]-item['box'][0],item['box'][3]-item['box'][1]),
                                  'path':deque(maxlen=self.max_points),'status':'tracking','label_id':self.next_label}
                self.next_label+=1
            state=self.states[key];position=center(item);dt=frame_id-state['frame']
            if dt>0:
                state['velocity']=.65*((position-state['center'])/dt)+.35*state['velocity']
            gap=dt>1 or state['status']=='lost'
            if not state['path'] or np.linalg.norm(position-np.array(state['path'][-1][:2]))>=1 or gap:
                state['path'].append((float(position[0]),float(position[1]),gap))
            state.update(center=position,frame=frame_id,status='tracking')
            if descriptors[j] is not None:
                old=state['appearance'];state['appearance']=descriptors[j] if old is None else .9*old+.1*descriptors[j]
            item['track_id']=key;item['ball_id']=state['label_id'];output.append(item);seen.add(key)
        for key,state in list(self.states.items()):
            if key not in seen:state['status']='lost'
            if frame_id-state['frame']>self.max_age:
                self.retired.add(key);del self.states[key]
        if len(self.states)>128:
            for key in sorted(self.states,key=lambda k:self.states[k]["frame"])[:len(self.states)-128]:
                self.retired.add(key);del self.states[key]
        if len(self.retired)>512:self.retired=set(sorted(self.retired)[-512:])
        return [dict(d) for d in items if d['name']=='billiards_table']+output

    def paths(self):return {key:list(state['path']) for key,state in self.states.items()}
