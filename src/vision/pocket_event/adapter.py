"""Convert raw detections (never flow predictions) and pocket pixels to one plane."""
import cv2
import numpy as np
from src.table_coordinates import TableCoordinates, TableDimensions
from src.ball_observation import BALL_NAMES
from src.ball_tracks import BallTracks
from .models import BallState
from . import BallObservation, EventConfig, PocketEventEngine, make_zones


class PocketEventAdapter:
    def __init__(self,pockets,dimensions=None,config=None,balls=('black_ball','white_ball'),track_target_balls=True):
        config=config or EventConfig()
        table=pockets['table']
        if not table.get('quality',{}).get('geometry_reliable'):
            raise ValueError('球桌几何不可靠，请先自动检测袋口或手动校准')
        polygon=table['playable_polygon']
        q=np.asarray(polygon,float)
        dimensions=dimensions or TableDimensions()
        # Match the long physical dimension to the long screen edge.
        a,b=sorted((dimensions.x_mm,dimensions.y_mm))
        axes=(b,a) if np.linalg.norm(q[1]-q[0])>=np.linalg.norm(q[3]-q[0]) else (a,b)
        self.coordinates=TableCoordinates(polygon,TableDimensions(*axes,reference=dimensions.reference))
        self.scale=min(axes);self.previous=None;self.last_timestamp=None
        self.track_target_balls=track_target_balls
        self.target_tracks=BallTracks(max_age=90)
        self.target_names={}
        self.target_pixels={}
        self.target_templates={}
        self.detector_seen={}
        self.pocket_centers=[]
        self.pocket_ids=[]
        self.occluded_pockets=set()
        converted=[]
        for p in pockets['pockets']:
            def point(v):
                if v is None:return None
                if isinstance(v,dict):v=[v['x'],v['y']]
                return self.project([v])[0].tolist()
            pixel_center=p.get('center') or p.get('candidate_center')
            if pixel_center is not None:self.pocket_centers.append(list(pixel_center.values()) if isinstance(pixel_center,dict) else pixel_center)
            if pixel_center is not None:self.pocket_ids.append(p['id'])
            converted.append({**p,'center':point(p.get('center')),'candidate_center':point(p.get('candidate_center')),
                              'entry_polygon':self.project(p['entry_polygon']).tolist() if p.get('entry_polygon') else []})
        center=np.asarray(axes)/(2*self.scale)
        self.engine=PocketEventEngine(make_zones(converted,center,config),config,balls)
        if not self.engine.zones:raise ValueError('没有可用袋口，请先检测或手动校准')

    def align_geometry(self,pockets,delta):
        """Move pixel geometry, preserving normalized event history and verdicts."""
        from src.camera_alignment import map_points
        self.coordinates=TableCoordinates(pockets['table']['playable_polygon'],self.coordinates.dimensions)
        self.pocket_centers=[]
        for p in pockets['pockets']:
            point=p.get('center') or p.get('candidate_center')
            if point:self.pocket_centers.append([point['x'],point['y']])
        factor=float(np.sqrt(abs(np.linalg.det(delta[:2,:2]))))
        for state in self.target_tracks.states.values():
            state['center']=np.asarray(map_points([state['center']],delta)[0])
            state['velocity']=delta[:2,:2]@state['velocity']
            state['size']*=factor
            points=list(state['path'])
            mapped=map_points([p[:2] for p in points],delta)
            state['path'].clear();state['path'].extend((x,y,p[2]) for (x,y),p in zip(mapped,points))
        for _,target in self.target_pixels.values():
            x1,y1,x2,y2=target['box']
            points=np.asarray(map_points([[x1,y1],[x2,y1],[x2,y2],[x1,y2]],delta))
            target['box']=[*points.min(axis=0),*points.max(axis=0)]

    def project(self,points):
        # Pocket entrances can be outside the cloth; do not clip to the table.
        return cv2.perspectiveTransform(np.float32([points]),self.coordinates.image_to_mm)[0]/self.scale

    def track_targets(self,timestamp,detections,frame):
        # The model shares one class for every coloured ball. Associate real
        # detections before the engine groups them, so two reds stay independent.
        if frame is None or not self.track_target_balls:return detections
        candidates=[{**d,'name':'billiard_ball','observed_name':d['name']} for d in detections if d['name'] in BALL_NAMES]
        for candidate in candidates:
            candidate.pop('selection_id',None)
            candidate.pop('_pocket_continuity',None)
        # Recover only a recently observed ball already approaching a pocket.
        # These are measured pixels, never a predicted centre used as evidence.
        from src.ball_observation import closest_ball
        for identity,(seen,target) in list(self.target_pixels.items()):
            raw_age=timestamp-self.detector_seen.get(identity,seen)
            if raw_age>.8 or timestamp-seen>.15 or identity not in self.target_tracks.states:
                del self.target_pixels[identity]
                self.target_templates.pop(identity,None);self.detector_seen.pop(identity,None)
                continue
            key=self.target_names[identity]
            state=self.engine.states[key]
            if state.pocket is None:continue
            extended=raw_age>.3
            zone=next((z for z in self.engine.zones if z.id==state.pocket),None)
            if extended and (not state.direction_seen or state.occlusion_seen or zone is None or not zone.trusted_shape):continue
            origin=np.mean(np.asarray(target['box']).reshape(2,2),axis=0)
            size=max(target['box'][2]-target['box'][0],target['box'][3]-target['box'][1])
            if any(np.linalg.norm(np.mean(np.asarray(d['box']).reshape(2,2),axis=0)-origin)<size*1.5 for d in candidates):continue
            recovered=self.pocket_pixels(frame,target)
            if recovered is None:recovered=closest_ball(frame,target)
            if recovered is None:recovered=self.pocket_template(frame,target,timestamp)
            if recovered is not None and extended:
                # Extend only uninterrupted, freshly measured motion into a
                # trusted pocket. Never extend a missing/predicted point or
                # let a template wander away along the cushion.
                point=np.mean(np.asarray(recovered['box']).reshape(2,2),axis=0)
                old_plane,new_plane=self.project([origin,point])
                track=self.target_tracks.states[identity]
                dt=max(1,round(timestamp*30)-track['frame'])
                predicted=track['center']+track['velocity']*dt
                if zone.distance(new_plane)>zone.distance(old_plane)+.003 or np.linalg.norm(point-predicted)>size:
                    recovered=None
            if recovered is not None:
                candidates.append({**recovered,'name':'billiard_ball','selection_id':identity,'_pocket_continuity':True,'_pocket_recovered':True})
        frame_id=round(timestamp*30)
        for item in candidates:
            matches=[]
            point=np.mean(np.asarray(item['box']).reshape(2,2),axis=0)
            for identity,track in self.target_tracks.states.items():
                state=self.engine.states[self.target_names[identity]]
                age=frame_id-track['frame']
                if state.pocket is None or not state.direction_seen or not 0<age<=3:continue
                distance=np.linalg.norm(point-track['center']-track['velocity']*age)
                if distance<track['size']*1.5:matches.append((distance/track['size'],identity))
            matches.sort()
            if matches and (len(matches)==1 or matches[1][0]-matches[0][0]>.5):
                item['selection_id']=matches[0][1];item['_pocket_continuity']=True
        tracked=self.target_tracks.update(frame_id,frame,candidates)
        output=[d for d in detections if d['name'] not in BALL_NAMES]
        for item in tracked:
            identity=item['track_id']
            if identity not in self.target_names:
                observed=item['observed_name']
                # Keep the first physical identity through later class flicker.
                # Only one track may own the special black/white display slot.
                key=observed if observed in self.engine.balls and observed not in self.target_names.values() else f"billiard_ball:{identity}"
                self.target_names[identity]=key
            key=self.target_names[identity]
            if key not in self.engine.states:
                self.engine.balls=(*self.engine.balls,key)
                self.engine.states[key]=BallState(reason='new_target')
            if not item.get('_pocket_recovered'):
                self.detector_seen[identity]=timestamp
                x1,y1,x2,y2=np.rint(item['box']).astype(int)
                crop=frame[max(0,y1):max(0,y2),max(0,x1):max(0,x2)]
                if crop.size:self.target_templates[identity]=crop.copy()
            self.target_pixels[identity]=(timestamp,item)
            output.append({**item,'name':key})
        return output

    def pocket_template(self,frame,target,timestamp):
        identity=target['track_id'];template=self.target_templates.get(identity)
        if template is None:return None
        track=self.target_tracks.states[identity]
        dt=max(1,round(timestamp*30)-track['frame'])
        center=track['center']+track['velocity']*dt
        h,w=template.shape[:2];radius=max(w,h)*1.5
        ax,ay=max(0,int(center[0]-radius)),max(0,int(center[1]-radius))
        bx,by=min(frame.shape[1],int(center[0]+radius)),min(frame.shape[0],int(center[1]+radius))
        if bx-ax<w or by-ay<h:return None
        scores=cv2.matchTemplate(frame[ay:by,ax:bx],template,cv2.TM_SQDIFF_NORMED)
        value,_,location,_=cv2.minMaxLoc(scores)
        if value>.35:return None
        x,y=location
        return {**target,'box':[ax+x,ay+y,ax+x+w,ay+y+h],'confidence':.75,'method':'pocket_template_pixels'}

    def pocket_pixels(self,frame,target):
        # Saturated coloured balls remain separable from white pocket liners
        # even when the general object detector drops its box at the cushion.
        identity=target['track_id'];descriptor=self.target_tracks.states[identity]['appearance']
        if descriptor is None:return None
        hue_bin,sat_bin=np.unravel_index(int(np.argmax(descriptor)),(18,4))
        if sat_bin<2:return None
        hue=hue_bin*10+5
        x1,y1,x2,y2=target['box'];size=max(x2-x1,y2-y1);cx,cy=(x1+x2)/2,(y1+y2)/2
        ax,ay=max(0,int(cx-2*size)),max(0,int(cy-2*size))
        bx,by=min(frame.shape[1],int(cx+2*size)),min(frame.shape[0],int(cy+2*size))
        hsv=cv2.cvtColor(frame[ay:by,ax:bx],cv2.COLOR_BGR2HSV)
        delta=np.abs(hsv[:,:,0].astype(float)-hue);delta=np.minimum(delta,180-delta)
        binary=((delta<12)&(hsv[:,:,1]>100)&(hsv[:,:,2]>45)).astype(np.uint8)*255
        contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        found=[]
        for contour in contours:
            x,y,w,h=cv2.boundingRect(contour);area=cv2.contourArea(contour)
            if min(w,h)<size*.4 or max(w,h)>size*1.6 or max(w,h)/min(w,h)>1.8:continue
            if area/max(w*h,1)<.5:continue
            center=np.array([ax+x+w/2,ay+y+h/2])
            if np.linalg.norm(center-[cx,cy])>size*1.8:continue
            found.append((np.linalg.norm(center-[cx,cy]),{**target,'box':[ax+x,ay+y,ax+x+w,ay+y+h],'confidence':.75,'method':'pocket_colour_pixels'}))
        return min(found,key=lambda pair:pair[0])[1] if found else None

    def observations(self,detections):
        observed=[];all_positions=[];alternatives=[]
        for d in detections:
            if d['name'] not in BALL_NAMES and d['name'] not in self.engine.balls:continue
            x1,y1,x2,y2=d['box'];cx,cy=(x1+x2)/2,(y1+y2)/2
            p=self.project([[cx,cy],[x1,cy],[x2,cy],[cx,y1],[cx,y2]])
            radius=float((np.linalg.norm(p[2]-p[1])+np.linalg.norm(p[4]-p[3]))/4)
            observation=BallObservation(d['name'],tuple(map(float,p[0])),radius,float(d.get('confidence',0)),d.get('track_id'))
            all_positions.append(observation)
            if d['name'] in self.engine.balls:observed.append(observation)
            else:alternatives.append(observation.center)
        return observed,all_positions,alternatives

    def clear_scene(self,frame,detections):
        self.occluded_pockets=set()
        if frame is None:return False
        factor=min(1,480/max(frame.shape[:2]));image=cv2.resize(frame,None,fx=factor,fy=factor)
        mask=np.zeros(image.shape[:2],np.uint8)
        q=self.coordinates.corners*factor
        cv2.fillConvexPoly(mask,np.rint(q).astype(np.int32),255)
        width=min(np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[3]-q[0]))
        inset=max(3,int(width*.035));mask=cv2.erode(mask,np.ones((inset,inset),np.uint8))
        for d in detections:
            if d['name'] not in BALL_NAMES and d['name'] not in self.engine.balls:continue
            x1,y1,x2,y2=np.asarray(d['box'])*factor
            margin=max(2,int(max(x2-x1,y2-y1)*.4))
            cv2.rectangle(mask,(int(x1)-margin,int(y1)-margin),(int(x2)+margin,int(y2)+margin),0,-1)
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);samples=hsv[mask>0]
        if len(samples)<100:return False
        colored=samples[(samples[:,1]>40)&(samples[:,2]>20)]
        if len(colored)<.8*len(samples):return False
        hue=int(np.argmax(np.bincount(colored[:,0],minlength=180)))
        delta=np.abs(samples[:,0].astype(float)-hue);delta=np.minimum(delta,180-delta)
        blocked=(delta>18)|(samples[:,1]<30)|(samples[:,2]<12)
        # Check skin-colored obstruction at pocket rims too: the inner-plane
        # erosion intentionally excludes rails, where a hand can hide an entry.
        h,s,v=cv2.split(hsv)
        difference=np.abs(h.astype(float)-hue);difference=np.minimum(difference,180-difference)
        skin=(h<25)&(s>35)&(s<230)&(v>135)&(difference>18)
        excluded=np.zeros_like(mask)
        for d in detections:
            if d['name'] not in BALL_NAMES and d['name'] not in self.engine.balls:continue
            x1,y1,x2,y2=np.asarray(d['box'])*factor
            cv2.rectangle(excluded,(int(x1)-3,int(y1)-3),(int(x2)+3,int(y2)+3),255,-1)
        for pocket_id,center in zip(self.pocket_ids,self.pocket_centers):
            zone=np.zeros_like(mask)
            cv2.circle(zone,tuple(np.rint(np.asarray(center)*factor).astype(int)),max(4,int(width*.08)),255,-1)
            region=(zone>0)&(excluded==0)
            if region.any() and np.count_nonzero(skin&region)>max(4,.025*np.count_nonzero(region)):self.occluded_pockets.add(pocket_id)
        # Conservative obstruction gate, not an arm/person classifier.
        return not self.occluded_pockets and float(np.mean(blocked))<.025

    def update(self,timestamp,detections,frame=None,scene_clear=None):
        tracked=self.track_targets(timestamp,detections,frame)
        observations,positions,alternatives=self.observations(tracked)
        stable=False;motion=False
        if self.previous is not None and self.last_timestamp is not None and timestamp>self.last_timestamp:
            dt=timestamp-self.last_timestamp
            if dt<=self.engine.config.max_observation_gap:
                # Exclude disappeared targets, whose absence the engine reviews separately.
                distances=[]
                for o in positions:
                    matches=[np.linalg.norm(np.asarray(o.center)-p.center) for p in self.previous if p.ball==o.ball]
                    distances.append(max(0.,min(matches)-o.radius*.25)/dt if matches else float('inf'))
                stable=all(v<self.engine.config.motion_speed for v in distances)
                motion=any(np.isfinite(v) and v>=self.engine.config.motion_speed for v in distances)
        clear=self.clear_scene(frame,tracked) if scene_clear is None else scene_clear
        self.last_scene_clear=clear
        results=self.engine.update(timestamp,observations,scene_clear=clear,scene_stable=stable,alternatives=alternatives,shot_motion=motion,pocket_clear={z.id:z.id not in self.occluded_pockets for z in self.engine.zones} if scene_clear is None else None)
        self.previous=positions;self.last_timestamp=timestamp
        return results
