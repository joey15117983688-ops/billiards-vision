"""Bounded geometric search driven by explicit correction feedback and image evidence."""
from dataclasses import replace
import cv2
import numpy as np
from src.table_coordinates import TableCoordinates
from src.vision.pocket import PocketDetector, TableDetector, PocketConfig, TableResult
from src.vision.pocket.table_detector import _line, _intersection


def same_polygon(a,b,tolerance=.006):
    if a is None or b is None:return False
    try:
        a=TableCoordinates(a).corners.astype(float);b=TableCoordinates(b).corners.astype(float)
    except (ValueError,cv2.error):return False
    short=min(np.linalg.norm(np.roll(a,-1,axis=0)-a,axis=1))
    return bool(np.max(np.linalg.norm(a-b,axis=1))<max(2.,short*tolerance))


def shifted_polygon(polygon,offsets):
    q=TableCoordinates(polygon).corners.astype(float)
    lines=[]
    for i,distance in enumerate(offsets):
        a,b=q[i],q[(i+1)%4];v=b-a
        inward=np.array([-v[1],v[0]])/np.linalg.norm(v)
        lines.append(_line(a+inward*distance,b+inward*distance))
    points=[_intersection(lines[i-1],lines[i]) for i in range(4)]
    if any(p is None for p in points):return None
    try:return TableCoordinates(points).corners.tolist()
    except (ValueError,cv2.error):return None


class ImageEvidence:
    def __init__(self,frame):
        self.scale=min(1.,900/max(frame.shape[:2]))
        image=cv2.resize(frame,None,fx=self.scale,fy=self.scale) if self.scale<1 else frame
        self.hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        value=self.hsv[:,:,2]
        self.edges=cv2.Canny(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),15,50)|cv2.Canny(self.hsv[:,:,1],20,60)
        self.edges=cv2.dilate(self.edges,np.ones((5,5),np.uint8))
        self.value=value

    def measure(self,polygon):
        q=TableCoordinates(polygon).corners.astype(float)*self.scale
        h,w=self.value.shape
        if np.any(q<0) or np.any(q[:,0]>=w) or np.any(q[:,1]>=h):raise ValueError('boundary_outside_frame')
        short=min(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1))
        mask=np.zeros((h,w),np.uint8);cv2.fillConvexPoly(mask,np.rint(q).astype(np.int32),255)
        if cv2.countNonZero(mask)<.08*mask.size:raise ValueError('table_too_small')
        radius=max(2,int(short*.018));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2)
        inner=cv2.erode(mask,kernel)>0;band=(mask>0)&~inner
        colorful=inner&(self.hsv[:,:,1]>35)&(self.value>20)
        if np.count_nonzero(colorful)<100:raise ValueError('no_cloth_evidence')
        histogram=np.bincount(self.hsv[:,:,0][colorful],minlength=180)
        hue=int(np.argmax(np.convolve(np.r_[histogram[-8:],histogram,histogram[:8]],np.ones(17),'valid')))%180
        delta=np.abs(self.hsv[:,:,0].astype(float)-hue);delta=np.minimum(delta,180-delta)
        cloth=(delta<22)&(self.hsv[:,:,1]>30)&(self.value>20)
        median=float(np.median(self.value[colorful]))
        black=(self.value<max(20,median*.35))
        fill=float(np.mean(cloth[mask>0]));black_band=float(np.mean(black[band])) if band.any() else 1.
        supports=[]
        for i in range(4):
            samples=q[i]+np.linspace(.12,.88,100)[:,None]*(q[(i+1)%4]-q[i])
            points=np.rint(samples).astype(int);points[:,0]=np.clip(points[:,0],0,w-1);points[:,1]=np.clip(points[:,1],0,h-1)
            supports.append(float(np.mean(self.edges[points[:,1],points[:,0]]>0)))
        return {'cloth_fill':fill,'black_band':black_band,'edge_support':supports,
                'edge_mean':float(np.mean(supports))}


def evaluate_candidate(frame,polygon,evidence,config):
    q=TableCoordinates(polygon).corners.tolist()
    metrics=evidence.measure(q)
    table=TableResult(detected=True,complete=True,corners=q,playable_polygon=q,scope_polygon=q,
                      quality={'geometry_reliable':True,**metrics},confidence=metrics['edge_mean'])
    pockets=PocketDetector(config).detect(frame,table=table).to_dict()
    observed=[p for p in pockets['pockets'] if p['status']=='observed']
    metrics['observed']=len(observed)
    metrics['side_observed']=sum('MIDDLE' in p['id'] for p in observed)
    metrics['score']=float(2*len(observed)+metrics['side_observed']+3*metrics['cloth_fill']+
                           2*metrics['edge_mean']-5*metrics['black_band'])
    strong_plane=(metrics['cloth_fill']>=.9 and metrics['black_band']<=.06 and
                  metrics['edge_mean']>=.8 and min(metrics['edge_support'])>=.6)
    metrics['reliable']=bool(metrics['cloth_fill']>=.7 and metrics['black_band']<=.22 and
                             metrics['edge_mean']>=.35 and (len(observed)>=3 or strong_plane))
    return {'polygon':q,'pockets':pockets,'metrics':metrics}


def choose_correction(candidates,baseline,rejected,manual=False):
    eligible=[c for c in candidates if c['metrics']['reliable'] and
              not any(same_polygon(c['polygon'],q) for q in rejected)]
    if baseline is None:
        eligible=[c for c in eligible if c['metrics']['observed']>=3]
    if baseline is not None:
        eligible=[c for c in eligible if c['metrics']['observed']>=baseline['metrics']['observed'] and
                  c['metrics']['side_observed']>=baseline['metrics']['side_observed'] and
                  c['metrics']['score']>=baseline['metrics']['score']+(1.25 if manual else .65)]
    if eligible:
        return max(eligible,key=lambda c:c['metrics']['score']-(.15 if ':edges' in c.get('strategy','') else 0.)),('updated' if baseline is not None else 'initial')
    if baseline is not None:return baseline,'unchanged'
    raise ValueError('无法可靠定位台呢内库：未找到通过台呢、黑边和袋口检查的候选，原结果保留')


def search_calibration(frame,progress,context=None):
    context=context or {};attempt=max(1,int(context.get('attempt',1)))
    config=PocketConfig(hsv_ranges=PocketConfig().hsv_ranges+(
        ((0,70,30),(15,255,255)),((126,50,30),(165,255,255)),
        ((166,70,30),(179,255,255)),((16,70,30),(29,255,255))))
    progress(f'2/4 第 {attempt} 次纠错：检查是否包含黑边、遗漏两侧，并生成不同边界…')
    seeds=[]
    previous=context.get('previous_polygon')
    if previous is not None:seeds.append(('previous',previous))
    for name,image,options in [
        ('color_and_edges',frame,config),
        ('fine_edges',frame,replace(config,max_side=1800)),
        ('low_exposure',np.clip(frame.astype(float)*.65,0,255).astype(np.uint8),config),
        ('high_exposure',np.clip(frame.astype(float)*1.35,0,255).astype(np.uint8),config)]:
        table=TableDetector(options).detect(image)
        if table.detected and table.quality.get('geometry_reliable'):seeds.append((name,table.playable_polygon))
    evidence=ImageEvidence(frame);baseline=None
    if previous is not None:
        try:baseline=evaluate_candidate(frame,previous,evidence,config)
        except (ValueError,cv2.error):pass
    diagnosis=[]
    if baseline:
        before=baseline['metrics']
        if before['black_band']>.22:diagnosis.append('旧范围可能包含深色库边，检查向内收缩')
        if before['side_observed']<2:diagnosis.append('两侧中袋证据不足，同时检查遗漏区域和边界偏移')
        if before['edge_mean']<.5:diagnosis.append('旧边界与可见直线不一致，重新拟合')
        if not diagnosis:diagnosis.append('用户再次否定当前结果，检查其他边界并提高替换门槛')
        progress(f"3/4 旧结果：边缘黑色占比 {before['black_band']:.0%}，中袋验证 {before['side_observed']}/2；对比其他方案…")
    else:diagnosis.append('当前无可验证的旧边界，比较多种自动定位方案')
    hypotheses=list(seeds)
    # Each retry genuinely broadens the search. Outward hypotheses recover omitted
    # sides; inward hypotheses remove rails. Individual edges cover asymmetric errors.
    radius=min(.10,.025*attempt)
    for name,seed in seeds[:2]:
        q=np.asarray(seed);short=min(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1))
        radii=[radius] if attempt==1 else [radius,radius/2]
        for amount in [short*r*sign for r in radii for sign in (1,-1)]:
            for group in [(0,1,2,3),(0,),(1,),(2,),(3,),(0,2),(1,3)]:
                offsets=[amount if i in group else 0. for i in range(4)]
                polygon=shifted_polygon(seed,offsets)
                if polygon is not None:hypotheses.append((f'{name}:edges{group}:{amount:.1f}',polygon))
    results=[];seen=set();evaluated=[]
    for strategy,polygon in hypotheses:
        key=tuple(np.rint(np.asarray(polygon).ravel()).astype(int))
        if key in seen:continue
        seen.add(key)
        if len(seen)%8==1:progress(f'3/4 比较候选 {len(seen)}：验证内沿、黑边占比和六袋口证据…')
        try:
            candidate=evaluate_candidate(frame,polygon,evidence,config)
            candidate['strategy']=strategy;results.append(candidate)
            evaluated.append({'strategy':strategy,**candidate['metrics']})
        except (ValueError,cv2.error):continue
    chosen,decision=choose_correction(results,baseline,context.get('rejected_polygons',[]),context.get('manual',False))
    report={'attempt':attempt,'candidate_count':len(seen),'decision':decision,
            'strategy':chosen.get('strategy','previous'),
            'before':baseline['metrics'] if baseline else None,'after':chosen['metrics'],
            'search_radius_ratio':radius,'evaluations':evaluated,'diagnosis':diagnosis,
            'reason':'找到证据更充分的新边界' if decision!='unchanged' else '没有更可靠的替代结果，保持原结果；可再次校准扩大搜索或手动修正'}
    return chosen['polygon'],chosen['pockets'],report
