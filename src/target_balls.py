"""Locate one black and one white ball inside the visible playing surface."""
import cv2
import numpy as np
from src.vision.pocket import TableDetector
from src.ball_observation import closest_ball


def playing_surface(frame):
    table = TableDetector().detect(frame)
    if not table.detected or not table.quality.get('geometry_reliable'):
        return None
    return table.playable_polygon


def surface_mask(frame, polygon):
    mask = np.zeros(frame.shape[:2], np.uint8)
    if polygon:
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
    return mask


def target_balls(frame, detections=(), polygon=None):
    polygon = playing_surface(frame) if polygon is None else polygon
    if not polygon:
        return []
    mask = surface_mask(frame, polygon)
    points = np.array(polygon, np.float32)
    short = min(np.linalg.norm(points[(i+1)%4]-points[i]) for i in range(4))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    channels = {'black_ball': ((hsv[:,:,2]<85)&((hsv[:,:,1]<170)|(hsv[:,:,2]<35))),
                'white_ball': ((hsv[:,:,2]>160)&(hsv[:,:,1]<85))}
    candidates = {name: [] for name in channels}
    for name, selected in channels.items():
        binary = (selected & (mask>0)).astype(np.uint8)*255
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
        contours,_ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            x,y,w,h = cv2.boundingRect(contour)
            if not .015*short <= min(w,h) <= .065*short:continue
            if max(w,h)/min(w,h)>2.1:continue
            if area/max(w*h,1)<.48:continue
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = area/max(hull_area,1)
            if solidity<.78:continue
            moments = cv2.moments(contour)
            if not moments['m00']:continue
            center = (moments['m10']/moments['m00'],moments['m01']/moments['m00'])
            # Reject cushion shadows and pocket cavities that straddle the boundary.
            if cv2.pointPolygonTest(points,center,True)<.42*min(w,h):continue
            perimeter = cv2.arcLength(contour,True)
            circularity = min(1.,4*np.pi*area/max(perimeter**2,1))
            if circularity<.48:continue
            # A ball must be surrounded by cloth, preventing cuffs/cues from qualifying.
            pad = max(3,int(min(w,h)*.35))
            ax,ay,bx,by=max(0,x-pad),max(0,y-pad),min(frame.shape[1],x+w+pad),min(frame.shape[0],y+h+pad)
            context = hsv[ay:by,ax:bx]
            green = (context[:,:,0]>=30)&(context[:,:,0]<=95)&(context[:,:,1]>40)&(context[:,:,2]>25)
            blue = (context[:,:,0]>=96)&(context[:,:,0]<=125)&(context[:,:,1]>=150)&(context[:,:,2]>=80)
            cloth = green|blue
            surround = np.ones(cloth.shape,bool);surround[y-ay:y+h-ay,x-ax:x+w-ax]=False
            cloth_fraction = float(np.mean(cloth[surround])) if surround.any() else 0.
            if cloth_fraction<.6:continue
            colored = (context[:,:,1]>90)&~cloth
            if name=='white_ball' and np.mean(colored[surround])>.15:continue
            overlap = any(item.get('name') in ('billiard_ball','white_ball','black_ball') and
                          item['box'][0]<=center[0]<=item['box'][2] and item['box'][1]<=center[1]<=item['box'][3]
                          for item in detections)
            score = .35*circularity+.25*solidity+.3*cloth_fraction+.1*overlap
            candidates[name].append({'name':name,'box':[x,y,x+w,y+h], 'confidence':float(score),
                                     'method':'color_shape_on_table','yolo_supported':overlap})
    return [max(items,key=lambda item:item['confidence']) for items in candidates.values() if items]


def scene_detections(frame, detections, calibration_polygon=None, scope_polygon=None):
    """Keep the table and all on-table balls, refining only black/white labels."""
    if calibration_polygon is not None:
        from src.table_coordinates import TableCoordinates
        from src.vision.pocket.models import TableResult
        polygon=TableCoordinates(calibration_polygon).corners.tolist()
        table=TableResult(detected=True,playable_polygon=polygon,scope_polygon=polygon,
                          quality={'geometry_reliable':True})
    else:
        table=TableDetector().detect(frame)
    if not table.detected or not table.quality.get('geometry_reliable'):
        return []
    inner_polygon=table.playable_polygon
    from src.cloth_region import cloth_region
    region=None if scope_polygon is not None else cloth_region(frame,inner_polygon)
    polygon=scope_polygon if scope_polygon is not None else region['polygon'] if region else inner_polygon
    points = np.array(polygon,np.float32)
    mask = surface_mask(frame,polygon)
    special = target_balls(frame,detections,polygon)
    table_scores = [item['confidence'] for item in detections if item['name']=='billiards_table']
    table = {'name':'billiards_table','confidence':max(table_scores) if table_scores else .8,
             'box':[float(points[:,0].min()),float(points[:,1].min()),
                    float(points[:,0].max()),float(points[:,1].max())],
             'polygon':polygon,'playable_polygon':inner_polygon,'geometry':'connected_cloth_scope',
             'scope_evidence':region,
             'method':'yolo_and_cloth_geometry' if table_scores else 'cloth_geometry'}
    if calibration_polygon is None:
        # Verify a first automatic result before the UI locks/persists its plane.
        from src.adaptive_calibration import evaluate_candidate,ImageEvidence
        from src.vision.pocket import PocketConfig
        try:
            evidence=evaluate_candidate(frame,inner_polygon,ImageEvidence(frame),PocketConfig())
            if evidence['metrics']['reliable'] and evidence['metrics']['observed']==6:
                evidence['pockets']['table']['scope_polygon']=polygon
                table['calibration_evidence']=evidence
        except (ValueError,cv2.error):pass
    balls = []
    def overlap(a,b):
        x1,y1=max(a[0],b[0]),max(a[1],b[1])
        x2,y2=min(a[2],b[2]),min(a[3],b[3])
        intersection=max(0,x2-x1)*max(0,y2-y1)
        return intersection/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)
    for item in detections:
        if item['name'] not in ('billiard_ball','white_ball','black_ball'):continue
        x1,y1,x2,y2=item['box'];center=((x1+x2)/2,(y1+y2)/2)
        if min(x2-x1,y2-y1)<=0 or max(x2-x1,y2-y1)/min(x2-x1,y2-y1)>1.8:continue
        rounded=closest_ball(frame,item,1.2)
        if rounded is None:continue
        rc=np.mean(np.array(rounded['box']).reshape(2,2),axis=0)
        if np.linalg.norm(rc-np.array(center))>.6*max(x2-x1,y2-y1):continue
        if cv2.pointPolygonTest(points,center,False)<0:continue
        ax,ay=max(0,int(x1)),max(0,int(y1))
        bx,by=min(frame.shape[1],int(np.ceil(x2))),min(frame.shape[0],int(np.ceil(y2)))
        if bx<=ax or by<=ay or np.mean(mask[ay:by,ax:bx]>0)<.65:continue
        if any(overlap(item['box'],p['box'])>.2 or
               (x1<=(p['box'][0]+p['box'][2])/2<=x2 and y1<=(p['box'][1]+p['box'][3])/2<=y2)
               for p in special):continue
        if any(overlap(item['box'],p['box'])>.65 for p in balls):continue
        balls.append({**item,'name':'billiard_ball'})
    return [table]+balls+special
