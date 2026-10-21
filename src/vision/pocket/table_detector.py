"""Find a plausible cloth quadrilateral and fit inner cushion boundary lines, without YOLO."""
import cv2
import numpy as np
from .models import PocketConfig, TableResult


def _line(a, b):
    delta = b-a
    vector = np.array([delta[1], -delta[0]], float)
    vector /= max(np.linalg.norm(vector), 1e-9)
    return np.r_[vector, -vector@a]


def _intersection(a, b):
    determinant = a[0]*b[1]-b[0]*a[1]
    if abs(determinant) < 1e-6:
        return None
    return np.array([(a[1]*b[2]-b[1]*a[2])/determinant,
                     (a[2]*b[0]-b[2]*a[0])/determinant],float)


class TableDetector:
    def __init__(self, config=None):
        self.config = config or PocketConfig()

    def detect(self, frame):
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) < 32:
            return TableResult(reasons=['invalid_bgr_frame'])
        height, width = frame.shape[:2]
        scale = min(1., self.config.max_side/max(height, width))
        image = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_masks=[cv2.inRange(hsv,np.array(low),np.array(high)) for low,high in self.config.hsv_ranges]
        # Select the dominant cloth colour instead of merging blue carpet and
        # green surroundings into one artificial table contour.
        mask=max(color_masks,key=lambda item:cv2.countNonZero(item[int(.15*h):int(.85*h),int(.15*w):int(.85*w)]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cloth_value=float(np.median(hsv[:,:,2][mask>0])) if np.any(mask) else 128.
        normalized_image=np.clip(image.astype(np.float32)*(128/max(cloth_value,1)),0,255).astype(np.uint8)
        normalized_value=cv2.cvtColor(normalized_image,cv2.COLOR_BGR2HSV)[:,:,2]
        edges = (cv2.Canny(cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY), 15, 50) |
                 cv2.Canny(hsv[:,:,0], 10, 35) | cv2.Canny(hsv[:,:,1], 20, 60))
        candidates = []
        rejection = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            if cv2.contourArea(contour) < mask.size*self.config.min_table_area:
                continue
            hull = cv2.convexHull(contour)
            quad = None
            for factor in [.008,.012,.016,.02,.025,.03,.04]:
                approx = cv2.approxPolyDP(hull, factor*cv2.arcLength(hull, True), True)
                if len(approx)==4 and cv2.isContourConvex(approx):
                    quad = approx.reshape(4,2).astype(float)
                    break
            if quad is None:
                rejection.append('no_reliable_quadrilateral')
                continue
            center = quad.mean(0)
            quad = quad[np.argsort(np.arctan2(quad[:,1]-center[1],quad[:,0]-center[0]))]
            lengths = np.linalg.norm(np.roll(quad,-1,axis=0)-quad,axis=1)
            pair = [(lengths[0]+lengths[2])/2, (lengths[1]+lengths[3])/2]
            aspect = max(pair)/max(min(pair),1)
            if not self.config.min_aspect <= aspect <= self.config.max_aspect:
                rejection.append('implausible_table_aspect')
                continue
            angles = []
            for i in range(4):
                a,b=quad[i-1]-quad[i],quad[(i+1)%4]-quad[i]
                angles.append(float(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1)))))
            if min(angles)<25 or max(angles)>155 or max(lengths[0]/lengths[2],lengths[2]/lengths[0],lengths[1]/lengths[3],lengths[3]/lengths[1])>3.5:
                rejection.append('implausible_perspective')
                continue
            quad_mask = np.zeros_like(mask)
            cv2.fillConvexPoly(quad_mask,np.rint(quad).astype(np.int32),255)
            fill = cv2.countNonZero(mask & quad_mask)/max(cv2.countNonZero(quad_mask),1)
            if fill<.75 or cv2.contourArea(quad.astype(np.float32))<.8*cv2.contourArea(hull):
                rejection.append('cloth_shape_not_supported')
                continue
            clipped = bool(np.any(quad[:,0]<max(3,.007*w)) or np.any(quad[:,0]>w-1-max(3,.007*w)) or
                           np.any(quad[:,1]<max(3,.007*h)) or np.any(quad[:,1]>h-1-max(3,.007*h)))
            lines, sources, coverages = [], [], []
            points = contour.reshape(-1,2).astype(float)
            for i in range(4):
                a,b=quad[i],quad[(i+1)%4]
                vector=b-a
                t=(points-a)@vector/max(vector@vector,1)
                distance=np.abs((points-a)@np.array([vector[1],-vector[0]]))/max(np.linalg.norm(vector),1)
                selected=points[(t>.12)&(t<.88)&(distance<max(2.5,.06*min(pair)))]
                if len(selected)>=12:
                    vx,vy,x,y=cv2.fitLine(selected.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel()
                    lines.append(np.array([vy,-vx,vx*y-vy*x],float))
                    sources.append('contour_line_fit')
                else:
                    lines.append(_line(a,b));sources.append('quadrilateral_edge')
                samples=a+np.linspace(.12,.88,60)[:,None]*vector
                coverage=[]
                for x,y in np.rint(samples).astype(int):
                    coverage.append(bool(edges[max(0,y-4):min(h,y+5),max(0,x-4):min(w,x+5)].any()))
                coverages.append(float(np.mean(coverage)))
            scope_corners=[_intersection(lines[i-1],lines[i]) for i in range(4)]
            if any(point is None for point in scope_corners):scope_corners=quad.copy()
            # Colour contours can include cushions even on fully visible tables.
            # Always look for supported inner transitions; a frame-border test
            # cannot distinguish a playing surface from its surrounding rails.
            segments=cv2.HoughLinesP(edges,1,np.pi/720,threshold=45,
                                    minLineLength=max(35,int(.20*min(pair))),maxLineGap=18)
            if segments is not None:
                for i in range(4):
                    a,b=quad[i],quad[(i+1)%4];direction=b-a
                    length=np.linalg.norm(direction);unit=direction/length
                    normal=np.array([-unit[1],unit[0]])
                    choices=[]
                    for raw in segments[:,0]:
                        u,v=raw[:2].astype(float),raw[2:].astype(float)
                        delta=v-u;extent=np.linalg.norm(delta)
                        if extent<.18*length or abs(delta@unit)/extent<.9995:continue
                        midpoint=(u+v)/2;along=(midpoint-a)@unit/length
                        inward=(midpoint-a)@normal
                        if not .12<along<.88 or not 2<inward<.12*min(pair):continue
                        # The adjacent segment must lie inside the cloth rectangle.
                        # Rank continuous cushion transitions, not weak cloth
                        # texture or the outer edge of a printed rail.
                        line=_line(u,v)
                        sample=a+np.linspace(.15,.85,100)[:,None]*direction
                        sample-=((sample@line[:2]+line[2])[:,None])*line[:2]
                        values=[]
                        for offset in (-4,0,4):
                            q=np.rint(sample+offset*normal).astype(int)
                            q[:,0]=np.clip(q[:,0],0,w-1);q[:,1]=np.clip(q[:,1],0,h-1)
                            values.append(normalized_value[q[:,1],q[:,0]].astype(float))
                        transition=float(np.median(np.abs(values[0]-values[2])))
                        ridge=float(np.median((values[0]+values[2])/2-values[1]))
                        strength=max(transition,2*ridge)
                        if strength<8:continue
                        choices.append((inward, strength, u, v))
                    if choices:
                        strongest=max(item[1] for item in choices)
                        # Inner seams can be much fainter than the outer black rail,
                        # especially under clipping/exposure changes. Retain supported
                        # weaker long seams before choosing the innermost boundary.
                        # Oblique views have broad cushion shadows that can mimic
                        # an interior seam. Use the conservative contrast gate there.
                        near_overhead=not clipped and max(abs(angle-90) for angle in angles)<3
                        supported=[item for item in choices if item[1]>=max(8,strongest*(.20 if near_overhead else .55))]
                        _,_,u,v=max(supported,key=lambda item:item[0])
                        seed=_line(u,v)
                        samples=a+np.linspace(.12,.88,100)[:,None]*direction
                        samples-=((samples@seed[:2]+seed[2])[:,None])*seed[:2]
                        offsets=np.arange(-max(3,int(.02*min(pair))),max(3,int(.02*min(pair)))+1)
                        responses=[]
                        for offset in offsets:
                            values=[]
                            for side in (-3,3):
                                q=np.rint(samples+(offset+side)*normal).astype(int)
                                q[:,0]=np.clip(q[:,0],0,w-1);q[:,1]=np.clip(q[:,1],0,h-1)
                                values.append(normalized_value[q[:,1],q[:,0]].astype(float))
                            responses.append(values[1]-values[0])
                        responses=np.array(responses)
                        best=np.argmax(responses,axis=0)
                        strength=responses[best,np.arange(len(samples))]
                        supported=strength>=max(8,float(np.median(strength))*.5)
                        refined=samples+offsets[best,None]*normal
                        if np.count_nonzero(supported)>=40:
                            vx,vy,cx,cy=cv2.fitLine(refined[supported].astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel()
                            fit=np.array([vy,-vx,vx*cy-vy*cx],float)
                            lines[i]=fit
                        else:
                            lines[i]=seed
                        sources[i]='cushion_image_line_fit'
            fitted=[_intersection(lines[i-1],lines[i]) for i in range(4)]
            if all(p is not None for p in fitted) and max(np.linalg.norm(np.array(fitted)-quad,axis=1))<.16*min(pair):
                quad=np.array(fitted)
            coverages=[]
            for i in range(4):
                a,b=quad[i],quad[(i+1)%4]
                samples=a+np.linspace(.12,.88,60)[:,None]*(b-a)
                coverage=[]
                for x,y in np.rint(samples).astype(int):
                    coverage.append(bool(edges[max(0,y-4):min(h,y+5),max(0,x-4):min(w,x+5)].any()))
                coverages.append(float(np.mean(coverage)))
            if not clipped and (np.mean(coverages)<.45 or min(coverages)<.20):
                rejection.append('insufficient_cushion_edge_evidence')
                continue
            # Recheck visibility after line intersections, which can extrapolate a clipped corner.
            clipped = clipped or bool(np.any(quad[:,0]<max(3,.007*w)) or np.any(quad[:,0]>w-1-max(3,.007*w)) or
                                      np.any(quad[:,1]<max(3,.007*h)) or np.any(quad[:,1]>h-1-max(3,.007*h)))
            # Keep geometric points in image space; partial tables are explicitly incomplete.
            corners=(quad/scale).tolist()
            reasons=['table_touches_image_border'] if clipped else []
            if 'quadrilateral_edge' in sources:
                reasons.append('some_boundaries_estimated_from_quadrilateral')
            complete=not clipped and sum(source in ('contour_line_fit','cushion_image_line_fit') for source in sources)==4
            quality={'cloth_fill':float(fill),'projected_aspect':float(aspect),
                     'edge_support':coverages,'interior_angles':angles,
                     'boundary_sources':sources,'geometry_reliable':bool(sum(source in ('contour_line_fit','cushion_image_line_fit') for source in sources)==4 and np.mean(coverages)>=.45 and min(coverages)>=.20),'area_fraction':float(cv2.contourArea(contour)/mask.size)}
            confidence=float(np.clip(.35*fill+.35*np.mean(coverages)+.3*(1 if complete else .4),0,1))
            boundaries=[]
            for i in range(4):
                original=_line(np.array(corners[i]),np.array(corners[(i+1)%4]))
                boundaries.append({'edge':i,'start':corners[i],'end':corners[(i+1)%4],
                                   'line_abc':original.tolist(),'support':coverages[i], 'source':sources[i]})
            original_mask=cv2.resize(mask,(width,height),interpolation=cv2.INTER_NEAREST)
            result=TableResult(True,complete,corners,corners,boundaries,confidence,quality,reasons,original_mask)
            # Recognition scope includes the visible cloth-covered cushion band.
            # Keep the inner-cushion plane unchanged for homography and pocket checks.
            scope=cv2.convexHull(np.vstack([np.array(scope_corners),quad]).astype(np.float32)).reshape(-1,2)/scale
            result.scope_polygon=scope.astype(float).tolist()
            from src.cloth_region import cloth_region
            connected=cloth_region(frame,corners)
            if connected:
                result.scope_polygon=connected['polygon']
                result.quality['scope_color_fill']=connected['color_fill']
                result.quality['scope_connected_coverage']=connected['connected_color_coverage']
            candidates.append(result)
        if not candidates:
            return TableResult(reasons=list(dict.fromkeys(rejection)) or ['no_plausible_cloth_region'])
        return max(candidates,key=lambda result: result.confidence+(.3 if result.complete else 0))
