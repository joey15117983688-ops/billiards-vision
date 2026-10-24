"""Conservative local cavity verification, constrained by the inner cushion geometry."""
# 在透视展开图的几何候选附近，分别验证暗袋腔或明亮袋网/袋衬。
# 暗色斑必须满足接近库口、接触台呢等结构证据，不能把台内黑球当袋口。
# 入口走廊与可见袋腔中心分开：轨迹需要穿过过渡区，不要求球心正好经过袋腔质心。

import cv2
import numpy as np
from .models import PocketResult


class PocketValidator:
    # 保存局部搜索尺度与启发式门槛；评分不视为经过统计校准的概率。
    def __init__(self, config):self.config=config

    # 默认 estimated；有效像素太少时 missing，缺少台呢上下文时保留估算。
    # 暗腔候选检查暗阈值、库外占比、对比度与形状；袋网候选还检查纹理及口部相交。
    # 只有局部证据达标才 observed，再把中心与入口从展开图映射回原图，供显示和事件判断。
    def validate(self, pocket_id, normalized, geometry, warped, valid):
        c=self.config
        point=np.array(normalized,float)*[c.canonical_width,c.canonical_height]+c.padding
        candidate=geometry.image_points([normalized])[0]
        result=PocketResult(pocket_id,'estimated',candidate.tolist(),list(normalized),
                            candidate.tolist(),list(normalized),confidence=min(.49,geometry.table_confidence*.45),
                            quality={"basis":"geometry_only","table_quality":geometry.table_confidence},
                            reasons=['no_verified_cavity_near_geometric_candidate'])
        radius=round(c.canonical_height*c.search_radius_ratio)
        x,y=np.rint(point).astype(int)
        x1,y1=max(0,x-radius),max(0,y-radius)
        x2,y2=min(warped.shape[1],x+radius+1),min(warped.shape[0],y+radius+1)
        roi=warped[y1:y2,x1:x2]
        support=valid[y1:y2,x1:x2]
        if roi.size==0 or np.mean(support>0)<.18:
            result.status='missing';result.center=None;result.normalized_center=None
            result.confidence=0.;result.reasons=['pocket_search_region_not_visible']
            return result
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
        gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        cloth=np.zeros(gray.shape,np.uint8)
        for low,high in c.hsv_ranges:
            cloth|=cv2.inRange(hsv,np.array(low),np.array(high))
        cloth_values=gray[(cloth>0)&(support>0)]
        if len(cloth_values)<20:
            result.reasons=['insufficient_local_cloth_context']
            return result
        threshold=float(np.clip(np.median(cloth_values)*.58,22,75))
        dark=((gray<threshold)&((hsv[:,:,1]<150)|(hsv[:,:,2]<30))&(support>0)).astype(np.uint8)*255
        dark=cv2.morphologyEx(dark,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        dark=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,_=cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        top=normalized[1]==0
        candidates=[]
        for contour in contours:
            area=cv2.contourArea(contour)
            if not (.0003*c.canonical_height**2<area<.35*roi.shape[0]*roi.shape[1]):continue
            bx,by,bw,bh=cv2.boundingRect(contour)
            if bx<=1 or by<=1 or bx+bw>=roi.shape[1]-1 or by+bh>=roi.shape[0]-1:continue
            if max(bw,bh)/max(1,min(bw,bh))>3.5:continue
            moments=cv2.moments(contour)
            if not moments['m00']:continue
            center=np.array([moments['m10']/moments['m00']+x1,moments['m01']/moments['m00']+y1])
            distance=float(np.linalg.norm(center-point))
            if distance>.9*radius:continue
            component=np.zeros_like(dark);cv2.drawContours(component,[contour],-1,255,-1)
            yy,xx=np.nonzero(component)
            global_x,global_y=xx+x1,yy+y1
            inside=(global_x>=c.padding)&(global_x<=c.padding+c.canonical_width)&(global_y>=c.padding)&(global_y<=c.padding+c.canonical_height)
            outside_fraction=float(1-np.mean(inside))
            # A black ball wholly on the cloth cannot satisfy this cushion-opening evidence.
            dilated=cv2.dilate(component,np.ones((9,9),np.uint8))
            clipped_component=bool(np.any((dilated>0)&(support==0)))
            if outside_fraction<(.08 if clipped_component else .15):continue
            offset=center-point
            tolerance=(.045 if clipped_component else .025)*c.canonical_height
            top=normalized[1]==0
            boundary_y=offset[1] if top else -offset[1]
            if boundary_y>tolerance:continue
            corner=normalized[0] in (0,1)
            if corner:
                boundary_x=offset[0] if normalized[0]==0 else -offset[0]
                if boundary_x>tolerance:continue
            nearest=float(np.min(np.linalg.norm(np.c_[global_x,global_y]-point,axis=1)))
            if nearest>.035*c.canonical_height:continue
            cloth_contact=int(np.count_nonzero((dilated>0)&(cloth>0)))
            if cloth_contact<6:continue
            ring=(cv2.dilate(component,np.ones((13,13),np.uint8))>0)&(component==0)&(support>0)
            contrast=float(np.median(gray[ring])-np.median(gray[component>0])) if ring.any() else 0.
            if contrast<15:continue
            solidity=float(area/max(cv2.contourArea(cv2.convexHull(contour)),1))
            if solidity<.55:continue
            score=float(np.clip(.3*(1-distance/radius)+.25*min(contrast/60,1)+.25*solidity+.2*min(cloth_contact/50,1),0,1))
            if score < c.observed_quality_threshold:continue
            candidates.append((score,center,area,{'distance_from_candidate_canonical':distance,
                'outside_cloth_fraction':outside_fraction,'cloth_contact_pixels':cloth_contact,
                'contrast_gray':contrast,'solidity':solidity,'dark_threshold':threshold,'partial_visibility':clipped_component,'visible_search_fraction':float(np.mean(support>0)),'basis':'local_cavity_and_cushion_geometry'}))
        # Bright, desaturated pocket nets / jaw liners are common in broadcast
        # crops where the dark cavity itself lies outside the frame. Require a
        # compact, textured component at the cushion mouth, not any bright ball.
        local_value=float(np.median(hsv[:,:,2][(cloth>0)&(support>0)]))
        liner=((hsv[:,:,1]<95)&(hsv[:,:,2]>max(40,local_value*.9))&(support>0)).astype(np.uint8)*255
        rows,cols=np.indices(liner.shape)
        local_distance=np.hypot(cols+x1-point[0],rows+y1-point[1])
        liner[local_distance>.6*radius]=0
        liner=cv2.morphologyEx(liner,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        liner=cv2.morphologyEx(liner,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        outlines,_=cv2.findContours(liner,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in outlines:
            area=cv2.contourArea(contour)
            if not .0002*c.canonical_height**2<area<.006*c.canonical_height**2:continue
            bx,by,bw,bh=cv2.boundingRect(contour)
            if max(bw,bh)/max(1,min(bw,bh))>2.3:continue
            moments=cv2.moments(contour)
            if not moments['m00']:continue
            center=np.array([moments['m10']/moments['m00']+x1,moments['m01']/moments['m00']+y1])
            distance=float(np.linalg.norm(center-point))
            if distance>.45*radius:continue
            component=np.zeros_like(liner);cv2.drawContours(component,[contour],-1,255,-1)
            yy,xx=np.nonzero(component);gx,gy=xx+x1,yy+y1
            # Weight visible liner evidence near the cushion intersection so
            # connected outer rail trim cannot drag the mouth position outward.
            weights=np.exp(-((gx-point[0])**2+(gy-point[1])**2)/(2*(.25*radius)**2))
            center=np.array([np.average(gx,weights=weights),np.average(gy,weights=weights)])
            distance=float(np.linalg.norm(center-point))
            inside=(gx>=c.padding)&(gx<=c.padding+c.canonical_width)&(gy>=c.padding)&(gy<=c.padding+c.canonical_height)
            outside_fraction=float(1-np.mean(inside))
            if outside_fraction<.12:continue
            # A corner liner must enter the playable side of both cushions;
            # an entirely exterior cuff/rail patch is not a pocket mouth.
            if normalized[0] in (0,1) and np.count_nonzero(inside)<6:continue
            nearest=float(np.min(np.linalg.norm(np.c_[gx,gy]-point,axis=1)))
            if nearest>.025*c.canonical_height:continue
            texture=float(np.std(gray[component>0])*128/max(local_value,1))
            if texture<8:continue
            contact=int(np.count_nonzero((cv2.dilate(component,np.ones((9,9),np.uint8))>0)&(cloth>0)))
            if contact<12:continue
            score=float(np.clip(.35*(1-distance/radius)+.25*min(texture/25,1)+.2*min(contact/50,1)+.2,0,1))
            if score<c.observed_quality_threshold:continue
            candidates.append((score,center,area,{'basis':'visible_pocket_net_or_jaw',
                'outside_cloth_fraction':outside_fraction,'cloth_contact_pixels':contact,
                'texture_gray_std':texture,'partial_visibility':bool(np.any((cv2.dilate(component,np.ones((5,5),np.uint8))>0)&(support==0))),
                'visible_search_fraction':float(np.mean(support>0))}))
        # 只在暗腔/袋网的局部结构证据通过时把 estimated 升级为 observed。
        if candidates:
            score,center,area,quality=max(candidates,key=lambda item:item[0])
            result.status='observed';result.center=geometry.canonical_to_image([center])[0].tolist()
            result.normalized_center=geometry.normalized_points([result.center])[0].tolist()
            quality['center_kind']='visible_cavity_centroid' if quality['basis']=='local_cavity_and_cushion_geometry' else 'visible_net_or_jaw_centroid'
            result.confidence=score;result.quality=quality;result.reasons=[]
        else:
            center=point
            area=(.035*c.canonical_height)**2*np.pi
        # Separate entrance corridor: retain the cavity and the transition into the cloth.
        inward=np.array([0 if normalized[0]==.5 else (1 if normalized[0]==0 else -1),1 if top else -1],float)
        inward/=np.linalg.norm(inward)
        tangent=np.array([-inward[1],inward[0]])
        half_width=float(np.clip(np.sqrt(area/np.pi)*.75,.025*c.canonical_height,.06*c.canonical_height))
        inner=point+inward*(.035*c.canonical_height)
        outer=center-inward*(.015*c.canonical_height)
        entry=[outer-tangent*half_width,outer+tangent*half_width,inner+tangent*half_width,inner-tangent*half_width]
        result.entry_polygon=geometry.canonical_to_image(entry).tolist()
        return result
