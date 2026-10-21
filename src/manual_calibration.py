"""Validate user-confirmed normalized geometry without claiming visual detection."""
from itertools import permutations
import numpy as np
from src.table_coordinates import TableCoordinates
from src.vision.pocket.models import TableResult, PocketResult, PocketDetectionResult
from src.vision.pocket.table_geometry import TableGeometry, CANDIDATES


def manual_geometry(frame,profile):
    corners=np.asarray(profile.get('corners'),float)
    centers=np.asarray(profile.get('pockets'),float)
    if corners.shape!=(4,2) or centers.shape!=(6,2) or not np.isfinite(corners).all() or not np.isfinite(centers).all():
        raise ValueError('需要四个内沿角点与六个不同的袋口中心')
    if np.any(corners<0) or np.any(corners>1) or np.any(centers<0) or np.any(centers>1):
        raise ValueError('校准点必须位于画面内')
    scale=np.array([frame.shape[1],frame.shape[0]])
    polygon=TableCoordinates(corners*scale).corners.tolist()
    q=np.asarray(polygon)
    short=min(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1))
    if short<30:raise ValueError('台面区域过小')
    table=TableResult(detected=True,complete=True,corners=polygon,playable_polygon=polygon,
                      scope_polygon=polygon,quality={'geometry_reliable':True,'user_confirmed':True})
    points=centers*scale
    options=[]
    for alternate in (False,True):
        geometry=TableGeometry(table,alternate=alternate)
        candidates=geometry.image_points(list(CANDIDATES.values()))
        costs=np.linalg.norm(points[:,None,:]-candidates[None,:,:],axis=2)
        assignment=min(permutations(range(6)),key=lambda order:sum(costs[i,j] for i,j in enumerate(order)))
        options.append((sum(costs[i,j] for i,j in enumerate(assignment)),geometry,candidates,costs,assignment))
    _,geometry,candidates,costs,assignment=min(options,key=lambda option:option[0])
    if max(costs[i,j] for i,j in enumerate(assignment))>.25*short:
        raise ValueError('袋口位置与台面四角不一致，请检查中袋和角袋')
    pockets=[]
    names=list(CANDIDATES)
    for i,j in enumerate(assignment):
        point=points[i].tolist()
        pockets.append(PocketResult(names[j],status='manual',center=point,candidate_center=candidates[j].tolist(),
                                   normalized_center=geometry.normalized_points([point])[0].tolist(),
                                   reasons=['user_confirmed_position'],quality={'user_confirmed':True}))
    result=PocketDetectionResult(frame.shape[1],frame.shape[0],table,pockets,valid=True,geometry=geometry.to_dict())
    data=result.to_dict();data['failed_pockets']=[]
    from src.cloth_region import cloth_region
    region=cloth_region(frame,polygon)
    if region:data['table']['scope_polygon']=region['polygon']
    data['confidence_kind']='user_confirmed_geometry_not_visual_validation'
    return polygon,data
