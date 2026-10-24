"""Bounded image registration; a saved calibration stays the immutable baseline."""
# 参考图配准与几何重投影；基准坐标和当前显示坐标必须分开保存。
# 先尝试固定参考光流，再用 ORB 回退；只有覆盖足够台面范围的匹配才可产生变换。
# lost 只描述配准失败；本模块不负责进球许可，也不自动写入校准文件。

from copy import deepcopy
import base64
import cv2
import numpy as np


# 将参考画面缩为最长边 720 的灰度 JPEG 再编码，减少持久化校准体积。
def encode_reference(frame):
    scale=min(1.,720/max(frame.shape[:2]))
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if scale<1:gray=cv2.resize(gray,None,fx=scale,fy=scale)
    ok,data=cv2.imencode('.jpg',gray,[cv2.IMWRITE_JPEG_QUALITY,92])
    if not ok:raise ValueError('无法保存校准参考画面')
    return base64.b64encode(data).decode('ascii')


# 以 3×3 单应矩阵批量映射像素点；空点集保持空，不生成默认坐标。
def map_points(points,matrix):
    if not len(points):return []
    return cv2.perspectiveTransform(np.float32([points]),np.asarray(matrix,float))[0].tolist()


# 深拷贝袋口数据，统一变换内库、范围、入口和双向坐标矩阵。
# 从固定基准投影，不对上一帧已变换结果反复累乘，也不修改调用方的基准对象。
def transform_geometry(polygon,scope,pockets,matrix):
    """Reproject copies; never progressively transform the already moved result."""
    polygon=map_points(polygon,matrix);scope=map_points(scope,matrix)
    result=deepcopy(pockets)
    if result is None:return polygon,scope,None
    table=result['table'];table.update(corners=polygon,playable_polygon=polygon,scope_polygon=scope,boundary_lines=[])
    for pocket in result['pockets']:
        for key in ('center','candidate_center'):
            point=pocket.get(key)
            if point:
                xy=map_points([[point['x'],point['y']]],matrix)[0]
                pocket[key]={'x':xy[0],'y':xy[1]}
        pocket['entry_polygon']=map_points(pocket.get('entry_polygon',[]),matrix)
    geometry=result.get('geometry')
    if geometry:
        geometry['image_to_normalized']=(np.asarray(geometry['image_to_normalized'])@np.linalg.inv(matrix)).tolist()
        geometry['normalized_to_image']=(matrix@np.asarray(geometry['normalized_to_image'])).tolist()
        geometry['ordered_corners']=map_points(geometry['ordered_corners'],matrix)
    return polygon,scope,result


class CameraAlignment:
    # 在台面边界带采集参考角点和 ORB 描述符，尽量避免把移动球当机位参照。
    def __init__(self,frame,scope,reference=None):
        self.shape=frame.shape[:2];self.scale=min(1.,720/max(self.shape))
        self.matrix=np.eye(3);self.tick=0;self.lost=False
        self.scope=np.asarray(scope,float)
        if reference:
            raw=np.frombuffer(base64.b64decode(reference,validate=True),np.uint8)
            gray=cv2.imdecode(raw,cv2.IMREAD_GRAYSCALE)
            if gray is None:raise ValueError('校准参考画面无法读取')
            # Reference images and restored polygons share the original aspect.
            size=(round(self.shape[1]*self.scale),round(self.shape[0]*self.scale))
            gray=cv2.resize(gray,size)
        else:gray=self.gray(frame)
        self.reference=gray
        q=np.rint(self.scope*self.scale).astype(np.int32)
        mask=np.zeros(gray.shape,np.uint8);cv2.fillConvexPoly(mask,q,255)
        short=min(np.linalg.norm(q[(i+1)%4]-q[i]) for i in range(4))
        radius=max(3,int(short*.12));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,)*2)
        mask=cv2.dilate(mask,kernel)-cv2.erode(mask,kernel)
        self.mask=mask
        self.orb=cv2.ORB_create(nfeatures=1400,edgeThreshold=9,fastThreshold=10)
        self.keys,self.descriptors=self.orb.detectAndCompute(gray,mask)
        self.points=cv2.goodFeaturesToTrack(gray,300,.015,6,mask=mask,blockSize=5)

    # 按构造时的尺寸比例生成灰度帧，保证参考图和当前图具有相同配准尺度。
    def gray(self,frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray,(round(self.shape[1]*self.scale),round(self.shape[0]*self.scale)))

    # 通过 RANSAC 拟合受限相似变换，并检查内点比例、四象限覆盖、尺度/角度和最大位移。
    def estimate(self,source,target):
        if len(source)<12:return None
        source=np.asarray(source,np.float32).reshape(-1,2);target=np.asarray(target,np.float32).reshape(-1,2)
        affine,inliers=cv2.estimateAffinePartial2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=1.5,maxIters=800,confidence=.995)
        if affine is None:return None
        keep=inliers.ravel()>0
        if keep.sum()<12 or keep.mean()<.5:return None
        # Evidence must span the table, rather than one arm or a moving ball.
        extent=np.ptp(source[keep],axis=0);table_extent=np.ptp(self.scope*self.scale,axis=0)
        if np.any(extent<table_extent*.55):return None
        center=np.mean(self.scope*self.scale,axis=0)
        quadrants=(source[keep,0]>center[0]).astype(int)+2*(source[keep,1]>center[1]).astype(int)
        if len(np.unique(quadrants))<3:return None
        scale=np.hypot(affine[0,0],affine[1,0]);angle=abs(np.arctan2(affine[1,0],affine[0,0]))
        if not .7<scale<1.4 or angle>np.deg2rad(8):return None
        matrix=np.vstack([affine,[0,0,1]]);matrix[:2,2]/=self.scale
        moved=np.asarray(map_points(self.scope,matrix))
        if np.max(np.linalg.norm(moved-self.scope,axis=1))>min(self.shape)*.25:return None
        return matrix

    # 默认每四帧实际检查一次；光流采用前后向一致性验证，失败再匹配 ORB。
    # 失败保留先前矩阵并设 lost；微小变化低于门槛时不发布新的几何，防止显示抖动。
    def update(self,frame,force=False):
        self.tick+=1
        if frame.shape[:2]!=self.shape:self.lost=True;return None
        if not force and self.tick%4:return None
        gray=self.gray(frame);matrix=None
        # Fixed-reference flow is cheap and cannot accumulate frame-to-frame drift.
        if self.points is not None:
            initial=np.float32(map_points((self.points.reshape(-1,2)/self.scale).tolist(),self.matrix))*self.scale
            found,ok,error=cv2.calcOpticalFlowPyrLK(self.reference,gray,self.points,initial.reshape(-1,1,2),winSize=(25,25),maxLevel=3,flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
            if found is not None:
                back,valid,_=cv2.calcOpticalFlowPyrLK(gray,self.reference,found,None,winSize=(25,25),maxLevel=3)
                keep=(ok.ravel()>0)&(valid.ravel()>0)&(np.linalg.norm(back-self.points,axis=2).ravel()<1.)&(error.ravel()<25)
                matrix=self.estimate(self.points[keep],found[keep])
        if matrix is None and self.descriptors is not None:
            keys,descriptors=self.orb.detectAndCompute(gray,None)
            if descriptors is not None:
                pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.descriptors,descriptors,k=2)
                matches=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.72*b.distance]
                matrix=self.estimate([self.keys[m.queryIdx].pt for m in matches],[keys[m.trainIdx].pt for m in matches])
        self.lost=matrix is None
        if matrix is None:return None
        delta=np.max(np.linalg.norm(np.asarray(map_points(self.scope,matrix))-np.asarray(map_points(self.scope,self.matrix)),axis=1))
        if delta<.6:return None
        self.matrix=matrix
        return matrix.copy()
