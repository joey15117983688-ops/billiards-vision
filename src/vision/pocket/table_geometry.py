"""Homographies between camera pixels and a normalized table plane."""
import cv2
import numpy as np
from .models import PocketConfig

CANDIDATES={'TOP_LEFT':(0.,0.),'TOP_MIDDLE':(.5,0.),'TOP_RIGHT':(1.,0.),
            'BOTTOM_LEFT':(0.,1.),'BOTTOM_MIDDLE':(.5,1.),'BOTTOM_RIGHT':(1.,1.)}


class TableGeometry:
    def __init__(self, table, config=None, alternate=False):
        self.config=config or PocketConfig()
        self.table_confidence=table.confidence
        if not table.detected or not (table.complete or table.quality.get("geometry_reliable",False)) or len(table.corners)!=4:
            raise ValueError('Four supported cushion boundaries are required for calibration')
        q=np.array(table.corners,float)
        edge=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
        longer=0 if edge[0]+edge[2]>=edge[1]+edge[3] else 1
        if self.config.long_axis=='edge_01':longer=0
        elif self.config.long_axis=='edge_12':longer=1
        elif self.config.long_axis!='auto':raise ValueError('long_axis must be auto, edge_01 or edge_12')
        if alternate:longer=1-longer
        q=np.roll(q,-longer,axis=0)
        a,b=(q[0]+q[1])/2,(q[2]+q[3])/2
        use_x=abs(a[1]-b[1])<.04*max(edge)
        if (b[0]<a[0]) if use_x else (b[1]<a[1]):q=np.roll(q,-2,axis=0)
        direction=q[1]-q[0]
        if (direction[0]<0) if abs(direction[0])>=abs(direction[1]) else (direction[1]<0):
            q=q[[1,0,3,2]]
        self.corners=q
        self.to_normalized=cv2.getPerspectiveTransform(q.astype(np.float32),np.float32([[0,0],[1,0],[1,1],[0,1]]))
        inverse_condition, self.to_image = cv2.invert(self.to_normalized, flags=cv2.DECOMP_SVD)
        if abs(cv2.determinant(self.to_normalized))<1e-12 or inverse_condition<1e-10:
            raise ValueError('Unstable perspective mapping')
        self.orientation_method='configured' if self.config.long_axis!='auto' else 'projected_edge_length'
        self.axis_edge=longer

    @staticmethod
    def _map(points, matrix):
        return cv2.perspectiveTransform(np.array(points,np.float32).reshape(1,-1,2),matrix)[0].astype(float)

    def image_points(self, points):return self._map(points,self.to_image)
    def normalized_points(self, points):return self._map(points,self.to_normalized)

    def canonical_to_image(self, points):
        q=np.array(points,float)
        q=(q-self.config.padding)/[self.config.canonical_width,self.config.canonical_height]
        return self.image_points(q)

    def warp(self, frame):
        c=self.config
        canonical=np.array([[c.canonical_width,0,c.padding],[0,c.canonical_height,c.padding],[0,0,1]],float)@self.to_normalized
        size=(c.canonical_width+2*c.padding,c.canonical_height+2*c.padding)
        image=cv2.warpPerspective(frame,canonical,size,borderValue=(127,127,127))
        valid=cv2.warpPerspective(np.full(frame.shape[:2],255,np.uint8),canonical,size,flags=cv2.INTER_NEAREST)
        return image,valid

    def to_dict(self):
        return {'image_to_normalized':self.to_normalized.tolist(),'normalized_to_image':self.to_image.tolist(),
                'ordered_corners':self.corners.tolist(),'canonical_width':self.config.canonical_width,
                'canonical_height':self.config.canonical_height,'padding':self.config.padding,
                'long_axis_edge':self.axis_edge,'orientation_method':self.orientation_method}
