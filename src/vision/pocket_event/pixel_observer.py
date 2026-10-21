"""Current-frame ball pixels on a calibrated cloth; no optical predictions."""
import cv2
import numpy as np


class PixelBallObserver:
    def __init__(self,polygon):
        self.polygon=np.asarray(polygon,np.float32)
        self.cached_shape=None

    def observe(self,frame):
        scale=min(1.,960/max(frame.shape[:2]))
        image=cv2.resize(frame,None,fx=scale,fy=scale) if scale<1 else frame
        if image.shape[:2]!=self.cached_shape:
            self.cached_shape=image.shape[:2]
            self.points=self.polygon*scale
            self.mask=np.zeros(image.shape[:2],np.uint8)
            cv2.fillConvexPoly(self.mask,np.rint(self.points).astype(np.int32),255)
            self.mask=cv2.dilate(self.mask,np.ones((9,9),np.uint8))
            self.short=min(np.linalg.norm(np.roll(self.points,-1,axis=0)-self.points,axis=1))
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        samples=hsv[(self.mask>0)&(hsv[:,:,1]>65)&(hsv[:,:,2]>40)]
        if len(samples)<100:return []
        hue=int(np.argmax(np.bincount(samples[:,0],minlength=180)))
        cloth_value=float(np.median(samples[:,2]))
        delta=np.abs(hsv[:,:,0].astype(float)-hue);delta=np.minimum(delta,180-delta)
        binary=(((delta>14)|(hsv[:,:,1]<60)|(hsv[:,:,2]<cloth_value-45))&(self.mask>0)).astype(np.uint8)*255
        binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        found=[]
        for index,contour in enumerate(contours):
            # Only selected foreground components are balls. Hole contours
            # (cloth/mesh inside a bright rim) are not separate objects.
            depth=0;parent=hierarchy[0][index][3]
            while parent>=0:
                depth+=1;parent=hierarchy[0][parent][3]
            if depth%2:continue
            x,y,w,h=cv2.boundingRect(contour);area=cv2.contourArea(contour)
            if min(w,h)<max(4,self.short*.009) or max(w,h)>self.short*.075:continue
            if max(w,h)/min(w,h)>2.8 or area/(w*h)<.5:continue
            perimeter=cv2.arcLength(contour,True)
            if 4*np.pi*area/max(perimeter*perimeter,1)<.4:continue
            m=cv2.moments(contour)
            if not m['m00']:continue
            cx,cy=m['m10']/m['m00'],m['m01']/m['m00']
            if cv2.pointPolygonTest(self.points,(cx,cy),True)<min(w,h)*.42:continue
            region=np.zeros((h,w),np.uint8)
            cv2.drawContours(region,[contour-np.array([[[x,y]]])],-1,255,-1)
            pixels=hsv[y:y+h,x:x+w][region>0]
            saturation,value=np.median(pixels[:,1:],axis=0)
            name='white_ball' if saturation<85 and value>160 else 'black_ball' if value<80 else 'billiard_ball'
            found.append({'name':name,'box':[x/scale,y/scale,(x+w)/scale,(y+h)/scale],
                          'confidence':.85,'method':'current_frame_cloth_pixels'})
        return found
