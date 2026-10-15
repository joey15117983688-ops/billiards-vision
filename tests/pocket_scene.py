"""Synthetic fixtures with known pocket centers, for geometry tests (not real-camera accuracy)."""
import cv2
import numpy as np


def make_scene(perspective=False, covered=None, brightness=1.0, black_ball=False):
    image=np.full((800,1280,3),(180,185,190),np.uint8)
    cv2.rectangle(image,(160,160),(1040,640),(50,80,110),-1)
    cv2.rectangle(image,(200,200),(1000,600),(45,145,45),-1)
    centers={'TOP_LEFT':(190,190),'TOP_MIDDLE':(600,192),'TOP_RIGHT':(1010,190),
             'BOTTOM_LEFT':(190,610),'BOTTOM_MIDDLE':(600,608),'BOTTOM_RIGHT':(1010,610)}
    for name,center in centers.items():
        if name != covered:
            cv2.circle(image,center,24,(8,8,8),-1)
    cv2.circle(image,(470,370),12,(250,250,250),-1)
    cv2.circle(image,(850,430),12,(20,20,220),-1)
    if black_ball:
        cv2.circle(image,(600,224),12,(5,5,5),-1)
    if covered:
        cx,cy=centers[covered]
        cv2.rectangle(image,(cx-35,cy-35),(cx+35,cy+35),(100,150,210),-1)
    if perspective:
        source=np.float32([[200,200],[1000,200],[1000,600],[200,600]])
        target=np.float32([[250,140],[1040,230],[1000,660],[170,590]])
        matrix=cv2.getPerspectiveTransform(source,target)
        image=cv2.warpPerspective(image,matrix,(1280,800),borderValue=(180,185,190))
        names=list(centers)
        transformed=cv2.perspectiveTransform(np.float32([list(centers.values())]),matrix)[0]
        centers={name:tuple(map(float,point)) for name,point in zip(names,transformed)}
    image=np.clip(image.astype(float)*brightness,0,255).astype(np.uint8)
    return image,centers
