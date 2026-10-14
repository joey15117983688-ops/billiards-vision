"""Convert audited LabelMe boxes and add reviewed table/demo-video labels."""
from pathlib import Path
import json
import shutil
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data' / 'dataset'


def table_box(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 35, 25]), np.array([95, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < image.shape[0] * image.shape[1] * .3:
        raise ValueError('Table annotation requires review')
    x, y, w, h = cv2.boundingRect(contour)
    return [x, y, x+w, y+h]


def demo_balls(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = [cv2.inRange(hsv, np.array([0, 100, 80]), np.array([12, 255, 255])) |
             cv2.inRange(hsv, np.array([170, 100, 80]), np.array([179, 255, 255])),
             cv2.inRange(hsv, np.array([15, 100, 100]), np.array([38, 255, 255]))]
    boxes=[]
    for mask in masks:
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area=cv2.contourArea(c); x,y,w,h=cv2.boundingRect(c)
            perimeter=cv2.arcLength(c,True)
            if (150 < area < 2200 and 0.65 < w/h < 1.5 and
                4*np.pi*area/max(perimeter**2,1) > .6 and w < 65 and h < 65):
                # Color mask misses dark/shaded edges; pad the reviewed box.
                if w <= 45 and h <= 45:
                    boxes.append([max(0,x-3),max(0,y-3),min(image.shape[1],x+w+3),min(image.shape[0],y+h+3)])
    green=cv2.inRange(hsv,np.array([30,35,25]),np.array([95,255,255]))
    contours,_=cv2.findContours(green,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    table=max(contours,key=cv2.contourArea)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    circles=cv2.HoughCircles(cv2.GaussianBlur(gray,(5,5),1),cv2.HOUGH_GRADIENT,
                            dp=1,minDist=23,param1=80,param2=18,minRadius=9,maxRadius=24)
    if circles is not None:
        for x,y,r in circles[0]:
            H,S,V=hsv[round(y),round(x)]
            if not (S>100 and V>100 and (H<=12 or H>=170 or 15<=H<=38)):
                continue
            if cv2.pointPolygonTest(table,(float(x),float(y)),False)<0:
                continue
            if any(np.hypot(x-(b[0]+b[2])/2,y-(b[1]+b[3])/2)<20 for b in boxes):
                continue
            radius=float(r)+3
            boxes.append([max(0,float(x)-radius),max(0,float(y)-radius),
                          min(image.shape[1],float(x)+radius),min(image.shape[0],float(y)+radius)])
    return boxes


def write_sample(stem, image_path, boxes, split):
    image=cv2.imread(str(image_path)); height,width=image.shape[:2]
    target=DATASET/'images'/split/(stem+'.jpg')
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(image_path,target)
    labels=[]
    for cls,(x1,y1,x2,y2) in boxes:
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        labels.append(f'{cls} {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}')
    target=DATASET/'labels'/split/(stem+'.txt');target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text('\n'.join(labels)+'\n')


def main():
    files=sorted((DATASET/'raw').glob('*.json'))
    groups={'december':[],'february':[]}
    for p in files:groups['february' if p.stem.startswith('177') else 'december'].append(p)
    manifest=[]
    table_previews=[]
    for group,items in groups.items():
        cut=round(len(items)*.8)
        for index,p in enumerate(items):
            d=json.loads(p.read_text()); image_path=p.with_suffix('.jpg');im=cv2.imread(str(image_path))
            boxes=[(0,table_box(im))]
            for shape in d['shapes']:
                assert shape['shape_type']=='rectangle'
                points=np.array(shape['points']);x1,y1=points.min(axis=0);x2,y2=points.max(axis=0)
                boxes.append((1,[x1,y1,x2,y2]))
            split='train' if index<cut else 'val'
            write_sample(p.stem,image_path,boxes,split)
            manifest.append({'stem':p.stem,'split':split,'source':group,'ball_count':len(boxes)-1,'table_box':boxes[0][1]})
            thumb=cv2.resize(im,(352,180));x1,y1,x2,y2=boxes[0][1]
            cv2.rectangle(thumb,(int(x1*352/im.shape[1]),int(y1*180/im.shape[0])),(min(351,int(x2*352/im.shape[1])),min(179,int(y2*180/im.shape[0]))),(255,0,255),2)
            cv2.putText(thumb,p.stem,(6,18),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
            table_previews.append(thumb)
    extra=DATASET/'supplemental';extra.mkdir(exist_ok=True)
    cap=cv2.VideoCapture(str(ROOT/'data/billiards_table.mp4'))
    previews=[]
    for sec in [0,3,6,9,12,15,18,21,24,30,33,36]:
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000);ok,im=cap.read();assert ok
        stem=f'demo_{sec:02d}';image_path=extra/(stem+'.jpg');cv2.imwrite(str(image_path),im)
        boxes=[(0,table_box(im))]+[(1,b) for b in demo_balls(im)]
        if sec == 12:  # Visually reviewed yellow ball missed by touching-mask proposals.
            boxes.append((1,[708,240,752,284]))
        split='train' if sec<=24 else 'val'
        write_sample(stem,image_path,boxes,split)
        manifest.append({'stem':stem,'split':split,'source':'demo_video','seconds':sec,'ball_count':len(boxes)-1,'table_box':boxes[0][1]})
        (extra/(stem+'.json')).write_text(json.dumps({'table_box':boxes[0][1],'ball_boxes':[b for cls,b in boxes if cls==1]},indent=2))
        preview=im.copy()
        for cls,(x1,y1,x2,y2) in boxes:cv2.rectangle(preview,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,255) if cls==0 else (0,255,255),3)
        thumb=cv2.resize(preview,(480,270));cv2.putText(thumb,stem+f' balls={len(boxes)-1}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);previews.append(thumb)
    cap.release()
    cv2.imwrite(str(DATASET/'supplemental_review.jpg'),np.vstack([np.hstack(previews[i:i+3]) for i in range(0,12,3)]))
    padded=table_previews+[np.zeros_like(table_previews[0]) for _ in range((-len(table_previews))%7)]
    cv2.imwrite(str(DATASET/'table_labels_review.jpg'),np.vstack([np.hstack(padded[i:i+7]) for i in range(0,len(padded),7)]))
    (DATASET/'data.yaml').write_text(yaml.safe_dump({'path':str(DATASET),'train':'images/train','val':'images/val','names':{0:'billiards_table',1:'billiard_ball'}},sort_keys=False))
    (DATASET/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('train',sum(m['split']=='train' for m in manifest),'val',sum(m['split']=='val' for m in manifest),'balls',sum(m['ball_count'] for m in manifest))
    print('Supplemental ball counts:',[(m['stem'],m['ball_count']) for m in manifest if m['source']=='demo_video'])


if __name__=='__main__':main()
