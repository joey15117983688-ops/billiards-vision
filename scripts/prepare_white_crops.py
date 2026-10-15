"""Add white-focused training crops; keep validation originals and all visible box labels."""
from pathlib import Path
import json
import shutil
import cv2
import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'data/dataset_white'
    dest=ROOT/'data/dataset_white_focused'
    if dest.exists():
        raise FileExistsError(f'Existing dataset preserved: {dest}')
    shutil.copytree(source,dest)
    manifest=[]
    for path in sorted((source/'images/train').glob('*.jpg')):
        image=cv2.imread(str(path));height,width=image.shape[:2]
        labels=[]
        for line in (source/'labels/train'/path.with_suffix('.txt').name).read_text().splitlines():
            cls,x,y,w,h=map(float,line.split())
            labels.append((int(cls),[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]))
        for index,(cls,box) in enumerate(labels):
            if cls!=2:continue
            cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
            # Half-image crop makes a tiny white ball larger without discarding other labels.
            cw,ch=width//2,height//2
            x0=max(0,min(width-cw,round(cx-cw/2)));y0=max(0,min(height-ch,round(cy-ch/2)))
            crop=image[y0:y0+ch,x0:x0+cw]
            output=[]
            for c,b in labels:
                x1,y1=max(x0,b[0]),max(y0,b[1]);x2,y2=min(x0+cw,b[2]),min(y0+ch,b[3])
                if x2-x1<2 or y2-y1<2:continue
                area=(b[2]-b[0])*(b[3]-b[1])
                if c!=0 and (x2-x1)*(y2-y1)/area<.5:continue
                output.append(f'{c} {((x1+x2)/2-x0)/cw:.8f} {((y1+y2)/2-y0)/ch:.8f} {(x2-x1)/cw:.8f} {(y2-y1)/ch:.8f}')
            stem=f'{path.stem}_white_crop_{index}'
            assert cv2.imwrite(str(dest/'images/train'/(stem+'.jpg')),crop)
            (dest/'labels/train'/(stem+'.txt')).write_text('\n'.join(output)+'\n')
            manifest.append({'crop':stem,'source':path.stem,'rectangle':[x0,y0,x0+cw,y0+ch]})
    for p in (dest/'labels').glob('*.cache'):p.unlink()
    cfg=yaml.safe_load((source/'data.yaml').read_text());cfg['path']=str(dest)
    (dest/'data.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    (dest/'white_crop_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Training-only white crops:',len(manifest),'Training images:',len(list((dest/'images/train').glob('*.jpg'))))
    for p in (source/'labels/val').glob('*.txt'):assert p.read_bytes()==(dest/'labels/val'/p.name).read_bytes()
    print('Validation originals unchanged: PASS')


if __name__=='__main__':main()
