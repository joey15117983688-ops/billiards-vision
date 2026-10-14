"""Validate trained weights and inspect held-out demo frames; no label remapping."""
from pathlib import Path
import argparse,json,os

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('YOLO_AUTOINSTALL','false')
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'logs'/'ultralytics'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    args=parser.parse_args()
    import cv2,torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model=YOLO(str(args.model))
    assert set(model.names.values())=={'billiards_table','billiard_ball'}
    metrics=model.val(data=str(ROOT/'data/dataset/data.yaml'),imgsz=768,batch=4,
                      device='cpu',workers=0,plots=True,
                      project=str(ROOT/'runs/val'),name='billiards_v1')
    report={'model':str(args.model),'names':model.names,'metrics':metrics.results_dict,'frames':[]}
    out=ROOT/'logs'/'trained_predictions';out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(ROOT/'data/billiards_table.mp4'))
    for sec in [0,5,27,30,33,36]:
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000);ok,image=cap.read();assert ok
        result=model.predict(image,device='cpu',imgsz=960,conf=.25,verbose=False)[0]
        objects=[{'class':model.names[int(b.cls[0])],'confidence':float(b.conf[0]),'box':b.xyxy[0].tolist()} for b in result.boxes]
        report['frames'].append({'seconds':sec,'objects':objects})
        cv2.imwrite(str(out/f'frame_{sec:02d}.jpg'),result.plot())
        print('Video seconds:',sec,'tables:',sum(o['class']=='billiards_table' for o in objects),'balls:',sum(o['class']=='billiard_ball' for o in objects))
    cap.release()
    (ROOT/'logs'/'billiards_evaluation.json').write_text(json.dumps(report,indent=2))
    print('Validation:',metrics.results_dict)


if __name__=='__main__':main()
