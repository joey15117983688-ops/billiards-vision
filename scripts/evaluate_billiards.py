"""Validate trained weights and inspect held-out demo frames; no label remapping."""
# 已有模型离线评估工具，部署门槛与界面共享；验证集成绩不能外推成真实球房准确率。

from pathlib import Path
import argparse,json,os,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.billiards_inference import CLASS_CONFIDENCE, filter_detections
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('YOLO_AUTOINSTALL','false')
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'logs'/'ultralytics'))


# 加载指定权重/数据划分，统计验证及原片抽样预测并写本地报告，不训练模型。
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=ROOT/'data/dataset/data.yaml')
    parser.add_argument('--report-name',default='billiards_evaluation')
    args=parser.parse_args()
    import cv2,torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model=YOLO(str(args.model))
    assert set(model.names.values()) in ({'billiards_table','billiard_ball'}, {'billiards_table','billiard_ball','white_ball'})
    metrics=model.val(data=str(args.data),imgsz=768,batch=2,
                      device='cpu',workers=0,plots=True,
                      project=str(ROOT/'runs/val'),name='billiards_v1')
    report={'model':str(args.model),'names':model.names,'metrics':metrics.results_dict,'per_class':{model.names[int(c)]:dict(zip(['precision','recall','mAP50','mAP50_95'],map(float,metrics.box.class_result(i)))) for i,c in enumerate(metrics.box.ap_class_index)},'inference_confidence':CLASS_CONFIDENCE,'frames':[]}
    out=ROOT/'logs'/(args.report_name+'_predictions');out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(ROOT/'data/billiards_table.mp4'))
    for sec in [0,5,27,30,33,36]:
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000);ok,image=cap.read();assert ok
        result=model.predict(image,device='cpu',imgsz=960,conf=.25,verbose=False)[0]
        raw_counts={name:sum(model.names[int(b.cls[0])]==name for b in result.boxes) for name in model.names.values()}
        result=filter_detections(result)
        objects=[{'class':model.names[int(b.cls[0])],'confidence':float(b.conf[0]),'box':b.xyxy[0].tolist()} for b in result.boxes]
        report['frames'].append({'seconds':sec,'raw_counts':raw_counts,'objects':objects})
        cv2.imwrite(str(out/f'frame_{sec:02d}.jpg'),result.plot())
        print('Video seconds:',sec,'tables:',sum(o['class']=='billiards_table' for o in objects),'balls:',sum(o['class']=='billiard_ball' for o in objects))
    cap.release()
    (ROOT/'logs'/(args.report_name+'.json')).write_text(json.dumps(report,indent=2))
    print('Validation:',metrics.results_dict)


if __name__=='__main__':main()
