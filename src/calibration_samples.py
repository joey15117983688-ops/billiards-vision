"""Archive exact human-labeled frames for future supervised geometry training."""
# 已有人工标注样本的本地归档，不执行训练，也不把未标注的球体当作负样本。
# 图像与标注摘要去重，临时目录准备完整后再发布，避免留下可被误读为完整的样本。

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import cv2
import numpy as np
from src.manual_calibration import manual_geometry


# 先验证人工几何，再保存原帧、台面 mask 与归一化标注；明确 balls=unlabeled。
# 内容已存在时复用；保存异常时清理临时目录，不上传图像或来源信息。
def save_calibration_sample(root,frame,profile,source):
    polygon,pockets=manual_geometry(frame,profile)
    height,width=frame.shape[:2]
    labels={'schema_version':1,'label_origin':'human_confirmed',
            'image':{'file':'image.png','width':width,'height':height},
            'source':source,
            'table':{'polygon':(np.asarray(polygon)/[width,height]).tolist(),'mask_file':'table_mask.png'},
            'pockets':[{'id':p['id'],'center':[p['center']['x']/width,p['center']['y']/height]}
                       for p in sorted(pockets['pockets'],key=lambda p:p['id'])],
            'annotation_coverage':{'table':'complete','pocket_centers':'complete','balls':'unlabeled'},
            'training_status':'collected_not_trained',
            'coordinate_system':'normalized_image_xy'}
    encoded=json.dumps(labels,ensure_ascii=False,sort_keys=True,indent=2).encode('utf-8')
    identity=hashlib.sha256(frame.tobytes()+encoded).hexdigest()
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    target=root/identity
    if (target/'annotation.json').is_file():return target
    temporary=Path(tempfile.mkdtemp(prefix='.sample-',dir=root))
    try:
        mask=np.zeros((height,width),np.uint8)
        cv2.fillConvexPoly(mask,np.rint(polygon).astype(np.int32),255)
        for name,image in [('image.png',frame),('table_mask.png',mask)]:
            if not cv2.imwrite(str(temporary/name),image):raise OSError(f'无法保存 {name}')
        (temporary/'annotation.json').write_bytes(encoded)
        if target.exists():
            if (target/'annotation.json').is_file():return target
            raise OSError('学习样本目录不完整')
        temporary.replace(target)
        return target
    finally:
        if temporary.exists():shutil.rmtree(temporary)
