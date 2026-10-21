"""Archive exact human-labeled frames for future supervised geometry training."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import cv2
import numpy as np
from src.manual_calibration import manual_geometry


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
