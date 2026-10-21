import json
import cv2
import numpy as np
import pytest
from src.calibration_samples import save_calibration_sample


def profile():
    return {'corners':[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],
            'pockets':[[.1,.1],[.5,.1],[.9,.1],[.1,.9],[.5,.9],[.9,.9]]}


def test_archive_keeps_exact_frame_and_supervision_without_false_ball_labels(tmp_path):
    frame=np.zeros((120,200,3),np.uint8);frame[:]=(30,100,40)
    source={'kind':'video','path':'camera-a.mp4','frame_index':17}
    folder=save_calibration_sample(tmp_path,frame,profile(),source)
    assert np.array_equal(cv2.imread(str(folder/'image.png')),frame)
    data=json.loads((folder/'annotation.json').read_text())
    assert data['source']==source and len(data['pockets'])==6
    assert data['table']['polygon']==profile()['corners']
    assert data['annotation_coverage']['balls']=='unlabeled'
    assert data['training_status']=='collected_not_trained'
    mask=cv2.imread(str(folder/'table_mask.png'),cv2.IMREAD_GRAYSCALE)
    assert mask[60,100]==255 and mask[0,0]==0
    assert save_calibration_sample(tmp_path,frame,profile(),source)==folder
    assert len(list(tmp_path.iterdir()))==1


def test_invalid_annotation_never_becomes_a_training_sample(tmp_path):
    bad=profile();bad['corners']=[[0,0]]*4
    with pytest.raises(ValueError):save_calibration_sample(tmp_path,np.zeros((120,200,3),np.uint8),bad,{})
    assert list(tmp_path.iterdir())==[]
