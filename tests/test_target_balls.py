import cv2
import numpy as np
from pocket_scene import make_scene
from src.target_balls import target_balls, playing_surface, surface_mask


def test_one_black_one_white_and_no_outside_targets():
    frame,_=make_scene()
    cv2.circle(frame,(750,400),12,(12,12,12),-1)
    cv2.circle(frame,(80,300),16,(8,8,8),-1)
    cv2.circle(frame,(1150,400),16,(250,250,250),-1)
    found=target_balls(frame)
    assert {item['name'] for item in found}=={'black_ball','white_ball'}
    assert len(found)==2
    for item in found:
        x1,y1,x2,y2=item['box']
        assert 200<x1<x2<1000 and 200<y1<y2<600


def test_pockets_cues_and_colored_balls_do_not_create_targets():
    frame,_=make_scene()
    cv2.circle(frame,(470,370),15,(45,145,45),-1)  # Remove white ball.
    cv2.circle(frame,(650,350),16,(10,220,235),-1)
    cv2.circle(frame,(645,345),5,(245,245,245),-1)
    cv2.line(frame,(300,300),(700,500),(240,240,240),5)
    assert target_balls(frame)==[]


def test_no_table_produces_no_visible_pixels_or_targets():
    frame=np.full((500,900,3),180,np.uint8)
    cv2.circle(frame,(300,300),15,(8,8,8),-1)
    assert playing_surface(frame) is None
    assert not surface_mask(frame,None).any()
    assert target_balls(frame)==[]


def test_mask_keeps_balls_and_hides_outside_slanted_table():
    frame,_=make_scene(perspective=True)
    polygon=playing_surface(frame)
    mask=surface_mask(frame,polygon)
    assert mask[350,600]==255 and mask[20,20]==0
    visible=cv2.bitwise_and(frame,frame,mask=mask)
    assert not visible[20,20].any()
    assert np.array_equal(visible[350,600],frame[350,600])


def test_local_black_and_white_regression():
    from pathlib import Path
    import pytest
    path=Path(__file__).resolve().parents[1]/'data/dataset/images/train/000001.jpg'
    if not path.exists():pytest.skip('Local media not checked in')
    found=target_balls(cv2.imread(str(path)))
    assert {item['name'] for item in found}=={'black_ball','white_ball'}
    centers={item['name']:np.mean(np.array(item['box']).reshape(2,2),axis=0) for item in found}
    assert np.linalg.norm(centers['black_ball']-[1706,438])<10
    assert np.linalg.norm(centers['white_ball']-[732,98])<10


def test_scene_retains_table_colored_balls_and_refines_black_white():
    from src.target_balls import scene_detections
    frame,_=make_scene()
    cv2.circle(frame,(750,400),12,(12,12,12),-1)
    raw=[{'name':'billiard_ball','box':[838,418,862,442],'confidence':.9},
         {'name':'white_ball','box':[458,358,482,382],'confidence':.9},
         {'name':'billiard_ball','box':[738,388,762,412],'confidence':.9},
         {'name':'billiard_ball','box':[60,60,90,90],'confidence':.9},
         {'name':'billiards_table','box':[140,140,1060,660],'confidence':.95}]
    found=scene_detections(frame,raw)
    assert [p['name'] for p in found].count('billiards_table')==1
    assert [p['name'] for p in found].count('billiard_ball')==1
    assert [p['name'] for p in found].count('white_ball')==1
    assert [p['name'] for p in found].count('black_ball')==1
    assert found[0]['polygon']
    assert all(item['box'][0]>100 for item in found)


def test_scene_reports_table_when_model_has_no_table_proposal():
    from src.target_balls import scene_detections
    frame,_=make_scene(perspective=True)
    found=scene_detections(frame,[])
    assert found[0]['name']=='billiards_table' and found[0]['polygon']
    assert found[0]['method']=='cloth_geometry'


def test_same_color_cushion_band_is_in_recognition_but_outside_coordinate_plane(monkeypatch):
    from pathlib import Path
    import pytest
    from src.vision.pocket import TableDetector
    from src.table_coordinates import TableCoordinates
    from src.target_balls import scene_detections
    path=Path(__file__).resolve().parents[1]/'data/dataset/images/train/1770626791.jpg'
    if not path.exists():pytest.skip('Local regression frame not checked into Git')
    frame=cv2.imread(str(path));table=TableDetector().detect(frame)
    assert cv2.pointPolygonTest(np.array(table.scope_polygon,np.float32),(905,240),False)>=0
    assert TableCoordinates(table.playable_polygon).millimetres(905,240) is None
    raw=[{'name':'billiard_ball','box':[898,234,910,246],'confidence':.9},
         {'name':'billiard_ball','box':[940,234,952,246],'confidence':.9}]
    # Isolate scope filtering from the independent image-shape validation.
    monkeypatch.setattr('src.target_balls.closest_ball',lambda frame,item,radius_factor:item)
    found=scene_detections(frame,raw)
    assert any(d['name']=='billiard_ball' and d['box']==raw[0]['box'] for d in found)
    assert not any(d.get('box')==raw[1]['box'] for d in found)
    assert found[0]['geometry']=='connected_cloth_scope'
    payload=table.to_dict()
    assert payload['scope_polygon'] and payload['playable_polygon']
