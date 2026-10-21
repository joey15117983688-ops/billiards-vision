import cv2
import numpy as np
import pytest
from pocket_scene import make_scene
from src.vision.pocket import TableDetector, PocketConfig


@pytest.mark.parametrize('perspective,brightness', [(False,1.), (True,1.), (True,.65)])
def test_table_and_four_supported_inner_boundaries(perspective,brightness):
    frame,_=make_scene(perspective=perspective,brightness=brightness)
    table=TableDetector().detect(frame)
    assert table.detected and table.complete, table.reasons
    assert len(table.corners)==4 and len(table.boundary_lines)==4
    assert all(edge['source']=='contour_line_fit' for edge in table.boundary_lines)
    assert min(table.quality['edge_support'])>.6


def test_cropped_table_cannot_be_declared_complete():
    frame,_=make_scene()
    result=TableDetector().detect(frame[210:590,210:990])
    assert result.detected and not result.complete
    assert 'table_touches_image_border' in result.reasons


def test_green_square_and_invalid_input_are_not_tables():
    frame=np.full((700,1000,3),180,np.uint8)
    cv2.rectangle(frame,(200,100),(700,600),(40,150,40),-1)
    assert not TableDetector().detect(frame).detected
    assert not TableDetector().detect(None).detected


def test_cloth_color_is_configurable():
    frame,_=make_scene()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    cloth=cv2.inRange(hsv,np.array([30,35,25]),np.array([95,255,255]))>0
    frame[cloth]=(170,80,40)
    config=PocketConfig(hsv_ranges=(((95,40,25),(125,255,255)),))
    assert TableDetector(config).detect(frame).complete

from src.vision.pocket import PocketDetector, TableGeometry


@pytest.mark.parametrize('perspective,brightness', [(False,1.),(True,1.),(True,.65)])
def test_six_visually_observed_pockets_and_refined_centers(perspective,brightness):
    frame,truth=make_scene(perspective=perspective,brightness=brightness)
    result=PocketDetector().detect(frame)
    assert result.valid, result.reasons
    assert len(result.pockets)==6 and all(p.status=='observed' for p in result.pockets)
    for p in result.pockets:
        assert np.linalg.norm(np.array(p.center)-truth[p.id])<4
        assert p.entry_polygon and p.quality['outside_cloth_fraction']>.15
    # A white/black-ball hole must not cut the playable-area mask.
    assert result.table_mask[370,470]==255
    assert result.table_mask[20,20]==0
    assert cv2.countNonZero(result.pocket_mask)>0


def test_covered_pocket_and_nearby_black_ball_are_not_observed():
    frame,_=make_scene(covered='TOP_MIDDLE',black_ball=True)
    result=PocketDetector().detect(frame)
    assert not result.valid
    pocket=next(p for p in result.pockets if p.id=='TOP_MIDDLE')
    assert pocket.status!='observed'
    assert pocket.reasons


def test_cropped_image_returns_six_explicit_failures_not_six_fake_centers():
    frame,_=make_scene()
    result=PocketDetector().detect(frame[210:590,210:990])
    payload=result.to_dict()
    assert not payload['valid'] and len(payload['failed_pockets'])==6
    assert all(p['center'] is None and p['status']=='missing' for p in payload['pockets'])


def test_homography_roundtrip_and_repeatable_detection():
    frame,_=make_scene(perspective=True)
    table=TableDetector().detect(frame)
    geometry=TableGeometry(table)
    points=np.array([[.1,.2],[.8,.65],[.5,0]],float)
    assert np.allclose(geometry.normalized_points(geometry.image_points(points)),points,atol=1e-5)
    first=PocketDetector().detect(frame);second=PocketDetector().detect(frame)
    assert [p.center for p in first.pockets]==[p.center for p in second.pockets]


def test_blank_scene_returns_structured_failure():
    result=PocketDetector().detect(np.full((480,640,3),128,np.uint8))
    assert not result.valid and not result.table.detected and len(result.pockets)==6


def test_saved_json_masks_and_source_identity(tmp_path):
    import json
    from src.vision.pocket.debug import save_debug_result
    frame,_=make_scene(perspective=True)
    result=PocketDetector().detect(frame)
    result.source={'kind':'image','path':'synthetic-fixture'}
    path=save_debug_result(frame,result,tmp_path)
    data=json.loads(path.read_text())
    assert data['valid'] and data['failed_pockets']==[]
    assert len(data['source']['frame_sha256'])==64
    for name in ['table_mask','pocket_regions','pockets','inner_cushions','pocket_candidates']:
        saved=cv2.imread(data['artifacts'][name],cv2.IMREAD_UNCHANGED)
        assert saved.shape[:2]==frame.shape[:2]
    assert 'not_calibrated' in data['confidence_kind']


def test_synthetic_table_translation_updates_all_pockets():
    frame,_=make_scene()
    shifted=cv2.warpAffine(frame,np.float32([[1,0,40],[0,1,30]]),(1280,800),borderValue=(180,185,190))
    first=PocketDetector().detect(frame)
    second=PocketDetector().detect(shifted)
    assert first.valid and second.valid
    for a,b in zip(first.pockets,second.pockets):
        assert np.allclose(np.array(b.center)-a.center,[40,30],atol=2)


def test_visible_partial_cavities_are_verified_without_full_search_window():
    frame,truth=make_scene()
    # Keep every mouth visible, clipping the outward halves of top pocket cavities.
    crop=frame[190:680,150:1060]
    result=PocketDetector().detect(crop)
    top=[p for p in result.pockets if p.id.startswith('TOP')]
    assert all(p.status=='observed' for p in top)
    assert any(p.quality.get('partial_visibility') for p in top)
    assert result.valid


def test_white_ball_inside_cloth_does_not_replace_occluded_pocket():
    frame,_=make_scene(covered='TOP_MIDDLE')
    cv2.circle(frame,(600,227),12,(245,245,245),-1)
    pocket=next(p for p in PocketDetector().detect(frame).pockets if p.id=='TOP_MIDDLE')
    assert pocket.status!='observed'


def test_user_reported_local_images():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'data/dataset/images/train'
    if not all((root/name).is_file() for name in ['000001.jpg','demo_15.jpg']):
        pytest.skip('Local regression media not checked into Git')
    cropped=PocketDetector().detect(cv2.imread(str(root/'000001.jpg')))
    assert sum(p.status=='observed' for p in cropped.pockets)==5
    assert cropped.pockets[-1].status=='estimated'  # Player covers bottom-right pocket.
    assert all(edge['source']=='cushion_image_line_fit' for edge in cropped.table.boundary_lines)
    demo=PocketDetector().detect(cv2.imread(str(root/'demo_15.jpg')))
    assert demo.valid and all(p.status=='observed' for p in demo.pockets)
    assert next(p for p in demo.pockets if p.id=='TOP_RIGHT').quality['partial_visibility']


def test_reported_right_cushion_and_pockets_across_lighting():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'data/dataset/images/train/1770626791.jpg'
    if not path.exists():pytest.skip('Local annotated regression frame not checked into Git')
    frame=cv2.imread(str(path));anchors=[]
    for brightness in [.55,1.,1.5]:
        image=np.clip(frame.astype(float)*brightness,0,255).astype(np.uint8)
        result=PocketDetector().detect(image)
        assert result.valid,[(p.id,p.status) for p in result.pockets]
        right_corners=np.array(result.table.corners)[[1,2]]
        # The user marked the inner cushion near x=885, not the outer rail at x=920.
        assert np.all(np.abs(right_corners[:,0]-885)<10)
        right=[p for p in result.pockets if 'RIGHT' in p.id]
        assert all(p.quality['basis']=='visible_pocket_net_or_jaw' for p in right)
        assert np.linalg.norm(np.array(right[0].center)-[890,16])<10
        assert np.linalg.norm(np.array(right[1].center)-[892,466])<10
        anchors.append([p.center for p in right])
    assert np.max(np.linalg.norm(np.array(anchors)-anchors[1],axis=2))<3


def test_blue_cloth_does_not_merge_green_surroundings_into_table():
    frame,_=make_scene()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    cloth=cv2.inRange(hsv,np.array([30,35,25]),np.array([95,255,255]))>0
    frame[cloth]=(185,141,48)
    floor=np.all(frame==[180,185,190],axis=2)
    frame[floor]=(65,110,65)
    result=PocketDetector().detect(frame)
    assert result.table.detected and result.table.complete
    corners=np.array(result.table.corners)
    assert corners[:,0].min()>180 and corners[:,0].max()<1020
    assert corners[:,1].min()>180 and corners[:,1].max()<620


@pytest.mark.parametrize('brightness',[.55,1.,1.5])
def test_reported_portrait_frame_excludes_rails_and_finds_both_side_pockets(brightness):
    from pathlib import Path
    frame=cv2.imread(str(Path(__file__).parent/'fixtures/calibration_portrait.png'))
    frame=np.clip(frame.astype(float)*brightness,0,255).astype(np.uint8)
    result=PocketDetector().detect(frame)
    corners=np.array(result.table.playable_polygon)
    assert np.all((corners[[0,3],0]>78)&(corners[[0,3],0]<98)),corners
    assert np.all((corners[[1,2],0]>592)&(corners[[1,2],0]<615)),corners
    if brightness==1.:
        assert result.valid,[(p.id,p.status) for p in result.pockets]
        for anchor in ([71,611],[619,609]):
            assert min(np.linalg.norm(np.array(p.center)-anchor) for p in result.pockets)<12
    # Exposure changes need not make every pocket visually verifiable; never
    # promote an estimate merely because the table edges were fitted.
