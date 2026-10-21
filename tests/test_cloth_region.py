from pathlib import Path
import cv2
import numpy as np
import pytest
from src.cloth_region import cloth_region
from src.target_balls import scene_detections
from src.table_coordinates import TableCoordinates

ROOT=Path(__file__).resolve().parents[1]


def inside(polygon,point):
    return cv2.pointPolygonTest(np.array(polygon,np.float32),point,False)>=0


@pytest.mark.parametrize('hue',[60,100])
def test_same_color_bands_and_shadows_remain_inside_but_frame_and_floor_do_not(hue):
    hsv=np.zeros((600,800,3),np.uint8);hsv[:]=(hue,200,170) # Same-colour floor.
    frame=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.rectangle(frame,(70,70),(730,530),(15,15,15),-1)
    cloth=cv2.cvtColor(np.uint8([[[hue,220,180]]]),cv2.COLOR_HSV2BGR)[0,0].tolist()
    shaded=cv2.cvtColor(np.uint8([[[hue,220,75]]]),cv2.COLOR_HSV2BGR)[0,0].tolist()
    cv2.rectangle(frame,(100,100),(700,500),shaded,-1)
    cv2.rectangle(frame,(125,125),(675,475),cloth,-1)
    cv2.circle(frame,(350,300),12,(255,255,255),-1)
    cv2.line(frame,(230,490),(260,360),(180,150,110),4)
    inner=[[125,125],[675,125],[675,475],[125,475]]
    region=cloth_region(frame,inner)
    assert region['connected_color_coverage']>.97
    for point in [(110,300),(690,300),(400,110),(400,490)]:assert inside(region['polygon'],point)
    for point in [(85,300),(715,300),(400,85),(400,515),(30,30)]:assert not inside(region['polygon'],point)
    result=scene_detections(frame,[],inner)[0]
    assert result['polygon']==region['polygon']
    assert result['playable_polygon']==inner
    assert TableCoordinates(inner).millimetres(110,300) is None


def video_frame(name):
    path=ROOT/'data'/name
    if not path.exists():pytest.skip('Local regression video unavailable')
    capture=cv2.VideoCapture(str(path));ok,frame=capture.read();capture.release()
    assert ok
    return frame


def test_reported_blue_video_recovers_full_colored_bands_from_shrunken_cache():
    frame=video_frame('overhead_pool.mp4')
    inner=[[171,102],[860,102],[856,479],[171,476]]
    region=cloth_region(frame,inner)
    for point in [(160,250),(870,250),(500,105),(500,473)]:assert inside(region['polygon'],point)
    for point in [(140,250),(900,250),(500,90),(500,495)]:assert not inside(region['polygon'],point)
    assert region['connected_color_coverage']>.97
    from scripts.yolo_click_worker import detect_selected
    table=detect_selected(None,frame,{'name':'billiards_table'},inner)[0]
    assert table['polygon']==region['polygon'] and table['playable_polygon']==inner


def test_reported_portrait_video_keeps_green_edges_without_black_rails():
    frame=video_frame('test.mp4')
    inner=[[138,149],[929,149],[936,1762],[143,1762]]
    region=cloth_region(frame,inner)
    for point in [(112,700),(950,700),(500,120),(500,1790)]:assert inside(region['polygon'],point)
    for point in [(70,700),(995,700),(500,75),(500,1840)]:assert not inside(region['polygon'],point)
    table=scene_detections(frame,[],inner)[0]
    assert inside(table['polygon'],(112,700))
    assert not inside(table['playable_polygon'],(112,700))


def test_reported_cropped_green_image_does_not_invent_a_smaller_table():
    path=ROOT/'data/dataset/images/train/0000.jpg'
    if not path.exists():pytest.skip('Local regression image unavailable')
    frame=cv2.imread(str(path));inner=[[19,27],[1734,26],[1733,865],[18,864]]
    region=cloth_region(frame,inner)
    assert inside(region['polygon'],(5,300))
    assert inside(region['polygon'],(1750,300))
    assert inside(region['polygon'],(800,5))
    assert region['connected_color_coverage']>.98


def test_old_cached_inner_outline_restores_scope_and_aligned_display_coordinates(tmp_path,monkeypatch):
    import hashlib
    import main
    from PyQt5.QtWidgets import QApplication
    from src.vision.pocket import PocketDetector,TableResult
    from src.calibration_state import save_automatic
    frame=video_frame('overhead_pool.mp4');inner=[[171,102],[860,102],[856,479],[171,476]]
    table=TableResult(detected=True,complete=True,corners=inner,playable_polygon=inner,scope_polygon=inner,
                      quality={'geometry_reliable':True})
    pockets=PocketDetector().detect(frame,table=table).to_dict()
    source=tmp_path/'blue.png';cv2.imwrite(str(source),frame)
    key=hashlib.sha256(str(source.resolve()).encode()).hexdigest()
    save_automatic(tmp_path/'config/calibration'/f'{key}.auto.json',frame,
                   {'polygon':inner,'pockets':pockets,'calibration_report':{'decision':'initial'}},None)
    # Model an old cache created before frozen scopes were persisted.
    import json
    cache=tmp_path/'config/calibration'/f'{key}.auto.json'
    legacy=json.loads(cache.read_text());legacy.pop('scope_frozen',None)
    legacy['pockets']['table'].pop('scope_polygon',None)
    cache.write_text(json.dumps(legacy))
    app=QApplication.instance() or QApplication([]);monkeypatch.setattr(main,'ROOT',tmp_path)
    window=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    assert np.allclose(window.canvas.calibration_polygon,inner)
    assert inside(window.canvas.recognition_polygon,(160,250))
    assert not inside(window.canvas.recognition_polygon,(140,250))
    assert window.canvas.coordinates.millimetres(160,250) is not None
    assert window.canvas.pocket_result['table']['scope_polygon']==window.canvas.recognition_polygon
    window.close()
