import cv2
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QPushButton
from main import MainWindow, VideoCanvas, CalibrationWorker
from pocket_scene import make_scene

@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication([])

@pytest.fixture
def window(app,tmp_path,monkeypatch):
    import main
    monkeypatch.setattr(main,'ROOT',tmp_path)
    cv2.imwrite(str(tmp_path/'frame.png'),np.zeros((240,400,3),np.uint8))
    w=MainWindow(image_dir=tmp_path,start_with_images=True)
    w.yolo_ready=True
    yield w
    w.close()

def test_cancel_during_inference_rejects_delayed_result(window):
    w=window
    w.identify_all_button.click()
    generation=w.yolo_generation
    assert w.identify_all_button.text()=='取消识别'
    assert w.yolo_busy
    w.identify_all_button.click()
    assert w.detection_mode=='idle' and w.identify_all_button.text()=='全部识别'
    w.on_yolo_all_result(generation,[{'name':'billiard_ball','box':[1,1,4,4]}])
    assert w.canvas.detections==[] and not w.live_tracking.detections

def test_track_modes_filter_without_erasing_history(window):
    w=window
    w.live_tracking.identities.states={
        1:{'name':'white_ball','path':[(10,10,False),(20,10,False)]},
        2:{'name':'billiard_ball','path':[(30,10,False),(40,10,False)]}}
    w.selected_targets={2:{}}
    for mode,expected in [(0,{1,2}),(1,{1}),(2,{2}),(3,{2})]:
        w.trails_button.setCurrentIndex(mode)
        if mode==3:w.selected_targets={2:{}}
        w.sync_tracks()
        assert set(w.canvas.track_paths)==expected
    assert len(w.live_tracking.identities.states)==2

def test_coordinates_and_single_outline_use_latest_detection(app):
    c=VideoCanvas();c.resize(400,240)
    c.set_frame(np.zeros((240,400,3),np.uint8))
    old=[[60,60],[340,60],[340,180],[60,180]]
    new=[[20,20],[380,20],[380,220],[20,220]]
    c.recognition_polygon=old
    c.pocket_result={'table':{'scope_polygon':old,'playable_polygon':old},'pockets':[]}
    c.coordinate_mode=True
    c.set_detections([{'name':'billiards_table','polygon':new,'box':[20,20,380,220]}])
    assert np.allclose(c.coordinates.corners,new)
    c.coordinate_mode=False
    image=c.grab().toImage()
    assert image.pixelColor(20,100).blue()>100
    assert image.pixelColor(60,100).blue()==0  # no obsolete inner border
    c.close()

def test_geometry_worker_never_loads_ball_model(app,tmp_path,monkeypatch):
    import main
    frame,_=make_scene();path=tmp_path/'scene.png';cv2.imwrite(str(path),frame)
    monkeypatch.setattr(main,'YoloWorker',lambda:pytest.fail('must not create a ball model'))
    w=CalibrationWorker();results=[];errors=[]
    w.done.connect(lambda *args:results.append(args));w.failed.connect(lambda *args:errors.append(args))
    w.run(frame,1,{'kind':'image','path':str(path)})
    assert not errors and len(results)==1
    assert [d['name'] for d in results[0][2]['detections']]==['billiards_table']
    assert len(results[0][2]['pockets']['pockets'])==6

def test_no_separate_calibration_button(window):
    assert '校准' not in [b.text() for b in window.findChildren(QPushButton)]

@pytest.mark.parametrize('mode,expected',[(0,{'white_ball','billiard_ball'}),(1,{'white_ball'}),(2,{'billiard_ball'})])
def test_automatic_trails_start_whole_frame_detection_without_selection(window,mode,expected):
    w=window;w.media_mode='video'
    # No decoder ticks are needed to verify request/response routing.
    w.timer.start(60000)
    requests=[]
    w.live_requested.connect(lambda payload,token:requests.append((payload,token)))
    w.trails_button.setCurrentIndex(mode)
    assert len(requests)==1 and isinstance(requests[0][0],np.ndarray)
    assert w.live_active() and w.selected_targets=={}
    assert w.identify_all_button.text()=='全部识别'
    detections=[{'name':'white_ball','confidence':.9,'box':[30,30,45,45]},
                {'name':'billiard_ball','confidence':.9,'box':[100,30,115,45]}]
    w.on_live_result(requests[0][1],detections)
    assert {w.live_tracking.identities.states[k]['name'] for k in w.canvas.track_paths}==expected
    assert not w.canvas.show_ball_boxes and w.canvas.selected_box is None
    w.live_frame_id+=1
    w.live_tracking.push(w.live_frame_id,w.canvas.frame)
    w.maybe_submit_live()
    moved=[{**d,'box':[d['box'][0]+4,d['box'][1],d['box'][2]+4,d['box'][3]]} for d in detections]
    w.on_live_result(requests[-1][1],moved)
    assert w.canvas.track_paths and all(len(path)>=2 for path in w.canvas.track_paths.values())
    w.trails_button.setCurrentIndex(4)
    assert not w.canvas.show_trails and not w.live_active() and not w.canvas.track_paths


def test_switching_from_selected_to_automatic_removes_crop_and_selection(window):
    w=window;w.media_mode='video';w.timer.start(60000)
    w.yolo_button.setChecked(True)
    w.detection_mode='single';w.selected_targets={7:{'name':'white_ball','box':[20,20,40,40]}}
    w.live_selected_id=7;w.canvas.selected_box=[20,20,40,40]
    requests=[];w.live_requested.connect(lambda payload,token:requests.append(payload))
    w.trails_button.setCurrentIndex(1)
    assert w.detection_mode=='idle' and w.selected_targets=={} and w.canvas.selected_box is None
    assert len(requests)==1 and isinstance(requests[0],np.ndarray)


def test_hiding_pockets_keeps_calibration_and_can_show_again(window):
    w=window
    frame,_=make_scene()
    from src.scene_calibration import calibrate_scene
    result=calibrate_scene(frame,None,lambda _:None,geometry_only=True)
    result['calibration_report']=None
    w.on_calibration_done(w.media_generation,frame,result)
    assert not w.pocket_visibility_button.isHidden()
    polygon=w.canvas.table_outline()
    coordinates=w.canvas.coordinates.corners.copy()
    w.pocket_visibility_button.click()
    assert not w.canvas.show_pocket_overlays
    assert w.pocket_visibility_button.text()=='显示袋口识别'
    assert w.canvas.pocket_result is result['pockets'] and w.canvas.table_outline()==polygon
    assert np.allclose(w.canvas.coordinates.corners,coordinates)
    w.pocket_visibility_button.click()
    assert w.canvas.show_pocket_overlays
    w.invalidate_results()
    assert w.pocket_visibility_button.isHidden()


@pytest.mark.parametrize('pot_enabled',[False,True])
@pytest.mark.parametrize('mode,expected',[(0,{'white_ball','billiard_ball'}),(1,{'white_ball'}),(2,{'billiard_ball'}),(3,{'white_ball'})])
def test_each_trail_mode_tracks_and_paints_with_or_without_pot_analysis(window,monkeypatch,mode,expected,pot_enabled):
    import main
    w=window;w.media_mode='video';w.capture=cv2.VideoCapture(str(main.DEFAULT_VIDEO))
    w.video_path=main.DEFAULT_VIDEO
    def scene(i):
        frame=np.full((240,400,3),(45,140,45),np.uint8)
        cv2.circle(frame,(40+3*i,65),9,(245,245,245),-1)
        cv2.circle(frame,(170+3*i,130),9,(20,20,225),-1)
        return frame
    def detections(i):
        return [{'name':name,'box':[x+3*i-9,y-9,x+3*i+9,y+9],'confidence':.95}
                for name,x,y in [('white_ball',40,65),('billiard_ball',170,130)]]
    w.canvas.set_frame(scene(0));w.yolo_button.setChecked(True)
    if mode==3:w.toggle_target(detections(0)[0])
    w.trails_button.setCurrentIndex(mode)
    w.pot_button.blockSignals(True);w.pot_button.setChecked(pot_enabled)
    w.timer.start(60000)
    for i in range(1,25):
        w.live_frame_id+=1;frame=scene(i)
        w.live_tracking.push(w.live_frame_id,frame);w.canvas.set_frame(frame)
        w.live_last_visual_request=0  # Let the display refresh on its next fair turn.
        w.maybe_submit_live();request=w.live_inflight
        assert request is not None
        w.on_live_result(request[0],detections(i))
        w.sync_tracks()
    names={w.live_tracking.identities.states[k]['name'] for k in w.canvas.track_paths}
    assert names==expected
    assert all(len(path)>=10 for path in w.canvas.track_paths.values())
    if pot_enabled:assert request[2]>=0  # Model work is only for the current display, never an old analysis cursor.
    painted=w.canvas.grab().toImage()
    w.canvas.show_trails=False
    assert painted!=w.canvas.grab().toImage()  # Verify pixels, not just a populated dictionary.
    w.trails_button.setCurrentIndex(4)
    assert not w.canvas.track_paths and not w.canvas.show_trails
