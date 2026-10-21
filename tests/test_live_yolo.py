import cv2
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from main import MainWindow
from src.live_tracking import LiveTracking


def scene():
    frame=np.full((100,160,3),(45,135,45),np.uint8)
    cv2.circle(frame,(50,40),10,(230,230,230),-1)
    cv2.circle(frame,(47,37),3,(60,60,60),-1)
    return frame


def test_flow_aligns_delayed_result_to_displayed_frame_and_preserves_id():
    frame=scene();tracker=LiveTracking(history_size=4)
    detection={'name':'white_ball','box':[40,30,61,51],'confidence':.9}
    tracker.push(0,frame);assert tracker.replace(0,[detection])
    original_id=tracker.detections[0]['track_id']
    shifted=cv2.warpAffine(frame,np.float32([[1,0,5],[0,1,3]]),(160,100))
    tracker.push(1,shifted)
    assert np.array(tracker.detections[0]['box'])-detection['box']==pytest.approx([5,3,5,3],abs=.5)
    assert tracker.replace(0,[detection])  # Inference arrived one frame late.
    assert tracker.detections[0]['track_id']==original_id
    assert np.array(tracker.detections[0]['box'])-detection['box']==pytest.approx([5,3,5,3],abs=.5)


def test_frame_history_is_bounded_and_old_result_rejected():
    tracker=LiveTracking(history_size=4)
    for i in range(20):tracker.push(i,scene())
    assert len(tracker.frames)==4
    assert not tracker.replace(0,[{'name':'white_ball','box':[40,30,61,51],'confidence':.9}])
    assert not tracker.detections
    tracker.clear();assert not tracker.frames


@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication([])


def test_running_video_queues_only_one_request_and_rejects_stale_source(app):
    window=MainWindow();requests=[]
    try:
        window.play_button.click()
        assert window.timer.isActive()
        window.yolo_ready=True
        window.live_requested.connect(lambda frame,token:requests.append(token))
        window.identify_all_button.click()
        assert window.timer.isActive() and len(requests)==1 and window.yolo_busy
        for _ in range(40):window.next_frame()
        assert window.timer.isActive() and len(requests)==1
        assert len(window.live_tracking.frames)==16
        window.on_live_result(requests[0],[{'name':'white_ball','box':[10,10,30,30],'confidence':.9}])
        assert not window.canvas.detections  # Too old for retained history.
        window.next_frame();assert len(requests)==2
        window.source_combo.setCurrentIndex(1)
        window.on_live_result(requests[1],[{'name':'white_ball','box':[10,10,30,30],'confidence':.9}])
        assert not window.canvas.detections and not window.yolo_busy
    finally:window.close()


def test_live_click_uses_overlay_without_pausing_or_new_inference(app):
    window=MainWindow()
    try:
        window.play_button.click()
        window.yolo_ready=True;window.detection_mode="all";window.yolo_button.setChecked(True)
        item={'name':'black_ball','box':[460,250,480,270],'confidence':.9,'track_id':17}
        window.live_tracking.detections=[item];window.canvas.set_detections([item])
        window.identify_element(window.canvas.frame.copy(),465,255)
        assert window.timer.isActive() and len(window.selected_targets)==1
        assert window.canvas.selected_label.startswith('黑球')
        window.yolo_button.setChecked(False)
        assert window.timer.isActive() and not window.canvas.detections
    finally:window.close()


def test_enabling_yolo_without_all_or_selection_does_not_start_inference(app):
    window=MainWindow();requests=[]
    try:
        window.play_button.click()
        window.yolo_ready=True;window.live_requested.connect(lambda frame,token:requests.append(token))
        window.yolo_button.setChecked(True)
        for _ in range(5):window.next_frame()
        assert window.timer.isActive() and not requests and window.detection_mode=='idle'
    finally:window.close()


def test_paused_ball_selection_survives_play_and_submits_only_crop(app):
    window=MainWindow();requests=[]
    try:
        window.pause_video();window.yolo_ready=True;window.yolo_button.setChecked(True)
        window.on_yolo_result(window.yolo_generation,'black_ball',.9,[465,150,485,170])
        selected=window.live_selected_id
        window.live_requested.connect(lambda payload,token:requests.append(payload))
        window.play_button.click()
        assert window.timer.isActive() and window.detection_mode=='single'
        assert window.live_selected_id==selected
        assert len(requests)==1 and isinstance(requests[0],dict)
        assert requests[0]['target']['name']=='black_ball'
        assert len(window.live_tracking.detections)==1
    finally:window.close()


def test_click_toggle_multiple_targets_and_cancel_all(app):
    window=MainWindow()
    try:
        window.pause_video();window.yolo_ready=True;window.yolo_button.setChecked(True)
        window.on_yolo_result(window.yolo_generation,'black_ball',.9,[465,150,485,170])
        window.on_yolo_result(window.yolo_generation,'white_ball',.9,[265,315,285,335])
        assert len(window.selected_targets)==2 and len(window.canvas.detections)==2
        window.identify_element(window.canvas.frame.copy(),475,160)
        assert len(window.selected_targets)==1
        assert next(iter(window.selected_targets.values()))['name']=='white_ball'
        window.identify_element(window.canvas.frame.copy(),275,325)
        assert not window.selected_targets and window.detection_mode=='idle'
        assert not window.canvas.detections
        requests=[];window.live_requested.connect(lambda p,t:requests.append(p))
        window.play_button.click()
        assert window.timer.isActive() and not requests
    finally:window.close()


def test_multiple_selected_targets_sent_together_without_all(app):
    window=MainWindow();requests=[]
    try:
        window.pause_video();window.yolo_ready=True;window.yolo_button.setChecked(True)
        for name,box in [('black_ball',[465,150,485,170]),('white_ball',[265,315,285,335])]:
            window.on_yolo_result(window.yolo_generation,name,.9,box)
        window.live_requested.connect(lambda payload,token:requests.append(payload))
        window.play_button.click()
        assert len(requests)==1 and len(requests[0]['targets'])==2
        ids={item['selection_id'] for item in requests[0]['targets']}
        reply=[{**target,'selection_id':target['selection_id']} for target in requests[0]['targets']]
        window.on_live_result(window.live_inflight[0],reply)
        assert {d['track_id'] for d in window.canvas.detections}==ids
        assert len(window.selected_targets)==2
    finally:window.close()
