"""Behavioural regressions for selection timing, coloured balls, cues and video EOF."""
import cv2
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
from main import MainWindow
from src.ball_observation import closest_ball
from src.live_tracking import LiveTracking
from src.ball_tracks import BallTracks


@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication([])


def write_video(path):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,100))
    for i in range(5):writer.write(np.full((100,160,3),(30+i*20,120,30),np.uint8))
    writer.release()


def test_video_navigation_and_end_stops_at_first_frame(app,tmp_path):
    for name in ('a.mp4','test.mp4','z.mp4'):write_video(tmp_path/name)
    window=MainWindow(tmp_path/'test.mp4')
    try:
        assert window.previous_button.isEnabled() and window.next_button.isEnabled()
        window.next_button.click()
        assert window.video_path.name=='z.mp4' and not window.next_button.isEnabled()
        window.previous_button.click()
        assert window.video_path.name=='test.mp4'
        first=window.canvas.frame.copy()
        window.yolo_generation+=1;old_generation=window.yolo_generation
        window.capture.set(cv2.CAP_PROP_POS_FRAMES,4)
        window.next_frame();window.next_frame()
        assert not window.timer.isActive() and window.play_button.isEnabled()
        assert window.capture.get(cv2.CAP_PROP_POS_FRAMES)==1
        assert np.array_equal(window.canvas.frame,first)
        assert window.detection_mode=='idle' and not window.canvas.track_paths
        assert window.yolo_generation>old_generation
        window.play_button.click();assert window.timer.isActive()
    finally:window.close()


def test_play_while_click_inference_pending_keeps_selection(app,tmp_path):
    write_video(tmp_path/'a.mp4');window=MainWindow(tmp_path/'a.mp4')
    try:
        window.pause_video();window.yolo_ready=True;window.yolo_button.setChecked(True)
        window.identify_element(window.canvas.frame.copy(),50,50)
        generation=window.yolo_generation
        assert window.selection_busy
        window.play_button.click()
        assert window.resume_after_selection and not window.timer.isActive()
        assert window.yolo_generation==generation
        window.on_yolo_result(generation,'billiard_ball',.9,[40,40,60,60])
        assert window.timer.isActive() and window.detection_mode=='single'
        assert len(window.selected_targets)==1 and not window.resume_after_selection
    finally:window.close()


@pytest.mark.parametrize('colour',[(20,20,230),(220,40,20),(15,210,235),(15,15,15),(240,240,240)])
def test_selected_ball_builds_continuous_trail_without_all(colour):
    tracker=LiveTracking()
    for i in range(20):
        frame=np.full((120,240,3),(45,140,45),np.uint8)
        cv2.circle(frame,(35+5*i,55),9,colour,-1)
        cv2.line(frame,(20,90),(180,110),(245,245,245),3)  # Unrelated cue.
        tracker.push(i,frame)
        if i==0:tracker.replace(0,[{'name':'billiard_ball','box':[26,46,44,64],'confidence':.9,'selection_id':101}])
    state=tracker.identities.states[101]
    assert state['status']=='tracking' and len(state['path'])>=18
    assert np.linalg.norm(state['center']-[130,55])<2
    assert all(not point[2] for point in list(state['path'])[1:])
    assert len(tracker.identities.states)==1


def test_cue_does_not_produce_ball_trail_even_if_model_mislabels_it():
    tracker=LiveTracking()
    for i in range(15):
        frame=np.full((120,240,3),(45,140,45),np.uint8)
        cv2.line(frame,(30+i*3,55),(180+i*3,55),(245,245,245),4)
        tracker.push(i,frame)
        if i==0:
            target={'name':'billiard_ball','box':[40,46,58,64],'confidence':.8,'selection_id':101}
            assert closest_ball(frame,target) is None
            tracker.replace(0,[target])
    assert len(tracker.identities.states[101]['path'])==1
    assert tracker.identities.states[101]['status']=='lost'
    assert not BallTracks().update(0,frame,[{'name':'cue','box':[30,50,180,60],'confidence':.9}])


def test_switch_video_during_pending_selection_does_not_wait_for_old_click(app,tmp_path):
    for name in ('a.mp4','b.mp4'):write_video(tmp_path/name)
    window=MainWindow(tmp_path/'a.mp4')
    try:
        window.pause_video();window.yolo_ready=True;window.yolo_button.setChecked(True)
        window.identify_element(window.canvas.frame.copy(),50,50)
        old=window.yolo_generation
        assert window.selection_busy
        window.next_button.click()
        assert window.video_path.name=='b.mp4' and not window.timer.isActive()
        assert window.play_button.text()=='播放'
        window.on_yolo_result(old,'billiard_ball',.9,[40,40,60,60])
        assert not window.selected_targets and window.detection_mode=='idle'
    finally:window.close()


@pytest.mark.parametrize('colour,name',[((240,240,240),'white_ball'),((20,20,225),'billiard_ball'),((15,15,15),'black_ball')])
def test_fast_ball_multiple_diameters_per_frame_keeps_id_and_continuous_path(colour,name):
    tracker=LiveTracking()
    for i in range(12):
        frame=np.full((240,800,3),(45,140,45),np.uint8)
        cv2.circle(frame,(50+50*i,90),9,colour,-1)
        tracker.push(i,frame)
        if i==0:tracker.replace(0,[{'name':name,'box':[41,81,59,99],'confidence':.95,'selection_id':101}])
    state=tracker.identities.states[101]
    assert state['status']=='tracking'
    assert len(state['path'])==12
    assert np.linalg.norm(state['center']-[600,90])<2
    assert not any(p[2] for p in list(state['path'])[1:])


def test_delayed_fast_ball_detection_backfills_real_observed_positions():
    tracker=LiveTracking(history_size=12)
    for i in range(10):
        frame=np.full((240,800,3),(45,140,45),np.uint8)
        cv2.circle(frame,(50+45*i,90),10,(240,240,240),-1)
        tracker.push(i,frame)
    assert tracker.replace(0,[{'name':'white_ball','box':[40,80,60,100],'confidence':.95}])
    assert len(tracker.identities.states)==1
    state=next(iter(tracker.identities.states.values()))
    assert len(state['path'])>=9
    assert np.linalg.norm(state['center']-[455,90])<2
    assert not any(p[2] for p in list(state['path'])[1:])


def test_true_occlusion_keeps_a_gap_without_fabricated_points():
    tracker=LiveTracking()
    for i in range(8):
        frame=np.full((240,800,3),(45,140,45),np.uint8)
        if i not in (3,4,5):cv2.circle(frame,(50+40*i,90),9,(240,240,240),-1)
        tracker.push(i,frame)
        if i==0:tracker.replace(0,[{'name':'white_ball','box':[41,81,59,99],'confidence':.95,'selection_id':101}])
    points=list(tracker.identities.states[101]['path'])
    assert len(points)==5
    assert points[3][2] is True
    assert not any(150<x<285 for x,y,gap in points)


def test_moderate_fast_ball_motion_blur_remains_a_measured_ball():
    tracker=LiveTracking()
    for i in range(10):
        frame=np.full((240,800,3),(45,140,45),np.uint8)
        cv2.ellipse(frame,(50+40*i,90),(9,9) if i==0 else (17,7),0,0,360,(240,240,240),-1)
        tracker.push(i,frame)
        if i==0:tracker.replace(0,[{'name':'white_ball','box':[41,81,59,99],'confidence':.95,'selection_id':101}])
    state=tracker.identities.states[101]
    assert state['status']=='tracking' and len(state['path'])==10
    assert np.linalg.norm(state['center']-[410,90])<2


def test_periodic_late_results_do_not_erase_existing_continuous_trail():
    tracker=LiveTracking()
    for i in range(30):
        frame=np.full((240,800,3),(45,140,45),np.uint8)
        cv2.circle(frame,(50+15*i,90),9,(240,240,240),-1)
        tracker.push(i,frame)
        if i==0 or i%6==0:
            start=max(0,i-3)
            x=50+15*start
            tracker.replace(start,[{'name':'white_ball','box':[x-9,81,x+9,99],'confidence':.95}])
    state=next(iter(tracker.identities.states.values()))
    assert len(state['path'])>=29
    assert np.linalg.norm(np.asarray(state['path'][0][:2])-[50,90])<2
    assert not any(point[2] for point in list(state['path'])[1:])
