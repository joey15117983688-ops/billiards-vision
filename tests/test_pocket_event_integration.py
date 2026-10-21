import json
import cv2
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
import main
from src.vision.pocket import PocketDetector
from src.vision.pocket_event.adapter import PocketEventAdapter
from src.vision.pocket_event.recording import EventRecording
from pocket_scene import make_scene

@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication([])


def test_adapter_projected_pockets_match_ball_coordinates_and_do_not_clip():
    frame,_=make_scene();pockets=PocketDetector().detect(frame).to_dict()
    a=PocketEventAdapter(pockets)
    p=pockets['pockets'][0]['center'];zone=a.engine.zones[0]
    assert np.allclose(a.project([[p['x'],p['y']]])[0],zone.center)
    assert zone.center[0]<0  # physical pocket outside the cloth stays outside
    assert zone.trusted_shape and len(zone.entry_polygon)>=3


def test_visual_gate_blocks_skin_occlusion_and_allows_clear_table():
    frame,_=make_scene();a=PocketEventAdapter(PocketDetector().detect(frame).to_dict())
    assert a.clear_scene(frame,[])
    occluded=frame.copy();cv2.rectangle(occluded,(200,200),(600,280),(100,150,210),-1)
    assert not a.clear_scene(occluded,[])


def test_event_recording_saves_real_input_frame_and_json(tmp_path):
    record=EventRecording(tmp_path,'fixture.mp4');frame=np.zeros((40,40,3),np.uint8)
    record.write({'timestamp':1.2,'balls':{}},[{'shot_id':'shot_0001','events':[]}],frame)
    record.close()
    payload=json.loads((record.path/'events.jsonl').read_text())
    assert (record.path/payload['key_frame']).exists() and payload['source']=='fixture.mp4'


def test_ui_uses_full_scene_for_pots_even_when_single_ball_selected(app,tmp_path,monkeypatch):
    monkeypatch.setattr(main,'ROOT',tmp_path)
    w=main.MainWindow(main.DEFAULT_VIDEO)
    try:
        w.play_button.click()
        frame,_=make_scene();w.canvas.set_frame(frame)
        w.canvas.set_pocket_result(PocketDetector().detect(frame).to_dict())
        w.yolo_ready=True
        w.detection_mode='single';w.selected_targets={1:{'name':'white_ball','box':[460,360,480,380]}}
        requests=[];w.live_requested.connect(lambda payload,token:requests.append((payload,token)))
        w.pot_button.click()
        assert requests and isinstance(requests[0][0],np.ndarray)
        assert w.pot_adapter is not None and w.timer.isActive()
        token=requests[0][1]
        w.on_live_result(token,[{'name':'black_ball','box':[590,220,610,240],'confidence':.9}])
        assert '黑球' in w.pot_readout.text()
        old=w.pot_adapter.engine
        w.pause_video();timestamp=old.timestamp
        for _ in range(3):app.processEvents()
        assert old.timestamp==timestamp
        w.invalidate_results()
        assert w.pot_adapter is None
        w.on_live_result(token,[])
        assert w.pot_adapter is None
        w.pot_button.click()
        assert '未开启' in w.pot_readout.text()
    finally:w.close()


def test_visual_gate_checks_occlusion_on_pocket_rim_outside_inner_plane():
    frame,_=make_scene();a=PocketEventAdapter(PocketDetector().detect(frame).to_dict())
    cv2.rectangle(frame,(590,170),(615,201),(100,150,210),-1)
    assert not a.clear_scene(frame,[])


def test_video_end_keeps_unknown_result_but_replay_clears_engine(app,tmp_path,monkeypatch):
    monkeypatch.setattr(main,'ROOT',tmp_path)
    w=main.MainWindow(main.DEFAULT_VIDEO)
    try:
        w.timer.stop();w.yolo_ready=True
        frame,_=make_scene();w.canvas.set_frame(frame)
        w.canvas.set_pocket_result(PocketDetector().detect(frame).to_dict())
        w.pot_button.click()
        w.observe_pot_events(0,[{'name':'black_ball','box':[590,220,610,240],'confidence':.9}],frame)
        assert w.pot_adapter is not None
        w.capture.set(cv2.CAP_PROP_POS_FRAMES,w.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        w.next_frame()
        assert w.pot_adapter is None and '本段结束' in w.pot_readout.text()
        assert '人工复核' in w.pot_readout.text()
    finally:w.close()


def test_two_same_class_target_balls_are_counted_independently(app,tmp_path,monkeypatch):
    from src.vision.pocket_event.models import State
    monkeypatch.setattr(main,'ROOT',tmp_path)
    frame,_=make_scene()
    adapter=PocketEventAdapter(PocketDetector().detect(frame).to_dict())
    zones=[adapter.engine.zones[0],adapter.engine.zones[-1]]
    table_center=np.mean([zone.center for zone in adapter.engine.zones],axis=0)
    results=[]
    for timestamp in np.arange(0,3.5,.04):
        detections=[]
        if timestamp<.8:
            for zone in zones:
                direction=table_center-np.asarray(zone.center)
                direction/=np.linalg.norm(direction)
                point=np.asarray(zone.center)+direction*max(.008,.25-.25*timestamp/.65)
                pixel=cv2.perspectiveTransform(np.float32([[point*adapter.scale]]),adapter.coordinates.mm_to_image)[0,0]
                x,y=map(float,pixel)
                detections.append({'name':'billiard_ball','box':[x-9,y-9,x+9,y+9],'confidence':.95})
        results+=adapter.update(float(timestamp),detections,np.zeros_like(frame),scene_clear=True)
    targets=[event for result in results for event in result['events'] if event['ball'].startswith('billiard_ball:') and event['event']==State.POTTED]
    assert len(targets)==2, [(r["shot_id"], r["review_timestamp"], [(e["ball"],e["event"]) for e in r["events"]]) for r in results]
    assert len({event['ball'] for event in targets})==2
    w=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    try:
        w.update_pot_count(results);w.update_pot_count(results)
        assert w.pot_count_label.text()=='（2）'
    finally:w.close()


def test_ball_class_flicker_preserves_physical_identity():
    frame,_=make_scene();adapter=PocketEventAdapter(PocketDetector().detect(frame).to_dict())
    names=[]
    for i,name in enumerate(('billiard_ball','black_ball','billiard_ball','black_ball')):
        detections=adapter.track_targets(i/30,[{'name':name,'box':[300+i,240,320+i,260],'confidence':.95}],frame)
        names.append(detections[0]['name'])
    assert len(set(names))==1
    assert len(adapter.target_tracks.states)==1


def test_pot_observations_follow_display_frames_even_when_yolo_is_busy(app,tmp_path,monkeypatch):
    monkeypatch.setattr(main,'ROOT',tmp_path)
    w=main.MainWindow(main.DEFAULT_VIDEO)
    try:
        frame,_=make_scene();w.canvas.set_pocket_result(PocketDetector().detect(frame).to_dict())
        w.pot_button.click();w.yolo_busy=True
        observed=[]
        monkeypatch.setattr(w,'observe_pot_events',lambda timestamp,detections,frame:observed.append(timestamp))
        position=w.capture.get(cv2.CAP_PROP_POS_FRAMES);fps=w.capture.get(cv2.CAP_PROP_FPS)
        for _ in range(8):w.next_frame()
        assert w.capture.get(cv2.CAP_PROP_POS_FRAMES)==position+8
        assert observed==pytest.approx([(position+i)/fps for i in range(8)])
    finally:w.close()


@pytest.mark.parametrize('segment_index',[0,1])
def test_recorded_real_pot_survives_detector_dropout_and_class_flicker(segment_index):
    from pathlib import Path
    root=Path(main.__file__).parent
    source=root/'data/test.mp4'
    if not source.exists():pytest.skip('Local original recording is unavailable')
    fixture=json.loads((Path(__file__).parent/'fixtures/recorded_pot_segments.json').read_text())
    segment=fixture['segments'][segment_index]
    capture=cv2.VideoCapture(str(source))
    try:
        ok,frame=capture.read();assert ok
        adapter=PocketEventAdapter(PocketDetector().detect(frame).to_dict())
        capture.set(cv2.CAP_PROP_POS_FRAMES,round(segment['start']*capture.get(cv2.CAP_PROP_FPS)))
        verdicts={}
        for row in segment['frames']:
            ok,frame=capture.read();assert ok
            for result in adapter.update(row['timestamp'],row['detections'],frame):
                for event in result['events']:
                    verdicts[(result['shot_id'],event['ball'])]=event['event']
        assert sum(event=='POTTED' for event in verdicts.values())==1
    finally:capture.release()


def test_eof_finishes_without_waiting_for_yolo(app,tmp_path,monkeypatch):
    monkeypatch.setattr(main,'ROOT',tmp_path)
    w=main.MainWindow(main.DEFAULT_VIDEO)
    try:
        frame,_=make_scene();w.canvas.set_pocket_result(PocketDetector().detect(frame).to_dict())
        w.pot_button.click();w.yolo_busy=True
        w.capture.set(cv2.CAP_PROP_POS_FRAMES,w.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        w.next_frame()
        assert not w.timer.isActive() and w.pot_adapter is None
        assert '本段结束' in w.pot_readout.text()
    finally:w.close()


def test_actual_video_continuous_playback_updates_both_pot_counts_without_yolo(app,tmp_path,monkeypatch):
    from pathlib import Path
    source=Path(main.__file__).parent/'data/test.mp4'
    if not source.exists():pytest.skip('Local original recording unavailable')
    monkeypatch.setattr(main,'ROOT',tmp_path)
    w=main.MainWindow(source)
    try:
        w.pause_video();w.canvas.set_pocket_result(PocketDetector().detect(w.canvas.frame).to_dict())
        w.pot_button.click();w.yolo_busy=True
        assert not w.yolo_ready  # No model, mocked detections, or accelerated analysis clock.
        fps=w.capture.get(cv2.CAP_PROP_FPS)
        while w.capture.get(cv2.CAP_PROP_POS_FRAMES)/fps<74:
            w.next_frame()
            timestamp=(w.capture.get(cv2.CAP_PROP_POS_FRAMES)-1)/fps
            if 49.5<timestamp<50:assert w.pot_count_label.text()=='（1）'
            if 67<timestamp<67.5:assert w.pot_count_label.text()=='（2）'
            assert w.pot_adapter.engine.timestamp==pytest.approx(timestamp)
        assert w.pot_count_label.text()=='（2）'
    finally:w.close()


@pytest.mark.parametrize('cloth_hue',[60,100])
def test_pixel_observer_finds_balls_but_rejects_arm_and_cue(cloth_hue):
    from src.vision.pocket_event.pixel_observer import PixelBallObserver
    frame,_=make_scene()
    cv2.circle(frame,(700,350),12,(15,15,15),-1)
    if cloth_hue==100:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        green=(hsv[:,:,0]>35)&(hsv[:,:,0]<85)&(hsv[:,:,1]>60)
        hsv[:,:,0][green]=100;frame=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    cv2.rectangle(frame,(220,260),(290,440),(100,150,210),-1)
    cv2.line(frame,(400,500),(850,550),(245,245,245),4)
    observer=PixelBallObserver([[200,200],[1000,200],[1000,600],[200,600]])
    detections=observer.observe(frame)
    assert {d['name'] for d in detections}=={'white_ball','black_ball','billiard_ball'}
    assert len(detections)==3


def test_switching_yolo_off_does_not_reset_pot_observation(app,tmp_path,monkeypatch):
    monkeypatch.setattr(main,'ROOT',tmp_path)
    w=main.MainWindow(main.DEFAULT_VIDEO)
    try:
        frame,_=make_scene();w.canvas.set_pocket_result(PocketDetector().detect(frame).to_dict())
        w.pot_button.click();observer=w.pot_observer;adapter=w.pot_adapter
        w.toggle_yolo(False)
        assert w.pot_observer is observer and w.pot_adapter is adapter
        w.next_frame()
        assert adapter.engine.timestamp is not None
    finally:w.close()


def test_real_late_black_and_next_red_count_after_the_reported_thirteenth(app,tmp_path,monkeypatch):
    """The lost black at 315.6s must count, then later pots must continue."""
    from pathlib import Path
    root=Path(main.__file__).parent;source=root/'data/test.mp4'
    if not source.is_file():pytest.skip('Local original recording unavailable')
    pockets=json.loads((Path(__file__).parent/'fixtures/late_pot_calibration.json').read_text())
    monkeypatch.setattr(main,'ROOT',tmp_path)
    w=main.MainWindow(source)
    try:
        w.pause_video();w.capture.set(cv2.CAP_PROP_POS_MSEC,312000)
        ok,frame=w.capture.read();assert ok
        w.canvas.set_frame(frame);w.canvas.set_pocket_result(pockets)
        w.pot_button.click();w.pause_video()
        assert w.pot_adapter is not None and not w.yolo_ready
        # Exercise the actual UI frame/observer/association/event/count pipeline.
        while (w.capture.get(cv2.CAP_PROP_POS_FRAMES)-1)/30<331:
            w.next_frame()
            t=(w.capture.get(cv2.CAP_PROP_POS_FRAMES)-1)/30
            if 317<t<320:assert w.pot_count_label.text()=='（1）'
            if 329.5<t<331:assert w.pot_count_label.text()=='（2）'
        events=[e for r in w.pot_adapter.engine.results for e in r['events'] if e['event']=='POTTED']
        assert any(e['ball']=='black_ball' and 315.5<e['timestamp']<315.8 for e in events)
        assert len(events)==2
    finally:w.close()


@pytest.mark.parametrize('case',['toward','receding','no_direction','occluded','interrupted','expired'])
def test_extended_pocket_recovery_requires_continuous_measured_approach(monkeypatch,case):
    """Association guard unit test; real-pixel accuracy is covered by the video above."""
    frame,_=make_scene();adapter=PocketEventAdapter(PocketDetector().detect(frame).to_dict())
    detection={'name':'black_ball','box':[230,210,250,230],'confidence':.95}
    adapter.track_targets(0,[detection],frame)
    identity=next(iter(adapter.target_tracks.states));target=adapter.target_pixels[identity][1]
    state=adapter.engine.states[adapter.target_names[identity]]
    state.pocket='TOP_LEFT';state.direction_seen=case!='no_direction';state.occlusion_seen=case=='occluded'
    now=1. if case=='expired' else .4
    seen=.1 if case=='interrupted' else now-.04
    adapter.target_pixels[identity]=(seen,target)
    adapter.target_tracks.states[identity]['frame']=round(seen*30)
    direction=5 if case=='receding' else -5
    measured={**target,'box':(np.array(target['box'])+[direction,direction,direction,direction]).tolist()}
    monkeypatch.setattr(adapter,'pocket_pixels',lambda *args:measured)
    monkeypatch.setattr(adapter,'pocket_template',lambda *args:None)
    import src.ball_observation
    monkeypatch.setattr(src.ball_observation,'closest_ball',lambda *args:None)
    result=adapter.track_targets(now,[],frame)
    assert bool(result)==(case=='toward')
