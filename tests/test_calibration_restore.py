"""Saved manual geometry must be ready on source entry, before another calibration."""
import hashlib
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication
import main


def profile(inset=.1):
    a,b=inset,1-inset
    return {'corners':[[a,a],[b,a],[b,b],[a,b]],
            'pockets':[[a,a],[.5,a],[b,a],[a,b],[.5,b],[b,b]]}


def save_profile(root,source,data):
    path=root/'config'/'calibration'/(hashlib.sha256(str(source.resolve()).encode()).hexdigest()+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))


def assert_restored(window,inset):
    h,w=window.canvas.frame.shape[:2]
    assert np.allclose(window.canvas.calibration_polygon,np.array(profile(inset)['corners'])*[w,h])
    assert window.single_table_polygon==window.canvas.calibration_polygon
    assert all(p['status']=='manual' for p in window.canvas.pocket_result['pockets'])
    if window.yolo_worker:
        assert window.yolo_worker.calibration_polygon==window.canvas.calibration_polygon


def test_images_restore_on_entry_and_click_without_recalibrating(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(main,'ROOT',tmp_path)
    for name in ['a.png','b.png','c.png']:
        cv2.imwrite(str(tmp_path/name),np.zeros((600,1000,3),np.uint8))
    save_profile(tmp_path,tmp_path/'a.png',profile(.1))
    save_profile(tmp_path,tmp_path/'b.png',profile(.2))
    window=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    monkeypatch.setattr(window,'calibrate_table',lambda: (_ for _ in ()).throw(AssertionError('must not recalibrate')))
    assert_restored(window,.1)
    window.yolo_worker=main.YoloWorker()
    window.show_image(1);assert_restored(window,.2)
    window.show_image(2)
    assert window.canvas.calibration_polygon is None
    assert window.yolo_worker.calibration_polygon is None
    assert window.canvas.pocket_result is None
    window.show_image(0);assert_restored(window,.1)
    window.yolo_button.blockSignals(True);window.yolo_button.setChecked(True);window.yolo_button.blockSignals(False)
    window.yolo_ready=True
    requests=[];window.identify_requested.connect(lambda *args:requests.append(args))
    window.identify_element(window.canvas.frame,500,300)
    assert len(requests)==1
    from src.target_balls import scene_detections
    table=scene_detections(window.canvas.frame,[],window.yolo_worker.calibration_polygon)[0]
    window.on_yolo_result(window.yolo_generation,table['name'],table['confidence'],table['box'],table['polygon'])
    assert next(iter(window.selected_targets.values()))['polygon']==window.canvas.calibration_polygon
    window.close()
    # A fresh application window restores the same persisted geometry as well.
    reopened=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    assert_restored(reopened,.1);reopened.close()


def test_video_switch_rewind_and_source_modes_restore_saved_geometry(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(main,'ROOT',tmp_path)
    for name,inset in [('a.avi',.1),('b.avi',.2)]:
        path=tmp_path/name
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(200,120))
        assert writer.isOpened()
        for _ in range(3):writer.write(np.zeros((120,200,3),np.uint8))
        writer.release();save_profile(tmp_path,path,profile(inset))
    cv2.imwrite(str(tmp_path/'image.png'),np.zeros((120,200,3),np.uint8))
    window=main.MainWindow(video_path=tmp_path/'a.avi',image_dir=tmp_path)
    window.pause_video();assert_restored(window,.1)
    window.navigate_media(1);window.pause_video();assert_restored(window,.2)
    window.navigate_media(-1);window.pause_video();assert_restored(window,.1)
    window.capture.set(cv2.CAP_PROP_POS_FRAMES,3)
    window.next_frame();assert_restored(window,.1)
    window.switch_source(1)
    assert window.canvas.calibration_polygon is None
    window.switch_source(0);window.pause_video();assert_restored(window,.1)
    window.close()


def test_accepted_auto_correction_is_restored_without_recalibrating(tmp_path,monkeypatch):
    from src.adaptive_calibration import search_calibration
    from src.calibration_state import save_automatic
    fixture=main.ROOT/'tests/fixtures/calibration_portrait.png'
    frame=cv2.imread(str(fixture))
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(main,'ROOT',tmp_path)
    source=tmp_path/'camera.png';cv2.imwrite(str(source),frame)
    polygon,pockets,report=search_calibration(frame,lambda _:None)
    key=hashlib.sha256(str(source.resolve()).encode()).hexdigest()
    path=tmp_path/'config/calibration'/f'{key}.auto.json'
    save_automatic(path,frame,{'polygon':polygon,'pockets':pockets,'calibration_report':report},None)
    for _ in range(2):
        window=main.MainWindow(image_dir=tmp_path,start_with_images=True)
        assert np.allclose(window.canvas.calibration_polygon,polygon)
        assert window.canvas.pocket_result['valid']
        window.close()


def test_first_verified_automatic_detection_locks_and_persists_without_button(tmp_path,monkeypatch):
    from src.target_balls import scene_detections
    frame=cv2.imread(str(main.ROOT/'tests/fixtures/calibration_portrait.png'))
    source=tmp_path/'camera.png';cv2.imwrite(str(source),frame)
    app=QApplication.instance() or QApplication([])
    monkeypatch.setattr(main,'ROOT',tmp_path)
    window=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    assert window.canvas.calibration_polygon is None
    detections=scene_detections(frame,[])
    assert detections[0]['calibration_evidence']['metrics']['observed']==6
    window.yolo_button.blockSignals(True);window.yolo_button.setChecked(True);window.yolo_button.blockSignals(False)
    window.on_yolo_all_result(window.yolo_generation,detections)
    polygon=window.canvas.calibration_polygon
    assert polygon and window.canvas.pocket_result['valid']
    assert window.calibration_profile_path().with_suffix('.auto.json').is_file()
    window.close()
    reopened=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    assert np.allclose(reopened.canvas.calibration_polygon,polygon)
    reopened.close()


def test_saved_auto_result_remains_the_baseline_under_an_obscured_startup_frame(tmp_path,monkeypatch):
    from src.adaptive_calibration import search_calibration
    from src.calibration_state import save_automatic
    frame=cv2.imread(str(main.ROOT/'tests/fixtures/calibration_portrait.png'))
    polygon,pockets,report=search_calibration(frame,lambda _:None)
    source=tmp_path/'camera.png';cv2.imwrite(str(source),np.zeros_like(frame))
    old={'corners':(np.array([[40,62],[638,60],[641,1159],[36,1159]])/[704,1210]).tolist(),
         'pockets':[[.05,.05],[.05,.5],[.05,.95],[.9,.05],[.9,.5],[.9,.95]]}
    save_profile(tmp_path,source,old)
    key=hashlib.sha256(str(source.resolve()).encode()).hexdigest()
    save_automatic(tmp_path/'config/calibration'/f'{key}.auto.json',frame,
                   {'polygon':polygon,'pockets':pockets,'calibration_report':report},old)
    app=QApplication.instance() or QApplication([]);monkeypatch.setattr(main,'ROOT',tmp_path)
    window=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    assert np.allclose(window.canvas.calibration_polygon,polygon)
    assert window.canvas.pocket_result['table']['scope_polygon']==pockets['table']['scope_polygon']
    window.close()
