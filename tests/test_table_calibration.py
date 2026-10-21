import cv2
import numpy as np
import pytest
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
import main
from main import MainWindow, CalibrationWorker
from src.scene_calibration import calibrate_scene
from src.target_balls import scene_detections
from pocket_scene import make_scene


def predict(frame, options):
    return {'detections':scene_detections(frame, [], options['calibration_polygon'])}


def test_full_pipeline_uses_one_frame_and_rebuilds_all_observations():
    frame,_=make_scene()
    stages=[]
    calls=[]
    def prediction(image, options):
        assert image is frame
        calls.append(options)
        return predict(image, options)
    result=calibrate_scene(frame,prediction,stages.append)
    assert stages[0].startswith('2/4') and stages[-1].startswith('4/4') and len(calls)==1
    assert result['calibration_report']['candidate_count']>1
    assert calls[0]['mode']=='all'
    assert len(result['pockets']['pockets'])==6
    assert result['pockets']['table']['scope_polygon']==result['scope_polygon']
    assert result['detections'][0]['polygon']==result['scope_polygon']
    assert result['detections'][0]['playable_polygon']==result['polygon']
    assert min(p[0] for p in result['scope_polygon'])>180
    assert any(d['name']=='white_ball' for d in result['detections'])
    assert min(p[0] for p in result['polygon'])>180  # excludes wooden frame at x=160


def test_bad_table_fails_instead_of_using_old_geometry():
    with pytest.raises(ValueError,match='无法可靠定位'):
        calibrate_scene(np.zeros((600,1000,3),np.uint8),
                        lambda *_:pytest.fail('must reject before ball prediction'),lambda _:None)


def test_worker_rereads_source_instead_of_reusing_old_display(tmp_path,monkeypatch):
    frame,_=make_scene()
    path=tmp_path/'fresh.png'
    cv2.imwrite(str(path),frame)
    owner=main.YoloWorker()
    class Runner:
        def __init__(self):
            self.failed=owner.failed
            self.process=type('Process',(),{'poll':lambda _:None})()
        def load(self):pass
        def shutdown(self):pass
        def request_prediction(self,image,options):
            assert np.array_equal(image,frame)
            return predict(image,options)
    monkeypatch.setattr(main,'YoloWorker',Runner)
    worker=CalibrationWorker()
    result=[];errors=[];stages=[]
    worker.done.connect(lambda *args:result.append(args))
    worker.failed.connect(lambda *args:errors.append(args))
    worker.progress.connect(lambda *args:stages.append(args))
    monkeypatch.setattr(main,'ROOT',tmp_path)
    _,centers=make_scene()
    manual={'corners':(np.array([[200,200],[1000,200],[1000,600],[200,600]])/[1280,800]).tolist(),
            'pockets':(np.array(list(centers.values()))/[1280,800]).tolist()}
    worker.run(np.zeros_like(frame),7,{'kind':'image','path':str(path),
               'manual_profile':manual,'collect_training_sample':True})
    assert not errors and len(result)==1 and len(stages)==4
    assert result[0][0]==7 and np.array_equal(result[0][1],frame)
    sample=result[0][2]['learning_sample']
    from pathlib import Path
    assert (Path(sample)/'annotation.json').is_file()
    assert np.array_equal(cv2.imread(str(Path(sample)/'image.png')),frame)


def test_pocket_button_runs_async_and_commits_geometry_only(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    frame,_=make_scene()
    path=tmp_path/'scene.png';cv2.imwrite(str(path),frame)
    class Worker(QObject):
        progress=pyqtSignal(int,str)
        done=pyqtSignal(int,object,object)
        failed=pyqtSignal(int,str)
        @pyqtSlot(object,int,object)
        def run(self,image,generation,source):
            self.progress.emit(generation,'3/4 验证袋口')
    monkeypatch.setattr(main,'CalibrationWorker',Worker)
    monkeypatch.setattr(main,'ROOT',tmp_path)
    window=MainWindow(image_dir=tmp_path,start_with_images=True)
    window.yolo_button.blockSignals(True);window.yolo_button.setChecked(True);window.yolo_button.blockSignals(False)
    window.canvas.set_detections([{'name':'billiards_table','box':[0,0,1279,799],'polygon':None,'confidence':.9}])
    window.pocket_button.click()
    assert window.calibration_busy and not window.pocket_button.isEnabled()
    assert window.canvas.detections==[]
    generation=window.media_generation
    window.on_yolo_all_result(window.yolo_generation,[{'name':'stale'}])
    assert window.canvas.detections==[]
    for _ in range(10):QTest.qWait(10)
    assert '3/4' in window.statusBar().currentMessage()
    result=calibrate_scene(frame,None,lambda _:None,geometry_only=True)
    window.on_calibration_done(generation-1,frame,result)
    assert window.calibration_busy and window.canvas.detections==[]
    window.on_calibration_done(generation,frame,result)
    assert not window.calibration_busy
    assert window.canvas.detections==result['detections']
    assert window.canvas.pocket_result==result['pockets']
    assert window.canvas.calibration_polygon==result['polygon']
    assert '纠错完成' in window.statusBar().currentMessage()
    window.canvas.coordinate_mode=True
    window.canvas.set_frame(frame,keep_pockets=True)
    assert np.allclose(window.canvas.coordinates.corners,result['scope_polygon'])
    window.invalidate_results()
    assert window.canvas.calibration_polygon is None
    window.close()


def test_failed_correction_restores_the_previous_result_and_tracks(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([])
    frame,_=make_scene();path=tmp_path/'scene.png';cv2.imwrite(str(path),frame)
    class Worker(QObject):
        progress=pyqtSignal(int,str);done=pyqtSignal(int,object,object);failed=pyqtSignal(int,str)
        @pyqtSlot(object,int,object)
        def run(self,image,generation,source):
            self.failed.emit(generation,'未找到更可靠的新边界')
    monkeypatch.setattr(main,'CalibrationWorker',Worker);monkeypatch.setattr(main,'ROOT',tmp_path)
    window=MainWindow(image_dir=tmp_path,start_with_images=True)
    polygon=[[200,200],[1000,200],[1000,600],[200,600]]
    old={'name':'billiards_table','box':[200,200,1000,600],'polygon':polygon,'confidence':.9}
    window.canvas.calibration_polygon=polygon
    window.canvas.set_detections([old]);window.detection_mode='all'
    window.calibrate_table()
    for _ in range(30):
        app.processEvents();QTest.qWait(5)
        if not window.calibration_busy:break
    assert not window.calibration_busy and window.canvas.detections==[old]
    assert window.canvas.calibration_polygon==polygon and window.detection_mode=='all'
    assert '原识别结果已保留' in window.statusBar().currentMessage()
    window.close()


def test_clicking_a_ball_delivers_verified_table_geometry_before_selection(monkeypatch):
    worker=main.YoloWorker();events=[]
    table={'name':'billiards_table','calibration_evidence':{'verified':True}}
    monkeypatch.setattr(worker,'request_prediction',lambda *args:{
        'name':'white_ball','confidence':.9,'box':[100,100,120,120],'polygon':None,'scene_geometry':table})
    worker.geometry_ready.connect(lambda generation,data:events.append(('geometry',generation,data)))
    worker.result.connect(lambda generation,*args:events.append(('ball',generation,args)))
    worker.identify(np.zeros((200,300,3),np.uint8),110,110,3)
    assert events[0]==('geometry',3,table)
    assert events[1][0]=='ball' and events[1][1]==3
