import cv2
import numpy as np
import pytest
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
from src.manual_calibration import manual_geometry
from src.calibration_editor import CalibrationEditor
from src.scene_calibration import calibrate_scene
from src.target_balls import scene_detections


def profile():
    return {'corners':[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],
            'pockets':[[.1,.1],[.5,.1],[.9,.1],[.1,.9],[.5,.9],[.9,.9]]}


def test_manual_geometry_excludes_rails_and_survives_resolution_changes():
    for width,height in [(1000,600),(2000,1200)]:
        frame=np.zeros((height,width,3),np.uint8)
        polygon,data=manual_geometry(frame,profile())
        assert np.allclose(np.array(polygon)/[width,height],profile()['corners'])
        assert data['valid'] and len(data['pockets'])==6
        assert {p['status'] for p in data['pockets']}=={'manual'}
        assert len({p['id'] for p in data['pockets']})==6
        assert data['failed_pockets']==[]


def test_manual_pipeline_does_not_reapply_bad_automatic_table():
    frame=np.zeros((600,1000,3),np.uint8)
    result=calibrate_scene(frame,lambda f,options:{'detections':scene_detections(f,[],options['calibration_polygon'])},
                           lambda _:None,profile())
    assert result['detections'][0]['box']==[100,60,900,540]
    assert all(p['status']=='manual' for p in result['pockets']['pockets'])


@pytest.mark.parametrize('key,value',[('corners',[[0,0]]*4),('pockets',[[.5,.5]]*6),('pockets',[[float('nan'),0]]*6)])
def test_invalid_manual_profiles_are_rejected(key,value):
    data=profile();data[key]=value
    with pytest.raises(ValueError):manual_geometry(np.zeros((600,1000,3),np.uint8),data)


def test_editor_steps_undo_and_save():
    app=QApplication.instance() or QApplication([])
    editor=CalibrationEditor(np.zeros((600,1000,3),np.uint8));editor.show();app.processEvents()
    assert not editor.save.isEnabled()
    area=editor.canvas.area()
    points=profile()['corners']+profile()['pockets']
    for x,y in points:
        QTest.mouseClick(editor.canvas,Qt.LeftButton,pos=QPoint(round(area.left()+x*area.width()),round(area.top()+y*area.height())))
    assert editor.save.isEnabled()
    editor.undo.click();assert len(editor.canvas.points)==9 and not editor.save.isEnabled()
    x,y=points[-1]
    QTest.mouseClick(editor.canvas,Qt.LeftButton,pos=QPoint(round(area.left()+x*area.width()),round(area.top()+y*area.height())))
    editor.save.click()
    assert editor.result()==editor.Accepted


def test_selected_table_keeps_confirmed_plane_without_running_detector():
    from scripts.yolo_click_worker import detect_selected
    points=[[100,100],[900,100],[900,500],[100,500]]
    result=detect_selected(None,np.zeros((600,1000,3),np.uint8),{'name':'billiards_table'},points)
    assert result[0]['polygon']==points and result[0]['box']==[100,100,900,500]


def test_save_and_restore_manual_profile_for_current_source(tmp_path,monkeypatch):
    import main
    import src.calibration_editor as editor_module
    app=QApplication.instance() or QApplication([])
    cv2.imwrite(str(tmp_path/'scene.png'),np.zeros((600,1000,3),np.uint8))
    window=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    path=tmp_path/'saved.json'
    monkeypatch.setattr(window,'calibration_profile_path',lambda:path)
    calls=[];monkeypatch.setattr(window,'calibrate_table',lambda **kwargs:calls.append(True))
    class Dialog:
        Accepted=1
        clear_saved=False
        def __init__(self,*args):pass
        def exec_(self):return 1
        def profile(self):return profile()
    monkeypatch.setattr(editor_module,'CalibrationEditor',Dialog)
    window.edit_calibration()
    import json
    assert json.loads(path.read_text())==profile()
    assert calls==[True]
    Dialog.clear_saved=True
    window.edit_calibration()
    assert not path.exists() and calls==[True,True]
    window.close()
