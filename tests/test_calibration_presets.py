"""Venue defaults persist independently of filenames and opening-frame occlusion."""
import hashlib
import json
import cv2
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication
import main
from src.calibration_presets import CalibrationPresets
from src.table_coordinates import TableDimensions
from pocket_scene import make_scene

@pytest.fixture(scope='module')
def app():return QApplication.instance() or QApplication([])


def manual(root,source,inset):
    path=root/'config/calibration'/(hashlib.sha256(str(source.resolve()).encode()).hexdigest()+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    a,b=inset,1-inset
    path.write_text(json.dumps({'corners':[[a,a],[b,a],[b,b],[a,b]],
                                'pockets':[[a,a],[.5,a],[b,a],[a,b],[.5,b],[b,b]]}))


def test_default_preset_reopens_for_new_filename_without_image_fitting(app,tmp_path,monkeypatch):
    frame,_=make_scene();source=tmp_path/'camera_a.png';cv2.imwrite(str(source),frame)
    manual(tmp_path,source,.15);monkeypatch.setattr(main,'ROOT',tmp_path)
    w=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    try:
        w.canvas.dimensions=TableDimensions(3000,1500,False)
        identity=w.create_calibration_preset('朝阳俱乐部 · 1号桌')
        polygon=w.canvas.calibration_polygon;scope=w.canvas.recognition_polygon
        pockets=w.canvas.pocket_result
        assert w.calibration_presets.data['default']==identity
    finally:w.close()
    source.rename(tmp_path/'new_recording.png')
    cv2.imwrite(str(tmp_path/'new_recording.png'),np.zeros_like(frame))
    import src.cloth_region,src.adaptive_calibration
    monkeypatch.setattr(src.cloth_region,'cloth_region',lambda *a:pytest.fail('must not refit saved camera setup'))
    monkeypatch.setattr(src.adaptive_calibration,'ImageEvidence',lambda *a:pytest.fail('must not reverify opening-frame pixels'))
    reopened=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    try:
        assert reopened.active_calibration_preset==identity
        assert np.allclose(reopened.canvas.calibration_polygon,polygon)
        assert reopened.canvas.recognition_polygon==scope
        assert reopened.canvas.pocket_result==pockets
        assert reopened.canvas.dimensions==TableDimensions(3000,1500,False)
        assert reopened.calibration_thread is None
    finally:reopened.close()


def test_venue_tables_are_independent_and_default_is_explicit(app,tmp_path,monkeypatch):
    frame=np.zeros((600,1000,3),np.uint8)
    for name,inset in [('a.png',.1),('b.png',.2)]:
        source=tmp_path/name;cv2.imwrite(str(source),frame);manual(tmp_path,source,inset)
    monkeypatch.setattr(main,'ROOT',tmp_path)
    w=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    try:
        a=w.create_calibration_preset('甲球房 · 1号桌');polygon_a=w.canvas.calibration_polygon
        w.calibration_preset_combo.setCurrentIndex(0);w.show_image(1)
        w.canvas.dimensions=TableDimensions(2500,1250,False)
        b=w.create_calibration_preset('乙球房 · 2号桌');polygon_b=w.canvas.calibration_polygon
        assert polygon_a!=polygon_b and w.calibration_presets.data['default']==a
        w.set_default_calibration_preset();assert w.calibration_presets.data['default']==b
        w.calibration_preset_combo.setCurrentIndex(w.calibration_preset_combo.findData(a))
        assert w.canvas.calibration_polygon==polygon_a
        assert w.canvas.dimensions!=TableDimensions(2500,1250,False)
        assert w.calibration_thread is None
    finally:w.close()
    reopened=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    try:
        assert reopened.active_calibration_preset==b
        assert reopened.canvas.calibration_polygon==polygon_b
        assert reopened.canvas.dimensions==TableDimensions(2500,1250,False)
    finally:reopened.close()


def test_saved_geometry_scales_only_for_same_aspect_ratio(app,tmp_path,monkeypatch):
    source=tmp_path/'a.png';cv2.imwrite(str(source),np.zeros((600,1000,3),np.uint8));manual(tmp_path,source,.1)
    monkeypatch.setattr(main,'ROOT',tmp_path)
    w=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    try:
        w.create_calibration_preset('固定机位')
        original=np.asarray(w.canvas.calibration_polygon)
    finally:w.close()
    cv2.imwrite(str(source),np.zeros((300,500,3),np.uint8))
    w=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    assert np.allclose(w.canvas.calibration_polygon,original*.5);w.close()
    cv2.imwrite(str(source),np.zeros((500,300,3),np.uint8))
    w=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    assert w.canvas.calibration_polygon is None
    assert w._calibration_restore_note
    w.close()


def test_corrupt_preset_index_is_preserved_and_duplicate_names_rejected(tmp_path):
    store=CalibrationPresets(tmp_path);a=store.create('球房 A')
    with pytest.raises(ValueError):store.create('球房 A')
    assert len(store.data['profiles'])==1 and store.data['default']==a
    store.index.write_text('broken index')
    bad=CalibrationPresets(tmp_path)
    with pytest.raises(ValueError):bad.create('新配置')
    assert bad.index.read_text()=='broken index'


def test_explicit_manual_update_overwrites_only_selected_preset(app,tmp_path,monkeypatch):
    from src.manual_calibration import manual_geometry
    frame=np.zeros((600,1000,3),np.uint8);source=tmp_path/'a.png'
    cv2.imwrite(str(source),frame);manual(tmp_path,source,.1)
    monkeypatch.setattr(main,'ROOT',tmp_path)
    w=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    try:
        identity=w.create_calibration_preset('甲球房')
        profile={'corners':[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],
                 'pockets':[[.2,.2],[.5,.2],[.8,.2],[.2,.8],[.5,.8],[.8,.8]]}
        w.calibration_profile_path().write_text(json.dumps(profile))
        polygon,pockets=manual_geometry(frame,profile)
        result={'polygon':polygon,'pockets':pockets,'detections':[],'calibration_report':None}
        w.on_calibration_done(w.media_generation,frame,result)
        assert w.canvas.calibration_polygon==polygon
    finally:w.close()
    w=main.MainWindow(image_dir=tmp_path,start_with_images=True)
    assert w.active_calibration_preset==identity and w.canvas.calibration_polygon==polygon
    w.close()


def test_failed_default_write_does_not_change_in_memory_default(tmp_path,monkeypatch):
    import src.calibration_presets
    store=CalibrationPresets(tmp_path);first=store.create('A');second=store.create('B')
    before=store.index.read_text()
    monkeypatch.setattr(src.calibration_presets,'write_json',lambda *a:(_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):store.set_default(second)
    assert store.data['default']==first and store.index.read_text()==before
