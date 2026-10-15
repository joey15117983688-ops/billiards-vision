"""A slanted/perspective surface must stay slanted and exclude surrounding floor."""
import cv2
import numpy as np
import pytest

from src.table_geometry import contains_point, table_polygon


@pytest.mark.parametrize('corners', [
    [[60, 160], [490, 40], [560, 310], [120, 420]],
    [[160, 70], [440, 80], [540, 400], [60, 400]],
])
def test_estimate_occluded_surface_without_axis_alignment(corners):
    frame = np.zeros((480, 640, 3), np.uint8)
    expected = np.array(corners, np.int32)
    cv2.fillConvexPoly(frame, expected, (40, 135, 40))
    cv2.circle(frame, (300, 200), 15, (255, 255, 255), -1)
    cv2.line(frame, (250, 0), (290, 140), (20, 20, 20), 20)
    polygon = table_polygon(frame, [20, 20, 600, 450])
    assert polygon is not None
    actual = np.zeros(frame.shape[:2], np.uint8)
    target = np.zeros_like(actual)
    cv2.fillConvexPoly(actual, np.rint(polygon).astype(np.int32), 1)
    cv2.fillConvexPoly(target, expected, 1)
    assert np.logical_and(actual, target).sum() / np.logical_or(actual, target).sum() > .95
    item = {'box': [20, 20, 600, 450], 'polygon': polygon}
    assert contains_point(item, 300, 220)
    assert not contains_point(item, 30, 30)


def test_no_green_surface_does_not_invent_corners():
    assert table_polygon(np.full((300, 600, 3), 150, np.uint8), [0, 0, 600, 300]) is None


def test_nested_table_candidates_merge_but_separate_tables_remain():
    import torch
    from ultralytics.engine.results import Results
    from src.table_geometry import prediction_detections
    frame = np.full((200, 400, 3), 128, np.uint8)
    prediction = Results(frame, path='tables.jpg', names={0: 'billiards_table'},
                         boxes=torch.tensor([[0,0,200,200,.9,0], [20,0,180,200,.9,0],
                                             [220,0,400,200,.9,0]]))
    result = prediction_detections(prediction, frame)
    assert len(result) == 2
    assert {tuple(d['box']) for d in result} == {(0,0,200,200), (220,0,400,200)}


def test_overlapping_partial_tables_recover_complete_surface():
    import torch
    from ultralytics.engine.results import Results
    from src.table_geometry import prediction_detections
    frame = np.full((200,400,3), (40,135,40), np.uint8)
    prediction = Results(frame, path='partial.jpg', names={0:'billiards_table'},
                         boxes=torch.tensor([[0,0,320,200,.9,0], [80,0,400,200,.9,0]]))
    items = prediction_detections(prediction, frame)
    assert len(items)==1 and items[0]['box']==[0,0,400,200]
    assert items[0]['polygon'] is not None
