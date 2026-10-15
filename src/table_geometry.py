"""Estimate the visible green table surface inside a YOLO table box (not an OBB model)."""
import cv2
import numpy as np


def table_polygon(frame, box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2-x1 < 30 or y2-y1 < 30:
        return None
    roi = frame[y1:y2, x1:x2]
    # Work at a bounded size; ball holes and small cue occlusions are closed below.
    scale = min(1.0, 900 / max(roi.shape[:2]))
    small = cv2.resize(roi, None, fx=scale, fy=scale) if scale < 1 else roi
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, np.array([30, 40, 25]), np.array([95, 255, 255])) |
            cv2.inRange(hsv, np.array([96, 150, 80]), np.array([125, 255, 255])))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < mask.size * .20:
        return None
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for fraction in [.01, .015, .02, .025, .03, .04, .05]:
        corners = cv2.approxPolyDP(hull, fraction * perimeter, True)
        if len(corners) != 4 or not cv2.isContourConvex(corners):
            continue
        area = cv2.contourArea(corners)
        if area < .80 * cv2.contourArea(hull):
            continue
        polygon_mask = np.zeros_like(mask)
        cv2.fillConvexPoly(polygon_mask, corners, 255)
        support = cv2.countNonZero(cv2.bitwise_and(mask, polygon_mask)) / max(1, cv2.countNonZero(polygon_mask))
        if support < .75:
            continue
        points = corners.reshape(4, 2).astype(float) / scale + [x1, y1]
        # Keep contour order, with a stable top-left-ish starting vertex.
        points = np.roll(points, -np.argmin(points.sum(axis=1)), axis=0)
        points[:, 0] = np.clip(points[:, 0], 0, width-1)
        points[:, 1] = np.clip(points[:, 1], 0, height-1)
        return points.tolist()
    return None


def prediction_detections(prediction, frame):
    detections = []
    for box in prediction.boxes:
        name = prediction.names[int(box.cls[0])]
        item = {'name': name, 'confidence': float(box.conf[0]), 'box': box.xyxy[0].cpu().tolist()}
        if name == 'billiards_table':
            item['polygon'] = table_polygon(frame, item['box'])
            item['geometry'] = 'table_surface' if item['polygon'] else 'bbox_fallback'
        detections.append(item)
    # Merge strongly overlapping partial-table proposals, then fit their combined surface.
    # Disjoint tables remain separate.
    tables = sorted((d for d in detections if d['name'] == 'billiards_table'),
                    key=lambda d: (d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]), reverse=True)
    kept = []
    for item in tables:
        x1,y1,x2,y2 = item['box']
        area = max((x2-x1)*(y2-y1), 1)
        nested = False
        for other in kept:
            a,b,c,d = other['box']
            intersection = max(0,min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))
            if intersection / area > .65:
                other['box'] = [min(x1,a), min(y1,b), max(x2,c), max(y2,d)]
                other['confidence'] = max(other['confidence'], item['confidence'])
                other['polygon'] = table_polygon(frame, other['box'])
                other['geometry'] = 'table_surface' if other['polygon'] else 'bbox_fallback'
                nested = True
                break
        if not nested:
            kept.append(item)
    return kept + [d for d in detections if d['name'] != 'billiards_table']


def contains_point(item, x, y):
    if item.get('polygon'):
        return cv2.pointPolygonTest(np.array(item['polygon'], np.float32), (float(x), float(y)), False) >= 0
    x1, y1, x2, y2 = item['box']
    return x1 <= x <= x2 and y1 <= y <= y2
