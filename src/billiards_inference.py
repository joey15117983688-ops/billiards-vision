"""Class confidence thresholds shared by the player and inference acceptance checks."""
CLASS_CONFIDENCE = {'billiards_table': 0.60, 'billiard_ball': 0.25, 'white_ball': 0.25}


def filter_detections(result):
    keep = [index for index, box in enumerate(result.boxes)
            if float(box.conf[0]) >= CLASS_CONFIDENCE.get(result.names[int(box.cls[0])], 1.0)]
    result.boxes = result.boxes[keep]
    return result
