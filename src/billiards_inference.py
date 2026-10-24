"""Class confidence thresholds shared by the player and inference acceptance checks."""
# 模型输出的类别置信度过滤入口；界面和离线评估复用同一门槛。
# 仅过滤已有检测，不补造框、不更换模型权重。

CLASS_CONFIDENCE = {'billiards_table': 0.60, 'billiard_ball': 0.25, 'white_ball': 0.25}


# 按 CLASS_CONFIDENCE 的类别门槛筛选 boxes，未知类别默认门槛为 1.0；直接更新传入结果。
def filter_detections(result):
    keep = [index for index, box in enumerate(result.boxes)
            if float(box.conf[0]) >= CLASS_CONFIDENCE.get(result.names[int(box.cls[0])], 1.0)]
    result.boxes = result.boxes[keep]
    return result
