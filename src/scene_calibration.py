"""Rebuild table, pocket and ball observations from one fresh frame."""
# 单张新鲜帧上的校准协调入口；几何、识别范围与球体检测分层返回。
# 人工确认、显式自动纠错和普通模型检测是不同来源，不能把一种来源伪装成另一种。

from copy import deepcopy


# 优先使用没有纠错请求的人工配置，否则搜索并比较候选；geometry_only 跳过球体推理。
# 内库 polygon 用于坐标和袋口，scope_polygon 用于显示/过滤相连库边，两者不可互换。
# 纠错为 unchanged 时尽量保留之前范围和袋口，最终返回报告说明是否真正发生替换。
def calibrate_scene(frame, predict, progress, manual_profile=None, correction_context=None, geometry_only=False):
    progress('2/4 重新定位台呢与内库边界…')
    report=None
    if manual_profile is not None and correction_context is None:
        from src.manual_calibration import manual_geometry
        polygon,pocket_data=manual_geometry(frame,manual_profile)
        progress('3/4 使用人工确认的六个袋口位置…')
    else:
        from src.adaptive_calibration import search_calibration
        polygon,pocket_data,report=search_calibration(frame,progress,correction_context)
        if report['decision']=='unchanged' and manual_profile is not None:
            from src.manual_calibration import manual_geometry
            manual_polygon,manual_pockets=manual_geometry(frame,manual_profile)
            from src.adaptive_calibration import same_polygon
            if same_polygon(polygon,manual_polygon):pocket_data=manual_pockets
    if geometry_only:
        from src.cloth_region import cloth_region
        progress('4/4 更新球桌范围、袋口与坐标…')
        region=cloth_region(frame,polygon)
        scope=region['polygon'] if region else polygon
        xs,ys=zip(*scope)
        detections=[{'name':'billiards_table','confidence':1.,'box':[min(xs),min(ys),max(xs),max(ys)],
                     'polygon':scope,'playable_polygon':polygon,'scope_evidence':region}]
    else:
        progress('4/4 重新识别台面上的球并更新坐标…')
        reply = predict(frame, {'mode':'all', 'calibration_polygon':polygon})
        detections = reply['detections']
    if not any(d['name']=='billiards_table' for d in detections):
        raise ValueError('台面识别失败，请重新校准')
    # The whole connected cloth is the recognition scope. Physical coordinates
    # and pocket geometry retain the separately calibrated inner plane.
    table=next(d for d in detections if d['name']=='billiards_table')
    scope=table.get('polygon') or polygon
    if report is not None:
        previous_scope=(correction_context or {}).get('previous_scope')
        from src.adaptive_calibration import same_polygon
        evidence=table.get('scope_evidence')
        improved=False
        if evidence and previous_scope is not None and not same_polygon(scope,previous_scope):
            from src.cloth_region import scope_improvement
            improved=scope_improvement(frame,previous_scope,scope,evidence['hue'])
        if report['decision']=='unchanged':
            # A failed correction must not silently replace the blue outline or
            # pocket centres while reporting that the old result was retained.
            if previous_scope is not None and not improved:scope=deepcopy(previous_scope)
            previous_pockets=(correction_context or {}).get('previous_pockets')
            if previous_pockets is not None:pocket_data=deepcopy(previous_pockets)
        report['scope_corrected']=bool(improved)
        report['scope_evidence']=evidence
    pocket_data['table']['scope_polygon']=scope
    table['polygon']=scope
    xs,ys=zip(*scope);table['box']=[min(xs),min(ys),max(xs),max(ys)]
    return {'polygon':polygon, 'pockets':pocket_data, 'detections':detections, 'calibration_report':report,'scope_polygon':scope}
