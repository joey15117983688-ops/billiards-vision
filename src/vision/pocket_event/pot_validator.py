"""A disappearance is never sufficient evidence of a pot."""
# 入袋最终资格检查；仅使用状态机已有的证据，不从“没看到”反推“已进球”。
# 所有门槛均通过才确认；证据缺失返回 UNKNOWN，时间/稳定复核尚未满足返回 POT_CANDIDATE。

from .models import State


# 依次检查历史可见性、实测入口方向、可信袋形、遮挡和身份歧义，再检查消失与稳定复核时长。
def validate(state,zone,now,config,scene_clear,stable,review):
    if state.last_seen is None:return State.UNKNOWN,'not_seen_before_shot'
    if not state.entry_seen or not state.direction_seen:return State.UNKNOWN,'no_observed_entry_trajectory'
    if zone is None or not zone.trusted_shape:return State.UNKNOWN,'pocket_geometry_requires_review'
    if state.occlusion_seen or not scene_clear:return State.UNKNOWN,'occlusion_requires_review'
    if state.ambiguous:return State.UNKNOWN,'possible_class_change_or_multiple_balls'
    if now-state.last_seen<config.min_absence_seconds:return State.POT_CANDIDATE,'waiting_for_absence_window'
    if not stable or not review:return State.POT_CANDIDATE,'waiting_for_fresh_stable_scene_review'
    return State.POTTED,'entry_trajectory_absence_and_stable_scene_verified'
