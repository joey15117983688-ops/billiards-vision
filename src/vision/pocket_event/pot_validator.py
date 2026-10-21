"""A disappearance is never sufficient evidence of a pot."""
from .models import State


def validate(state,zone,now,config,scene_clear,stable,review):
    if state.last_seen is None:return State.UNKNOWN,'not_seen_before_shot'
    if not state.entry_seen or not state.direction_seen:return State.UNKNOWN,'no_observed_entry_trajectory'
    if zone is None or not zone.trusted_shape:return State.UNKNOWN,'pocket_geometry_requires_review'
    if state.occlusion_seen or not scene_clear:return State.UNKNOWN,'occlusion_requires_review'
    if state.ambiguous:return State.UNKNOWN,'possible_class_change_or_multiple_balls'
    if now-state.last_seen<config.min_absence_seconds:return State.POT_CANDIDATE,'waiting_for_absence_window'
    if not stable or not review:return State.POT_CANDIDATE,'waiting_for_fresh_stable_scene_review'
    return State.POTTED,'entry_trajectory_absence_and_stable_scene_verified'
