"""Preflight measurements and temporal eligibility, not a recognition verdict.

A passed assessment only permits downstream analysis; actual pot evidence is
still required. Defaults are provisional and must be tuned with venue samples.
"""
from dataclasses import dataclass
import time
import cv2
import numpy as np
from src.live_source import StreamFrame


@dataclass(frozen=True)
class ReadinessPolicy:
    max_frame_age_ms: float=500
    max_observation_gap: float=.2
    recovery_seconds: float=1.
    min_mean_luma: float=20
    max_mean_luma: float=235
    min_focus_variance: float | None=None

    def __post_init__(self):
        if self.max_frame_age_ms<=0 or self.max_observation_gap<=0 or self.recovery_seconds<0:
            raise ValueError('画面时效／连续性参数无效')
        if not 0<=self.min_mean_luma<self.max_mean_luma<=255:raise ValueError('亮度参数无效')
        if self.min_focus_variance is not None and self.min_focus_variance<0:raise ValueError('清晰度参数无效')


@dataclass(frozen=True)
class FrameReadiness:
    can_detect: bool
    can_measure: bool
    can_commit_event: bool
    reasons: tuple[str,...]
    receive_to_consume_ms: float
    skipped_frames: int
    continuity_broken: bool
    mean_luma: float
    focus_variance: float
    dark_fraction: float
    highlight_fraction: float


class LiveReadiness:
    def __init__(self,policy=None):
        self.policy=policy or ReadinessPolicy()
        self.previous=None;self.stable_since=None

    def assess(self,packet:StreamFrame,geometry_ready=False,polygon=None,now=None):
        now=time.monotonic() if now is None else now
        age=(now-packet.received_at)*1000
        reasons=[];continuity=False;skipped=0
        previous=self.previous
        if previous is None or packet.discontinuity or packet.session!=previous.session:
            reasons.append('new_session');continuity=True
        else:
            skipped=max(0,packet.sequence-previous.sequence-1)
            if packet.sequence<=previous.sequence or packet.received_at<=previous.received_at:
                reasons.append('repeated_or_reordered_frame');continuity=True
            if packet.received_at-previous.received_at>self.policy.max_observation_gap:
                reasons.append('observation_gap');continuity=True
            if packet.image.shape!=previous.image.shape:
                reasons.append('resolution_changed');continuity=True
        if age>self.policy.max_frame_age_ms or age<0:
            reasons.append('stale_frame' if age>=0 else 'invalid_timestamp');continuity=True
        image=packet.image
        scale=min(1.,640/max(image.shape[:2]))
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if scale<1:gray=cv2.resize(gray,None,fx=scale,fy=scale)
        mask=np.ones(gray.shape,np.uint8)
        if polygon is not None:
            q=np.asarray(polygon,float)
            if q.shape!=(4,2) or not np.isfinite(q).all():raise ValueError('无效的球桌范围')
            mask[:]=0;cv2.fillConvexPoly(mask,np.rint(q*scale).astype(np.int32),1)
        pixels=gray[mask>0]
        if not pixels.size:raise ValueError('球桌范围不在当前画面内')
        luma=float(pixels.mean());focus=float(cv2.Laplacian(gray,cv2.CV_32F)[mask>0].var())
        if luma<self.policy.min_mean_luma:reasons.append('underexposed')
        if luma>self.policy.max_mean_luma:reasons.append('overexposed')
        if self.policy.min_focus_variance is not None and focus<self.policy.min_focus_variance:reasons.append('blurred')
        bad_pixels=any(reason in reasons for reason in ('stale_frame','invalid_timestamp','repeated_or_reordered_frame','underexposed','overexposed','blurred'))
        can_detect=not bad_pixels
        can_measure=can_detect and geometry_ready
        if not geometry_ready:reasons.append('geometry_not_ready')
        if continuity or not can_measure:self.stable_since=None
        if can_measure and self.stable_since is None:self.stable_since=packet.received_at
        recovered=self.stable_since is not None and packet.received_at-self.stable_since>=self.policy.recovery_seconds
        can_commit=bool(can_measure and recovered and not continuity)
        if can_measure and not can_commit:reasons.append('recovering_continuity')
        self.previous=packet
        return FrameReadiness(can_detect,can_measure,can_commit,tuple(reasons),max(0,age),skipped,continuity,
                              luma,focus,float(np.mean(pixels<20)),float(np.mean(pixels>245)))
