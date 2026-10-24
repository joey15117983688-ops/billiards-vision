"""Decoder/model-free, timestamp-based black/white ball event state machine."""
# 纯时间戳状态机：接收实测观测和可见性信号，不读取视频、调用模型或操作 UI。
# TRACKING → APPROACHING_POCKET → POT_CANDIDATE 的证据需连续；消失本身不足以确认 POTTED。
# 时间单位为源视频秒；同一暂停帧不增加证据，回退/长间隔打断当前杆并要求复核。

import math
import numpy as np
from .models import BALLS, BallState, EventConfig, PocketEvent, State
from .pot_validator import validate


class PocketEventEngine:
    # 注入几何入口、阈值配置和待跟踪身份列表，然后建立干净事件会话。
    def __init__(self,zones=(),config=None,balls=BALLS):
        self.config=config or EventConfig();self.balls=tuple(balls);self.zones=list(zones)
        self.reset()

    # 重置每球状态、稳定窗、杆号与结果；换源时必须调用以免混用旧时间线。
    def reset(self,reason='new_source'):
        self.states={ball:BallState(reason=reason) for ball in self.balls}
        self.timestamp=None;self.stable_since=None;self.started=None
        self.shot_number=1;self.completed=False;self.results=[];self.reason=reason

    # 生成当前杆的稳定标识；调用方用它与球身份组合去重、应用修订。
    @property
    def shot_id(self):return f'shot_{self.shot_number:04d}'

    # 验证源时间单调性，将合格实测观测按身份分组，并区分运动与稳定复核窗口。
    # 有观测时积累朝向和线段穿入口证据；无观测时检查消失窗、遮挡及其它球/类别替代的歧义。
    # 已完成杆中的球重新出现可产生 NOT_POTTED 修订；必须替换旧事件而非再加一个计数。
    def update(self,timestamp,observations,*,scene_clear=True,scene_stable=False,final_review=True,alternatives=(),shot_motion=None,pocket_clear=None):
        now=float(timestamp)
        if not math.isfinite(now) or now<0:raise ValueError('timestamp must be finite video seconds')
        # 暂停重复帧不算新证据；倒放或过大时间间隔先截断上一段观测。
        if self.timestamp is not None:
            if now==self.timestamp:return []  # paused/repeated frame: no extra evidence
            if now<self.timestamp or now-self.timestamp>self.config.max_observation_gap:
                interrupted=self.finish(self.timestamp,force_unknown='video_timeline_discontinuity') if not self.completed else []
                self.states={ball:BallState(reason='video_timeline_discontinuity') for ball in self.balls}
                self.shot_number+=1;self.completed=False;self.started=None;self.stable_since=None
            else:interrupted=[]
        else:interrupted=[]
        self.timestamp=now
        if self.started is None:self.started=now
        grouped={ball:[] for ball in self.balls}
        for o in observations:
            if o.ball in grouped and o.confidence>=self.config.min_confidence and np.isfinite(o.center).all():grouped[o.ball].append(o)
        moving=not scene_stable if shot_motion is None else shot_motion
        # 已完成杆的晚到实测可能否定旧确认；只有真正运动才开启下一杆。
        if self.completed:
            # Begin another shot only on actual motion, not every stationary review.
            if not moving:
                # A late real observation contradicts an earlier pot decision.
                revised=False
                for ball,state in self.states.items():
                    if state.state==State.POTTED and grouped[ball]:
                        state.state=State.NOT_POTTED;state.last_seen=now;state.reason='reappeared_after_confirmation'
                        event=next(e for e in self.results[-1]['events'] if e['ball']==ball)
                        event.update(event='NOT_POTTED',pocket=None,timestamp=None,reason=state.reason,
                                     evidence=event['evidence']+['reappeared_after_confirmation'],signal='NOT_POTTED')
                        revised=True
                if revised:
                    self.results[-1]['revision']=self.results[-1].get('revision',0)+1
                    self.results[-1]['review_timestamp']=now
                    from copy import deepcopy
                    return interrupted+[deepcopy(self.results[-1])]
                return interrupted
            self.shot_number+=1;self.completed=False;self.started=now;self.stable_since=None
            for state in self.states.values():
                state.state=State.TRACKING;state.pocket=None;state.evidence=[];state.missing_since=None
                state.entry_seen=False;state.direction_seen=False;state.occlusion_seen=False;state.ambiguous=False;state.clear_since=None
                state.history=state.history[-2:]
        # 稳定复核必须是一段连续清晰且静止的时间，不能累计零散好帧。
        if scene_clear and scene_stable:
            if self.stable_since is None:self.stable_since=now
        else:self.stable_since=None
        stable=self.stable_since is not None and now-self.stable_since>=self.config.stable_seconds
        for ball,state in self.states.items():
            candidates=grouped[ball]
            local_clear=(pocket_clear or {}).get(state.pocket,scene_clear)
            if len(candidates)>1:
                state.ambiguous=True;state.reason='multiple_same_class_observations';continue
            # 看见球：保留有限历史，在附近袋口检查运动朝向和线段入口交集。
            if candidates:
                o=candidates[0]
                if state.missing_since is not None:
                    state.evidence.append('reappeared_after_disappearance')
                    state.entry_seen=False;state.direction_seen=False;state.pocket=None
                    state.occlusion_seen=False;state.ambiguous=False
                state.missing_since=None;state.clear_since=None
                history=state.history
                state.last_seen=now
                state.history=(history+[(now,tuple(o.center),o.radius,o.track_id)])[-32:]
                state.state=State.TRACKING;state.reason='visible_on_table'
                nearby=[z for z in self.zones if z.near(o.center,o.radius)]
                if nearby:
                    z=min(nearby,key=lambda z:z.distance(o.center))
                    if state.pocket!=z.id:state.entry_seen=False;state.direction_seen=False
                    state.pocket=z.id;state.state=State.APPROACHING_POCKET
                    state.evidence=list(dict.fromkeys(state.evidence+['approached_pocket']))
                    recent=[h for h in history if .04<=now-h[0]<=.8]
                    if recent and now-recent[0][0]>=self.config.min_track_seconds:
                        start=recent[0][1]
                        receding=z.distance(o.center)>z.distance(recent[-1][1])+.002
                        toward=not receding and z.toward(start,o.center)
                        if receding and not z.touches_entry(o.center,o.radius):
                            state.direction_seen=False;state.entry_seen=False
                        elif toward:
                            state.direction_seen=True
                        state.entry_seen=state.entry_seen or (toward and z.segment_enters(start,o.center,o.radius))
                        if toward:state.evidence=list(dict.fromkeys(state.evidence+['trajectory_toward_entry']))
                        if state.entry_seen:state.evidence=list(dict.fromkeys(state.evidence+['observed_entry_intersection']))
                else:
                    state.pocket=None;state.entry_seen=False;state.direction_seen=False
                if nearby and not (pocket_clear or {}).get(state.pocket,scene_clear):state.occlusion_seen=True
            # 球消失：保留最后实测时间，遮挡/类别替换风险会阻止直接确认。
            elif state.last_seen is not None:
                if state.missing_since is None:state.missing_since=now
                if state.pocket is not None and not local_clear:
                    state.occlusion_seen=True;state.clear_since=None
                elif state.clear_since is None:state.clear_since=now
                z=next((z for z in self.zones if z.id==state.pocket),None)
                # A nearby ball with another identity may be a fragmented track
                # or a class switch. Do not count its former identity as a pot.
                other_positions=list(alternatives)+[o.center for key,items in grouped.items() if key!=ball for o in items]
                if state.history and any(np.linalg.norm(np.asarray(p)-state.history[-1][1])<max(.04,state.history[-1][2]*3) for p in other_positions):state.ambiguous=True
                if z and state.entry_seen and state.direction_seen:
                    state.evidence=list(dict.fromkeys(state.evidence+['disappeared_near_entry']))
                    local_review=(pocket_clear is not None and state.clear_since is not None and now-state.clear_since>=self.config.stable_seconds)
                    verdict,reason=validate(state,z,now,self.config,local_clear,stable or local_review,final_review)
                    state.state=verdict;state.reason=reason
                else:state.state=State.UNKNOWN;state.reason='disappeared_without_entry_evidence'
        if any(s.state==State.POTTED for s in self.states.values()):
            return interrupted+self.finish(now)
        # 不能在最短消失窗尚未完成时提前用整桌稳定判定结束候选。
        waiting=any(s.state==State.POT_CANDIDATE and s.last_seen is not None and now-s.last_seen<self.config.min_absence_seconds for s in self.states.values())
        if stable and final_review and not waiting and now-self.started>=self.config.stable_seconds:
            return interrupted+self.finish(now)
        if any(s.pocket is not None and s.missing_since is not None and now-s.missing_since>=self.config.candidate_timeout for s in self.states.values()):
            return interrupted+self.finish(now,force_unknown='confirmation_timeout')
        return interrupted

    # 为整杆输出一次结构化裁决：当前可见为 NOT_POTTED，证据确认才保留 POTTED，其余 UNKNOWN。
    # force_unknown 用于换源/跳时/超时收尾；结果保留最近 128 杆并深拷贝返回，防止外部修改内部历史。
    def finish(self,timestamp=None,force_unknown=None):
        if self.completed:return []
        now=self.timestamp if timestamp is None else timestamp
        events=[]
        for ball,state in self.states.items():
            visible=state.last_seen is not None and state.last_seen==now
            if force_unknown:verdict=State.UNKNOWN;reason=force_unknown
            elif visible:verdict=State.NOT_POTTED;reason='visible_after_shot'
            elif state.state==State.POTTED:verdict=State.POTTED;reason=state.reason
            else:verdict=State.UNKNOWN;reason=state.reason
            state.state=verdict;state.reason=reason
            evidence=list(dict.fromkeys(state.evidence+(['visible_after_shot'] if visible else ['not_detected_after_shot'] if verdict==State.POTTED else [])))
            event=PocketEvent(ball,verdict.value,state.pocket if verdict==State.POTTED else None,
                              state.last_seen if verdict==State.POTTED else None,evidence,reason,
                              'confirmed' if verdict!=State.UNKNOWN else 'requires_review',
                              'WHITE_BALL_POTTED' if ball=='white_ball' and verdict==State.POTTED else verdict.value)
            events.append(event.to_dict())
        result={'schema_version':1,'shot_id':self.shot_id,'review_timestamp':now,'events':events}
        self.results.append(result);self.results=self.results[-128:];self.completed=True
        from copy import deepcopy
        return [deepcopy(result)]

    # 返回当前杆、视频时间和各身份状态/原因/证据，供 UI 与本地诊断使用。
    def snapshot(self):
        return {'shot_id':self.shot_id,'timestamp':self.timestamp,'balls':{
            b:{'state':s.state.value,'candidate_pocket':s.pocket,'reason':s.reason,'evidence':s.evidence} for b,s in self.states.items()}}
