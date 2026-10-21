"""Deterministic time-based evidence tests; no inference predictions are fabricated as real accuracy."""
import numpy as np
import pytest
from src.vision.pocket_event import *

CENTERS={'TOP_LEFT':(0,0),'TOP_MIDDLE':(1,0),'TOP_RIGHT':(2,0),
         'BOTTOM_LEFT':(0,1),'BOTTOM_MIDDLE':(1,1),'BOTTOM_RIGHT':(2,1)}


def zones(status='observed'):
    pockets=[]
    for name,(x,y) in CENTERS.items():
        r=.035
        pockets.append({'id':name,'center':(x,y),'status':status,
                        'entry_polygon':[[x-r,y-r],[x+r,y-r],[x+r,y+r],[x-r,y+r]]})
    return make_zones(pockets,np.array([1,.5]),EventConfig())


def observation(ball,point,id=1):return BallObservation(ball,tuple(point),.015,.95,id)


def run_case(pocket='TOP_RIGHT',ball='black_ball',kind='pot',status='observed',frequency=10):
    e=PocketEventEngine(zones(status))
    c=np.asarray(CENTERS[pocket]);direction=(np.array([1,.5])-c);direction/=np.linalg.norm(direction)
    results=[]
    for t in np.arange(0,3.5,1/frequency):
        others=[observation('white_ball' if ball=='black_ball' else 'black_ball',(.7,.5),5)]
        if t<.8:
            point=c+direction*max(.008,.25-.25*t/.65)
            others.append(observation(ball,point,1 if t<.4 else 99))
        elif kind in ('bounce','stop','miss','occlusion'):
            if kind in ('bounce','stop') or t>=1.1:
                point=c+direction*(.12 if kind=='bounce' else .035)
                others.append(observation(ball,point))
        if kind=='elsewhere' and t<.8:others[-1]=observation(ball,(.7+t*.15,.5))
        clear=not(kind=='occlusion' and .7<t<1.2)
        results+=e.update(float(t),others,scene_clear=clear,scene_stable=t>=1.2)
    return e,results

@pytest.mark.parametrize('pocket',list(CENTERS))
@pytest.mark.parametrize('frequency',[5,10,25])
def test_black_pots_all_six_pockets_by_time_not_frame_count(pocket,frequency):
    engine,results=run_case(pocket,frequency=frequency)
    assert len(results)==1
    black,white=results[0]['events']
    assert black['event']=='POTTED' and black['pocket']==pocket
    assert white['event']=='NOT_POTTED' and white['pocket'] is None
    assert 'observed_entry_intersection' in black['evidence']

@pytest.mark.parametrize('kind',['bounce','stop','miss','occlusion'])
def test_no_pot_on_reappearance_or_rebound(kind):
    engine,results=run_case(kind=kind)
    assert results[-1]['events'][0]['event']=='NOT_POTTED'
    assert not any(event['event']=='POTTED' for r in results for event in r['events'])


def test_mid_table_disappearance_unknown():
    _,results=run_case(kind='elsewhere')
    assert results[-1]['events'][0]['event']=='UNKNOWN'
    assert results[-1]['events'][0]['pocket'] is None


def test_estimated_geometry_never_confirms():
    _,results=run_case(status='estimated')
    assert results[-1]['events'][0]['event']=='UNKNOWN'


def test_white_has_independent_scratch_signal():
    _,results=run_case(ball='white_ball',pocket='BOTTOM_MIDDLE')
    assert results[0]['events'][1]['signal']=='WHITE_BALL_POTTED'
    assert results[0]['events'][1]['pocket']=='BOTTOM_MIDDLE'


def test_pause_and_video_jump_do_not_add_evidence():
    e=PocketEventEngine(zones());o=observation('black_ball',(.9,.1))
    e.update(1,[o]);history=list(e.states['black_ball'].history)
    for _ in range(20):assert not e.update(1,[],scene_stable=True)
    assert e.states['black_ball'].history==history
    result=e.update(10,[],scene_stable=True)
    assert result[0]['events'][0]['event']=='UNKNOWN'
    assert result[0]['events'][0]['reason']=='video_timeline_discontinuity'
    e.update(.1,[o]);assert e.states['black_ball'].last_seen==.1


def test_occluded_near_entry_does_not_confirm_after_occluder_remains():
    e=PocketEventEngine(zones())
    for t,point in [(0,(1,.25)),(.2,(1,.15)),(.4,(1,.025))]:
        e.update(t,[observation('black_ball',point)],scene_clear=t<.4)
    results=[]
    for t in np.arange(.6,3,.2):results+=e.update(float(t),[],scene_clear=True,scene_stable=True)
    assert results[0]['events'][0]['event']=='UNKNOWN'
    assert results[0]['events'][0]['reason']=='occlusion_requires_review'


def test_missing_without_baseline_and_duplicate_class_unknown():
    e=PocketEventEngine(zones())
    e.update(0,[])
    e.update(.2,[observation('black_ball',(1,.2)),observation('black_ball',(1.1,.2))])
    results=[]
    for t in np.arange(.4,2,.2):results+=e.update(float(t),[],scene_stable=True)
    assert all(x['event']=='UNKNOWN' for x in results[0]['events'])


def test_class_confusion_near_entry_unknown():
    e=PocketEventEngine(zones())
    for t,y in [(0,.25),(.2,.15),(.4,.025)]:e.update(t,[observation('black_ball',(1,y))])
    results=[]
    for t in np.arange(.6,2.5,.2):results+=e.update(float(t),[],scene_stable=True,alternatives=[(1,.025)])
    assert results[0]['events'][0]['event']=='UNKNOWN'


def test_visible_rebound_followed_by_disappearance_does_not_confirm():
    e=PocketEventEngine(zones())
    for t,y in [(0,.25),(.2,.15),(.4,.025),(.6,.10)]:e.update(t,[observation('black_ball',(1,y))])
    results=[]
    for t in np.arange(.8,3,.2):results+=e.update(float(t),[],scene_stable=True)
    assert results[0]['events'][0]['event']=='UNKNOWN'


def test_recording_ending_before_stable_review_never_confirms():
    e=PocketEventEngine(zones())
    for t,y in [(0,.25),(.2,.15),(.4,.025)]:e.update(t,[observation('black_ball',(1,y))])
    e.update(.6,[])
    r=e.finish(force_unknown='recording_ended_before_final_review')
    assert r[0]['events'][0]['event']=='UNKNOWN'


def test_late_reappearance_revises_same_shot_without_duplicate_pot():
    e,results=run_case()
    assert results[0]['events'][0]['event']=='POTTED'
    revised=e.update(3.6,[observation('black_ball',(1.9,.05))],scene_stable=True)
    assert revised[0]['shot_id']==results[0]['shot_id']
    assert revised[0]['revision']==1 and revised[0]['events'][0]['event']=='NOT_POTTED'


def test_missing_manual_entry_width_requires_review():
    from src.vision.pocket_event.pocket_zone import make_zones
    z=make_zones([{'id':'TOP_MIDDLE','center':(1,0),'status':'manual'}],np.array([1,.5]),EventConfig())
    assert not z[0].trusted_shape


def test_both_balls_can_pot_independently_in_one_shot():
    e=PocketEventEngine(zones());results=[]
    for t in np.arange(0,3.5,.1):
        seen=[]
        if t<.8:
            gap=max(.008,.25-.25*t/.65)
            seen=[observation('black_ball',(1,gap)),observation('white_ball',(1,1-gap),2)]
        results+=e.update(float(t),seen,scene_stable=t>=1.2)
    assert len(results)==1
    assert [(v['event'],v['pocket']) for v in results[0]['events']]==[('POTTED','TOP_MIDDLE'),('POTTED','BOTTOM_MIDDLE')]


def test_second_shot_has_new_id_and_final_events_are_not_repeated():
    e,results=run_case(kind='both_visible')
    assert len(results)==1
    # Constant stable input after the final review must not create more events.
    for t in np.arange(3.6,4,.1):assert not e.update(float(t),[],scene_stable=True)
    assert not e.update(4.1,[observation('black_ball',(1,.2))],scene_stable=False)
    assert e.shot_id=='shot_0002'
    more=[]
    for t in np.arange(4.2,5.5,.1):more+=e.update(float(t),[observation('black_ball',(1,.2))],scene_stable=True)
    assert len(more)==1 and more[0]['shot_id']=='shot_0002'


def test_low_confidence_observation_cannot_supply_entry_evidence():
    e=PocketEventEngine(zones())
    e.update(0,[observation('black_ball',(1,.2))])
    e.update(.2,[BallObservation('black_ball',(1,.02),confidence=.1)])
    results=[]
    for t in np.arange(.4,2,.2):results+=e.update(float(t),[],scene_stable=True)
    assert results[0]['events'][0]['event']=='UNKNOWN'


def test_clear_pocket_can_confirm_while_other_balls_keep_moving():
    e=PocketEventEngine(zones())
    results=[]
    for t in np.arange(0,3,.1):
        observations=[observation('white_ball',(.5+t*.05,.5),5)]
        if t<.8:observations.append(observation('black_ball',(2-max(.008,.25-.25*t/.65),.005)))
        results+=e.update(float(t),observations,scene_clear=False,scene_stable=False,
                          pocket_clear={z.id:True for z in e.zones},shot_motion=True)
    pots=[event for r in results for event in r['events'] if event['event']=='POTTED']
    assert len(pots)==1 and pots[0]['ball']=='black_ball'


def test_missing_midtable_observation_does_not_permanently_poison_later_entry():
    e=PocketEventEngine(zones())
    e.update(0,[observation('black_ball',(.8,.5))])
    e.update(.2,[],scene_clear=False)
    results=[]
    for t,point in [(.4,(1.75,.005)),(.6,(1.9,.005)),(.8,(1.99,.005))]:
        results+=e.update(t,[observation('black_ball',point)],scene_clear=True)
    for t in np.arange(1,3,.1):
        results+=e.update(float(t),[],scene_clear=True,scene_stable=True)
    assert any(event['event']=='POTTED' for r in results for event in r['events'])
