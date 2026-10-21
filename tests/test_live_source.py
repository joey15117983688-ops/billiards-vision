import threading
import time
import numpy as np
import pytest
from src.live_source import LiveSourceConfig,LatestFrameBuffer,LatestFrameStream,StreamFrame
from src.live_readiness import LiveReadiness,ReadinessPolicy


def packet(sequence=1,session=1,at=None,shape=(40,60,3),value=100,discontinuity=False):
    return StreamFrame(np.full(shape,value,np.uint8),sequence,session,time.monotonic() if at is None else at,discontinuity)


def test_one_slot_never_delivers_an_old_queue_or_duplicate():
    buffer=LatestFrameBuffer()
    for n in range(10000):buffer.put(packet(n))
    assert buffer.snapshot()==dict(received=10000,delivered=0,overwritten=9999,stale=0,pending=1)
    assert buffer.take().sequence==9999
    assert buffer.take() is None
    buffer.put(packet(at=time.monotonic()-2));assert buffer.take(max_age_ms=500) is None
    assert buffer.snapshot()['stale']==1


def test_identity_ignores_changed_passwords_and_tokens():
    a=LiveSourceConfig('club/table1/cam1','rtsp://name:private@host/camera?token=secret')
    b=LiveSourceConfig('club/table1/cam1','https://other/stream?token=new')
    assert a.calibration_key==b.calibration_key
    assert a.calibration_key!=LiveSourceConfig('club/table2/cam1',b.source).calibration_key
    assert 'private' not in repr(a) and 'secret' not in repr(a)


def test_capture_reader_reconnects_with_new_session_and_clears_old_frame():
    calls=[];recovered=threading.Event();release_gate=threading.Event();released=[]
    class Capture:
        def __init__(self,number):self.number=number;self.reads=0
        def isOpened(self):return True
        def read(self):
            self.reads+=1
            if self.number==1:
                if self.reads==1:return True,np.full((8,8,3),1,np.uint8)
                return False,None
            if self.reads==1:return True,np.full((8,8,3),2,np.uint8)
            recovered.set();release_gate.wait(1)
            return False,None
        def release(self):released.append(self.number)
    def factory(config):calls.append(1);return Capture(len(calls))
    stream=LatestFrameStream(LiveSourceConfig('cam','rtsp://private',reconnect_initial=.01,reconnect_max=.02),factory)
    try:
        stream.start();assert recovered.wait(1)
        latest=stream.read_latest()
        assert latest.session==2 and latest.discontinuity and latest.sequence==2
        assert latest.image[0,0,0]==2
        assert stream.read_latest() is None
        assert stream.snapshot()['reconnections']==1
    finally:
        stream.stop_event.set();release_gate.set();assert stream.stop(timeout=1)
    assert released==[1,2]


def test_open_errors_do_not_expose_source_or_exception_message():
    attempted=threading.Event()
    def factory(_):attempted.set();raise RuntimeError('rtsp://name:PASSWORD@host?token=SECRET')
    stream=LatestFrameStream(LiveSourceConfig('cam','rtsp://name:PASSWORD@host',reconnect_initial=.01,reconnect_max=.02),factory)
    stream.start();assert attempted.wait(1);assert stream.stop(1)
    assert 'PASSWORD' not in str(stream.snapshot()) and 'SECRET' not in str(stream.snapshot())


def test_reconnect_and_gap_never_immediately_allow_event_decisions():
    policy=ReadinessPolicy(recovery_seconds=.5,max_observation_gap=.2)
    gate=LiveReadiness(policy)
    first=gate.assess(packet(1,at=10,discontinuity=True),True,now=10.01)
    assert first.can_detect and first.can_measure and not first.can_commit_event
    for n in range(2,8):result=gate.assess(packet(n,at=10+(n-1)*.1),True,now=10+(n-1)*.1+.01)
    assert result.can_commit_event
    gap=gate.assess(packet(40,at=12),True,now=12.01)
    assert not gap.can_commit_event and gap.continuity_broken and 'observation_gap' in gap.reasons
    assert gap.skipped_frames==32
    reconnected=gate.assess(packet(41,session=2,at=12.1),True,now=12.11)
    assert not reconnected.can_commit_event and 'new_session' in reconnected.reasons


def test_stale_dark_blur_missing_geometry_and_resolution_change():
    gate=LiveReadiness(ReadinessPolicy(min_focus_variance=5))
    dark=gate.assess(packet(at=1,value=0),True,now=1.1)
    assert not dark.can_detect and 'underexposed' in dark.reasons
    blur=gate.assess(packet(2,at=1.1),True,now=1.11)
    assert not blur.can_detect and 'blurred' in blur.reasons
    stale=gate.assess(packet(3,at=1.2),True,now=3)
    assert not stale.can_commit_event and 'stale_frame' in stale.reasons
    missing=LiveReadiness().assess(packet(at=2),False,now=2.01)
    assert missing.can_detect and not missing.can_measure and not missing.can_commit_event
    gate=LiveReadiness(ReadinessPolicy(recovery_seconds=0))
    gate.assess(packet(at=10),True,now=10.01)
    changed=gate.assess(packet(2,at=10.1,shape=(60,80,3)),True,now=10.11)
    assert 'resolution_changed' in changed.reasons and not changed.can_commit_event
    repeated=gate.assess(packet(2,at=10.1,shape=(60,80,3)),True,now=10.12)
    assert 'repeated_or_reordered_frame' in repeated.reasons and not repeated.can_detect


@pytest.mark.parametrize('kwargs',[{'read_timeout_ms':0},{'reconnect_initial':3,'reconnect_max':1},{'camera_id':''}])
def test_invalid_config(kwargs):
    arguments={'camera_id':'cam','source':'rtsp://host'};arguments.update(kwargs)
    with pytest.raises(ValueError):LiveSourceConfig(**arguments)
