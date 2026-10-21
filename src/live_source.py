"""Experimental live ingestion: bounded latest-frame delivery and reconnect epochs.

Timestamps measure local decode receipt, not camera-to-screen latency. No events
or calibration updates happen here. The capture is owned by one reader thread.
"""
from dataclasses import dataclass,field
from enum import Enum
import hashlib
import threading
import time
from urllib.parse import urlsplit
import cv2
import numpy as np


class StreamState(str,Enum):
    STOPPED='stopped'
    CONNECTING='connecting'
    LIVE='live'
    RECONNECTING='reconnecting'


@dataclass(frozen=True)
class LiveSourceConfig:
    camera_id: str
    source: str=field(repr=False)
    open_timeout_ms: int=3000
    read_timeout_ms: int=1500
    max_frame_age_ms: float=500
    reconnect_initial: float=.5
    reconnect_max: float=5.

    def __post_init__(self):
        if not self.camera_id.strip() or len(self.camera_id)>160:raise ValueError('需要稳定的球房／球桌／摄像机标识')
        if not self.source:raise ValueError('视频流地址不能为空')
        if min(self.open_timeout_ms,self.read_timeout_ms,self.max_frame_age_ms)<=0:raise ValueError('超时必须大于零')
        if not 0<self.reconnect_initial<=self.reconnect_max:raise ValueError('重连间隔无效')

    @property
    def calibration_key(self):
        # Tokens/passwords/hostnames may change; camera identity must not.
        return hashlib.sha256(self.camera_id.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class StreamFrame:
    image: np.ndarray=field(repr=False)
    sequence: int
    session: int
    received_at: float
    discontinuity: bool=False


class LatestFrameBuffer:
    """At most one pending frame; a frame is delivered at most once."""
    def __init__(self):
        self.condition=threading.Condition();self.pending=None
        self.received=0;self.delivered=0;self.overwritten=0;self.stale=0

    def put(self,packet):
        with self.condition:
            self.received+=1
            if self.pending is not None:self.overwritten+=1
            self.pending=packet;self.condition.notify_all()

    def clear(self):
        with self.condition:self.pending=None;self.condition.notify_all()

    def take(self,max_age_ms=500,timeout=0):
        deadline=time.monotonic()+max(0,timeout)
        with self.condition:
            while self.pending is None:
                remaining=deadline-time.monotonic()
                if remaining<=0:return None
                self.condition.wait(remaining)
            packet=self.pending;self.pending=None
            if (time.monotonic()-packet.received_at)*1000>max_age_ms:
                self.stale+=1;return None
            self.delivered+=1;return packet

    def snapshot(self):
        with self.condition:
            return dict(received=self.received,delivered=self.delivered,overwritten=self.overwritten,
                        stale=self.stale,pending=int(self.pending is not None))


def open_network_capture(config):
    if urlsplit(config.source).scheme.lower() not in ('rtsp','rtsps','http','https'):
        raise ValueError('仅接受 RTSP／HTTP 视频流地址')
    # Open-only timeouts must be supplied before opening FFmpeg. Fail rather
    # than silently switching to a backend without bounded reads.
    capture=cv2.VideoCapture(config.source,cv2.CAP_FFMPEG,[
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,config.open_timeout_ms,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,config.read_timeout_ms])
    capture.set(cv2.CAP_PROP_BUFFERSIZE,1)  # Advisory; upstream buffers can remain.
    return capture


class LatestFrameStream:
    def __init__(self,config,capture_factory=open_network_capture):
        self.config=config;self.capture_factory=capture_factory
        self.buffer=LatestFrameBuffer();self.stop_event=threading.Event()
        self.lock=threading.Lock();self.thread=None
        self.state=StreamState.STOPPED;self.session=0;self.sequence=0
        self.reconnections=0;self.last_error=None

    def _state(self,state,error=None):
        with self.lock:self.state=state;self.last_error=error

    def start(self):
        if self.thread is not None and self.thread.is_alive():return
        self.stop_event.clear();self.buffer.clear()
        self.thread=threading.Thread(target=self._run,name='latest-live-frame',daemon=True)
        self.thread.start()

    def _run(self):
        delay=self.config.reconnect_initial
        try:
            while not self.stop_event.is_set():
                capture=None;session_started=False
                self._state(StreamState.CONNECTING if self.session==0 else StreamState.RECONNECTING)
                try:
                    capture=self.capture_factory(self.config)
                    if not capture.isOpened():raise RuntimeError('open_failed')
                    first=True
                    while not self.stop_event.is_set():
                        ok,image=capture.read()
                        if not ok or image is None or image.ndim!=3 or image.shape[2]!=3 or image.size==0:
                            raise RuntimeError('read_failed')
                        now=time.monotonic()
                        if self.stop_event.is_set():break
                        if first:
                            with self.lock:
                                self.session+=1;self.reconnections=max(0,self.session-1)
                            session_started=True
                        self.sequence+=1
                        self.buffer.put(StreamFrame(image,self.sequence,self.session,now,first))
                        first=False;delay=self.config.reconnect_initial
                        self._state(StreamState.LIVE)
                except Exception as exc:
                    # Exception messages and source URLs may contain credentials.
                    self.buffer.clear()
                    self._state(StreamState.RECONNECTING,type(exc).__name__)
                finally:
                    if capture is not None:
                        try:capture.release()
                        except Exception:pass
                if self.stop_event.wait(delay):break
                if not session_started:delay=min(self.config.reconnect_max,delay*2)
        finally:
            self.buffer.clear();self._state(StreamState.STOPPED)

    def read_latest(self,timeout=0):
        return self.buffer.take(self.config.max_frame_age_ms,timeout)

    def snapshot(self):
        with self.lock:
            status=dict(camera_id=self.config.camera_id,calibration_key=self.config.calibration_key,
                        state=self.state.value,session=self.session,reconnections=self.reconnections,
                        last_error=self.last_error)
        return {**status,**self.buffer.snapshot()}

    def stop(self,timeout=None):
        self.stop_event.set();self.buffer.clear()
        if self.thread is not None:
            bound=timeout if timeout is not None else (self.config.open_timeout_ms+self.config.read_timeout_ms)/1000+1
            self.thread.join(bound)
            # A custom backend may ignore timeouts: report it, never create
            # additional reader threads or call release concurrently with read.
            return not self.thread.is_alive()
        return True
