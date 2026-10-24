"""Experimental live ingestion: bounded latest-frame delivery and reconnect epochs.

Timestamps measure local decode receipt, not camera-to-screen latency. No events
or calibration updates happen here. The capture is owned by one reader thread.
"""
# 实时源前期独立模块，尚未接入主窗口，不把本地 MP4 处理称为真实设备联调。
# 使用后台读取与单槽最新帧缓存，消费慢时丢弃中间帧，不积压完整录像。
# 连续性使用 session/sequence/单调接收时间判断；网络地址可能含凭据，不应公开运行配置。

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

    # 检查地址、稳定 camera_id、超时和重连参数，非法配置在拉流前失败。
    def __post_init__(self):
        if not self.camera_id.strip() or len(self.camera_id)>160:raise ValueError('需要稳定的球房／球桌／摄像机标识')
        if not self.source:raise ValueError('视频流地址不能为空')
        if min(self.open_timeout_ms,self.read_timeout_ms,self.max_frame_age_ms)<=0:raise ValueError('超时必须大于零')
        if not 0<self.reconnect_initial<=self.reconnect_max:raise ValueError('重连间隔无效')

    # 以稳定来源标识生成配置键，避免临时连接状态影响校准归属。
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
    # 建立受锁保护的单帧槽，统计覆盖/消费情况，不创建无限队列。
    def __init__(self):
        self.condition=threading.Condition();self.pending=None
        self.received=0;self.delivered=0;self.overwritten=0;self.stale=0

    # 新帧覆盖旧的未消费帧并更新统计，延迟优先于保留每个历史画面。
    def put(self,packet):
        with self.condition:
            self.received+=1
            if self.pending is not None:self.overwritten+=1
            self.pending=packet;self.condition.notify_all()

    # 清空最新帧槽，换会话时不把旧帧递交给下游。
    def clear(self):
        with self.condition:self.pending=None;self.condition.notify_all()

    # 原子取出并清空待消费帧；超过 max_age_ms 的旧帧丢弃，等待最多 timeout 秒。
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

    # 受锁读取缓冲统计，用于诊断生产/消费速率及覆盖情况。
    def snapshot(self):
        with self.condition:
            return dict(received=self.received,delivered=self.delivered,overwritten=self.overwritten,
                        stale=self.stale,pending=int(self.pending is not None))


# 为网络地址设置 OpenCV 后端与打开/读取超时，不能套用于所有设备 SDK。
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
    # 注入 capture 工厂与状态回调，便于使用本地回放进行无硬件测试。
    def __init__(self,config,capture_factory=open_network_capture):
        self.config=config;self.capture_factory=capture_factory
        self.buffer=LatestFrameBuffer();self.stop_event=threading.Event()
        self.lock=threading.Lock();self.thread=None
        self.state=StreamState.STOPPED;self.session=0;self.sequence=0
        self.reconnections=0;self.last_error=None

    # 统一发布连接/运行/重试状态；只通知，不在回调里做视觉分析。
    def _state(self,state,error=None):
        with self.lock:self.state=state;self.last_error=error

    # 启动独立读取线程；已在运行时不重复创建生产者。
    def start(self):
        if self.thread is not None and self.thread.is_alive():return
        self.stop_event.clear();self.buffer.clear()
        self.thread=threading.Thread(target=self._run,name='latest-live-frame',daemon=True)
        self.thread.start()

    # 循环打开、读取和发布最新帧，失败时清理 capture 并按配置等待重连。
    # 每次新连接使用新会话标识，恢复帧不能被当作前一段连续观测直接确认事件。
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

    # 向消费者提供最新帧，分析线程无需参与网络读取或重连。
    def read_latest(self,timeout=0):
        return self.buffer.take(self.config.max_frame_age_ms,timeout)

    # 汇总流状态和缓存统计，不把连接成功当作识别质量达标。
    def snapshot(self):
        with self.lock:
            status=dict(camera_id=self.config.camera_id,calibration_key=self.config.calibration_key,
                        state=self.state.value,session=self.session,reconnections=self.reconnections,
                        last_error=self.last_error)
        return {**status,**self.buffer.snapshot()}

    # 请求停止并回收读取资源；底层读取是否能及时退出还受后端超时支持影响。
    def stop(self,timeout=None):
        self.stop_event.set();self.buffer.clear()
        if self.thread is not None:
            bound=timeout if timeout is not None else (self.config.open_timeout_ms+self.config.read_timeout_ms)/1000+1
            self.thread.join(bound)
            # A custom backend may ignore timeouts: report it, never create
            # additional reader threads or call release concurrently with read.
            return not self.thread.is_alive()
        return True
