#!/usr/bin/env python3
"""Read-only production baseline, with isolated configs and real local inputs.

No source algorithms, model weights, thresholds, exam logic or live inputs are
changed. Timing wrappers are installed only in this benchmark process. RSS is
sampled every 250 ms for this process and its model child; one sampler thread is
measurement overhead. Run one scenario per process, sequentially, when idle.
"""
# 第一阶段统一性能口径：隔离配置运行已有模块/Qt/模型，不更改生产阈值或返回伪结果。
# CPU 100% 表示一个核心时间；RSS 为主进程与子进程采样和，可能重复计入共享页。
# 耗时有包含关系，不可把 next_frame、适配器、局部可见性各项直接相加；短测不能证明无泄漏。

import argparse
from collections import defaultdict
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


# 严格序列化测量结果到本次指定目录，拒绝非有限数值。
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


# 保留样本数、均值、中位、P95 和极值，避免只挑最好的一次表现。
def distribution(values):
    if not values:return {'count':0}
    values=np.asarray(values,float)
    return dict(count=len(values),mean=float(values.mean()),median=float(np.median(values)),
                p95=float(np.percentile(values,95)),maximum=float(values.max()),minimum=float(values.min()))


# 记录提交、架构、包版本及线程/内存条件，性能比较必须核对这些上下文。
def environment():
    names=['numpy','opencv-python','PyQt5','torch','torchvision','ultralytics','scipy','psutil','pytest']
    versions={}
    for name in names:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'python':sys.version,'platform':platform.platform(),'architecture':platform.machine(),
            'logical_cpus':psutil.cpu_count(),'physical_cpus':psutil.cpu_count(False),
            'memory_gib':psutil.virtual_memory().total/1024**3,'opencv_threads':cv2.getNumThreads(),
            'load_average':list(os.getloadavg()),'packages':versions}


class Timings:
    def __init__(self):self.active=False;self.data=defaultdict(list)
    # 仅在测量窗口启用计时；finally 保证异常调用也计入实际耗时。
    def wrap(self,label,method):
        def measured(*args,**kwargs):
            start=time.perf_counter()
            try:return method(*args,**kwargs)
            finally:
                if self.active:self.data[label].append((time.perf_counter()-start)*1000)
        return measured
    # 给本次诊断进程的方法装计时包装，不写回生产源文件。
    def patch(self,owner,name,label):setattr(owner,name,self.wrap(label,getattr(owner,name)))
    # 按标签分别汇总，标签之间可能是嵌套调用，不能视为互斥时间。
    def summary(self):return {name:distribution(values) for name,values in self.data.items()}


class Sampler:
    def __init__(self):
        self.process=psutil.Process();self.stop_event=threading.Event();self.rows=[];self.known={}
    # 汇总本进程与当前子进程资源，消失/退出的进程允许跳过；RSS 不是瞬时绝对峰值。
    def sample(self):
        row={'wall':time.monotonic(),'rss_mib':0.,'cpu_percent':0.,'os_threads':0,'fds':0,'children':0,'main_rss_mib':0.}
        try:processes=[self.process,*self.process.children(recursive=True)]
        except psutil.Error:processes=[self.process]
        row['children']=len(processes)-1
        for process in processes:
            try:
                if process.pid not in self.known:
                    process.cpu_percent(None);self.known[process.pid]=process
                process=self.known[process.pid]
                rss=process.memory_info().rss/1024**2
                row['rss_mib']+=rss;row['cpu_percent']+=process.cpu_percent(None)
                row['os_threads']+=process.num_threads()
                row['fds']+=process.num_fds() if hasattr(process,'num_fds') else 0
                if process.pid==self.process.pid:row['main_rss_mib']=rss
            except psutil.Error:continue
        self.rows.append(row)
    # 后台约每 250 ms 采样；采样器本身额外使用一个线程。
    def run(self):
        while not self.stop_event.is_set():self.sample();self.stop_event.wait(.25)
    # 启动本次采集线程，不改变生产 worker 的执行线程数。
    def start(self):
        self.thread=threading.Thread(target=self.run,name='baseline-sampler',daemon=True);self.thread.start()
    # 通知停止后补一次采样，收尾资源仍可记录在完整会话中。
    def stop(self):self.stop_event.set();self.thread.join(2);self.sample()
    # 仅用实际测量窗中的采样计算均值/分位，同时单列完整会话峰值及分段内存变化。
    def summary(self,start,end):
        rows=[r for r in self.rows if start<=r['wall']<=end]
        result={k:distribution([r[k] for r in rows]) for k in ['rss_mib','main_rss_mib','cpu_percent','os_threads','fds','children']}
        first=[r['rss_mib'] for r in rows if r['wall']<=start+(end-start)/3]
        last=[r['rss_mib'] for r in rows if r['wall']>=end-(end-start)/3]
        result['rss_last_third_minus_first_third_mib']=float(np.mean(last)-np.mean(first)) if first and last else None
        result['overall_sampled_rss_peak_mib']=max((r['rss_mib'] for r in self.rows),default=0.)
        result['semantics']='CPU 100%=one core; RSS is sampled sum of main+children (shared pages may be double counted); sampler adds one Python thread; no leak conclusion from a short run.'
        return result


# 对固定原片测台面/袋口/完整校准，并用受控身份重建检查容器增长；不是实战准确率。
def run_modules(args):
    from src.vision.pocket import TableDetector,PocketDetector
    from src.scene_calibration import calibrate_scene
    from src.vision.pocket_event.adapter import PocketEventAdapter
    from src.vision.pocket_event.pixel_observer import PixelBallObserver
    cases=[]
    for source in ['data/test.mp4','data/overhead_pool.mp4','data/billiards_table.mp4']:
        path=ROOT/source
        if not path.exists():cases.append({'source':source,'missing':True});continue
        capture=cv2.VideoCapture(str(path));ok,frame=capture.read();capture.release()
        if not ok:cases.append({'source':source,'decode_failed':True});continue
        timings=defaultdict(list);table=None;pockets=None;calibrated=None;errors=[]
        # One warmup, then three identical-frame repetitions. Calibration is a
        # separate whole pipeline measurement, not added to table/pocket times.
        for iteration in range(4):
            start=time.perf_counter();table=TableDetector().detect(frame)
            if iteration:timings['table_detector_ms'].append((time.perf_counter()-start)*1000)
            start=time.perf_counter();pockets=PocketDetector().detect(frame,table=table).to_dict()
            if iteration:timings['pocket_detector_given_table_ms'].append((time.perf_counter()-start)*1000)
            start=time.perf_counter()
            try:
                calibrated=calibrate_scene(frame,None,lambda _:None,geometry_only=True)
                if iteration:timings['full_automatic_calibration_ms'].append((time.perf_counter()-start)*1000)
            except (ValueError,cv2.error) as exc:errors.append(type(exc).__name__+': '+str(exc))
        name=path.stem
        write_json(args.output/(name+'_pockets.json'),pockets)
        if calibrated:write_json(args.output/(name+'_calibration.json'),calibrated)
        drawn=frame.copy()
        if table.detected:cv2.polylines(drawn,[np.int32(table.playable_polygon)],True,(255,200,0),3)
        for p in pockets['pockets']:
            center=p.get('center') or p.get('candidate_center')
            if center:cv2.circle(drawn,(round(center['x']),round(center['y'])),8,(0,255,0) if p['status']=='observed' else (0,165,255),2)
        scale=min(1.,1100/max(drawn.shape[:2]));cv2.imwrite(str(args.output/(name+'_geometry.jpg')),cv2.resize(drawn,None,fx=scale,fy=scale))
        cases.append({'source':source,'shape':list(frame.shape),'timings_ms':{k:distribution(v) for k,v in timings.items()},
                      'table_detected':table.detected,'pocket_statuses':{status:sum(p['status']==status for p in pockets['pockets']) for status in ['observed','estimated','manual','missing']},
                      'full_calibration_report':calibrated.get('calibration_report') if calibrated else None,'errors':errors})
    # Re-identification stress isolates retained metadata, not accuracy.
    sys.path.insert(0,str(ROOT/'tests'))
    from pocket_scene import make_scene
    frame,_=make_scene();adapter=PocketEventAdapter(PocketDetector().detect(frame).to_dict())
    for i in range(200):adapter.update(i*4.,[{'name':'white_ball','box':[460,360,480,380],'confidence':.95}],frame,scene_clear=True)
    churn={'synthetic':True,'description':'200 re-identifications separated by 4s virtual gaps; no real accuracy claim',
           'active_tracks':len(adapter.target_tracks.states),'name_mappings':len(adapter.target_names),'engine_states':len(adapter.engine.states),
           'engine_balls':len(adapter.engine.balls),'retained_results':len(adapter.engine.results)}
    write_json(args.output/'module_baseline.json',{'environment':environment(),'cases':cases,'identity_churn':churn})


# 创建隔离目录，包装真实 Qt 入口和模型 worker，预热后按单调时钟测量。
# 区分播放、像素观察及 YOLO 回包频率，末尾检查暂停/恢复/EOF/换源/资源释放。
def run_ui(args):
    # Select Qt platform before importing PyQt5 or main.
    os.environ['QT_QPA_PLATFORM']=args.qt_platform
    from PyQt5.QtCore import QTimer
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    import main
    from src.cloth_region import cloth_region
    from src.calibration_state import save_automatic
    from src.vision.pocket import TableResult,PocketDetector
    from src.vision.pocket_event.adapter import PocketEventAdapter
    from src.vision.pocket_event.pixel_observer import PixelBallObserver
    from src.live_tracking import LiveTracking
    from src.camera_alignment import CameraAlignment

    args.output.mkdir(parents=True,exist_ok=True)
    sandbox=args.output/'isolated_runtime';(sandbox/'logs').mkdir(parents=True,exist_ok=True)
    image_dir=sandbox/'images';image_dir.mkdir(exist_ok=True)
    main.ROOT=sandbox
    timings=Timings()
    for owner,method,label in [(PixelBallObserver,'observe','ball_pixels_ms'),(PocketEventAdapter,'update','pot_adapter_inclusive_ms'),
                               (PocketEventAdapter,'clear_scene','pocket_visibility_ms'),(LiveTracking,'push','tracking_push_ms'),
                               (LiveTracking,'replace','tracking_replace_ms'),(CameraAlignment,'update','camera_alignment_check_ms')]:
        timings.patch(owner,method,label)
    worker_base=main.YoloWorker
    class LocalModelWorker(worker_base):
        def load(self):
            try:
                self.shutdown();self.log=(sandbox/'logs/yolo_worker.log').open('a')
                self.process=subprocess.Popen([sys.executable,str(ROOT/'scripts/yolo_click_worker.py')],cwd=ROOT,
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
                if not self.read_reply().get('ready'):raise RuntimeError('model_not_ready')
                self.ready.emit()
            except Exception as exc:self.failed.emit(-1,str(exc))
    main.YoloWorker=LocalModelWorker
    timings.patch(LocalModelWorker,'request_prediction','model_request_roundtrip_ms')
    app=QApplication([]);window=main.MainWindow(args.video,image_dir=image_dir)
    window.setWindowTitle('工程基线采集（独立配置）');window.resize(1200,800)
    window.pause_video();window.capture.set(cv2.CAP_PROP_POS_MSEC,args.start*1000)
    window.next_frame();frame=window.canvas.frame
    if frame is None:raise ValueError('no_decoded_frame')
    cv2.imwrite(str(image_dir/'current_frame.png'),frame)
    window.image_paths=[image_dir/'current_frame.png']
    if args.video.name=='test.mp4':
        pockets=json.loads((ROOT/'tests/fixtures/late_pot_calibration.json').read_text())
        polygon=pockets['table']['playable_polygon'];scope=pockets['table']['scope_polygon']
    else:
        polygon=[[171,102],[860,102],[856,479],[171,476]]
        table=TableResult(detected=True,complete=True,corners=polygon,playable_polygon=polygon,scope_polygon=polygon,quality={'geometry_reliable':True})
        pockets=PocketDetector().detect(frame,table=table).to_dict()
        region=cloth_region(frame,polygon);scope=region['polygon'] if region else polygon
        pockets['table']['scope_polygon']=scope
    window.canvas.calibration_polygon=deepcopy(polygon);window.canvas.recognition_polygon=deepcopy(scope)
    window.canvas.set_pocket_result(deepcopy(pockets));window.start_camera_alignment()
    save_automatic(window.calibration_profile_path().with_suffix('.auto.json'),frame,
                   {'polygon':polygon,'scope_polygon':scope,'pockets':pockets,'calibration_report':{'decision':'baseline_fixture'}},None)
    window.single_table_polygon=deepcopy(polygon)
    class CaptureProbe:
        def __init__(self,capture):self.capture=capture
        def read(self):return timings.wrap('decode_ms',self.capture.read)()
        def __getattr__(self,name):return getattr(self.capture,name)
    window.capture=CaptureProbe(window.capture)
    sampler=Sampler();sampler.start()
    context={'start':None,'end':None,'frames':0,'paints':0,'last_paint':None,'model_results':0,'events':[],
             'started_model':time.monotonic(),'startup_model_s':None,'error':None,'observation_calls':0}
    original_next=window.next_frame
    def next_frame():
        original_next()
        if timings.active:context['frames']+=1
    window.timer.timeout.disconnect();window.timer.timeout.connect(timings.wrap('ui_next_frame_ms',next_frame))
    original_paint=window.canvas.paintEvent
    def paint(event):
        original_paint(event)
        if timings.active:
            position=window.capture.get(cv2.CAP_PROP_POS_FRAMES)
            if position!=context['last_paint']:context['paints']+=1;context['last_paint']=position
    window.canvas.paintEvent=paint
    original_count=window.update_pot_count
    def count(results=(),clear=False):
        original_count(results,clear)
        if timings.active and results:context['events'].extend(deepcopy(results))
    window.update_pot_count=count
    # Count actual async model responses; never substitute inference results.
    original_live_result=window.on_live_result
    def live_result(*values):
        if timings.active:context['model_results']+=1
        return original_live_result(*values)
    window.on_live_result=live_result
    # Connections to worker slots are established only when YOLO is enabled.
    window.show()

    # 预热结束后清空测量窗口计数并记录单调起点，不将模型加载时间混入稳态 FPS。
    def begin_measurement():
        context['start']=time.monotonic();context['media_start']=window.capture.get(cv2.CAP_PROP_POS_MSEC)/1000
        timings.active=True;QTimer.singleShot(round(args.seconds*1000),finish)

    def ready_to_play():
        if args.scenario in ('pots','combined'):window.pot_button.click()
        if args.scenario in ('tracks','combined'):
            window.trails_button.setCurrentIndex(0);window.canvas.show_trails=True
            window.detection_mode='all'
        window.play_video()
        QTimer.singleShot(1000,begin_measurement)

    def await_model():
        if window.yolo_ready:
            context['startup_model_s']=time.monotonic()-context['started_model'];ready_to_play()
        elif time.monotonic()-context['started_model']>30:
            context['error']='model_start_timeout';window.close();sampler.stop();app.quit()
        else:QTimer.singleShot(50,await_model)

    # 结束测量后执行正常生命周期检查、落盘指标并退出应用；不冒充挂死故障注入验收。
    def finish():
        timings.active=False;context['end']=time.monotonic()
        context['media_end']=window.capture.get(cv2.CAP_PROP_POS_MSEC)/1000
        window.grab().save(str(args.output/'ui_overlay.png'))
        write_json(args.output/'event_timeline.json',context['events'])
        if window.pot_adapter:write_json(args.output/'final_pot_state.json',window.pot_adapter.engine.snapshot())
        lifecycle={};window.pause_video();position=window.capture.get(cv2.CAP_PROP_POS_FRAMES)
        QTest.qWait(200)
        lifecycle['pause_position_stable']=position==window.capture.get(cv2.CAP_PROP_POS_FRAMES)
        lifecycle['pause_button_text']=window.play_button.text()
        window.play_button.click();lifecycle['resume_starts_timer']=window.timer.isActive();window.pause_video()
        window.capture.set(cv2.CAP_PROP_POS_FRAMES,window.capture.get(cv2.CAP_PROP_FRAME_COUNT));window.next_frame()
        lifecycle['eof_returns_to_first_paused']=not window.timer.isActive() and window.capture.get(cv2.CAP_PROP_POS_FRAMES)==1
        window.source_combo.setCurrentIndex(1)
        lifecycle['switch_clears_states']=not window.timer.isActive() and window.pot_adapter is None and not window.live_tracking.frames
        window.source_combo.setCurrentIndex(0)
        lifecycle['return_video_waits_for_manual_play']=not window.timer.isActive() and window.play_button.text()=='播放'
        children=[p.pid for p in psutil.Process().children(recursive=True)]
        capture=window.capture
        close_start=time.perf_counter();window.close();lifecycle['close_ms']=(time.perf_counter()-close_start)*1000
        QTest.qWait(300)
        lifecycle['capture_released']=not capture.isOpened()
        lifecycle['children_alive_after_close']=[pid for pid in children if psutil.pid_exists(pid)]
        lifecycle['qt_threads_running_after_close']={name:bool(getattr(window,name).isRunning()) for name in ['yolo_thread','calibration_thread','pocket_thread'] if getattr(window,name) is not None}
        lifecycle['event_recording_cleared']=window.pot_recording is None
        sampler.stop()
        duration=context['end']-context['start'];module_stats=timings.summary()
        result={'environment':environment(),'scenario':args.scenario,'source':str(args.video.relative_to(ROOT)),
                'source_start_seconds':args.start,'qt_platform':args.qt_platform,'window_size':[1200,800],
                'requested_seconds':args.seconds,'measured_seconds':duration,'model_startup_s':context['startup_model_s'],
                'decoded_frames':context['frames'],'painted_unique_frames':context['paints'],
                'playback_fps':context['frames']/duration,'painted_unique_fps':context['paints']/duration,
                'ball_analysis_fps':module_stats.get('ball_pixels_ms',{}).get('count',0)/duration,
                'tracking_fps':module_stats.get('tracking_push_ms',{}).get('count',0)/duration,
                'model_result_fps':context['model_results']/duration,'media_lag_growth_s':duration-(context['media_end']-context['media_start']),
                'timings_ms':module_stats,'resources':sampler.summary(context['start'],context['end']),
                'lifecycle':lifecycle,'resource_samples':sampler.rows,'instrumentation':'diagnostic wrappers; no frame/observation is synthesized',
                'measurement_error':context['error']}
        files=[p for p in sandbox.rglob('*') if p.is_file()]
        result['isolated_output_bytes']=sum(p.stat().st_size for p in files if '/images/' not in str(p))
        result['isolated_output_files']=len(files)
        write_json(args.output/'ui_baseline.json',result)
        app.quit()

    if args.scenario in ('tracks','combined'):
        window.yolo_button.setChecked(True);QTimer.singleShot(50,await_model)
    else:ready_to_play()
    # Independent fail-safe timer; a hard-blocked UI still requires the caller's
    # subprocess timeout. No claim of production watchdog coverage.
    QTimer.singleShot(round((args.seconds+60)*1000),app.quit)
    app.exec_()
    if context['end'] is None:
        if not sampler.stop_event.is_set():sampler.stop()
        raise RuntimeError(context['error'] or 'measurement_incomplete')


# 选择 modules/playback/pots/tracks/combined 等场景及独立输出目录；应串行运行避免互相抢资源。
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=['modules','playback','pots','tracks','combined'],required=True)
    parser.add_argument('--video',type=Path,default=ROOT/'data/test.mp4')
    parser.add_argument('--start',type=float,default=44)
    parser.add_argument('--seconds',type=float,default=12)
    parser.add_argument('--qt-platform',choices=['offscreen','cocoa'],default='offscreen')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output=args.output.resolve();args.video=args.video.resolve()
    if args.seconds<=0 or args.start<0:parser.error('seconds must be positive and start nonnegative')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.scenario=='modules':run_modules(args)
    else:run_ui(args)


if __name__=='__main__':main()
