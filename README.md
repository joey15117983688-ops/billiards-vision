# 台球视觉识别与训练辅助系统

基于 Python 3.10、PyQt5、YOLO 与 OpenCV，分析本地台球录像和样本图片，支持球桌校准、球体轨迹、袋口定位、进球事件观察与已有考题标注。当前主窗口尚未接入真实摄像头视频流。

## 当前功能与限制

- 视频打开后暂停，播放/暂停共用一个按钮；结束后回到首帧暂停。视频旋转只在视频区域内完成，操作栏保持可见。
- 支持单球、多球点选和全部识别；轨迹可筛选全部运动球、白球、其他球或所选球。短时恢复使用当前像素复核，无法可靠关联时保留缺失区间。
- 自动/人工校准内库和六袋，按来源或命名球房/球桌配置保存，并可设置启动默认值。显示范围 `scope_polygon` 与物理比赛平面 `playable_polygon` 分开。
- 参考画面对齐支持机位平移、缩放和小角度旋转；它不是任意机位或严重遮挡下的校准保证。
- 进球观察综合入口轨迹、方向、消失窗、可见性与稳定复核，支持 UNKNOWN、判定修订和去重计数；不是完整斯诺克裁判/计分系统。
- 已有考题圆圈支持创建、拖动、调整大小与序号显示，位置随视频显示变换同步；本轮未新增考题功能。
- 实时源准备模块提供独立读取、最新帧缓存、重连和连续性预检，仍需后续现场验证与主界面接入。

现有光照测试已复现彩灯下台面重检偏移、黑白球观测及袋口验证退化。**第二阶段仅完成 A（测试集建立），B → C → D → E 尚未执行**；不能宣称真实彩灯识别已经可靠。没有人工真值时不将球数变化当准确率。

## 环境与启动

参考环境：Python 3.10.20、NumPy 1.26.4、opencv-python 4.10.0.84、PyQt5 5.15.11、PyTorch 2.6.0、Ultralytics 8.3.203。当前只在 macOS ARM64 上验收；跨平台需另行验证。

从克隆得到的仓库目录准备环境：

```bash
cd billiards-vision
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

`requirements-lock.txt` 是既有 macOS ARM64 环境快照，包含平台专属 wheel，不能直接视为 Windows/Linux 通用安装清单。原有测试还依赖被忽略的本地素材；干净克隆不包含全部数据。依赖及素材准备缺口见工程基线。

```bash
# 命令行默认进入样本图片模式
./scripts/start_dev.sh

# 进入视频模式，首帧保持暂停
./scripts/start_dev.sh --source video

# 指定自己的本地视频或图片目录（以下为相对仓库路径）
./scripts/start_dev.sh --video data/example.mp4
./scripts/start_dev.sh --images-dir data/example_images
```

默认视频优先 `data/test.mp4`，缺失时使用 `data/overhead_pool.mp4`；默认样本目录为 `data/dataset/images/train`。素材不存在时需要先准备，代码不会凭空生成现场录像。当前视频播放无声。

模型使用 `models/billiards_v1.pt`，YOLO 在独立进程执行 CPU 推理；任意时刻最多一个在途请求，不要求每个视频帧都运行模型。视频播放帧率、逐帧像素观察频率和 YOLO 更新率是不同指标。

macOS 可显式运行 `.venv/bin/python scripts/install_desktop_app.py` 创建本地桌面入口；入口依赖当前仓库环境，移动项目后需重新生成。本操作不接入摄像头硬件。

## 操作与坐标

1. 载入视频或图片，选择已有球桌配置；没有配置时显式自动检测袋口或人工校准。接受的校准可保存，下次恢复使用。
2. 需要模型时开启 YOLO，等待就绪后点选目标；显式“全部识别”才进入全目标模式。连续点击已有球可选择/取消多颗球。
3. 选择轨迹范围并手动开始播放。暂停不推进视频时间，也不增加入袋消失证据。
4. 开启进球观察前需要可信台面/袋口。隐藏袋口仅隐藏图形，不等于关闭分析；UNKNOWN 需要复核，不能直接当已进球。
5. 坐标按可靠内库四角映射为毫米，默认 **3569 × 1778 mm 为参考尺寸**，并非所有球桌的实测尺寸。原点/轴按当前屏幕排序，台面外不返回毫米坐标。

球心为检测框中心向台面平面的投影，未补偿镜头畸变与球心高度视差；小数位不代表物理测量精度。帧间预测/补偿只可用于寻找实际像素，不等于进球实测证据。

## 代码入口与注释

主要 Python 模块现有中文注释，说明输入输出、坐标单位、异步世代、判断证据和失败边界。2026-10-24 的注释整理保持可执行语法树不变，没有借此调整算法/阈值或训练模型；整理后全量 **271 项测试通过，0 失败、0 跳过**。

| 入口 | 职责 |
|---|---|
| `main.py` | Qt 界面、视频解码、异步请求、校准与事件会话协调 |
| `scripts/yolo_click_worker.py` | 独立进程的 JSON 行协议与模型推理 |
| `src/scene_calibration.py`、`src/calibration_state.py`、`src/calibration_presets.py` | 校准搜索协调、持久化及命名配置 |
| `src/cloth_region.py`、`src/camera_alignment.py`、`src/table_coordinates.py` | 可见范围、固定参考配准与毫米映射 |
| `src/live_tracking.py`、`src/ball_tracks.py`、`src/ball_observation.py` | 延迟结果对齐、身份关联与局部像素复核 |
| `src/vision/pocket/` | 台面定位、透视展开、局部袋口验证与调试导出 |
| `src/vision/pocket_event/` | 当前帧观察、观测适配、入口证据、多帧入袋状态机和记录 |
| `src/live_source.py`、`src/live_readiness.py` | 独立实时源预备模块，尚未接主窗口 |
| `scripts/build_lighting_cases.py`、`scripts/evaluate_lighting_cases.py` | 确定性光照输入与原算法诊断 |
| `scripts/benchmark_engineering_baseline.py` | 第一阶段统一性能与生命周期测量 |
| `tests/` | 原有业务与光照测试；合成规则测试和真实视频测试需区分 |

## 保留的项目文档

| 文档 | 用途 |
|---|---|
| [工程基线](docs/ENGINEERING_BASELINE.md) | 代码/环境、功能、性能、资源与维护风险 |
| [光照样本与复跑说明](docs/LIGHTING_CASES.md) | 光照矩阵、参数、来源边界、生成和测试命令 |
| [A 阶段结果](docs/LIGHTING_ROBUSTNESS_REPORT.md) | 35 个片段现有输出、失败案例和后续建议 |
| [第二阶段任务书](docs/台球视觉识别系统——第二阶段：彩色灯光抗干扰与识别可靠性开发任务书.md) | A → B → C → D → E 的开发顺序及验收边界 |
| [实时视频准备方案](docs/live_video_preparation.md) | 无新硬件条件下的拉流/缓存/连续性准备 |

早期独立的环境、训练、袋口及 SDK 查找报告已清理；当前操作以本 README 为准。仍需保留的素材/模型来源见下文，原有历史报告可通过 Git 历史查阅。

## 测试与测量

准备哈希一致的本地素材后：

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests -q

# 可重复构建连续光照片段并运行原算法；输出目录必须不存在
.venv/bin/python scripts/build_lighting_cases.py \
  --output data/lighting_cases/NEW_RUN \
  --evaluate logs/lighting_phase2/NEW_RUN

# 显式使用已生成数据执行光照集成入口
LIGHTING_MANIFEST="$PWD/data/lighting_cases/NEW_RUN/manifest.json" \
  QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_lighting_scenarios.py -q
```

阶段 A 曾分别运行原有 **249 项**与新增 **22 项**，全部通过；未提供光照 manifest 时新增测试中一个本地集成用例明确跳过。测试数量不是识别准确率，未标注片段不能计算真实 FP/FN。

性能比较复用工程基线的脚本、素材、环境、预热和统计口径；原始性能数据仍是历史对照，注释整理不宣称性能改善。数据生成和性能测量不要并行运行，避免互相抢占资源。

## 素材来源与隐私

| 相对路径 | 来源与边界 |
|---|---|
| `data/test.mp4` | 用户提供的竖向俯拍录像，1080×1922、30 FPS；不推断拍摄来源，不随仓库分发 |
| `data/overhead_pool.mp4` | [FedericoPivotto/billiard-video-analysis](https://github.com/FedericoPivotto/billiard-video-analysis)，提交 `13fa4f15b41614c894e4cdd49a7557461680efaf` 的 `dataset/game1_clip1/game1_clip1.mp4`；蓝台呢八球短片，不是斯诺克 |
| `data/billiards_table.mp4` | [Louis Geisler / Billard](https://github.com/louisgeisler/Billard)，提交 `b1dff416c06f9a311b77aac4a73715219b8fd31b` 的 `NOdebug.mp4`；历史倾斜视角素材 |
| `data/snooker_match.mp4` | [Internet Archive — Snooker](https://archive.org/details/youtube-u7PyyW8CI64)，历史保留转播片段；不据文件名推断比赛日期 |

视频哈希见工程基线与光照参数模板。`.venv/bin/python scripts/download_demo_assets.py` 只下载其固定公共蓝台呢片段并核验体积、SHA1 和解码，不会下载用户录像，也不会获取全部测试数据。上游代码仓库的许可证不能直接视为转播画面的独立商业授权。

视频、派生图像、原始测量日志和私人球房配置保存在已忽略的 `data/`、`logs/`、`config/calibration/`、`config/venues/`。运行日志可能包含真实源路径；公开同步只包含代码、测试、参数模板和脱敏文档，不包含这些产物。文档使用仓库相对路径，不写个人主目录、私有设备地址或凭据。

## 既有模型与训练记录摘要

部署权重 `models/billiards_v1.pt` 为已有 `yolov8n.pt` 微调结果，类别为 `billiards_table`、`billiard_ball`、`white_ball`。SHA256：

```text
6f9f9a0cce4e54c952993a63a485a59505b4ab1a8bc77e67bc2ade28d88a94e0
```

历史训练完成过 CPU 40 轮、白球全图 20 轮及训练集局部图强化 15 轮；训练/验证白球原始划分为 67/17，局部图是派生样本，不是新独立采集。验证图保持 20 张原图；旧报告的有限验证集指标不代表新球房、彩灯、摄像机或实时比赛准确率。旧两类权重可追溯 Git 提交 `e942237`，当前权重哈希也保存在工程基线。

`scripts/train_billiards.py`、`scripts/prepare_billiards_dataset.py`、`scripts/prepare_white_crops.py` 为历史工具，需显式调用；日常播放、光照样本构建和测试都不启动训练。训练脚本保留 CPU 小批次、文件锁、采样内存预算及完整优化器检查点；软件内存检查不是系统硬性上限，部署 best.pt 与完整 resume 检查点不能互换。本轮未训练或修改权重。

海康 SDK 尚未安装或接入，当前播放器不依赖它；后续设备/平台选型应从[官方资源入口](https://open.hikvision.com/osp)核对。现阶段不下载 SDK、不采购设备、不把本地录像回放称为真实设备联调。
