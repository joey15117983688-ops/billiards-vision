# 袋口自动检测验收报告

日期：2026-10-15。本次按任务书实现独立 OpenCV 模块，使用本机现有图片、俯拍视频、斯诺克转播视频和合成场景验证。正式摄像头尚未接入，实机定位精度尚未验收。

## 球桌范围恢复（最新用户修正）

用户撤回“红线右侧不属于球桌”的定义。现在 `scope_polygon` 包含可见台面与相连库边／带字边条，桌框与通用球目标的桌外过滤使用该范围；旧图 x=905 的右侧边条重新在范围内。正常袋口界面以实线绘制这个范围。

内库比赛平面 `playable_polygon`、四条内库线与 Homography 保留，坐标模式仍用比赛平面换算 mm；局部袋口视觉验证不因范围恢复而回退。JSON 同时输出两种区域，原有 table_mask 仍是比赛平面 mask，不能把它当作包含边条的识别范围。原图六袋与新蓝台呢视频经过 Qt 联调，56 项测试通过。

## 内库边缘与局部光照修正（历史实现）

针对用户标注的 `1770626791.jpg`，修复右侧将带字外库边当作台面边缘的问题。贴边四边形周围搜索连续库边线，比较沿线亮度过渡与暗脊，剔除弱台呢纹理，再沿法线寻找台呢到库边的过渡并鲁棒拟合。台面遮罩、球体过滤、台球桌框与袋口模块统一使用相同的边界检测配置。

原图右侧边界从约 x=912～926 修正到 x=883～886（原图宽 928，用户标记内库约 x=885）。六袋均通过验证。袋网／衬垫采用相对台呢亮度、归一化纹理与交点附近的空间证据；对连接外饰条的轮廓，靠近内库交点的证据权重更高，避免定位锚点被外饰条拉偏。角袋衬垫必须有进入台面侧的证据，保持被球员遮住的 `000001.jpg` 右下袋为估算。

在同一原图的 0.55、1、1.5 倍亮度版本中均六袋通过，右侧两个定位锚点相对原图变化小于 1.1 像素。该结果只验证数字亮度变化，不代表不同实拍光照或真实袋腔中心精度。调试结果保存于 `data/debug/pocket_detection/edge_fixed_*/`，汇总为 `edge_lighting_acceptance.json`；界面截图 `logs/cushion_pockets_fixed_ui.png`。

最新 49 项自动测试通过；真实 Qt 联调验证该图六袋、遮罩边界一致、台球桌和彩球输出；已有黑白球、彩球、桌外过滤及旧袋口回归保留。

## 贴边画面优化（上一版）

针对用户提供的 `000001.jpg` 和 `demo_15.jpg` 修复：绿色库边与台呢连通时，使用长直线图像证据恢复内库；局部验证允许可见的部分袋腔，并增加紧邻内库的低饱和度、有纹理袋网／衬垫证据。保留尺寸、接触、距离、台呢外侧面积及白球／黑球防误判约束，不填造画外像素。

- `demo_15.jpg`：从 5 observed / 1 missing 提升为 6 observed，右上袋口可见部分验证通过。
- `000001.jpg`：从 6 missing 提升为 5 observed / 1 estimated；右下被球员遮挡，显示几何估算。
- 最新自动测试 41 项通过，原图回归、贴边袋腔及白球防误判覆盖；真实 Qt 联调通过，YOLO 权重哈希未变。
- `valid` 当前表示可靠球桌几何及六袋均有视觉证据，不再要求袋口周围整个搜索窗口都在画面中。`table.complete` 仍独立表示台面边界是否贴边。
- `quality.partial_visibility` 表明可见证据贴近裁切边界；`center_kind` 区分可见袋腔重心与袋网／衬垫重心。这些是袋口定位锚点，不能宣称是完整隐藏袋腔的真实中心。
- 输出样例：`data/debug/pocket_detection/improve_demo_15/`、`improve_000001/`；界面截图在 `logs/pocket_real_ui.png`、`logs/pocket_cropped_improved_ui.png`。

以下素材统计为优化前的首次基线记录，当前结果以上述原图回归为准。

## 实现与使用

界面底部新增“自动检测袋口”。点击时暂停视频并对当前帧运行一次后台检测，显示四条内库边、随透视倾斜的袋口入口多边形、六袋名称及状态；再次点击可重算。切图、切源、继续播放会清除旧标注，过期后台结果不应用到新画面。YOLO 开关和识别按钮与袋口结果独立。

算法路径：可配置 HSV 台呢筛选 → 多候选四边形检查 → 四条内库边鲁棒拟合 → 长轴定向及单应映射 → 六个几何候选 → 局部暗腔、台呢接触、轮廓形状和图像覆盖率验证。自动长轴定向结合中袋证据；特殊视角可配置 `long_axis`。默认绿色台呢，其他台呢通过 `PocketConfig.hsv_ranges` 配置。

`src/vision/pocket/` 包含 TableDetector、TableGeometry、PocketDetector、局部验证、数据结构和调试输出。没有引入袋口 YOLO 模型，没有改动本次开始时的模型权重，没有安装或升级框架。

```bash
.venv/bin/python scripts/test_pocket_detector.py --video data/billiards_table.mp4 --at 5
.venv/bin/python scripts/test_pocket_detector.py --image data/dataset/images/train/demo_00.jpg
.venv/bin/python scripts/test_pocket_detector.py --image /absolute/input.jpg --require-valid
.venv/bin/python -m pytest tests -q
```

命令默认保存到 `data/debug/pocket_detection/<时间戳>/`，可用 `--output` 指定目录。`--require-valid` 在结果不完整时返回退出码 2。

## 输出契约

- `observed`：几何与局部图像验证通过，绿色；分数是启发式质量分，不是经过校准的概率。
- `estimated`：只有几何候选或视觉证据不足，黄色，不能作为可靠定位使用。
- `missing`：几何不可靠、视野不足等，红色；真实中心为 null，可能单独提供候选点供调试。
- 可靠球桌几何下六袋全部 observed 时 `valid=true`；局部贴边不会直接否决有充分视觉证据的袋口。

JSON 提供原图坐标、归一化坐标、单应矩阵、四条边、入口多边形、失败原因、帧哈希、来源及文件路径。TOP/BOTTOM 等名称按规范化球桌长轴定义，旋转视角下不一定与屏幕方位一致。袋腔位于台呢外侧时归一化坐标允许略超出 0～1。

每次保存 original、table_mask、pocket_regions、inner_cushions、pocket_candidates、pockets、warped（几何可用时）及 result.json。球桌 mask 填充内库四边形，不挖去球造成的颜色空洞；袋口 mask 独立保存，包含 observed/estimated 的入口区域，下游必须同时检查单袋状态。

## 验证结果

自动测试 38 项通过，覆盖透视、低亮度、蓝台呢配置、裁切、无球桌、遮挡、近库黑球、坐标往返、重复帧、移动台面、mask/JSON、界面失效结果和异常恢复。

另完成真实 Qt 窗口联调：后台线程运行、界面心跳、重复检测另存、YOLO 识别与袋口共存、切图丢弃过期结果、视频暂停/播放、关闭线程及模型进程释放。修复本机 NumPy LAPACK 在 Qt 线程中的 SIGBUS：直线交点采用解析计算，单应矩阵逆采用 OpenCV SVD。修复 PyQt 临时多边形点引用导致标签位置异常的问题。

| 本地素材 | 样本数 | 结果 |
| --- | ---: | --- |
| billiards_table.mp4：0、5、12、21、33 秒 | 5 | 每帧 5 observed / 1 missing，右上袋口超出画面；valid=false |
| snooker_match.mp4：5、60、300 秒 | 3 | 每帧 0 observed / 6 estimated，转播局部证据不足；valid=false |
| 裁切训练图 | 1 | 6 missing；valid=false |
| 合成完整、透视、低亮度 | 3 | 均 6 observed；valid=true；最大中心误差分别 0.347、0.399、0.334 像素 |
| 合成遮挡中袋并加入近库黑球 | 1 | 5 observed / 1 estimated；valid=false |

这 13 个样例的检测耗时约 7～17 ms（不含 PNG/JSON 写盘和 GUI 绘制），结果见 `data/debug/pocket_detection/acceptance_summary.json`。合成误差仅验证几何计算，不能代表真实相机精度。真实素材没有人工标注真值，observed 数量不是准确率统计。

## 当前限制与后续验收

优化前的斯诺克转播抽样尚不能可靠定位六袋；贴边优化后仍需逐个样本验证，不会将六个几何候选直接当作成功。强阴影、反光、球员遮挡、非绿色台呢、极端透视及不完整画面均可能失败。当前版本不含进球判定、球轨迹或运动期间动态重定位。

正式相机接入后，应采集完整六袋的固定机位图像、标注袋口中心和入口边界，对多光照、多遮挡帧统计误差、误检率及重复定位抖动，再调整 HSV、局部阈值和长轴配置。现阶段不宣称达到正式机位像素误差要求。

本次权重 SHA256：`6f9f9a0cce4e54c952993a63a485a59505b4ab1a8bc77e67bc2ade28d88a94e0`。
