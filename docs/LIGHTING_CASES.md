# 第二阶段 A：光照测试集

日期：2026-10-22（Asia/Shanghai）。范围仅 A，生产算法未修改。任务书已归档至本目录；B → C → D → E 等待后续指令。

## 来源与真值边界

参数模板：`tests/fixtures/lighting/cases.json`；当前算法脱敏输出基线：`tests/fixtures/lighting/baseline_v1.json`（不是人工真值，也不要求后续保留错误输出）。以第一阶段 `docs/ENGINEERING_BASELINE.md` 冻结的三个原片及 SHA256 为输入，不复制或覆盖原片，不读取私人球房/考题配置。

| 片段 ID | 原片 | 时间（左闭右开） | 用途 | 变体数 |
|---|---|---|---|---:|
| green_approach | data/test.mp4 | 44–52 秒 | 运动、袋口入口和消失后复核窗口；既有回归的最后可见时间约 47.8 秒 | 13 |
| green_late | data/test.mp4 | 312–320 秒 | 后期黑球接近袋口、消失和复核窗口，既有回归约 315.6 秒 | 5 |
| blue_motion | data/overhead_pool.mp4 | 0–3 秒 | 蓝色台呢、运动与真实手臂/球杆 | 7 |
| angled_static | data/billiards_table.mp4 | 0–3 秒 | 倾斜视角，抽查 0/1/2 秒球的位置近似静止 | 5 |
| green_arm | data/test.mp4 | 0–3 秒 | 开局真实手臂、球杆进入台面 | 5 |

共 35 个连续片段，均保留原分辨率及帧率，不插帧、不降采样；时间由 `start_frame / source_fps` 计算，manifest 精确记录实际边界。静止两球完整序列和无遮挡完整片段没有独立逐帧确认，不硬贴标签。

最小真值模板可手工维护，位于参数模板各段的 `truth`。本轮仅 Codex 稀疏截图审阅，尚非人类逐帧标注：

- 扰动仅改变像素外观，没有空间变换，因此**相对同一原帧的几何应保持不变**。这不代表原视频本身一定没有机位变化。
- 接触台面的真实手臂/球杆在抽查帧可见；不据此推断整个片段或具体袋口都被遮挡。`real_occlusion=true` 仅表示片段包含已看到的遮挡，不表示每帧为真。
- 黑白球逐帧可见性、彩灯后可见性、独立真实进球事件均暂记 `unknown`；`manual_dense_ball_annotations=[]`。现有自动观测 fixture 和旧回归期望不是新人工真值。
- 不由算法输出补写真值，不以 `POTTED` 数量或球数增加作为改善证据，不计算缺少人工标注的准确率、召回率或 FP/FN。

三个原片的 SHA256 分别为：

- test.mp4：`feb3ac6c3cb0a8eb52b02d08cc050f29cce04a8b6da78dd0c63eed245ad38ed9`
- overhead_pool.mp4：`a158bdc4a241f55e92bf87953b34334e6e03ce952b0e0cb1da1e5c217a8400d5`
- billiards_table.mp4：`867b761b0afdf91f7ade34946927b85c4b80581eeaa795909c5b876031218a07`

## 扰动设计

所有参数写在模板和 manifest 中。种子 `20261022`；噪声由 `(seed, frame_index)` 决定，与处理顺序和墙钟无关。原片内容是真实视频，只有 normal 的 `synthetic_illumination=false`，其余均为 `true`。不模拟新的真实球运动，不凭空添加真实遮挡。

| profile | 方法 | 覆盖 |
|---|---|---|
| normal | 像素原样写入无损容器 | 5 段 |
| red / blue / purple / cyan | RGB 通道增益；红 `[3,.3,.35]`，蓝 `[.35,.4,3]`，紫 `[1.8,.3,2]`，青 `[.4,1.1,1.8]` | 红全部；蓝/紫早期绿色与蓝色；青早期绿色 |
| local_red | 归一化中心 `(.86,.10)`，半径 `(.22,.16)` 高斯色斑，RGB 加 `[220,0,0]` | 5 段 |
| slow_color | 全片相位正弦通道变化，振幅 `[.8,-.65,.6]` | 早期绿色 |
| flicker | 2 Hz，低/高亮度乘数 `.15/1.4`，低亮度占周期 `.6` | 5 段 |
| pocket_highlight | 中心 `(.87,.50)`、半径 `(.18,.14)`，RGB 加 `[420,420,420]` 后裁切 | 5 段；竖向绿色对应右侧中袋附近，其它机位仅保证同图像位置 |
| shadow | 中心 `(.5,.5)`，半径 `(.35,.3)`，最大衰减 `.94` | 早期绿色 |
| gradient | 从上到下亮度乘数 `.15→1.1` | 早期绿色 |
| mixed | 从左至右 RGB 增益 `[2.5,.4,.4]→[.4,.6,2.5]` | 早期绿色 |
| noise | 独立高斯噪声，sigma=18（8 bit 像素尺度） | 早期绿色 |

每个扰动片段前后各约 0.5 秒正常，之间 0.25 秒渐入/渐出。改变作用于**全部连续帧**，包括运动、入口及复核窗；不是只调整截图。使用非线性显示空间上的测试扰动，不声称物理照明仿真。高光、欠曝造成的信息丢失不可视作仍然“可见”。

## 一条命令构建并诊断

在项目根目录，已安装现有 `.venv` 依赖并准备哈希一致的本地原片：

```bash
.venv/bin/python scripts/build_lighting_cases.py \
  --output data/lighting_cases/NEW_A_RUN \
  --evaluate logs/lighting_phase2/NEW_A_RUN
```

输出目录必须不存在，防止覆盖历史结果；缺片或哈希不符直接失败，不暗中跳过。只用现有 OpenCV 的 FFV1 编码器，编码器不可用即失败，不回退有损编码。manifest 记录源分辨率/FPS、帧范围、参数、种子、校准引用、真值、输入/生成脚本哈希、OpenCV/NumPy 版本、容器哈希和解码 BGR 全序列哈希。

构建器逐帧验证生成前/解码后像素哈希一致及总帧数；测试还在同机同环境连续构建两份独立小序列，验证 manifest、容器与解码哈希一致。跨 OpenCV/NumPy/编码器版本时以**解码像素哈希**为核验口径，不保证跨版本容器字节相同。生成时间依赖 CPU/磁盘；先检查可用磁盘，原分辨率噪声片段压缩率较低。

只重跑诊断而不重建：

```bash
.venv/bin/python scripts/evaluate_lighting_cases.py \
  --manifest data/lighting_cases/NEW_A_RUN/manifest.json \
  --output logs/lighting_phase2/NEW_EVALUATION
```

## 测量契约

- 对每个连续帧运行原 `PixelBallObserver` 和 `PocketEventAdapter`，记录像素检测、遮挡、状态与事件；不注入旧的 YOLO 预测，不训练/替换权重。本轮未测 YOLO 在彩灯下的独立性能/准确率。
- 在原片开头建立几何对照；后期绿色直接读既有 `late_pot_calibration.json`（引用时间 300 秒，不重新保存或覆盖）。这是算法对照几何，**不是人工入口真值**，旧 fixture 与当前片段仍可能存在偏差。
- CameraAlignment 独立顺序诊断，每四帧按原实现检查。它的位移/失效仅记录，不接入下游许可；本轮不实现 B 的质量保护层。适配器沿用固定参考几何，不等价于完整 UI 对齐链路。
- 每源视频每秒的 0/.36 秒两个相位及最后一帧记录一次（避免只在 2 Hz 闪烁同一相位取样）台面重新检测、重新检测的六袋状态，以及复用固定台面的袋口验证。与**同一源帧**的 normal 对比四角最大偏移（消除轮廓起始角/顺序差异）、各袋中心偏移和球数变化；不是与整段第一帧混比。
- 原始逐帧 trace、各快照完整结构和最小真值本地保存。报告中的识别数量、事件状态只是算法结果，没有人工真值时 `accuracy=null`。
- `observer_adapter_ms_not_ui` 是离线单进程局部耗时，不能和第一阶段 Qt FPS、YOLO 更新率或综合 CPU 直接比较，也不宣称性能提升。

## 测试入口及缺口

```bash
# 无现场视频也能运行的扰动/无损/不可变性测试；本地集成用例明确 skip
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/test_lighting_scenarios.py -q

# 本地数据准备完成后，连同原有测试原样回归
LIGHTING_MANIFEST="$PWD/data/lighting_cases/NEW_A_RUN/manifest.json" \
  QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests -q
```

新增测试不要求彩灯识别全部成功，不修改原有测试期望，不通过 xfail 隐藏已有失败。第一阶段报告、脚本、249 项已有用例保留；其性能证据仍为 20261021 对照。A 只做输入/诊断基线，不提前执行 E 的全套性能验收。

缺失的实拍及人工标签：真实红/蓝/紫/青灯、缓慢混光、规则/不规则闪烁、袋口反射高光、欠曝和人物阴影；逐帧黑白身份/可见性与入口几何；真实进球、反弹不进、擦袋、白球入袋、袋口手臂遮挡和恢复窗。当前合成矩阵不能替代这些现场验收。明确无遮挡和两球静止完整序列也待人工确认。仅稀疏截图审阅，不能宣布真实球房识别准确率。

本地生成视频、manifest、截图、trace 置于已忽略的 `data/lighting_cases/` 和 `logs/lighting_phase2/`；不上传公开 GitHub。仅脚本、无视频的参数模板、用例和脱敏汇总属于可入库内容。

A 阶段实测结果见 `docs/LIGHTING_ROBUSTNESS_REPORT.md`。
