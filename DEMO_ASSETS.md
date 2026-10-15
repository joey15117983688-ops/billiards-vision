# 当前默认俯拍八球视频

用户要求先替换当前视频，现使用球桌与六袋完整入镜的俯拍美式八球短片（蓝台呢），不是斯诺克。原视频保留，无透视拉正、裁剪或预置识别框。

来源：[FedericoPivotto/billiard-video-analysis](https://github.com/FedericoPivotto/billiard-video-analysis)，固定提交 `13fa4f15b41614c894e4cdd49a7557461680efaf`，文件 `dataset/game1_clip1/game1_clip1.mp4`。画面比分牌为 Van Boening / Kun Lin，Predator World 8-Ball Championship；不推断比赛日期。仓库附带 MIT 许可证，原始转播素材不据此宣称有独立商业授权。

- 本地：`data/overhead_pool.mp4`。
- 分辨率：1024 × 576；帧率 29.97 FPS；187 帧，约 6.24 秒，循环播放。
- 大小：951516 bytes。
- SHA1：`3fe9c16245b82fd987f237ba869d4838bf7137cd`。
- SHA256：`a158bdc4a241f55e92bf87953b34334e6e03ce952b0e0cb1da1e5c217a8400d5`。

默认视频路径已更新。桌面应用双击直接进入视频模式，仍可通过下拉框选择样本图片。命令行启动仍保留此前默认图片模式，可用 `--source video` 查看新视频。下载脚本已同步源文件并验证首、中、末帧。

本次仅补充蓝台呢支持，不改动模型权重。台面颜色按中央区域的主色选择，避免合并蓝色地毯；起始画面六袋验证通过，中、末帧抽样有一个袋口估算，不宣称所有帧均六袋可靠。当前毫米配置仍为用户此前确认的 3569 × 1778 mm **参考尺寸**，不代表该八球桌实测尺寸。

## 保留的斯诺克素材（上一默认视频）


来源：[Internet Archive — Snooker](https://archive.org/details/youtube-u7PyyW8CI64)，存档视频为 `u7PyyW8CI64.mp4`，原 YouTube 视频 ID 为 `u7PyyW8CI64`。本次检查实际画面，比分栏为潘晓婷／奥沙利文，包含红球、六色球、白球、连续击球及换局，也有球员近景和镜头切换。未根据文件名推断比赛日期。

本地：`data/snooker_match.mp4`。

- 实际时长：2194.80 秒，约 36 分 35 秒。
- 分辨率：854 × 480；帧率：25.00000 FPS；帧数：54870。
- 文件大小：218191097 bytes，约 218.2 MB。
- SHA1（与存档元数据一致）：`bbe13b54cb545168a3b98fa950ccbf8dde66ca1f`。
- SHA256：`c54606c87c2266dd24714aab39556584c1b15e76b21caf62b974c440afd39b5c`。

保留源文件内容，没有拼接、裁剪或插入识别框；这是约 36 分钟的连续录像，不声明它覆盖整场比赛。检查起始、中段、末帧解码以及 0／300／900／1500／2100 秒画面通过。

启动：

```bash
./scripts/start_dev.sh
```

在其他机器获取同一视频：

```bash
.venv/bin/python scripts/download_demo_assets.py
```

下载脚本流式写入磁盘、支持服务器允许时的断点下载，并核验大小、校验和及起始／中段／末帧解码。视频保存在本地，日常播放无需联网。播放器无声显示，自动循环，保留播放／暂停及现有 YOLO 点选功能。录像不加入 Git。

## 原测试素材

原红黄球录像保留在 `data/billiards_table.mp4`，约 38 秒，1920 × 1080，仍用于训练报告中的历史验收。来源：[Louis Geisler / Billard](https://github.com/louisgeisler/Billard)，提交 `b1dff416c06f9a311b77aac4a73715219b8fd31b` 的 `NOdebug.mp4`。

切换回原录像：

```bash
./scripts/start_dev.sh --video /Users/joey/projects/billiards-vision/data/billiards_table.mp4
```

更早的海康演示素材为 `data/hikvision_demo.mp4`；SDK 说明见 `HIKVISION_SDK.md`。
