# 台球球桌俯拍测试视频

来源：[Louis Geisler / Billard 项目](https://github.com/louisgeisler/Billard)，作者提供的摄像头调试视频 `NOdebug.mp4`。
固定源码提交：`b1dff416c06f9a311b77aac4a73715219b8fd31b`。

本地文件：`data/billiards_table.mp4`，1920 × 1080，约 30 FPS，1139 帧，约 38.0 秒，约 86.2 MB。
保留原始录像，没有裁剪、透视校正或识别标注。摄像头从球桌上方拍摄，视角略有倾斜；主要画面是绿色球桌、红黄球及击球过程，边缘有少量拍摄设备与球员手部。
播放器无声显示，启动后自动循环播放，保留右上角白底黑字圆圈“1”和播放/暂停按钮。

SHA256: `867b761b0afdf91f7ade34946927b85c4b80581eeaa795909c5b876031218a07`

启动：

```bash
./scripts/start_dev.sh
```

视频已经下载至本地，日常启动不联网。
在其他机器下载同一份测试录像：

```bash
.venv/bin/python scripts/download_demo_assets.py
```

作者仓库未声明明确的视频再分发许可，素材仅用于本项目本地测试，不加入 Git 版本库。
之前的海康威视演示素材仍保存在 `data/hikvision_demo.mp4`，已不作为默认视频。其官方来源为 [滴漏检测相机效果视频](https://partners.hikvision.com/material-product/detail/964027327022809088?type=mp4)。
SDK 官方来源及平台限制仍见 `HIKVISION_SDK.md`。
