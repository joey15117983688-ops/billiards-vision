# 海康威视 SDK 查找记录

选择：海康威视设备网络 SDK（HCNetSDK），用于海康网络摄像机、NVR 等设备的登录、预览、录像回放和控制。

## 官方来源

- [设备网络 SDK 平台说明](https://open.hikvision.com/osp)
- [Device Network SDK_Linux64 官方下载页](https://www.hikvision.com/en/support/tools/hitools/clf4633a00e385d6ea/)
- [HiTools 官方列表](https://hsrc.hikvision.com/en/support/tools/hitools/)
- [国内官方资源入口](https://open.hikvision.com/download/5cda567cf47ae80dd41a54b3)
- [官方 C 接口使用手册](https://open.hikvision.com/hardware/definitions/NET_DVR_Init.html)

官方列表当前标示 Linux64 版本为 V6.1.9.48。下载时以页面提供的实际包为准。

## 本机适用性与下载状态

本机为 macOS ARM64。官方公开说明的设备网络 SDK 平台包含 Windows、Linux、Android、iOS，未列出 macOS；Linux64 包不能直接作为本机原生动态库使用。

本次完成 SDK 选型及官方来源记录，**未下载 SDK 二进制包，也未安装或接入 SDK**。
国际下载页当前访问遇到网页验证/访问限制，国内下载流程需要登录并接受最终用户许可。
没有使用第三方重打包版本，也没有将本地 MP4 播放称为 SDK 接入。

后续真实摄像头接入前，需要确认摄像头型号（海康网络摄像机还是海康机器人工业相机）、部署设备操作系统、CPU 架构，以及对应 SDK 版本。工业相机可能需要另一套 MVS SDK。
当前视频播放使用已有 OpenCV 4.10.0 + PyQt5，不依赖 HCNetSDK。
