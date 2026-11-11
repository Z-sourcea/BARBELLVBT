[app]

# App 名称
title = BarbellVBT
package.name = barbellvbt
package.domain = com.barbellvbt

# 源代码目录
source.dir = .
source.include_exts = py,png,jpg,kv,yaml,txt
source.exclude_exts = spec

# 主程序入口
main.filename = main

# 版本
version = 1.0

# 依赖包
requirements = python3==3.11.9,kivy==2.3.0,opencv-python-headless==4.8.1.78,numpy==1.26.4,scipy==1.13.1,pyyaml==6.0.1

# 权限
android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE
android.features = android.hardware.camera

# Android API 级别
android.api = 33
android.minapi = 21
android.ndk = 25b
android.sdk = 33
android.accept_sdk_license = True

# 架构（手机用 arm64-v8a 即可）
android.archs = arm64-v8a,armeabi-v7a

# 图标和启动画面
icon.filename = assets/icon.png
presplash.filename = assets/splash.png
presplash_color = 1B5E20

# 方向
orientation = portrait

# 全屏
fullscreen = 0

# 日志
android.logcat_filters = *:S python:D

[buildenv]
# 构建环境
git = https://github.com/AndroidPlatform/manifest.git

[app@macosx:macosx]
# macOS 相关配置（如果需要在mac上编译）
type = macosx

[app@linux:linux]
# Linux 相关配置
type = linux

[app@win:windows]
# Windows 相关配置
type = win

[buildozer]
# 构建选项
log_level = 2
warn_on_root = 0

android.debug = True
android.release = False
