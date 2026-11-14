[app]

# 手机上显示的名称
title = BarbellVBT
package.name = barbellvbt
package.domain = com.barbellvbt

# 项目源代码
source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,txt,md
source.exclude_exts = spec
source.exclude_dirs = .git,.github,.buildozer,bin,venv,__pycache__,.venv,tests

version = 0.3.3

# python-for-android develop 当前的 Python 与 hostpython 必须同版本。
# Kivy 不锁死版本，使用 p4a 自带并测试过的 Android recipe。
requirements = python3==3.14.2,hostpython3==3.14.2,kivy,numpy,opencv

# 视频文件读取权限
# Android 通过系统 ACTION_OPEN_DOCUMENT 选择视频，使用单文件 URI 授权，不申请整库媒体权限。

# Android 构建配置
android.api = 35
android.minapi = 24
android.ndk = 28c
android.accept_sdk_license = True
android.archs = arm64-v8a

# 当前 GitHub 仓库根目录中的图标文件
icon.filename = icon.png
presplash.filename = splash.png
presplash_color = #101820

orientation = portrait
fullscreen = 0
android.logcat_filters = *:S python:D

# 与 Python 3.14 的 python-for-android develop 分支配套
p4a.branch = develop

[buildozer]
log_level = 2
warn_on_root = 0
