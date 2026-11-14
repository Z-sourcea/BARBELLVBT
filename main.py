#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BarbellVBT 简洁增强版：训练视频分析 + 训练记录管理（测试版本）。"""

__version__ = "0.3.4"

import csv
import json
import math
import os
import threading
import time
import traceback
import uuid
from datetime import datetime

import cv2
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button as _KivyButton
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.slider import Slider
from kivy.graphics import Color, RoundedRectangle
from kivy.graphics.texture import Texture
from kivy.uix.label import Label as _KivyLabel
from kivy.uix.popup import Popup as _KivyPopup
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner as _KivySpinner, SpinnerOption as _KivySpinnerOption
from kivy.uix.textinput import TextInput as _KivyTextInput
from kivy.utils import platform as kivy_platform

# Android 上 Kivy 默认 Roboto 字体可能缺少简体中文汉字，导致文字显示为空白方框。
# 优先使用手机系统自带的 Noto CJK 中文字体，不在 APK 中额外打包字体文件。
# 统一界面主题：深海军蓝背景 + 蓝色强调色，保留高对比度，方便训练场景查看。
UI_THEME = {
    "background": (0.025, 0.055, 0.120, 1),
    "surface": (0.055, 0.100, 0.190, 1),
    "surface_alt": (0.075, 0.140, 0.250, 1),
    "surface_soft": (0.100, 0.190, 0.320, 1),
    "text": (0.935, 0.960, 1.000, 1),
    "muted": (0.640, 0.735, 0.860, 1),
    "accent": (0.245, 0.585, 0.985, 1),
    "accent_dark": (0.105, 0.330, 0.675, 1),
    "blue": (0.245, 0.585, 0.985, 1),
    "amber": (0.920, 0.650, 0.245, 1),
    "red": (0.850, 0.315, 0.335, 1),
}


ANDROID_CJK_FONT_CANDIDATES = (
    "/system/fonts/NotoSansCJK-Regular.ttc",
    "/system/fonts/NotoSansSC-Regular.otf",
    "/system/fonts/NotoSansSC-VF.ttf",
    "/system/fonts/NotoSansCJKsc-Regular.otf",
    "/system/fonts/DroidSansFallback.ttf",
)


def find_ui_font():
    if kivy_platform == "android":
        for font_path in ANDROID_CJK_FONT_CANDIDATES:
            if os.path.isfile(font_path) and os.access(font_path, os.R_OK):
                return font_path
    return "Roboto"


UI_FONT_NAME = find_ui_font()


class Label(_KivyLabel):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("font_name", UI_FONT_NAME)
        kwargs.setdefault("color", UI_THEME["text"])
        super().__init__(*args, **kwargs)


class Button(_KivyButton):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("font_name", UI_FONT_NAME)
        kwargs.setdefault("background_normal", "")
        kwargs.setdefault("background_down", "")
        kwargs.setdefault("background_color", UI_THEME["accent_dark"])
        kwargs.setdefault("color", UI_THEME["text"])
        kwargs.setdefault("bold", True)
        super().__init__(*args, **kwargs)


class TextInput(_KivyTextInput):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("font_name", UI_FONT_NAME)
        kwargs.setdefault("background_normal", "")
        kwargs.setdefault("background_active", "")
        kwargs.setdefault("background_color", UI_THEME["surface_alt"])
        kwargs.setdefault("foreground_color", UI_THEME["text"])
        kwargs.setdefault("hint_text_color", UI_THEME["muted"])
        kwargs.setdefault("cursor_color", UI_THEME["accent"])
        kwargs.setdefault("padding", (dp(10), dp(10)))
        super().__init__(*args, **kwargs)


class SpinnerOption(_KivySpinnerOption):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("font_name", UI_FONT_NAME)
        kwargs.setdefault("background_normal", "")
        kwargs.setdefault("background_color", UI_THEME["surface_alt"])
        kwargs.setdefault("color", UI_THEME["text"])
        super().__init__(*args, **kwargs)


class Spinner(_KivySpinner):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("font_name", UI_FONT_NAME)
        kwargs.setdefault("option_cls", SpinnerOption)
        kwargs.setdefault("background_normal", "")
        kwargs.setdefault("background_down", "")
        kwargs.setdefault("background_color", UI_THEME["surface_alt"])
        kwargs.setdefault("color", UI_THEME["text"])
        super().__init__(*args, **kwargs)


class Popup(_KivyPopup):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("title_font", UI_FONT_NAME)
        super().__init__(*args, **kwargs)


class Panel(BoxLayout):
    """Rounded card background for the mobile UI."""
    def __init__(self, *args, panel_color=None, radius_dp=16, **kwargs):
        self._panel_color = panel_color or UI_THEME["surface"]
        self._radius_dp = radius_dp
        super().__init__(*args, **kwargs)
        with self.canvas.before:
            Color(*self._panel_color)
            self._panel_rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(self._radius_dp)])
        self.bind(pos=self._update_panel_rect, size=self._update_panel_rect)

    def _update_panel_rect(self, *_args):
        self._panel_rect.pos = self.pos
        self._panel_rect.size = self.size



class CalibrationImage(KivyImage):
    """Image widget that converts taps on a contain-fitted frame to source pixel coordinates."""
    def __init__(self, on_point, source_width, source_height, **kwargs):
        self.on_point = on_point
        self.source_width = int(source_width)
        self.source_height = int(source_height)
        kwargs.setdefault("fit_mode", "contain")
        super().__init__(**kwargs)

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        if self.source_width <= 0 or self.source_height <= 0 or self.width <= 0 or self.height <= 0:
            return super().on_touch_down(touch)
        scale = min(self.width / self.source_width, self.height / self.source_height)
        shown_w = self.source_width * scale
        shown_h = self.source_height * scale
        left = self.x + (self.width - shown_w) / 2.0
        bottom = self.y + (self.height - shown_h) / 2.0
        if not (left <= touch.x <= left + shown_w and bottom <= touch.y <= bottom + shown_h):
            return True
        x = (touch.x - left) / scale
        y = self.source_height - ((touch.y - bottom) / scale)
        x = min(max(float(x), 0.0), self.source_width - 1.0)
        y = min(max(float(y), 0.0), self.source_height - 1.0)
        self.on_point((x, y))
        return True



COLOR_PRESETS = {
    "红色或橙色（推荐）": "red",
    "蓝色": "blue",
    "黄色": "yellow",
    "绿色": "green",
}

TRACKING_MODES = (
    "单端标记（正侧面推荐）",
    "双端标记（斜向辅助）",
)

EXERCISES = ["深蹲", "前蹲", "硬拉", "高翻", "抓举", "卧推", "其他"]

# 手机端逐帧追踪时先缩小检测画面，避免对 4K 原始帧直接做 HSV/形态学处理导致内存峰值过高。
# 检测坐标会映射回原视频像素坐标，再使用原视频标定矩阵计算。
ANDROID_ANALYSIS_MAX_DIMENSION = 1280


def prepare_detection_frame(frame, max_dimension=ANDROID_ANALYSIS_MAX_DIMENSION):
    """Return a memory-bounded frame for detection and scale factors back to source pixels."""
    if frame is None or getattr(frame, "ndim", 0) < 2:
        raise ValueError("视频帧为空或格式无效")
    source_h, source_w = frame.shape[:2]
    if source_w <= 0 or source_h <= 0:
        raise ValueError("视频帧尺寸无效")
    scale = min(1.0, float(max_dimension) / float(max(source_w, source_h)))
    if scale < 1.0:
        detect_w = max(1, int(round(source_w * scale)))
        detect_h = max(1, int(round(source_h * scale)))
        detection_frame = cv2.resize(frame, (detect_w, detect_h), interpolation=cv2.INTER_AREA)
    else:
        detection_frame = frame
        detect_w, detect_h = source_w, source_h
    return detection_frame, source_w / float(detect_w), source_h / float(detect_h)

# 阶段说明用于帮助训练者理解动作和人工回看视频。
# 当前算法只追踪标记点并切分连续运动片段，不会自动判断具体技术阶段。
EXERCISE_GUIDES = {
    "深蹲": [
        ("1. 下放（离心阶段）", "从站立开始，髋关节和膝关节屈曲，杠铃随身体下降至目标深度。回看时观察下放是否平稳、躯干是否稳定，以及杠铃是否大致保持在中足上方。"),
        ("2. 底部转换", "从下降结束到开始起立的转向位置。回看时观察是否失去平衡、是否出现明显停顿，以及髋部和躯干是否保持可控。"),
        ("3. 起立（向心阶段）", "从底部向上伸展髋、膝关节，回到站立。回看时观察杠铃上升速度、髋部是否先窜起，以及左右是否对称。"),
    ],
    "前蹲": [
        ("1. 下放（离心阶段）", "从站立下蹲至目标深度。前架位置保持稳定，肘部尽量维持抬起，躯干通常比后蹲更直立。"),
        ("2. 底部转换", "在目标深度由下降转为上升。观察上背和前架是否松动、脚掌是否稳定，以及是否出现明显塌腰或失衡。"),
        ("3. 起立（向心阶段）", "伸展髋、膝关节回到站立。重点观察肘部是否持续抬起、胸部是否塌陷，以及杠铃上升是否连续。"),
    ],
    "硬拉": [
        ("1. 起始与离地", "从杠铃静止、建立身体张力开始，到杠铃离开地面。观察起拉前是否先建立背部和腹部张力，杠铃是否贴近身体。"),
        ("2. 过膝与拉起", "杠铃经过膝部并继续上升。观察杠铃路径是否贴近腿部、髋和膝是否协调伸展，避免杠铃明显向前绕。"),
        ("3. 锁定", "髋、膝完全伸展，身体站直并稳定控制杠铃。不要用过度后仰代替髋部锁定。"),
        ("4. 下放（离心阶段）", "有控制地屈髋、屈膝，将杠铃沿接近上升时的路径放回地面。"),
    ],
    "高翻": [
        ("1. 第一拉", "杠铃从地面上升至膝部附近。目标是稳定离地、保持杠铃靠近身体，并建立后续加速所需的位置。"),
        ("2. 过渡阶段", "杠铃经过膝部，身体调整到适合第二拉的位置。观察杠铃是否贴近身体、重心是否稳定。"),
        ("3. 第二拉", "从大腿附近到髋、膝、踝快速伸展并完成加速。通常是杠铃速度明显增加的阶段，回看时关注伸展时机和杠铃垂直轨迹。"),
        ("4. 翻肘与接杠", "杠铃上升后快速翻肘，将杠铃接到前架位置。观察接杠时肘部是否及时穿过、躯干是否稳定。"),
        ("5. 起立", "从接杠位置站起并稳定完成动作。"),
    ],
    "抓举": [
        ("1. 第一拉", "杠铃从地面上升至膝部附近。保持起始姿势和杠铃靠近身体，建立稳定的拉起轨迹。"),
        ("2. 过渡阶段", "杠铃经过膝部，身体重新调整到第二拉位置。观察是否过早远离身体或失去平衡。"),
        ("3. 第二拉", "髋、膝、踝快速伸展，对杠铃进行主要加速。回看时关注发力时机和杠铃路径。"),
        ("4. 下钻与接杠", "主动下钻到杠铃下方，在头顶伸臂锁定位置接住杠铃。观察接杠位置、稳定性和左右对称。"),
        ("5. 起立", "保持头顶锁定，将身体站起并稳定结束动作。"),
    ],
    "卧推": [
        ("1. 下放（离心阶段）", "从肘关节伸展位置将杠铃有控制地下降至胸前。观察肩胛稳定、手腕和前臂位置，以及下降路径是否一致。"),
        ("2. 胸前转换", "杠铃到达胸前触点并由下降转为推起。避免借助弹胸制造反弹。"),
        ("3. 推起（向心阶段）", "向上推起杠铃直至肘关节伸展。观察两侧速度是否一致、杠铃是否明显偏斜。"),
        ("4. 顶端稳定", "在肘部伸展位置控制住杠铃，再安全回架。"),
    ],
    "其他": [
        ("1. 准备位置", "确认站位、握法、器械和身体姿势稳定。"),
        ("2. 向心阶段", "肌肉主动克服阻力、器械沿主要发力方向移动的阶段。"),
        ("3. 离心阶段", "在控制下抵抗外力、器械向回程方向移动的阶段。"),
        ("4. 转换与稳定", "向心和离心之间的方向转换，以及动作结束后的稳定控制。"),
    ],
}


def safe_float(value, default=None):
    try:
        number = float(str(value).strip())
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default



def build_two_point_homography(points, reference_length_cm):
    """Build a metric image-to-plane transform from two taps along a known vertical length.

    This corrects image roll and scale only. It does NOT correct perspective/parallax.
    points must be clicked top endpoint first, bottom endpoint second.
    """
    if len(points) != 2:
        raise ValueError("两点标定需要恰好选择 2 个端点。")
    length_m = safe_float(reference_length_cm)
    if length_m is None or length_m <= 0:
        raise ValueError("参考长度必须是大于 0 的厘米数。")
    p1 = np.asarray(points[0], dtype=np.float64)
    p2 = np.asarray(points[1], dtype=np.float64)
    delta = p2 - p1
    pixel_length = float(np.linalg.norm(delta))
    if pixel_length < 60:
        raise ValueError("两个标定点距离太近，请选择更长的已知物体。")
    down = delta / pixel_length
    right = np.asarray([down[1], -down[0]], dtype=np.float64)
    ppm = pixel_length / (length_m / 100.0)
    # World X points right relative to the selected vertical reference; world Y points up.
    matrix = np.asarray([
        [right[0] / ppm, right[1] / ppm, -float(np.dot(p1, right)) / ppm],
        [-down[0] / ppm, -down[1] / ppm, float(np.dot(p1, down)) / ppm],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return matrix, {"mode": "two_point", "reference_length_cm": float(reference_length_cm), "pixel_length": pixel_length, "pixels_per_meter": ppm}


def build_four_point_homography(points, width_cm, height_cm):
    """Map an image quadrilateral to a known rectangle in the marker's motion plane.

    Point order: top-left, top-right, bottom-right, bottom-left.
    This compensates planar perspective only; it cannot reconstruct out-of-plane motion.
    """
    if len(points) != 4:
        raise ValueError("四点透视标定需要按左上、右上、右下、左下选择 4 个点。")
    width_cm_value = safe_float(width_cm)
    height_cm_value = safe_float(height_cm)
    if width_cm_value is None or height_cm_value is None or width_cm_value <= 0 or height_cm_value <= 0:
        raise ValueError("参考矩形的宽度和高度都必须大于 0 厘米。")
    width_m = width_cm_value / 100.0
    height_m = height_cm_value / 100.0
    src = np.asarray(points, dtype=np.float32)
    contour = src.reshape((-1, 1, 2))
    if not cv2.isContourConvex(contour):
        raise ValueError("四个点没有形成凸四边形，请按左上、右上、右下、左下顺序重新选择。")
    area = abs(float(cv2.contourArea(contour)))
    if area < 2500:
        raise ValueError("标定矩形在画面中太小，请选择更大的参考区域（建议至少约 50×50 像素）。")
    dst = np.asarray([
        [0.0, height_m],
        [width_m, height_m],
        [width_m, 0.0],
        [0.0, 0.0],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) < 1e-12:
        raise ValueError("透视变换矩阵无效，请重新选择四个角点。")
    return matrix, {
        "mode": "four_point_plane",
        "reference_width_cm": width_cm_value,
        "reference_height_cm": height_cm_value,
        "image_quad_area_px": area,
        "image_points": [[float(x), float(y)] for x, y in points],
    }


def camera_setup_warnings(camera_setup, calibration):
    """Return transparent setup notes. Approximate angle labels are never used as numeric correction."""
    setup = camera_setup or {}
    warnings = []
    yaw = setup.get("horizontal_angle", "不确定")
    height = setup.get("camera_height", "不确定")
    roll = setup.get("level", "不确定")
    mode = (calibration or {}).get("mode", "")
    if yaw == "明显斜拍（从前方或后方看）":
        warnings.append("拍摄方向明显偏斜：单目二维视频的视差风险较高。优先重新拍成正侧面；只有在能放置与标记运动平面完全重合的已知矩形时，才考虑四点透视标定。")
    elif yaw == "轻微斜拍（能看到部分前/后侧面）":
        warnings.append("拍摄方向有轻微斜角：两点比例标定不能消除透视误差。如参考矩形与标记运动平面重合，可使用四点透视标定；否则优先重新摆位。")
    elif yaw in ("不确定", "正面视角/无法判断"):
        warnings.append("拍摄方向不确定：请检查标记是否始终清晰可见，尽量改为运动平面的正侧面视角。")
    if height == "明显高于杠铃活动范围中点，向下拍" or height == "明显低于杠铃活动范围中点，向上拍":
        warnings.append("镜头俯仰角较大：可能引入透视和视差。建议把镜头调到杠铃活动范围中部附近。")
    if roll == "画面明显倾斜":
        warnings.append("手机未保持水平：建议重新固定手机并校平。两点标定可按参考线方向校正简单画面旋转，但不会消除透视。")
    if mode == "four_point_plane":
        warnings.append("已使用四点平面透视标定：仅在四个角点与杠铃标记运动处于同一平面时有效；标记明显前后移动时仍会有误差。")
    elif mode == "two_point":
        warnings.append("当前使用两点比例标定：可校正简单画面倾斜和像素比例，但不能校正斜拍造成的透视/视差。请尽量正侧面拍摄。")
    if setup.get("tracking_mode") == "dual":
        warnings.append(
            "当前为双端辅助模式：软件同时追踪两端不同颜色标记，并用两端图像坐标的中点近似杠铃中心；"
            "它可减少单端标记偏离中心造成的代表性问题，并显示杆轴方向/两端间距变化，但不是三维重建，"
            "不能消除明显深度视差。正侧面拍摄仍建议使用单端模式。"
        )
        if mode == "four_point_plane":
            warnings.append("双端模式使用四点标定时，参考矩形应尽量位于杠铃中心的主要运动平面，并覆盖中心轨迹；不要把矩形仅贴在某一端标记所在的深度平面。")
        else:
            warnings.append("双端模式使用两点标定时，结果仍是近似二维中心轨迹；斜拍透视影响不会被两端标记自动消除。若斜角明显，请优先调整机位或改用与中心运动平面重合的四点参考矩形。")
    return warnings


class BarbellTracker:
    """Color-marker tracker with continuity gating and bounded re-acquisition."""
    def __init__(self, color="red", fps=30.0):
        self.color = color
        self.fps = max(float(fps), 1.0)
        self.last_center = None
        self.last_frame = None
        self.last_area = 0.0
        self.lost_frames = 0

    def _mask(self, hsv):
        sat_low, val_low = 85, 65
        if self.color == "red":
            a = cv2.inRange(hsv, np.array([0, sat_low, val_low]), np.array([22, 255, 255]))
            b = cv2.inRange(hsv, np.array([170, sat_low, val_low]), np.array([180, 255, 255]))
            return cv2.bitwise_or(a, b)
        bounds = {
            "blue": ([92, sat_low, val_low], [132, 255, 255]),
            "yellow": ([22, sat_low, val_low], [36, 255, 255]),
            "green": ([40, sat_low, val_low], [86, 255, 255]),
        }
        low, high = bounds.get(self.color, bounds["blue"])
        return cv2.inRange(hsv, np.array(low), np.array(high))

    def detect(self, frame, frame_index=None):
        if frame is None:
            return None
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._mask(hsv)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = max(35.0, (w * h) * 0.000015)
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            candidates.append((cx, cy, area))

        if not candidates:
            self.lost_frames += 1
            if self.lost_frames > max(5, int(self.fps * 0.25)):
                self.last_center = None
                self.last_frame = None
                self.last_area = 0.0
            return None

        if self.last_center is None:
            chosen = max(candidates, key=lambda c: c[2])
        else:
            delta_frames = max(1, (frame_index - self.last_frame) if frame_index is not None and self.last_frame is not None else 1)
            diagonal = math.hypot(w, h)
            jump_limit = max(45.0, diagonal * 0.075) * math.sqrt(delta_frames)
            lx, ly = self.last_center
            nearby = [c for c in candidates if math.hypot(c[0] - lx, c[1] - ly) <= jump_limit]
            if not nearby:
                self.lost_frames += 1
                if self.lost_frames > max(5, int(self.fps * 0.25)):
                    self.last_center = None
                    self.last_frame = None
                    self.last_area = 0.0
                return None
            # Prefer a spatially continuous candidate while mildly penalising implausible area changes.
            def score(c):
                distance = math.hypot(c[0] - lx, c[1] - ly) / max(jump_limit, 1.0)
                area_ratio = abs(math.log(max(c[2], 1.0) / max(self.last_area, 1.0)))
                return distance + 0.12 * area_ratio
            chosen = min(nearby, key=score)

        self.last_center = (chosen[0], chosen[1])
        self.last_area = float(chosen[2])
        self.last_frame = frame_index
        self.lost_frames = 0
        return (int(round(chosen[0])), int(round(chosen[1])))


class VideoAnalyzer:
    """Track a colored marker and calculate calibrated 2D in-plane vertical kinematics."""

    def __init__(self, calibration=None, marker_color="red", tracking_mode="single", marker2_color="blue"):
        self.calibration = calibration or {}
        self.marker_color = marker_color
        self.tracking_mode = "dual" if tracking_mode == "dual" else "single"
        self.marker2_color = marker2_color
        if self.tracking_mode == "dual" and self.marker_color == self.marker2_color:
            raise ValueError("双端标记必须使用两种不同颜色，请分别选择第一端和第二端颜色。")
        self.results = []
        self.summary = {}
        self.homography = np.asarray(self.calibration.get("homography", []), dtype=np.float64)
        if (self.homography.shape != (3, 3) or not np.all(np.isfinite(self.homography))
                or abs(float(np.linalg.det(self.homography))) < 1e-12):
            self.homography = None

    @staticmethod
    def _smooth(values, window=5):
        values = np.asarray(values, dtype=np.float64)
        if len(values) < 3:
            return values
        window = min(window, len(values) if len(values) % 2 else len(values) - 1)
        if window < 3:
            return values
        # Median filter suppresses isolated detector jumps; moving average smooths residual jitter.
        padded = np.pad(values, (window // 2, window // 2), mode="edge")
        med = np.asarray([np.median(padded[i:i + window]) for i in range(len(values))])
        smooth_window = 3 if len(med) >= 3 else 1
        if smooth_window > 1:
            p2 = np.pad(med, (1, 1), mode="edge")
            med = np.convolve(p2, np.ones(3) / 3.0, mode="valid")
        return med

    def _to_world(self, image_points):
        if self.homography is None:
            raise ValueError("缺少有效标定矩阵。请先完成两点或四点标定。")
        pts = np.asarray(image_points, dtype=np.float32).reshape((-1, 1, 2))
        mapped = cv2.perspectiveTransform(pts, self.homography.astype(np.float64)).reshape((-1, 2))
        if not np.all(np.isfinite(mapped)) or np.max(np.abs(mapped)) > 1000:
            raise ValueError("标定转换产生异常坐标，请检查四个角点顺序和实际尺寸后重新标定。")
        return mapped.astype(np.float64)

    def _segment_metrics(self, segment, number):
        times = np.asarray([p[0] for p in segment], dtype=np.float64)
        world_x_raw = np.asarray([p[4] for p in segment], dtype=np.float64)
        world_y_raw = np.asarray([p[5] for p in segment], dtype=np.float64)
        if len(times) < 5 or np.any(np.diff(times) <= 0):
            return None
        world_x = self._smooth(world_x_raw, 3)
        world_y = self._smooth(world_y_raw, 3)
        signed_velocity = np.gradient(world_y, times)
        smooth_signed = self._smooth(signed_velocity, 5)
        speeds = np.abs(smooth_signed)
        if not np.all(np.isfinite(speeds)):
            return None
        duration = float(times[-1] - times[0])
        if duration <= 0:
            return None
        dy = np.diff(world_y)
        dx = np.diff(world_x)
        path_length = float(np.sum(np.hypot(dx, dy)))
        moving_signs = smooth_signed[np.abs(smooth_signed) >= 0.04]
        if len(moving_signs) == 0:
            vertical_direction = "方向不明确"
        else:
            upward_ratio = float(np.mean(moving_signs > 0))
            if upward_ratio >= 0.80:
                vertical_direction = "上升为主"
            elif upward_ratio <= 0.20:
                vertical_direction = "下降为主"
            else:
                vertical_direction = "上升与下降混合"

        # 将单个连续运动片段沿实际运动方向（而非固定图像上下方向）按垂直行程均分为起始/中段/末端。
        # 这是“行程分区速度”，不是自动识别的高翻第一拉/第二拉或解剖学阶段。
        range_zone_speeds = {"lower": None, "middle": None, "upper": None}
        range_zone_counts = {"lower": 0, "middle": 0, "upper": 0}
        vertical_range = float(np.max(world_y) - np.min(world_y))
        if vertical_direction in ("上升为主", "下降为主") and vertical_range >= 0.02:
            if vertical_direction == "上升为主":
                progress = (world_y - float(np.min(world_y))) / vertical_range
            else:
                progress = (float(np.max(world_y)) - world_y) / vertical_range
            zone_masks = {
                "lower": (progress >= 0.0) & (progress < (1.0 / 3.0)),
                "middle": (progress >= (1.0 / 3.0)) & (progress < (2.0 / 3.0)),
                "upper": (progress >= (2.0 / 3.0)) & (progress <= 1.0),
            }
            for zone_name, zone_mask in zone_masks.items():
                sample_count = int(np.count_nonzero(zone_mask))
                range_zone_counts[zone_name] = sample_count
                if sample_count >= 3:
                    range_zone_speeds[zone_name] = round(float(np.mean(speeds[zone_mask])), 3)
        return {
            "rep": number,
            "direction": vertical_direction,
            "peak_velocity": round(float(np.max(speeds)), 3),
            "mean_velocity": round(float(np.mean(speeds)), 3),
            "displacement": round(float(np.max(world_y) - np.min(world_y)), 3),
            "horizontal_drift": round(float(np.max(world_x) - np.min(world_x)), 3),
            "path_length_2d": round(path_length, 3),
            "duration": round(duration, 2),
            "start_time": round(float(times[0]), 2),
            "end_time": round(float(times[-1]), 2),
            "tracked_frames": int(len(segment)),
            "range_zone_speeds_m_s": range_zone_speeds,
            "range_zone_sample_counts": range_zone_counts,
            "range_zone_method": "垂直行程等分三段；不是技术阶段自动识别",
        }

    def analyze(self, video_path, callback=None, camera_setup=None):
        if self.homography is None:
            return {"error": "尚未完成有效标定。请先在视频画面中完成两点比例标定或四点透视标定。"}
        try:
            cv2.setNumThreads(2)
        except Exception:
            pass
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return {"error": "无法打开视频。请尝试使用手机相册导出 H.264 编码的 MP4 文件，并确认视频没有损坏。"}
        fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 30.0) or 30.0
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if video_width <= 0 or video_height <= 0:
            cap.release()
            return {"error": "无法读取视频分辨率。请尝试将视频转换为常规 H.264 MP4 后重新导入。"}
        if callback:
            callback({"stage": "video_opened", "fps": fps, "width": video_width, "height": video_height,
                      "total_frames": total_frames})
        tracker = BarbellTracker(self.marker_color, fps=fps)
        tracker2 = BarbellTracker(self.marker2_color, fps=fps) if self.tracking_mode == "dual" else None
        # Each row: time, representative image center, frame, primary area,
        # endpoint-1 x/y, endpoint-2 x/y, endpoint distance, projected bar-axis angle.
        raw_points = []
        frame_count = 0
        marker1_found = 0
        marker2_found = 0
        paired_found = 0
        too_close_pairs = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index = frame_count
                timestamp = frame_index / fps
                detect_frame, restore_x, restore_y = prepare_detection_frame(
                    frame, ANDROID_ANALYSIS_MAX_DIMENSION
                )

                center1_small = tracker.detect(detect_frame, frame_index=frame_index)
                center2_small = tracker2.detect(detect_frame, frame_index=frame_index) if tracker2 is not None else None
                center1 = (int(round(center1_small[0] * restore_x)), int(round(center1_small[1] * restore_y))) if center1_small is not None else None
                center2 = (int(round(center2_small[0] * restore_x)), int(round(center2_small[1] * restore_y))) if center2_small is not None else None
                if center1 is not None:
                    marker1_found += 1
                if tracker2 is not None and center2 is not None:
                    marker2_found += 1

                if tracker2 is None:
                    if center1 is not None:
                        raw_points.append((timestamp, center1[0], center1[1], frame_index,
                                           tracker.last_area, None, None, None, None, None, None))
                elif center1 is not None and center2 is not None:
                    dx = float(center2[0] - center1[0])
                    dy = float(center2[1] - center1[1])
                    separation = math.hypot(dx, dy)
                    frame_diagonal = math.hypot(frame.shape[1], frame.shape[0])
                    # Avoid interpreting two overlapping or implausibly distant color blobs as bar ends.
                    if separation < max(12.0, frame_diagonal * 0.012):
                        too_close_pairs += 1
                    elif separation > frame_diagonal * 0.85:
                        pass
                    else:
                        midpoint_x = (center1[0] + center2[0]) / 2.0
                        midpoint_y = (center1[1] + center2[1]) / 2.0
                        axis_angle = math.degrees(math.atan2(dy, dx))
                        raw_points.append((timestamp, midpoint_x, midpoint_y, frame_index,
                                           min(tracker.last_area, tracker2.last_area),
                                           center1[0], center1[1], center2[0], center2[1],
                                           separation, axis_angle))
                        paired_found += 1
                frame_count += 1
                if callback and frame_count % 30 == 0 and total_frames > 0:
                    callback({"progress": min(100.0, frame_count / total_frames * 100.0)})
                if callback and frame_count % 300 == 0:
                    callback({"stage": "frames", "frame_count": frame_count, "total_frames": total_frames})
        except Exception as exc:
            return {"error": f"读取视频时出错：{type(exc).__name__}: {exc}"}
        finally:
            cap.release()

        if frame_count < 2 or len(raw_points) < 8:
            if self.tracking_mode == "dual":
                return {"error": "双端标记有效配对太少。请确认两端颜色不同、同时出现在画面中、没有被身体或器械遮挡，并检查光线。双端模式不会用单端位置伪造杠铃中心。"}
            return {"error": "有效标记点太少，无法分析。请检查标记颜色、光线、遮挡和标定画面。"}
        if self.tracking_mode == "dual" and len(raw_points) / max(frame_count, 1) < 0.15:
            return {"error": "双端标记同时识别的帧数不足视频总帧数的 15%。请检查两种颜色、标记大小、遮挡和拍摄距离，或改用单端模式。"}

        world = self._to_world([(p[1], p[2]) for p in raw_points])
        # World coordinates are computed from the midpoint of the two image-space endpoints in dual mode.
        # Under perspective, this is an approximation, not a 3-D reconstruction.
        points = []
        for raw, mapped in zip(raw_points, world):
            points.append((
                float(raw[0]), int(round(raw[1])), int(round(raw[2])), int(raw[3]),
                float(mapped[0]), float(mapped[1]), float(raw[4]),
                None if raw[5] is None else int(raw[5]),
                None if raw[6] is None else int(raw[6]),
                None if raw[7] is None else int(raw[7]),
                None if raw[8] is None else int(raw[8]),
                None if raw[9] is None else float(raw[9]),
                None if raw[10] is None else float(raw[10]),
            ))
        times = np.asarray([p[0] for p in points], dtype=np.float64)
        world_y = np.asarray([p[5] for p in points], dtype=np.float64)
        frame_numbers = np.asarray([p[3] for p in points], dtype=np.int32)
        moving_threshold = 0.04  # m/s; practical segmentation threshold, not a validated threshold.
        stop_samples = max(6, int(round(fps * 0.25)))
        max_gap_frames = max(1, int(round(fps * 0.35)))
        # Calculate derivatives separately inside continuous detection windows; never differentiate across a long dropout.
        split_points = np.where(np.diff(frame_numbers) > max_gap_frames)[0] + 1
        continuous_groups = np.split(np.arange(len(points)), split_points)
        smooth_speed = np.zeros(len(points), dtype=np.float64)
        for group_indices in continuous_groups:
            if len(group_indices) < 3:
                continue
            group_y = self._smooth(world_y[group_indices], 3)
            group_speed = np.abs(np.gradient(group_y, times[group_indices]))
            smooth_speed[group_indices] = self._smooth(group_speed, 5)
        segments, active, low_indices = [], [], []

        def close_segment():
            nonlocal active, low_indices
            if len(active) >= 5:
                segments.append(active)
            active, low_indices = [], []

        for i, point in enumerate(points):
            if i > 0 and point[3] - points[i - 1][3] > max_gap_frames:
                close_segment()
            if smooth_speed[i] >= moving_threshold:
                if low_indices and active:
                    active.extend(low_indices)
                low_indices = []
                active.append(point)
            elif active:
                low_indices.append(point)
                if len(low_indices) >= stop_samples:
                    close_segment()
        close_segment()

        results = []
        for segment in segments:
            item = self._segment_metrics(segment, len(results) + 1)
            if item:
                results.append(item)
        if not results:
            return {"error": "没有识别到足够长的运动片段。请检查标记、视频帧率和标定平面。"}

        comparable_groups = {}
        for item in results:
            if item["direction"] in ("上升为主", "下降为主"):
                comparable_groups.setdefault(item["direction"], []).append(item)
        for item in results:
            peers = comparable_groups.get(item["direction"], [])
            if len(peers) < 2:
                item["velocity_loss_pct"] = None
            else:
                fastest_same_direction = max(peer["mean_velocity"] for peer in peers)
                item["velocity_loss_pct"] = round(
                    max(0.0, (fastest_same_direction - item["mean_velocity"]) / fastest_same_direction * 100.0), 1
                ) if fastest_same_direction > 0 else 0.0

        detected_frames = np.asarray([p[3] for p in points], dtype=np.int32)
        missing_gaps = np.diff(detected_frames) - 1 if len(detected_frames) > 1 else np.asarray([], dtype=np.int32)
        max_missing_frames = int(np.max(missing_gaps)) if len(missing_gaps) else 0
        max_missing_seconds = max_missing_frames / fps
        coverage = min(100.0, len(points) / max(frame_count, 1) * 100.0)
        marker1_coverage = min(100.0, marker1_found / max(frame_count, 1) * 100.0)
        marker2_coverage = (min(100.0, marker2_found / max(frame_count, 1) * 100.0)
                            if self.tracking_mode == "dual" else None)
        pair_distances = np.asarray([p[9] for p in raw_points if p[9] is not None], dtype=np.float64)
        pair_angles = np.asarray([p[10] for p in raw_points if p[10] is not None], dtype=np.float64)
        if len(pair_distances):
            pair_distance_median = float(np.median(pair_distances))
            pair_distance_cv_pct = float(np.std(pair_distances) / pair_distance_median * 100.0) if pair_distance_median > 0 else 999.0
            pair_distance_range_pct = float((np.max(pair_distances) - np.min(pair_distances)) / pair_distance_median * 100.0) if pair_distance_median > 0 else 999.0
        else:
            pair_distance_median = pair_distance_cv_pct = pair_distance_range_pct = None
        if len(pair_angles):
            unwrapped_angles = np.rad2deg(np.unwrap(np.deg2rad(pair_angles)))
            bar_axis_angle_median = float(np.median(unwrapped_angles))
            bar_axis_angle_range = float(np.max(unwrapped_angles) - np.min(unwrapped_angles))
        else:
            bar_axis_angle_median = bar_axis_angle_range = None

        if coverage >= 85 and max_missing_seconds <= 0.10:
            tracking_quality = "较好"
        elif coverage >= 65 and max_missing_seconds <= 0.25:
            tracking_quality = "一般，建议核对轨迹"
        else:
            tracking_quality = "较低，不建议直接使用速度数值"
        if self.tracking_mode == "dual" and pair_distance_cv_pct is not None:
            if pair_distance_cv_pct > 20 or coverage < 65:
                tracking_quality = "较低：双端配对或间距稳定性不足"
            elif pair_distance_cv_pct > 10 or coverage < 85:
                tracking_quality = "一般：请核对两端标记配对"

        camera_warnings = camera_setup_warnings({**(camera_setup or {}), "tracking_mode": self.tracking_mode}, self.calibration)
        if self.tracking_mode == "dual" and pair_distance_cv_pct is not None:
            if pair_distance_cv_pct > 20:
                camera_warnings.append(f"两端标记间距变异系数为 {pair_distance_cv_pct:.1f}%，偏高。可能存在误追踪、遮挡、颜色串扰或杠铃杆轴方向变化；建议先检查回放，不要直接采信速度结果。")
            elif pair_distance_cv_pct > 10:
                camera_warnings.append(f"两端标记间距变异系数为 {pair_distance_cv_pct:.1f}%。请回看两端标记是否始终贴在同一根杠铃两端。")
            else:
                camera_warnings.append(f"两端标记间距变异系数为 {pair_distance_cv_pct:.1f}%，配对间距相对稳定；这只表示追踪几何关系稳定，不代表测量准确度已经验证。")
        calibration_region_coverage = None
        if self.calibration.get("mode") == "four_point_plane":
            image_quad = self.calibration.get("image_points")
            if image_quad and len(image_quad) == 4:
                polygon = np.asarray(image_quad, dtype=np.float32).reshape((-1, 1, 2))
                inside_count = sum(
                    1 for raw in raw_points
                    if cv2.pointPolygonTest(polygon, (float(raw[1]), float(raw[2])), False) >= 0
                )
                calibration_region_coverage = round(inside_count / max(len(raw_points), 1) * 100.0, 1)
                if calibration_region_coverage < 80.0:
                    camera_warnings.append(
                        f"只有 {calibration_region_coverage:.1f}% 的标记识别点落在四点标定矩形内部。当前轨迹大量位于标定区域外，属于外推，尺度误差风险较高；建议扩大参考矩形，使它在同一平面内包围整个杠铃运动范围后重新标定。"
                    )
                else:
                    camera_warnings.append(
                        f"标记点有 {calibration_region_coverage:.1f}% 落在四点标定矩形内；仍需确认标记与参考矩形处于同一平面。"
                    )
        self.results = results
        self.summary = {
            "fps": round(float(fps), 2),
            "video_width": int(video_width),
            "video_height": int(video_height),
            "analysis_max_dimension": int(ANDROID_ANALYSIS_MAX_DIMENSION),
            "total_frames": int(frame_count),
            "tracked_frames": int(len(points)),
            "tracking_coverage_pct": round(float(coverage), 1),
            "max_missing_seconds": round(float(max_missing_seconds), 3),
            "tracking_quality": tracking_quality,
            "tracking_mode": self.tracking_mode,
            "marker1_coverage_pct": round(float(marker1_coverage), 1),
            "marker2_coverage_pct": None if marker2_coverage is None else round(float(marker2_coverage), 1),
            "dual_pair_coverage_pct": round(float(coverage), 1) if self.tracking_mode == "dual" else None,
            "endpoint_distance_median_px": None if pair_distance_median is None else round(pair_distance_median, 2),
            "endpoint_distance_cv_pct": None if pair_distance_cv_pct is None else round(pair_distance_cv_pct, 2),
            "endpoint_distance_range_pct": None if pair_distance_range_pct is None else round(pair_distance_range_pct, 2),
            "bar_axis_angle_median_deg": None if bar_axis_angle_median is None else round(bar_axis_angle_median, 2),
            "bar_axis_angle_range_deg": None if bar_axis_angle_range is None else round(bar_axis_angle_range, 2),
            "paired_frames": int(paired_found) if self.tracking_mode == "dual" else None,
            "marker1_detected_frames": int(marker1_found),
            "marker2_detected_frames": int(marker2_found) if self.tracking_mode == "dual" else None,
            "too_close_pair_frames": int(too_close_pairs) if self.tracking_mode == "dual" else None,
            "mean_velocity": round(float(np.mean([r["mean_velocity"] for r in results])), 3),
            "peak_velocity": round(float(max(r["peak_velocity"] for r in results)), 3),
            "best_mean_velocity": round(float(max(r["mean_velocity"] for r in results)), 3),
            "last_velocity_loss_pct": results[-1]["velocity_loss_pct"],
            "segment_count": len(results),
            "calibration_mode": self.calibration.get("mode", "unknown"),
            "calibration_region_coverage_pct": calibration_region_coverage,
            "camera_warnings": camera_warnings,
        }
        trajectory = []
        for p in points:
            item = {
                "frame": int(p[3]), "time": round(float(p[0]), 4), "x": int(p[1]), "y": int(p[2]),
                "world_x": round(float(p[4]), 5), "world_y": round(float(p[5]), 5),
            }
            if self.tracking_mode == "dual":
                item.update({
                    "marker1_x": int(p[7]), "marker1_y": int(p[8]),
                    "marker2_x": int(p[9]), "marker2_y": int(p[10]),
                    "endpoint_distance_px": round(float(p[11]), 2),
                    "bar_axis_angle_deg": round(float(p[12]), 2),
                })
            trajectory.append(item)
        if callback:
            callback({"progress": 100.0})
        return {
            "results": results,
            "summary": self.summary,
            "trajectory": trajectory,
            "tracking_mode": self.tracking_mode,
            "video_info": {"fps": round(float(fps), 3), "frame_count": int(frame_count)},
        }

    @staticmethod
    def export_csv(filepath, session, results, summary):
        with open(filepath, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["BarbellVBT 训练视频分析记录"])
            writer.writerow(["日期", session.get("date", "")])
            writer.writerow(["训练者", session.get("athlete", "")])
            writer.writerow(["动作", session.get("exercise", "")])
            writer.writerow(["负荷(kg)", session.get("load_kg", "")])
            writer.writerow(["追踪模式", session.get("tracking_mode", "single")])
            writer.writerow(["第一端标记颜色", session.get("marker_color", "")])
            writer.writerow(["第二端标记颜色", session.get("marker2_color", "不适用")])
            writer.writerow(["标定方式", session.get("calibration_mode", "")])
            writer.writerow(["拍摄水平角", session.get("camera_horizontal_angle", "")])
            writer.writerow(["镜头高度", session.get("camera_height", "")])
            writer.writerow(["手机水平状态", session.get("camera_level", "")])
            writer.writerow(["识别覆盖率(%)", summary.get("tracking_coverage_pct", "")])
            writer.writerow(["最大连续漏检(s)", summary.get("max_missing_seconds", "")])
            writer.writerow(["追踪质量提示", summary.get("tracking_quality", "")])
            writer.writerow(["第一端识别覆盖率(%)", summary.get("marker1_coverage_pct", "")])
            writer.writerow(["第二端识别覆盖率(%)", summary.get("marker2_coverage_pct", "不适用")])
            writer.writerow(["双端配对覆盖率(%)", summary.get("dual_pair_coverage_pct", "不适用")])
            writer.writerow(["两端标记间距中位数(px)", summary.get("endpoint_distance_median_px", "不适用")])
            writer.writerow(["两端标记间距变异系数(%)", summary.get("endpoint_distance_cv_pct", "不适用")])
            writer.writerow(["投影杆轴角中位数(deg)", summary.get("bar_axis_angle_median_deg", "不适用")])
            writer.writerow(["投影杆轴角变化范围(deg)", summary.get("bar_axis_angle_range_deg", "不适用")])
            writer.writerow(["四点标定矩形内识别点比例(%)", summary.get("calibration_region_coverage_pct", "不适用")])
            writer.writerow(["平均垂直速度(m/s)", summary.get("mean_velocity", "")])
            writer.writerow(["峰值垂直速度(m/s)", summary.get("peak_velocity", "")])
            writer.writerow([])
            writer.writerow(["运动片段", "主要方向", "峰值垂直速度(m/s)", "平均垂直速度(m/s)", "垂直位移(m)", "水平漂移(m)", "二维路径长度(m)", "持续时间(s)", "开始时间(s)", "结束时间(s)", "相对最快同方向片段速度下降(%)", "行程起始1/3平均速度(m/s)", "行程中段1/3平均速度(m/s)", "行程末端1/3平均速度(m/s)"])
            for item in results:
                writer.writerow([
                    item["rep"], item["direction"], item["peak_velocity"], item["mean_velocity"],
                    item["displacement"], item["horizontal_drift"], item["path_length_2d"],
                    item["duration"], item["start_time"], item["end_time"], item["velocity_loss_pct"],
                    item.get("range_zone_speeds_m_s", {}).get("lower"),
                    item.get("range_zone_speeds_m_s", {}).get("middle"),
                    item.get("range_zone_speeds_m_s", {}).get("upper"),
                ])
        return filepath


class BarbellVBTApp(App):
    ANDROID_VIDEO_REQUEST_CODE = 7312

    def _diagnostic_path(self):
        base_dir = getattr(self, "user_data_dir", None)
        if not base_dir:
            base_dir = os.path.join(os.path.expanduser("~"), ".barbellvbt")
        return os.path.join(base_dir, "analysis_debug.log")

    def _write_diagnostic(self, message):
        """Write a small local diagnostic log; this survives an app restart after Android kills the process."""
        try:
            os.makedirs(self.user_data_dir, exist_ok=True)
            with open(self._diagnostic_path(), "a", encoding="utf-8") as log_file:
                log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
        except Exception:
            pass

    def _show_diagnostics(self, _instance=None):
        """Show recent diagnostic lines so a user can screenshot them if analysis still exits."""
        try:
            path = self._diagnostic_path()
            if not os.path.isfile(path):
                text = "目前没有诊断日志。"
            else:
                with open(path, "r", encoding="utf-8") as log_file:
                    lines = log_file.readlines()[-80:]
                text = "".join(lines) or "诊断日志为空。"
        except Exception as exc:
            text = f"读取诊断日志失败：{type(exc).__name__}: {exc}"
        scroll = ScrollView()
        label = Label(text=text, font_size="11sp", halign="left", valign="top", size_hint_y=None)
        label.bind(size=lambda obj, size: setattr(obj, "text_size", (max(size[0] - dp(12), 0), None)))
        label.bind(texture_size=lambda obj, size: setattr(obj, "height", size[1] + dp(20)))
        scroll.add_widget(label)
        Popup(title="分析诊断日志（最近 80 行）", content=scroll, size_hint=(0.96, 0.88)).open()

    def build(self):
        self.title = "BarbellVBT 篮球力量表现分析"
        self.calibration = None
        self.camera_setup = {}
        self.calibration_label = "尚未标定：先选择视频，再完成两点或四点标定。"
        self.marker_label = "红色或橙色（推荐）"
        self.marker2_label = "蓝色"
        self.tracking_mode = "single"
        self.analyzer = None
        self.results = []
        self.summary = {}
        self.selected_video = None
        self.session = {}
        self.trajectory = []
        self._preview_cap = None
        self._preview_event = None
        self._preview_is_playing = False
        self._preview_updating_slider = False
        self._preview_fps = 30.0
        self._preview_frame_stride = 1
        self._preview_total_frames = 0
        self._preview_frame_numbers = np.asarray([], dtype=np.int32)
        self._preview_coords = np.empty((0, 2), dtype=np.int32)
        self._preview_marker1_coords = np.empty((0, 2), dtype=np.int32)
        self._preview_marker2_coords = np.empty((0, 2), dtype=np.int32)
        self._preview_marker1_segments = []
        self._preview_marker2_segments = []
        self._preview_is_dual = False
        self._preview_display_segments = []
        self._write_diagnostic(f"APP_BUILD version={__version__} platform={kivy_platform}")

        root = BoxLayout(orientation="vertical", padding=(dp(10), dp(8)), spacing=dp(6))
        with root.canvas.before:
            Color(*UI_THEME["background"])
            self._root_bg = RoundedRectangle(pos=root.pos, size=root.size, radius=[0])
        root.bind(pos=lambda obj, *_: setattr(self._root_bg, "pos", obj.pos),
                  size=lambda obj, *_: setattr(self._root_bg, "size", obj.size))

        self.scroll = ScrollView(do_scroll_x=False, bar_width=dp(4),
                                  bar_color=UI_THEME["accent"], bar_inactive_color=UI_THEME["surface_soft"])
        content = BoxLayout(orientation="vertical", spacing=dp(11), size_hint_y=None,
                            padding=(dp(2), dp(4), dp(2), dp(18)))
        content.bind(minimum_height=content.setter("height"))

        def panel(title=None, subtitle=None, panel_color=None):
            card = Panel(orientation="vertical", spacing=dp(8), padding=(dp(13), dp(12)),
                         size_hint_y=None, panel_color=panel_color or UI_THEME["surface"])
            card.bind(minimum_height=card.setter("height"))
            if title:
                head = Label(text=title, bold=True, font_size="15sp", color=UI_THEME["text"],
                             halign="left", valign="middle", size_hint_y=None, height=dp(25))
                self._wrap_label(head)
                card.add_widget(head)
            if subtitle:
                sub = Label(text=subtitle, font_size="11sp", color=UI_THEME["muted"],
                            halign="left", valign="middle", size_hint_y=None, height=dp(34))
                self._wrap_label(sub)
                card.add_widget(sub)
            return card

        def small_label(text, height=28, font_size="12sp", color=None, bold=False, halign="left"):
            label = Label(text=text, size_hint_y=None, height=dp(height), font_size=font_size,
                          color=color or UI_THEME["muted"], bold=bold, halign=halign, valign="middle")
            self._wrap_label(label)
            return label

        # Header / product identity
        hero = panel(panel_color=UI_THEME["surface"])
        hero_top = BoxLayout(size_hint_y=None, height=dp(24), spacing=dp(6))
        hero_top.add_widget(Label(text="BARBELLVBT", bold=True, font_size="13sp",
                                  color=UI_THEME["accent"], halign="left"))
        hero_top.add_widget(Label(text="VERSION 0.3.3", font_size="10sp",
                                  color=UI_THEME["muted"], halign="right"))
        hero.add_widget(hero_top)
        hero.add_widget(Label(text="力量训练表现分析", font_size="23sp", bold=True,
                              color=UI_THEME["text"], halign="left", size_hint_y=None, height=dp(36)))
        hero.add_widget(small_label("视频测量 · 杠铃轨迹 · 行程分区速度 · 训练记录", height=dp(24),
                                    font_size="12sp", color=UI_THEME["muted"]))
        content.add_widget(hero)

        # Session setup
        setup = panel("01 训练信息", "先填写运动员和训练内容；负荷、训练者和备注可以留空。")
        self.athlete_input = self._input("运动员姓名或编号（可不填）", height=42)
        setup.add_widget(self.athlete_input)
        row = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
        self.exercise_spinner = Spinner(text="深蹲", values=EXERCISES, size_hint_x=0.58, font_size="13sp")
        self.load_input = self._input("负荷 kg", height=44, input_filter="float")
        row.add_widget(self.exercise_spinner)
        row.add_widget(self.load_input)
        setup.add_widget(row)
        self.notes_input = TextInput(hint_text="训练备注：RPE、训练目标或其他说明", multiline=True,
                                     size_hint_y=None, height=dp(58), font_size="13sp")
        setup.add_widget(self.notes_input)
        content.add_widget(setup)

        # Video and tracking workflow
        workflow = panel("02 视频与追踪设置", "建议先固定手机，再选择视频和完成标定；正侧面优先使用单端标记。")
        self.marker_mode_spinner = Spinner(text=TRACKING_MODES[0], values=TRACKING_MODES,
                                           size_hint_y=None, height=dp(44), font_size="13sp")
        self.marker_mode_spinner.bind(text=self._on_tracking_mode_change)
        workflow.add_widget(self.marker_mode_spinner)

        self.marker1_row = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(7))
        self.marker1_caption = Label(text="第一端颜色", size_hint_x=0.32, font_size="12sp",
                                     color=UI_THEME["muted"], halign="left")
        self.marker_spinner = Spinner(text=self.marker_label, values=list(COLOR_PRESETS.keys()),
                                      size_hint_x=0.68, font_size="12sp")
        self.marker1_row.add_widget(self.marker1_caption)
        self.marker1_row.add_widget(self.marker_spinner)
        workflow.add_widget(self.marker1_row)

        self.marker2_row = BoxLayout(size_hint_y=None, height=0, spacing=dp(7), opacity=0, disabled=True)
        self.marker2_caption = Label(text="第二端颜色", size_hint_x=0.32, font_size="12sp",
                                     color=UI_THEME["muted"], halign="left")
        self.marker2_spinner = Spinner(text=self.marker2_label, values=list(COLOR_PRESETS.keys()),
                                       size_hint_x=0.68, font_size="12sp")
        self.marker2_row.add_widget(self.marker2_caption)
        self.marker2_row.add_widget(self.marker2_spinner)
        workflow.add_widget(self.marker2_row)

        self.marker_mode_help = small_label(
            "单端：正侧面推荐，只贴一个清晰彩色标记。双端：斜向辅助，杠铃两端使用不同颜色；双端中点仍是二维近似，不是三维重建。",
            height=48, font_size="11sp")
        workflow.add_widget(self.marker_mode_help)

        self.select_btn = Button(text="选择训练视频", size_hint_y=None, height=dp(48),
                                 font_size="15sp", background_color=UI_THEME["blue"])
        self.select_btn.bind(on_press=self.open_filechooser)
        workflow.add_widget(self.select_btn)
        self.video_label = small_label("尚未选择视频", height=34, font_size="11sp")
        workflow.add_widget(self.video_label)

        self.calibration_status_label = small_label(self.calibration_label, height=46, font_size="11sp")
        workflow.add_widget(self.calibration_status_label)
        action_row = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(8))
        self.calibrate_btn = Button(text="拍摄角度与标定", disabled=True, font_size="13sp",
                                    background_color=UI_THEME["surface_soft"])
        self.calibrate_btn.bind(on_press=self.open_calibration)
        self.analyze_btn = Button(text="开始分析", disabled=True, font_size="14sp",
                                  background_color=UI_THEME["accent_dark"])
        self.analyze_btn.bind(on_press=self.start_analysis)
        action_row.add_widget(self.calibrate_btn)
        action_row.add_widget(self.analyze_btn)
        workflow.add_widget(action_row)

        self.camera_hint_label = small_label(
            "拍摄建议：手机固定在三脚架上，镜头尽量正对主要运动平面，高度接近杠铃活动范围中点；避免手持跟拍和中途变焦。",
            height=45, font_size="11sp")
        workflow.add_widget(self.camera_hint_label)

        self.status_label = small_label(
            "已加载中文字体。请选择训练视频并完成标定。分析结果受拍摄角度、帧率、标记识别和标定质量影响。",
            height=55, font_size="11sp", color=UI_THEME["accent"])
        workflow.add_widget(self.status_label)
        diagnostics_btn = Button(text="查看分析诊断信息", size_hint_y=None, height=dp(36),
                                 font_size="11sp", background_color=UI_THEME["surface_alt"])
        diagnostics_btn.bind(on_press=self._show_diagnostics)
        workflow.add_widget(diagnostics_btn)
        content.add_widget(workflow)

        # Summary / KPI cards
        summary_card = panel("03 本次表现概览", "结果只对本次视频和当前标定条件负责；追踪覆盖率不等于测量准确度。")
        self.metrics_grid = GridLayout(cols=2, spacing=dp(7), size_hint_y=None, height=dp(258))
        self.metric_labels = {}
        metric_specs = [
            ("avg", "平均垂直速度", "m/s"),
            ("peak", "峰值垂直速度", "m/s"),
            ("count", "连续运动片段", "段"),
            ("coverage", "追踪覆盖率", "%"),
            ("dualcoverage", "双端配对率", "%"),
            ("spacingcv", "两端间距 CV", "%"),
        ]
        for key, caption, unit in metric_specs:
            tile = Panel(orientation="vertical", padding=(dp(9), dp(7)), spacing=dp(2),
                         size_hint_y=None, height=dp(78), panel_color=UI_THEME["surface_alt"])
            tile.add_widget(Label(text=caption, font_size="10sp", color=UI_THEME["muted"],
                                  halign="left", size_hint_y=None, height=dp(20)))
            value = Label(text="—", font_size="19sp", bold=True, color=UI_THEME["text"],
                          halign="left", size_hint_y=None, height=dp(30))
            tile.add_widget(value)
            tile.add_widget(Label(text=unit, font_size="9sp", color=UI_THEME["muted"],
                                  halign="left", size_hint_y=None, height=dp(13)))
            self.metric_labels[key] = value
            self.metrics_grid.add_widget(tile)
        summary_card.add_widget(self.metrics_grid)
        self.loss_label = small_label("速度变化：等待分析", height=42, font_size="12sp", color=UI_THEME["amber"])
        summary_card.add_widget(self.loss_label)
        self.advice_label = small_label(
            "完成分析后，这里会显示追踪质量、速度变化和拍摄风险提示。", height=90, font_size="11sp")
        summary_card.add_widget(self.advice_label)
        content.add_widget(summary_card)

        # Segment velocity panel: normalized vertical travel thirds, not technical-phase recognition.
        segment_card = panel("04 行程分区速度", "新增：把每个连续运动片段按垂直行程分为起始、中段、末端三段，分别计算平均垂直速度。")
        segment_card.add_widget(small_label(
            "注意：这是按位移范围均分的行程分区，不等同于高翻第一拉/第二拉，也不等同于精确的解剖学或技术阶段。",
            height=44, font_size="10sp", color=UI_THEME["amber"]))
        self.result_layout = BoxLayout(orientation="vertical", spacing=dp(8), size_hint_y=None)
        self.result_layout.bind(minimum_height=self.result_layout.setter("height"))
        segment_card.add_widget(self.result_layout)
        content.add_widget(segment_card)

        # Trajectory preview
        preview_card = panel("05 杠铃轨迹回放", "在原视频上查看识别轨迹，拖动进度条可定位到不同时间。")
        self.preview_help = small_label(
            "单端：绿色线显示标记点轨迹。双端：蓝线/橙线显示两端，绿线显示两端图像中点；杆轴连线仅为画面投影。",
            height=43, font_size="10sp")
        preview_card.add_widget(self.preview_help)
        self.preview_image = KivyImage(size_hint_y=None, height=dp(225), fit_mode="contain")
        preview_card.add_widget(self.preview_image)
        self.preview_status = small_label("分析完成后，这里会显示叠加轨迹的原视频。", height=38, font_size="10sp")
        preview_card.add_widget(self.preview_status)
        self.preview_slider = Slider(min=0, max=1, value=0, size_hint_y=None, height=dp(30), disabled=True)
        self.preview_slider.bind(value=self._on_preview_seek)
        preview_card.add_widget(self.preview_slider)
        self.preview_time_label = Label(text="00:00.00 / 00:00.00", size_hint_y=None, height=dp(20),
                                        font_size="10sp", color=UI_THEME["muted"])
        preview_card.add_widget(self.preview_time_label)
        preview_buttons = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(7))
        self.preview_play_btn = Button(text="播放", disabled=True, font_size="12sp", background_color=UI_THEME["accent_dark"])
        self.preview_pause_btn = Button(text="暂停", disabled=True, font_size="12sp", background_color=UI_THEME["surface_soft"])
        self.preview_restart_btn = Button(text="从头播放", disabled=True, font_size="12sp", background_color=UI_THEME["surface_soft"])
        self.preview_play_btn.bind(on_press=self._play_preview)
        self.preview_pause_btn.bind(on_press=self._pause_preview)
        self.preview_restart_btn.bind(on_press=self._restart_preview)
        preview_buttons.add_widget(self.preview_play_btn)
        preview_buttons.add_widget(self.preview_pause_btn)
        preview_buttons.add_widget(self.preview_restart_btn)
        preview_card.add_widget(preview_buttons)
        content.add_widget(preview_card)

        # Exercise guide and interpretation
        guide_card = panel("06 动作阶段参考", "按所选动作切换观察重点；当前版本不会自动判定这些技术阶段。")
        self.guide_label = Label(text="", size_hint_y=None, height=dp(220), font_size="11sp",
                                 halign="left", valign="top", color=UI_THEME["text"])
        self._wrap_label(self.guide_label)
        self.guide_label.bind(texture_size=lambda obj, size: setattr(obj, "height", max(dp(170), size[1] + dp(14))))
        guide_card.add_widget(self.guide_label)
        content.add_widget(guide_card)
        self.exercise_spinner.bind(text=self._on_exercise_change)
        self._on_exercise_change(self.exercise_spinner, self.exercise_spinner.text)

        guide_metrics_card = panel("07 指标与测量边界")
        self.metric_guide_label = Label(
            text=(
                "平均垂直速度：该连续运动片段内，标记点垂直速度绝对值的平均值。\n"
                "峰值垂直速度：该片段内平滑后垂直速度的最高值。\n"
                "行程分区速度：沿本片段运动方向，把起点到终点的垂直行程均分为起始、中段、末端三段，分别求该区域的平均垂直速度；不是技术阶段自动识别。\n"
                "速度损失：当前同方向片段相对本视频最快同方向片段的平均速度下降，不等于生理疲劳百分比。\n"
                "垂直位移：标定平面内标记点最高与最低位置之间的距离。水平漂移不是三维深度位移。\n"
                "识别覆盖率反映追踪连续性，不等于速度准确度。双端中点是二维近似，不是三维重建。"
            ),
            size_hint_y=None, height=dp(155), font_size="10sp", halign="left", valign="top", color=UI_THEME["muted"])
        self._wrap_label(self.metric_guide_label)
        self.metric_guide_label.bind(texture_size=lambda obj, size: setattr(obj, "height", max(dp(145), size[1] + dp(12))))
        guide_metrics_card.add_widget(self.metric_guide_label)
        self.camera_result_label = small_label(
            "拍摄建议：正侧面优先；两点标定适合正侧面比例标定；四点标定只校正与标记运动同一平面内的透视，无法恢复前后深度位移。",
            height=65, font_size="10sp")
        guide_metrics_card.add_widget(self.camera_result_label)
        content.add_widget(guide_metrics_card)

        # Save / export / history
        data_card = panel("08 保存与导出", "训练记录保存在本机；建议定期导出 CSV 并备份。")
        row3 = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(8))
        self.save_btn = Button(text="保存训练记录", disabled=True, font_size="12sp",
                               background_color=UI_THEME["blue"])
        self.save_btn.bind(on_press=self.save_session)
        self.export_btn = Button(text="导出 CSV", disabled=True, font_size="12sp",
                                 background_color=UI_THEME["accent_dark"])
        self.export_btn.bind(on_press=self.export_results)
        row3.add_widget(self.save_btn)
        row3.add_widget(self.export_btn)
        data_card.add_widget(row3)
        history_row = BoxLayout(size_hint_y=None, height=dp(42), spacing=dp(8))
        history_btn = Button(text="查看本机历史记录", font_size="11sp", background_color=UI_THEME["surface_soft"])
        history_btn.bind(on_press=self.show_history)
        clear_btn = Button(text="清空当前结果", font_size="11sp", background_color=UI_THEME["red"])
        clear_btn.bind(on_press=self.clear_results)
        history_row.add_widget(history_btn)
        history_row.add_widget(clear_btn)
        data_card.add_widget(history_row)
        content.add_widget(data_card)

        footer = Label(text="BARBELLVBT · TRAINING DATA, BETTER DECISIONS", size_hint_y=None,
                       height=dp(26), font_size="9sp", color=UI_THEME["muted"], halign="center")
        content.add_widget(footer)

        self.scroll.add_widget(content)
        root.add_widget(self.scroll)
        return root

    @staticmethod
    def _wrap_label(label):
        label.bind(size=lambda obj, size: setattr(obj, "text_size", (max(size[0] - dp(8), 0), None)))
        return label

    @staticmethod
    def _section_label(text):
        return Label(text=text, bold=True, halign="left", size_hint_y=None, height=dp(30), font_size="15sp")

    @staticmethod
    def _input(hint, height=42, input_filter=None):
        field = TextInput(hint_text=hint, multiline=False, size_hint_y=None, height=dp(height))
        if input_filter:
            field.input_filter = input_filter
        return field

    def on_start(self):
        """Android 使用系统文件选择器，不再直接遍历 /storage/emulated/0。"""
        if kivy_platform == "android":
            try:
                from android import activity
                activity.bind(on_activity_result=self._on_android_activity_result)
                self._android_activity_module = activity
                self._write_diagnostic("APP_START Android SAF document picker registered")
            except Exception as exc:
                self.status_label.text = f"系统视频选择器初始化失败：{type(exc).__name__}: {exc}"
                self._write_diagnostic(f"ANDROID_ACTIVITY_BIND_ERROR {type(exc).__name__}: {exc}")

    def on_pause(self):
        # Android 打开系统文件选择器时会暂停当前 Activity；允许应用正常恢复。
        return True

    def open_filechooser(self, _instance):
        if kivy_platform == "android":
            try:
                from jnius import autoclass
                Intent = autoclass("android.content.Intent")
                PythonActivity = autoclass("org.kivy.android.PythonActivity")
                intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
                intent.addCategory(Intent.CATEGORY_OPENABLE)
                intent.setType("video/*")
                intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                intent.addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
                PythonActivity.mActivity.startActivityForResult(intent, self.ANDROID_VIDEO_REQUEST_CODE)
                self.status_label.text = "已打开系统视频选择器。请选择手机中的原始视频文件。"
                self._write_diagnostic("OPEN_DOCUMENT launched type=video/*")
            except Exception as exc:
                self.status_label.text = f"无法打开系统视频选择器：{type(exc).__name__}: {exc}"
                self._write_diagnostic(f"OPEN_DOCUMENT_LAUNCH_ERROR {type(exc).__name__}: {exc}")
            return

        # 电脑测试时保留 Kivy 文件选择器；Android 不使用这个直接路径浏览器。
        start_path = os.path.expanduser("~")
        chooser = FileChooserListView(
            path=start_path,
            filters=["*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV", "*.mkv", "*.MKV", "*.3gp", "*.3GP"],
            font_name=UI_FONT_NAME,
        )
        buttons = BoxLayout(size_hint_y=0.13, spacing=dp(6))
        ok_btn = Button(text="选中这个视频")
        cancel_btn = Button(text="取消")
        buttons.add_widget(ok_btn)
        buttons.add_widget(cancel_btn)
        box = BoxLayout(orientation="vertical", spacing=dp(6))
        box.add_widget(chooser)
        box.add_widget(buttons)
        popup = Popup(title="选择训练视频", content=box, size_hint=(0.95, 0.92))

        def confirm(_btn):
            if chooser.selection:
                self._set_selected_video(chooser.selection[0])
            popup.dismiss()

        ok_btn.bind(on_press=confirm)
        cancel_btn.bind(on_press=popup.dismiss)
        popup.open()

    def _on_android_activity_result(self, request_code, result_code, intent):
        """Receive the SAF URI, then copy it into app-private storage before OpenCV reads it."""
        try:
            if int(request_code) != self.ANDROID_VIDEO_REQUEST_CODE:
                return
            from jnius import autoclass
            Activity = autoclass("android.app.Activity")
            if int(result_code) != int(Activity.RESULT_OK) or intent is None:
                Clock.schedule_once(lambda _dt: setattr(self.status_label, "text", "已取消视频选择。"), 0)
                return
            uri = intent.getData()
            if uri is None:
                Clock.schedule_once(lambda _dt: setattr(self.status_label, "text", "没有获取到视频文件，请重新选择。"), 0)
                return
            uri_string = str(uri.toString())
            Clock.schedule_once(lambda _dt, value=uri_string: self._begin_video_import(value), 0)
        except Exception as exc:
            self._write_diagnostic(f"ACTIVITY_RESULT_ERROR {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            Clock.schedule_once(lambda _dt, msg=f"读取视频选择结果失败：{type(exc).__name__}: {exc}": setattr(self.status_label, "text", msg), 0)

    def _begin_video_import(self, uri_string):
        self.select_btn.disabled = True
        self.analyze_btn.disabled = True
        self.calibrate_btn.disabled = True
        self.status_label.text = "正在把视频安全导入应用目录，请不要关闭 APP……"
        threading.Thread(target=self._copy_video_uri_worker, args=(uri_string,), daemon=True).start()

    def _copy_video_uri_worker(self, uri_string):
        try:
            destination = self._copy_android_uri_to_private_file(uri_string)
            size_mb = os.path.getsize(destination) / (1024.0 * 1024.0)
            self._write_diagnostic(f"VIDEO_IMPORT_OK path={destination} size_mb={size_mb:.2f}")
            Clock.schedule_once(lambda _dt, p=destination: self._set_selected_video(p), 0)
        except Exception as exc:
            self._write_diagnostic(f"VIDEO_IMPORT_ERROR {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            message = f"视频导入失败：{type(exc).__name__}: {exc}。请尝试先把视频保存到手机本地‘下载’文件夹，再重新选择。"
            Clock.schedule_once(lambda _dt, msg=message: self._video_import_failed(msg), 0)

    def _copy_android_uri_to_private_file(self, uri_string):
        """Copy a content:// document to a normal seekable file for OpenCV/VideoCapture."""
        from jnius import autoclass
        PythonActivity = autoclass("org.kivy.android.PythonActivity")
        Uri = autoclass("android.net.Uri")
        activity_obj = PythonActivity.mActivity
        resolver = activity_obj.getContentResolver()
        uri = Uri.parse(uri_string)
        input_stream = resolver.openInputStream(uri)
        if input_stream is None:
            raise IOError("Android 文件提供器没有返回可读取的视频流")

        mime = str(resolver.getType(uri) or "video/mp4").lower()
        extension_map = {
            "video/mp4": ".mp4",
            "video/quicktime": ".mov",
            "video/x-matroska": ".mkv",
            "video/3gpp": ".3gp",
            "video/webm": ".webm",
            "video/avi": ".avi",
            "video/x-msvideo": ".avi",
        }
        extension = extension_map.get(mime, ".mp4")
        import_dir = os.path.join(self.user_data_dir, "imports")
        os.makedirs(import_dir, exist_ok=True)
        destination = os.path.join(import_dir, f"video_{int(time.time())}_{uuid.uuid4().hex[:8]}{extension}")
        # PyJNIus 会把 Python bytearray 转换为 Java byte[]，无需导入 jarray。
        buffer = bytearray(65536)
        total = 0
        try:
            with open(destination, "wb") as output:
                while True:
                    count = int(input_stream.read(buffer))
                    if count < 0:
                        break
                    if count == 0:
                        continue
                    output.write(buffer[:count])
                    total += count
            if total < 1024:
                raise IOError("选择的文件为空或不是有效视频文件")
            return destination
        except Exception:
            try:
                if os.path.exists(destination):
                    os.remove(destination)
            except Exception:
                pass
            raise
        finally:
            try:
                input_stream.close()
            except Exception:
                pass

    def _video_import_failed(self, message):
        self.select_btn.disabled = False
        self.calibrate_btn.disabled = self.selected_video is None
        self.analyze_btn.disabled = True if self.selected_video is None or self.calibration is None else False
        self.status_label.text = message

    def _set_selected_video(self, video_path):
        if not video_path or not os.path.isfile(video_path):
            self._video_import_failed("无法访问所选视频文件。请重新选择手机本地保存的视频。")
            return
        try:
            file_size = os.path.getsize(video_path)
        except OSError as exc:
            self._video_import_failed(f"无法读取视频文件信息：{exc}")
            return
        if file_size <= 0:
            self._video_import_failed("所选视频文件大小为 0，请重新选择。")
            return
        self.clear_results(None)
        self.selected_video = os.path.abspath(video_path)
        self.video_label.text = f"已选：{os.path.basename(self.selected_video)}（{file_size / (1024*1024):.1f} MB）"
        self.calibration = None
        self.camera_setup = {}
        self.calibration_label = "新视频尚未标定：请点击‘拍摄角度与标定’。"
        self.calibration_status_label.text = self.calibration_label
        self.camera_result_label.text = (
            "角度信息只用于质量提示，不会把用户估计的角度直接当作数学修正。\n"
            "四点标定只校正指定平面内的透视；标记离开该平面产生的深度位移无法用单目视频恢复。\n"
            "如果镜头明显斜拍，优先重新摆位。"
        )
        self.calibrate_btn.disabled = False
        self.analyze_btn.disabled = True
        self.select_btn.disabled = False
        self.status_label.text = "视频已安全导入应用目录。下一步请完成标定。"
        self._write_diagnostic(f"VIDEO_SELECTED path={self.selected_video} size_bytes={file_size}")

    def open_calibration(self, _instance):
        """Interactive two-point scale or four-point planar perspective calibration."""
        if not self.selected_video:
            self.status_label.text = "请先选择训练视频，再进行标定。"
            return
        cap = cv2.VideoCapture(self.selected_video)
        if not cap.isOpened():
            self.status_label.text = "无法读取视频首帧，无法进行标定。"
            return
        ok, calibration_frame = cap.read()
        cap.release()
        if not ok or calibration_frame is None:
            self.status_label.text = "无法读取视频首帧，请检查视频格式。"
            return

        points = []
        frame_h, frame_w = calibration_frame.shape[:2]
        preview_max_dim = 1280
        preview_scale = min(1.0, preview_max_dim / float(max(frame_w, frame_h)))
        if preview_scale < 1.0:
            display_w = max(1, int(round(frame_w * preview_scale)))
            display_h = max(1, int(round(frame_h * preview_scale)))
            calibration_display_frame = cv2.resize(
                calibration_frame, (display_w, display_h), interpolation=cv2.INTER_AREA
            )
        else:
            display_w, display_h = frame_w, frame_h
            calibration_display_frame = calibration_frame
        preview_scale_x = display_w / float(frame_w)
        preview_scale_y = display_h / float(frame_h)
        box = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(5))
        mode_spinner = Spinner(
            text="两点比例标定（正侧面推荐）",
            values=("两点比例标定（正侧面推荐）", "四点透视标定（进阶）"),
            size_hint_y=None, height=dp(42)
        )
        box.add_widget(mode_spinner)
        dual_selected = str(getattr(self, "marker_mode_spinner", None).text if getattr(self, "marker_mode_spinner", None) else "").startswith("双端")
        if dual_selected:
            first_instruction = (
                "当前为双端辅助模式：请把两种不同颜色标记分别贴在杠铃两端，并确保整个动作中两端都能看见。"
                "软件用两端图像坐标的中点近似杠铃中心，同时显示杆轴方向与间距稳定性。"
                "这不是三维重建，不能消除明显的前后深度视差。"
            )
        else:
            first_instruction = (
                "当前为单端模式（正侧面推荐）：只需在靠近镜头的一端贴一个清晰彩色标记。"
                "如果相机正对杠铃中心运动平面，不需要启用双端模式。"
            )
        instruction = Label(
            text=first_instruction + " 参照物必须出现在视频第一帧。",
            size_hint_y=None, height=dp(72), font_size="11sp", halign="left", valign="middle"
        )
        self._wrap_label(instruction)
        box.add_widget(instruction)

        width_row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(5))
        width_field = TextInput(text="", hint_text="已知竖直长度 cm", multiline=False, input_filter="float")
        height_field = TextInput(text="", hint_text="第二项无需填写", multiline=False, input_filter="float")
        width_row.add_widget(width_field)
        width_row.add_widget(height_field)
        box.add_widget(width_row)

        yaw_caption = Label(text="拍摄方向（大致估计）", size_hint_y=None, height=dp(22), font_size="11sp", halign="left")
        box.add_widget(yaw_caption)
        yaw_spinner = Spinner(text="大致正侧面", values=(
            "大致正侧面", "轻微斜拍（能看到部分前/后侧面）", "明显斜拍（从前方或后方看）", "正面视角/无法判断"
        ), size_hint_y=None, height=dp(38), font_size="12sp")
        box.add_widget(yaw_spinner)
        height_caption = Label(text="镜头高度", size_hint_y=None, height=dp(22), font_size="11sp", halign="left")
        box.add_widget(height_caption)
        height_spinner = Spinner(text="镜头约与活动范围中点同高", values=(
            "镜头约与活动范围中点同高", "明显高于杠铃活动范围中点，向下拍", "明显低于杠铃活动范围中点，向上拍"
        ), size_hint_y=None, height=dp(38), font_size="12sp")
        box.add_widget(height_spinner)
        level_spinner = Spinner(text="手机保持水平", values=("手机保持水平", "画面明显倾斜"), size_hint_y=None, height=dp(38), font_size="12sp")
        box.add_widget(level_spinner)
        setup_text = (
            "拍摄要点：手机固定不跟拍；镜头高度接近杠铃活动范围中点；不要变焦或移动手机。"
            + ("双端模式要求两端不同颜色、尽量同时可见；参考矩形应对齐杠铃中心的主要运动平面。双端中点仅为二维近似。"
               if dual_selected else "单端模式只追踪一个颜色标记；正侧面时无需双端标记。")
            + "角度选项只用于风险提示，不会把估计角度直接换算成速度修正。"
        )
        setup_note = Label(
            text=setup_text,
            size_hint_y=None, height=dp(82), font_size="11sp", halign="left", valign="middle"
        )
        self._wrap_label(setup_note)
        box.add_widget(setup_note)

        point_hint = Label(text="已选择 0 个点。", size_hint_y=None, height=dp(25), font_size="12sp")
        box.add_widget(point_hint)
        image_widget = CalibrationImage(
            on_point=lambda p: add_point(p), source_width=frame_w, source_height=frame_h,
            size_hint_y=1,
        )
        box.add_widget(image_widget)

        buttons = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(5))
        undo_btn = Button(text="撤销上一点")
        reset_btn = Button(text="清空点位")
        save_btn = Button(text="保存标定")
        buttons.add_widget(undo_btn)
        buttons.add_widget(reset_btn)
        buttons.add_widget(save_btn)
        box.add_widget(buttons)
        popup = Popup(title="拍摄角度与视频标定", content=box, size_hint=(0.98, 0.97))

        def draw_calibration_frame():
            canvas = calibration_display_frame.copy()
            for idx, (px, py) in enumerate(points, start=1):
                center = (int(round(px * preview_scale_x)), int(round(py * preview_scale_y)))
                marker_radius = max(6, int(min(display_w, display_h) * 0.018))
                cv2.circle(canvas, center, marker_radius, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, center, marker_radius, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(canvas, str(idx), (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(canvas, str(idx), (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            texture = Texture.create(size=(display_w, display_h), colorfmt="rgb")
            texture.flip_vertical()
            texture.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
            image_widget.texture = texture
            if mode_spinner.text.startswith("四点"):
                order_text = ("左上 → 右上 → 右下 → 左下" if not points else f"下一个点：{('右上','右下','左下')[min(len(points)-1,2)]}")
                point_hint.text = f"已选择 {len(points)}/4 点。顺序：{order_text}。"
            else:
                point_hint.text = f"已选择 {len(points)}/2 点。先点已知竖直长度的上端，再点下端。"

        def add_point(point):
            required = 4 if mode_spinner.text.startswith("四点") else 2
            if len(points) >= required:
                return
            points.append((float(point[0]), float(point[1])))
            draw_calibration_frame()

        def update_mode(*_args):
            points.clear()
            if mode_spinner.text.startswith("四点"):
                instruction.text = (
                    "进阶四点透视标定：仅当能在与标记实际运动相同的平面内放置已知矩形时使用。"
                    "按左上、右上、右下、左下点四角并输入实际宽、高（cm）。标记运动最好落在矩形范围内。"
                    "它只校正该平面内的透视，不能恢复离开该平面的前后深度运动。"
                )
                width_field.hint_text = "参考宽度 cm"
                height_field.hint_text = "参考高度 cm"
                width_field.text, height_field.text = "", ""
            else:
                instruction.text = (
                    "两点快速标定（正侧面推荐）：在与杠铃标记同一平面/深度内，先点已知竖直长度的上端，再点下端。"
                    "例如使用可读数的竖直标尺。此方法校正比例和简单画面旋转，不校正透视；必须输入真实测量长度。"
                )
                width_field.hint_text = "已知竖直长度 cm"
                height_field.hint_text = "第二项无需填写"
                width_field.text, height_field.text = "", ""
            draw_calibration_frame()

        def undo_point(_btn):
            if points:
                points.pop()
            draw_calibration_frame()

        def reset_points(_btn):
            points.clear()
            draw_calibration_frame()

        def save_calibration(_btn):
            try:
                if mode_spinner.text.startswith("四点"):
                    matrix, meta = build_four_point_homography(points, width_field.text, height_field.text)
                else:
                    matrix, meta = build_two_point_homography(points, width_field.text)
                self.calibration = {**meta, "homography": matrix.tolist(), "frame_width": frame_w, "frame_height": frame_h}
                self.camera_setup = {
                    "horizontal_angle": yaw_spinner.text,
                    "camera_height": height_spinner.text,
                    "level": level_spinner.text,
                    "tracking_mode": "dual" if self.marker_mode_spinner.text.startswith("双端") else "single",
                    "marker1_color": self.marker_spinner.text,
                    "marker2_color": self.marker2_spinner.text if self.marker_mode_spinner.text.startswith("双端") else "",
                }
                self.analyze_btn.disabled = False
                self.calibration_label = f"已完成{'四点透视' if meta['mode'] == 'four_point_plane' else '两点比例'}标定；可以开始分析。"
                self.calibration_status_label.text = self.calibration_label
                warnings = camera_setup_warnings(self.camera_setup, self.calibration)
                if warnings:
                    self.camera_result_label.text = "拍摄/标定提示：\n" + "\n".join("• " + item for item in warnings)
                self.status_label.text = "标定已保存。请确认参考矩形/线段与标记运动在同一平面。"
                popup.dismiss()
            except Exception as exc:
                self.status_label.text = f"标定未保存：{exc}"
                point_hint.text = str(exc)

        mode_spinner.bind(text=update_mode)
        undo_btn.bind(on_press=undo_point)
        reset_btn.bind(on_press=reset_points)
        save_btn.bind(on_press=save_calibration)
        draw_calibration_frame()
        popup.open()

    def _on_tracking_mode_change(self, _spinner, mode):
        dual = str(mode).startswith("双端")
        self.tracking_mode = "dual" if dual else "single"
        self.marker2_row.height = dp(42) if dual else 0
        self.marker2_row.opacity = 1 if dual else 0
        self.marker2_row.disabled = not dual
        if dual:
            self.marker1_caption.text = "第一端颜色"
            self.marker_mode_help.text = (
                "斜向拍摄可选双端辅助：在杠铃左右两端分别贴不同颜色标记，画面中都要清晰可见。"
                "软件用两端图像中点近似杠铃中心，并检查两端间距稳定性；不能恢复三维深度，也不能消除明显视差。"
            )
        else:
            self.marker1_caption.text = "追踪标记颜色"
            self.marker_mode_help.text = (
                "正侧面拍摄推荐单端标记：在靠近镜头的一侧选择一个清晰标记即可。"
                "不需要贴两端标记，流程更简单；请尽量让相机正对杠铃中心运动平面。"
            )

    def start_analysis(self, _instance):
        if not self.selected_video:
            self.status_label.text = "请先选择训练视频。"
            return
        if not os.path.isfile(self.selected_video) or not os.access(self.selected_video, os.R_OK):
            self.status_label.text = "视频文件当前不可读取。请重新选择视频；Android 版本会先把视频复制到应用目录。"
            self._write_diagnostic(f"PREFLIGHT_FILE_NOT_READABLE path={self.selected_video}")
            return
        try:
            selected_size = os.path.getsize(self.selected_video)
            if selected_size < 1024:
                self.status_label.text = "视频文件过小或已损坏，请重新选择。"
                self._write_diagnostic(f"PREFLIGHT_FILE_TOO_SMALL size_bytes={selected_size}")
                return
        except Exception as exc:
            self.status_label.text = f"读取视频文件信息失败：{exc}"
            self._write_diagnostic(f"PREFLIGHT_FILE_INFO_ERROR {type(exc).__name__}: {exc}")
            return
        if not self.calibration or "homography" not in self.calibration:
            self.status_label.text = "请先完成视频标定。没有标定值时，速度和位移单位不可靠。"
            return
        load = safe_float(self.load_input.text, None)
        if self.load_input.text.strip() and (load is None or load < 0):
            self.status_label.text = "负荷请输入大于或等于 0 的数字，或留空。"
            return

        marker_label = self.marker_spinner.text
        marker_color = COLOR_PRESETS.get(marker_label, "red")
        dual_mode = self.marker_mode_spinner.text.startswith("双端")
        marker2_label = self.marker2_spinner.text
        marker2_color = COLOR_PRESETS.get(marker2_label, "blue") if dual_mode else None
        if dual_mode and marker2_color == marker_color:
            self.status_label.text = "双端模式必须选择两种不同颜色。请给杠铃两端贴不同颜色标记。"
            return
        tracking_mode = "dual" if dual_mode else "single"
        self.analyzer = VideoAnalyzer(self.calibration, marker_color, tracking_mode, marker2_color or "blue")
        self.marker_label = marker_label
        self.marker2_label = marker2_label
        self.tracking_mode = tracking_mode
        self.results = []
        self.summary = {}
        self._clear_preview("正在分析。完成后可回放叠加完整轨迹的原视频。")
        self.result_layout.clear_widgets()
        for value in self.metric_labels.values():
            value.text = "-"
        self.loss_label.text = "速度变化：分析中..."
        self.advice_label.text = "正在分析视频，请稍等。"
        self.save_btn.disabled = True
        self.export_btn.disabled = True
        self.analyze_btn.disabled = True
        self.select_btn.disabled = True
        self.calibrate_btn.disabled = True
        self.status_label.text = "正在逐帧分析视频……"
        self.session = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "athlete": self.athlete_input.text.strip(),
            "exercise": self.exercise_spinner.text,
            "load_kg": load,
            "marker_color": marker_label,
            "marker2_color": marker2_label if dual_mode else "不适用",
            "tracking_mode": tracking_mode,
            "calibration_mode": self.calibration.get("mode", ""),
            "calibration_meta": {k: v for k, v in self.calibration.items() if k != "homography"},
            "camera_horizontal_angle": self.camera_setup.get("horizontal_angle", ""),
            "camera_height": self.camera_setup.get("camera_height", ""),
            "camera_level": self.camera_setup.get("level", ""),
            "video_name": os.path.basename(self.selected_video),
            "notes": self.notes_input.text.strip(),
        }
        threading.Thread(target=self._worker, args=(self.selected_video,), daemon=True).start()

    def _worker(self, path):
        def callback(data):
            if "progress" in data:
                p = data["progress"]
                Clock.schedule_once(lambda _dt, value=p: self._set_progress(value))
            if data.get("stage") == "video_opened":
                self._write_diagnostic(
                    f"VIDEO_OPENED fps={data.get('fps')} resolution={data.get('width')}x{data.get('height')} "
                    f"total_frames={data.get('total_frames')}"
                )
            elif data.get("stage") == "frames":
                self._write_diagnostic(
                    f"ANALYSIS_FRAMES frame={data.get('frame_count')} total={data.get('total_frames')}"
                )
        try:
            file_size = os.path.getsize(path) if os.path.isfile(path) else -1
            self._write_diagnostic(f"ANALYSIS_BEGIN path={path} size_bytes={file_size} tracking_mode={self.tracking_mode}")
            result = self.analyzer.analyze(path, callback, camera_setup=self.camera_setup)
            if isinstance(result, dict) and "error" in result:
                self._write_diagnostic(f"ANALYSIS_RETURNED_ERROR {result['error']}")
            else:
                self._write_diagnostic(
                    f"ANALYSIS_SUCCESS frames={result.get('video_info', {}).get('frame_count')} "
                    f"segments={len(result.get('results', []))}"
                )
        except Exception as exc:
            self._write_diagnostic(f"ANALYSIS_EXCEPTION {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            result = {"error": f"分析异常：{type(exc).__name__}: {exc}。可点击‘查看分析诊断信息’查看详细记录。"}
        Clock.schedule_once(lambda _dt, value=result: self._finish_analysis(value))

    def _set_progress(self, progress):
        self.status_label.text = f"正在分析视频……{progress:.0f}%"

    def _finish_analysis(self, result):
        self.analyze_btn.disabled = False
        self.calibrate_btn.disabled = False
        self.select_btn.disabled = False
        if isinstance(result, dict) and "error" in result:
            self.status_label.text = result["error"]
            self.loss_label.text = "同方向速度比较：没有可用结果"
            self.advice_label.text = "请检查视频、颜色标记、光线和标定值后重试。"
            self._clear_preview("本次没有得到有效轨迹，暂时无法回放。")
            return
        self.results = result["results"]
        self.summary = result["summary"]
        self.trajectory = result.get("trajectory", [])
        try:
            self._load_trajectory_preview(self.selected_video, result)
        except Exception as exc:
            self._write_diagnostic(f"PREVIEW_INIT_ERROR {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            self._clear_preview("分析已完成，但预览初始化失败。数据结果仍可查看；请稍后重试回放。")
        self.metric_labels["avg"].text = f"{self.summary['mean_velocity']:.3f} m/s"
        self.metric_labels["peak"].text = f"{self.summary['peak_velocity']:.3f} m/s"
        self.metric_labels["count"].text = str(self.summary["segment_count"])
        coverage = self.summary["tracking_coverage_pct"]
        self.metric_labels["coverage"].text = f"{coverage:.1f}%"
        if self.summary.get("tracking_mode") == "dual":
            self.metric_labels["dualcoverage"].text = f"{self.summary.get('dual_pair_coverage_pct', 0):.1f}%"
            cv_value = self.summary.get("endpoint_distance_cv_pct")
            self.metric_labels["spacingcv"].text = f"{cv_value:.1f}%" if cv_value is not None else "无数据"
        else:
            self.metric_labels["dualcoverage"].text = "单端模式"
            self.metric_labels["spacingcv"].text = "不适用"

        if self.summary["last_velocity_loss_pct"] is None:
            self.loss_label.text = "同方向速度比较：可比较片段不足 2 段"
        else:
            self.loss_label.text = f"最后同方向片段较最快同方向片段平均垂直速度低：{self.summary['last_velocity_loss_pct']:.1f}%"

        advice = self._make_advice(coverage, self.summary["last_velocity_loss_pct"])
        advice += f"\n追踪质量：{self.summary.get('tracking_quality', '未知')}；最大连续漏检：{self.summary.get('max_missing_seconds', 0):.3f} s。"
        region_pct = self.summary.get("calibration_region_coverage_pct")
        if region_pct is not None:
            advice += f"\n四点标定矩形内识别点比例：{region_pct:.1f}%。"
        fps_value = self.summary.get("fps", 0)
        advice += f"\n视频报告帧率：{fps_value:.1f} fps。时间按文件帧率估算；若视频经过可变帧率剪辑/转码或慢动作导出，速度可能有额外误差。"
        advice += f"\n原视频分辨率：{self.summary.get('video_width', 0)}×{self.summary.get('video_height', 0)}；追踪检测画面最长边限制为 {self.summary.get('analysis_max_dimension', 1280)} px，识别坐标已映射回原视频坐标。"
        if fps_value and fps_value < 50:
            advice += " 建议保留手机原始视频；若设备支持，可使用原始 60 fps 或更高帧率拍摄并避免社交平台二次压缩。"
        if self.summary.get("tracking_mode") == "dual":
            advice += (
                f"\n双端辅助模式：第一端颜色 {self.marker_label}，第二端颜色 {self.marker2_label}。"
                f"第一端识别率 {self.summary.get('marker1_coverage_pct', 0):.1f}%，"
                f"第二端识别率 {self.summary.get('marker2_coverage_pct', 0):.1f}%，"
                f"双端同时配对率 {self.summary.get('dual_pair_coverage_pct', 0):.1f}%。"
            )
            if self.summary.get("endpoint_distance_median_px") is not None:
                advice += (
                    f" 两端间距中位数 {self.summary['endpoint_distance_median_px']:.1f} px，"
                    f"间距变异系数 {self.summary.get('endpoint_distance_cv_pct', 0):.1f}%，"
                    f"投影杆轴角中位数 {self.summary.get('bar_axis_angle_median_deg', 0):.1f}°。"
                )
            advice += " 双端中点只是图像平面近似，不能解释为三维质心或完整透视补偿。"
        if self.summary.get("camera_warnings"):
            advice += "\n" + "\n".join(self.summary["camera_warnings"])
        self.advice_label.text = advice
        self.result_layout.clear_widgets()
        for item in self.results:
            row = Panel(orientation="vertical", size_hint_y=None, spacing=dp(7),
                        padding=(dp(10), dp(9)), panel_color=UI_THEME["surface_alt"])
            row.bind(minimum_height=row.setter("height"))
            velocity_loss = (f"速度差 {item['velocity_loss_pct']:.1f}%"
                             if item.get("velocity_loss_pct") is not None else "速度差：暂不可比较")
            title_text = f"运动片段 {item['rep']} · {item['direction']}"
            title_label = Label(text=title_text, size_hint_y=None, height=dp(24), font_size="12sp",
                                bold=True, halign="left", valign="middle", color=UI_THEME["text"])
            self._wrap_label(title_label)
            row.add_widget(title_label)
            speed_text = (f"平均速度 {item['mean_velocity']:.3f} m/s  ·  "
                          f"峰值速度 {item['peak_velocity']:.3f} m/s")
            speed_label = Label(text=speed_text, size_hint_y=None, height=dp(24), font_size="11sp",
                                halign="left", valign="middle", color=UI_THEME["accent"])
            self._wrap_label(speed_label)
            row.add_widget(speed_label)
            detail_text = (f"垂直位移 {item['displacement']:.3f} m  ·  水平漂移 {item['horizontal_drift']:.3f} m  ·  "
                           f"时长 {item['duration']:.2f} s  ·  {velocity_loss}")
            detail_label = Label(text=detail_text, size_hint_y=None, height=dp(35), font_size="10sp",
                                 halign="left", valign="middle", color=UI_THEME["muted"])
            self._wrap_label(detail_label)
            row.add_widget(detail_label)

            zone_values = item.get("range_zone_speeds_m_s", {})
            zone_grid = GridLayout(cols=3, spacing=dp(6), size_hint_y=None, height=dp(58))
            for zone_key, zone_title in (("lower", "起始 1/3"), ("middle", "中段 1/3"), ("upper", "末端 1/3")):
                zone_tile = Panel(orientation="vertical", spacing=dp(1), padding=(dp(5), dp(4)),
                                  size_hint_y=None, height=dp(56), panel_color=UI_THEME["surface"])
                zone_tile.add_widget(Label(text=zone_title, font_size="9sp", color=UI_THEME["muted"],
                                           size_hint_y=None, height=dp(19), halign="center"))
                zone_value = zone_values.get(zone_key)
                zone_text = f"{zone_value:.3f}" if zone_value is not None else "—"
                zone_tile.add_widget(Label(text=zone_text, font_size="14sp", bold=True,
                                           color=UI_THEME["accent"] if zone_value is not None else UI_THEME["muted"],
                                           size_hint_y=None, height=dp(24), halign="center"))
                zone_grid.add_widget(zone_tile)
            row.add_widget(zone_grid)
            if all(v is None for v in zone_values.values()):
                note = Label(text="该片段方向不够单一或有效采样点不足，暂不计算行程分区速度。",
                             size_hint_y=None, height=dp(27), font_size="9sp", color=UI_THEME["amber"],
                             halign="left", valign="middle")
                self._wrap_label(note)
                row.add_widget(note)
            self.result_layout.add_widget(row)
        self.status_label.text = f"分析完成：得到 {len(self.results)} 个连续运动片段。已计算起始/中段/末端三段行程速度；它们不是自动识别的技术阶段，请结合轨迹视频核对。"
        self.save_btn.disabled = False
        self.export_btn.disabled = False

    @staticmethod
    def _make_advice(coverage, loss):
        if coverage < 45:
            return "识别覆盖率偏低：可能有遮挡、标记颜色不明显或光线不足。本次速度结果可信度较低，请重新拍摄。"
        if coverage < 75:
            base = "识别覆盖率一般，请检查是否有漏检或误识别。"
        else:
            base = "标记识别覆盖率较高，但仍需人工确认追踪点是否始终跟随杠铃。"
        if loss is None:
            return base + " 同方向可比较片段不足两个，暂不计算速度差。"
        if loss < 10:
            return base + " 最后动作段相对最快段的速度下降较小。"
        if loss < 20:
            return base + " 片段间速度有一定下降，建议回看视频确认动作质量。"
        return base + " 片段间速度下降较明显，建议回看视频确认分段和标记识别，不要仅凭这个数值决定训练负荷。"

    def _on_exercise_change(self, _spinner, exercise):
        """根据选择的动作显示阶段名称、含义和回看重点。"""
        phases = EXERCISE_GUIDES.get(exercise, EXERCISE_GUIDES["其他"])
        lines = [f"当前动作：{exercise}", ""]
        for name, description in phases:
            lines.append(name)
            lines.append(description)
            lines.append("")
        if getattr(self, "tracking_mode", "single") == "dual":
            tracking_note = "当前使用双端辅助模式：追踪两端不同颜色标记并用图像中点近似杠铃中心。"
        else:
            tracking_note = "当前使用单端标记模式：追踪一个彩色标记点。"
        lines.append("说明：以上是动作教学/观察提示。" + tracking_note + "软件只切分连续运动片段，不会自动判断第一拉、第二拉等具体技术阶段；请结合轨迹视频人工核对。双端模式不等于三维重建。")
        self.guide_label.text = "\n".join(lines)

    def _clear_preview(self, message="请先完成视频分析。这里会显示叠加轨迹后的原视频。"):
        self._pause_preview(None)
        if self._preview_cap is not None:
            try:
                self._preview_cap.release()
            except Exception:
                pass
            self._preview_cap = None
        self.preview_image.texture = None
        self.trajectory = []
        self._preview_frame_numbers = np.asarray([], dtype=np.int32)
        self._preview_coords = np.empty((0, 2), dtype=np.int32)
        self._preview_marker1_coords = np.empty((0, 2), dtype=np.int32)
        self._preview_marker2_coords = np.empty((0, 2), dtype=np.int32)
        self._preview_marker1_segments = []
        self._preview_marker2_segments = []
        self._preview_is_dual = False
        self._preview_display_segments = []
        self._preview_total_frames = 0
        self.preview_slider.disabled = True
        self.preview_slider.max = 1
        self.preview_slider.value = 0
        self.preview_play_btn.disabled = True
        self.preview_pause_btn.disabled = True
        self.preview_restart_btn.disabled = True
        self.preview_status.text = message
        self.preview_time_label.text = "00:00.00 / 00:00.00"

    def _load_trajectory_preview(self, video_path, result):
        """重新打开原视频，在回放画面上叠加整段轨迹和当前标记位置。"""
        self._clear_preview("正在准备轨迹回放……")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            self.preview_status.text = "无法重新打开原视频，轨迹数据已计算，但暂时不能回放。"
            return

        self._preview_cap = cap
        self.trajectory = result.get("trajectory", [])
        self._preview_fps = safe_float(cap.get(cv2.CAP_PROP_FPS), 30.0) or 30.0
        self._preview_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self._preview_total_frames <= 0:
            self._preview_total_frames = int(result.get("video_info", {}).get("frame_count", 0))
        if not self.trajectory:
            self.preview_status.text = "没有保存到有效轨迹点，请检查标记颜色和视频。"
            self._clear_preview(self.preview_status.text)
            return

        self._preview_frame_numbers = np.asarray([p["frame"] for p in self.trajectory], dtype=np.int32)
        self._preview_coords = np.asarray([[p["x"], p["y"]] for p in self.trajectory], dtype=np.int32)
        self._preview_is_dual = all(p.get("marker1_x") is not None and p.get("marker2_x") is not None for p in self.trajectory)
        if self._preview_is_dual:
            self._preview_marker1_coords = np.asarray([[p["marker1_x"], p["marker1_y"]] for p in self.trajectory], dtype=np.int32)
            self._preview_marker2_coords = np.asarray([[p["marker2_x"], p["marker2_y"]] for p in self.trajectory], dtype=np.int32)
        # 按漏检间隔拆分轨迹，避免在视频中用一条直线连接两个相距很远的断点。
        gap_limit = max(1, int(round(self._preview_fps * 0.35)))
        split_at = np.where(np.diff(self._preview_frame_numbers) > gap_limit)[0] + 1
        raw_segments = np.split(self._preview_coords, split_at)
        # 轨迹显示最多抽样约 2500 个点，计算结果仍使用全部原始识别点。
        stride = max(1, int(math.ceil(len(self._preview_coords) / 2500.0)))
        self._preview_display_segments = []
        for segment in raw_segments:
            if len(segment) == 0:
                continue
            sampled = segment[::stride]
            if len(segment) > 1 and not np.array_equal(sampled[-1], segment[-1]):
                sampled = np.vstack((sampled, segment[-1]))
            self._preview_display_segments.append(sampled.astype(np.int32, copy=False))

        if self._preview_is_dual:
            for coords, target in ((self._preview_marker1_coords, self._preview_marker1_segments),
                                   (self._preview_marker2_coords, self._preview_marker2_segments)):
                endpoint_segments = np.split(coords, split_at)
                for segment in endpoint_segments:
                    if len(segment) == 0:
                        continue
                    sampled = segment[::stride]
                    if len(segment) > 1 and not np.array_equal(sampled[-1], segment[-1]):
                        sampled = np.vstack((sampled, segment[-1]))
                    target.append(sampled.astype(np.int32, copy=False))

        self.preview_slider.max = max(self._preview_total_frames - 1, 1)
        self.preview_slider.value = 0
        self.preview_slider.disabled = False
        self.preview_play_btn.disabled = False
        self.preview_pause_btn.disabled = False
        self.preview_restart_btn.disabled = False
        self._preview_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self._preview_cap.read()
        if not ok:
            self._clear_preview("视频已分析，但无法读取回放首帧。")
            return
        self._render_preview_frame(frame, 0)
        duration = (self._preview_total_frames / self._preview_fps) if self._preview_fps > 0 else 0
        if self._preview_is_dual:
            self.preview_status.text = f"双端回放：已生成 {len(self.trajectory)} 组配对点。绿线=两端图像中点轨迹，蓝线/橙线=两端各自轨迹；当前帧显示杆轴连线。可拖动进度条回看。"
        else:
            self.preview_status.text = f"单端回放：已生成 {len(self.trajectory)} 个识别点。绿线=标记点轨迹，黄点=起点，橙点=终点，红圈=当前追踪点；可拖动进度条回看。"
        self._set_preview_time(0)
        self.preview_time_label.text = f"00:00.00 / {self._format_time(duration)}"

    @staticmethod
    def _format_time(seconds):
        seconds = max(float(seconds), 0.0)
        minutes = int(seconds // 60)
        return f"{minutes:02d}:{seconds - minutes * 60:05.2f}"

    def _set_preview_time(self, frame_index):
        current = frame_index / self._preview_fps if self._preview_fps > 0 else 0
        total = self._preview_total_frames / self._preview_fps if self._preview_fps > 0 else 0
        self.preview_time_label.text = f"{self._format_time(current)} / {self._format_time(total)}"

    def _render_preview_frame(self, frame, frame_index):
        if frame is None:
            return
        # 先缩放预览画面，再绘制轨迹，降低高分辨率视频在中端手机上的 CPU 负担。
        original_h, original_w = frame.shape[:2]
        max_width = 960
        if original_w > max_width:
            display_h = max(1, int(original_h * max_width / original_w))
            canvas = cv2.resize(frame, (max_width, display_h), interpolation=cv2.INTER_AREA)
        else:
            canvas = frame
        scale_x = canvas.shape[1] / max(original_w, 1)
        scale_y = canvas.shape[0] / max(original_h, 1)
        thickness = max(2, int(round(min(canvas.shape[:2]) / 300)))

        # 整条轨迹分段绘制；不同连续识别区间之间不强行连线。
        display_segments = []
        for segment in self._preview_display_segments:
            scaled = np.rint(segment * np.asarray([scale_x, scale_y])).astype(np.int32)
            display_segments.append(scaled)
            if len(scaled) >= 2:
                polyline = scaled.reshape((-1, 1, 2))
                cv2.polylines(canvas, [polyline], False, (0, 0, 0), thickness + 3, cv2.LINE_AA)
                cv2.polylines(canvas, [polyline], False, (40, 220, 70), thickness, cv2.LINE_AA)
                # 少量方向箭头帮助判断标记移动方向。
                count = len(scaled)
                arrow_step = max(1, count // 9)
                for idx in range(0, count - 1, arrow_step):
                    p1 = tuple(int(v) for v in scaled[idx])
                    p2 = tuple(int(v) for v in scaled[min(idx + max(1, arrow_step // 2), count - 1)])
                    if p1 != p2:
                        cv2.arrowedLine(canvas, p1, p2, (255, 190, 0), max(2, thickness - 1), cv2.LINE_AA, tipLength=0.35)

        if self._preview_is_dual:
            # Endpoint-specific paths are drawn separately from the representative midpoint path.
            for source_segments, color in ((self._preview_marker1_segments, (255, 120, 30)),
                                           (self._preview_marker2_segments, (0, 150, 255))):
                for segment in source_segments:
                    scaled_endpoint = np.rint(segment * np.asarray([scale_x, scale_y])).astype(np.int32)
                    if len(scaled_endpoint) >= 2:
                        cv2.polylines(canvas, [scaled_endpoint.reshape((-1, 1, 2))], False,
                                      (0, 0, 0), thickness + 2, cv2.LINE_AA)
                        cv2.polylines(canvas, [scaled_endpoint.reshape((-1, 1, 2))], False,
                                      color, max(2, thickness - 1), cv2.LINE_AA)

        if display_segments:
            first_segment = next((seg for seg in display_segments if len(seg)), None)
            last_segment = next((seg for seg in reversed(display_segments) if len(seg)), None)
            if first_segment is not None:
                start_xy = tuple(int(v) for v in first_segment[0])
                cv2.circle(canvas, start_xy, max(6, thickness * 3), (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, "START", (max(2, start_xy[0] + 8), max(22, start_xy[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            if last_segment is not None:
                end_xy = tuple(int(v) for v in last_segment[-1])
                cv2.circle(canvas, end_xy, max(6, thickness * 3), (0, 165, 255), -1, cv2.LINE_AA)
                cv2.putText(canvas, "END", (max(2, end_xy[0] + 8), max(22, end_xy[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)

            # 当前帧位置使用最近的有效检测点；若离当前帧太远则不显示，避免误导。
            pos = int(np.searchsorted(self._preview_frame_numbers, frame_index, side="right") - 1)
            if pos >= 0:
                tracked_frame = int(self._preview_frame_numbers[pos])
                allowed_gap = max(3, int(round(self._preview_fps * 0.30)))
                if frame_index - tracked_frame <= allowed_gap:
                    x, y = (int(v) for v in self._preview_coords[pos])
                    x = int(round(x * scale_x))
                    y = int(round(y * scale_y))
                    if self._preview_is_dual:
                        p1 = self._preview_marker1_coords[pos]
                        p2 = self._preview_marker2_coords[pos]
                        p1s = (int(round(p1[0] * scale_x)), int(round(p1[1] * scale_y)))
                        p2s = (int(round(p2[0] * scale_x)), int(round(p2[1] * scale_y)))
                        cv2.line(canvas, p1s, p2s, (255, 255, 255), max(3, thickness + 2), cv2.LINE_AA)
                        cv2.line(canvas, p1s, p2s, (220, 220, 220), max(1, thickness), cv2.LINE_AA)
                        cv2.circle(canvas, p1s, max(7, thickness * 3), (255, 120, 30), -1, cv2.LINE_AA)
                        cv2.circle(canvas, p2s, max(7, thickness * 3), (0, 150, 255), -1, cv2.LINE_AA)
                        cv2.circle(canvas, (x, y), max(7, thickness * 3), (0, 255, 80), 2, cv2.LINE_AA)
                    else:
                        cv2.circle(canvas, (x, y), max(10, thickness * 4), (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.circle(canvas, (x, y), max(7, thickness * 3), (0, 0, 255), -1, cv2.LINE_AA)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        texture = self.preview_image.texture
        if texture is None or tuple(texture.size) != (width, height):
            texture = Texture.create(size=(width, height), colorfmt="rgb")
            texture.flip_vertical()
            self.preview_image.texture = texture
        texture.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        texture.ask_update()
        self._set_preview_time(frame_index)

    def _play_preview(self, _instance):
        if self._preview_cap is None:
            self.preview_status.text = "请先完成分析，生成轨迹视频回放。"
            return
        if self._preview_event is not None:
            return
        if self._preview_cap.get(cv2.CAP_PROP_POS_FRAMES) >= self._preview_total_frames - 1:
            self._restart_preview(None)
        display_fps = min(max(self._preview_fps, 1.0), 30.0)
        self._preview_frame_stride = max(1, int(round(self._preview_fps / display_fps)))
        self._preview_event = Clock.schedule_interval(self._preview_tick, 1.0 / display_fps)
        self._preview_is_playing = True
        self.preview_status.text = ("正在播放：绿线为两端图像中点轨迹，蓝线/橙线为两端轨迹，连线表示当前杆轴投影。"
                                    if self._preview_is_dual else "正在播放：绿线显示整段轨迹，红圈显示当前追踪位置。")

    def _pause_preview(self, _instance):
        if self._preview_event is not None:
            self._preview_event.cancel()
            self._preview_event = None
        self._preview_is_playing = False

    def _restart_preview(self, _instance):
        self._pause_preview(None)
        if self._preview_cap is None:
            return
        self._preview_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self._preview_cap.read()
        if ok:
            self._render_preview_frame(frame, 0)
            self._preview_updating_slider = True
            self.preview_slider.value = 0
            self._preview_updating_slider = False

    def _preview_tick(self, _dt):
        if self._preview_cap is None:
            self._pause_preview(None)
            return False
        # 高帧率视频按比例跳过解码帧，尽量保持接近原视频的播放速度。
        for _ in range(max(0, self._preview_frame_stride - 1)):
            if not self._preview_cap.grab():
                self._pause_preview(None)
                self.preview_status.text = "视频播放结束。点击‘从头播放’可再次查看。"
                return False
        ok, frame = self._preview_cap.read()
        if not ok:
            self._pause_preview(None)
            self.preview_status.text = "视频播放结束。点击‘从头播放’可再次查看。"
            return False
        frame_index = max(0, int(self._preview_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self._render_preview_frame(frame, frame_index)
        self._preview_updating_slider = True
        self.preview_slider.value = min(frame_index, self.preview_slider.max)
        self._preview_updating_slider = False
        return True

    def _on_preview_seek(self, _slider, value):
        if self._preview_updating_slider or self._preview_is_playing or self._preview_cap is None:
            return
        frame_index = int(value)
        self._preview_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._preview_cap.read()
        if ok:
            actual_index = max(0, int(self._preview_cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            self._render_preview_frame(frame, actual_index)

    def _history_path(self):
        return os.path.join(self.user_data_dir, "training_history.json")

    def save_session(self, _instance):
        if not self.results:
            self.status_label.text = "目前没有可保存的分析结果。"
            return
        record = {
            **self.session,
            "summary": self.summary,
            "results": self.results,
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = self._history_path()
        try:
            history = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as file:
                    loaded = json.load(file)
                    if isinstance(loaded, list):
                        history = loaded
            history.append(record)
            history = history[-100:]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temp_path = path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as file:
                json.dump(history, file, ensure_ascii=False, indent=2)
            os.replace(temp_path, path)
            self.status_label.text = f"已保存训练记录。本机共保存 {len(history)} 次。"
        except Exception as exc:
            self.status_label.text = f"保存失败：{type(exc).__name__}: {exc}"

    def show_history(self, _instance):
        path = self._history_path()
        try:
            if not os.path.exists(path):
                content = "还没有保存过训练记录。分析完成后点击‘保存训练记录’即可。"
            else:
                with open(path, "r", encoding="utf-8") as file:
                    history = json.load(file)
                lines = []
                for item in reversed(history[-20:]):
                    summary = item.get("summary", {})
                    lines.append(
                        f"{item.get('saved_at', item.get('date', ''))} / {item.get('exercise', '')} / "
                        f"负荷 {item.get('load_kg') or '未填'} kg / "
                        f"平均 {summary.get('mean_velocity', '')} m/s / "
                        f"{summary.get('segment_count', 0)} 段"
                    )
                content = "\n\n".join(lines) if lines else "历史记录为空。"
        except Exception as exc:
            content = f"读取历史记录失败：{type(exc).__name__}: {exc}"
        scroll = ScrollView()
        label = Label(text=content, font_size="12sp", halign="left", valign="top", size_hint_y=None)
        label.bind(size=lambda obj, size: setattr(obj, "text_size", (max(size[0] - dp(10), 0), None)))
        label.bind(texture_size=lambda obj, size: setattr(obj, "height", size[1] + dp(20)))
        scroll.add_widget(label)
        popup = Popup(title="最近 20 次训练记录", content=scroll, size_hint=(0.92, 0.78))
        popup.open()

    def export_results(self, _instance):
        if not self.results:
            self.status_label.text = "没有可导出的结果。"
            return
        filename = f"BarbellVBT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            if kivy_platform == "android":
                out_dir = os.path.join(self.user_data_dir, "exports")
            else:
                out_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, filename)
            VideoAnalyzer.export_csv(path, self.session, self.results, self.summary)
            message = f"CSV 已导出：\n{path}\n\nAndroid 版本暂时保存在 App 私有文件夹内。"
            self.status_label.text = f"CSV 已导出：{filename}"
        except Exception as exc:
            message = f"导出失败：{type(exc).__name__}: {exc}"
            self.status_label.text = "CSV 导出失败。"
        Popup(title="导出结果", content=Label(text=message, font_size="12sp"), size_hint=(0.9, 0.38)).open()

    def clear_results(self, _instance):
        self.results = []
        self.summary = {}
        self.trajectory = []
        self._clear_preview("当前结果已清空。重新分析后可查看轨迹回放。")
        self.result_layout.clear_widgets()
        for label in self.metric_labels.values():
            label.text = "-"
        self.loss_label.text = "同方向速度比较：等待分析"
        self.advice_label.text = "分析提示会在完成后显示。"
        self.save_btn.disabled = True
        self.export_btn.disabled = True
        self.status_label.text = "当前结果已清空；已保存的历史记录不会删除。"



    def on_stop(self):
        self._pause_preview(None)
        if self._preview_cap is not None:
            try:
                self._preview_cap.release()
            except Exception:
                pass
            self._preview_cap = None


def run_desktop_mode():
    print("BarbellVBT 增强简洁版：桌面简易测试模式")
    video = input("把训练视频拖到这里并按回车：").strip().strip('"')
    if not os.path.isfile(video):
        print("找不到视频文件。")
        return
    raw_ppm = input("输入已经实测的标定值 px/m（不要直接使用默认猜测值）：").strip()
    ppm = safe_float(raw_ppm, None)
    if ppm is None or ppm <= 0:
        print("标定值无效。")
        return
    matrix = [[1.0 / ppm, 0.0, 0.0], [0.0, -1.0 / ppm, 0.0], [0.0, 0.0, 1.0]]
    calibration = {"mode": "manual_scale", "homography": matrix, "pixels_per_meter": ppm}
    analyzer = VideoAnalyzer(calibration, "red")
    result = analyzer.analyze(video, lambda d: print(f"分析进度：{d['progress']:.0f}%") if "progress" in d else None,
                              camera_setup={"horizontal_angle": "不确定", "camera_height": "不确定", "level": "不确定"})
    if "error" in result:
        print(result["error"])
        return
    print("摘要：", result["summary"])
    for row in result["results"]:
        print(row)
    session = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "athlete": "", "exercise": "未指定", "load_kg": None,
        "marker_color": "红色或橙色（推荐）", "calibration_mode": "manual_scale",
        "video_name": os.path.basename(video),
    }
    output = f"BarbellVBT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    VideoAnalyzer.export_csv(output, session, result["results"], result["summary"])
    print(f"CSV 已保存：{os.path.abspath(output)}")


if __name__ == "__main__":
    if kivy_platform == "android":
        BarbellVBTApp().run()
    else:
        run_desktop_mode()
