#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BarbellVBT - 上传视频分析模式
用户上传一段杠铃训练视频，App自动分析每一组的杠铃速度
"""

import os
import cv2
import numpy as np
from datetime import datetime

# Kivy 相关（手机端UI）
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.clock import Clock
from kivy.graphics.texture import Texture

try:
    from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class VelocityCalculator:
    """速度计算器：从杠铃位移计算速度"""

    def __init__(self, fps=30, pixels_per_meter=1000):
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter  # 标定系数
        self.dt = 1.0 / fps
        self.positions = []  # 存储每次检测的杠铃Y坐标
        self.timestamps = []
        self.velocities = []
        self.reps = []  # 存储每组的分析数据

    def add_position(self, y_pixel, timestamp):
        """添加一个新位置点"""
        y_meter = y_pixel / self.pixels_per_meter
        self.positions.append(y_meter)
        self.timestamps.append(timestamp)

        if len(self.positions) >= 3:
            # 用中心差分计算瞬时速度
            idx = len(self.positions) - 1
            v = (self.positions[idx] - self.positions[idx - 2]) / (2 * self.dt)
            self.velocities.append(abs(v))
        else:
            self.velocities.append(0)

    def get_peak_velocity(self):
        """获取峰值速度"""
        if self.velocities:
            return max(self.velocities)
        return 0

    def get_mean_velocity(self):
        """获取平均速度"""
        if self.velocities:
            return np.mean(self.velocities)
        return 0

    def reset(self):
        """重置"""
        self.positions = []
        self.timestamps = []
        self.velocities = []


class BarbellTracker:
    """杠铃追踪器：从视频帧中检测杠铃位置"""

    def __init__(self):
        # 用颜色追踪杠铃（默认追踪银色/金属色）
        self.lower_bound = np.array([80, 20, 100])   # HSV 下限
        self.upper_bound = np.array([130, 80, 255])  # HSV 上限
        self.last_y = None

    def detect(self, frame):
        """在帧中检测杠铃，返回中心Y坐标"""
        if frame is None:
            return None

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)

        # 形态学操作去噪
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self.last_y  # 没检测到就返回上次位置

        # 找最大的轮廓
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area < 100:  # 太小忽略
            return self.last_y

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return self.last_y

        cy = int(M["m01"] / M["m00"])
        self.last_y = cy
        return cy


class VideoAnalyzer:
    """视频分析器：读取视频→逐帧分析→输出报告"""

    def __init__(self):
        self.tracker = BarbellTracker()
        self.calculator = VelocityCalculator()
        self.rep_data = []
        self.video_path = None

    def analyze(self, video_path, callback=None):
        """
        分析视频，callback 用于实时更新UI
        返回分析结果列表
        """
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {"error": "无法打开视频文件"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.calculator = VelocityCalculator(fps=fps)

        frame_count = 0
        results = []
        current_rep = {"frames": [], "positions": [], "velocities": []}
        rep_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            timestamp = frame_count / fps

            # 检测杠铃位置
            y = self.tracker.detect(frame)

            if y is not None:
                self.calculator.add_position(y, timestamp)
                current_rep["frames"].append(frame_count)
                current_rep["positions"].append(y)
                current_rep["velocities"].append(self.calculator.velocities[-1])

            # 检测一组结束（速度接近0且持续多帧）
            if len(self.calculator.velocities) > 10:
                recent = self.calculator.velocities[-10:]
                if all(v < 0.05 for v in recent) and len(current_rep["frames"]) > 5:
                    # 一组结束
                    rep_number += 1
                    peak_v = max(current_rep["velocities"]) if current_rep["velocities"] else 0
                    mean_v = np.mean(current_rep["velocities"]) if current_rep["velocities"] else 0
                    displacement = 0
                    if len(current_rep["positions"]) > 1:
                        displacement = abs(max(current_rep["positions"]) - min(current_rep["positions"])) / self.calculator.pixels_per_meter

                    rep_result = {
                        "rep": rep_number,
                        "peak_velocity": round(peak_v, 3),
                        "mean_velocity": round(mean_v, 3),
                        "displacement": round(displacement, 3),
                        "duration": round(timestamp - current_rep["frames"][0] / fps, 1),
                    }
                    results.append(rep_result)
                    current_rep = {"frames": [], "positions": [], "velocities": []}

                    if callback:
                        callback(rep_result)

            # 进度回调
            if callback and frame_count % 30 == 0:
                callback({"progress": frame_count / total_frames * 100})

        cap.release()

        # 保存结果
        self.rep_data = results
        return results

    def export_csv(self, filepath):
        """导出为 CSV"""
        import csv
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["组数", "峰值速度(m/s)", "平均速度(m/s)", "位移(m)", "持续时间(s)"])
            for r in self.rep_data:
                writer.writerow([r["rep"], r["peak_velocity"], r["mean_velocity"], r["displacement"], r["duration"]])
        return filepath


class BarbellVBTApp(App):
    """Kivy App 主界面"""

    def build(self):
        self.title = "BarbellVBT - 杠铃速度分析"
        self.analyzer = VideoAnalyzer()
        self.results = []

        # 主布局
        self.root = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # 标题
        title = Label(
            text="🏋️ BarbellVBT\n杠铃速度视频分析",
            font_size='24sp',
            size_hint_y=0.15,
            bold=True
        )
        self.root.add_widget(title)

        # 选择视频按钮
        self.select_btn = Button(
            text="📁 选择视频文件",
            font_size='20sp',
            size_hint_y=0.1,
            background_color=(0.2, 0.6, 0.9, 1)
        )
        self.select_btn.bind(on_press=self.open_filechooser)
        self.root.add_widget(self.select_btn)

        # 开始分析按钮
        self.analyze_btn = Button(
            text="▶️ 开始分析",
            font_size='20sp',
            size_hint_y=0.1,
            background_color=(0.2, 0.8, 0.3, 1),
            disabled=True
        )
        self.analyze_btn.bind(on_press=self.start_analysis)
        self.root.add_widget(self.analyze_btn)

        # 状态标签
        self.status_label = Label(
            text="请先选择一个训练视频",
            font_size='16sp',
            size_hint_y=0.08,
            color=(0.7, 0.7, 0.7, 1)
        )
        self.root.add_widget(self.status_label)

        # 结果显示区域（可滚动）
        self.scroll = ScrollView(size_hint_y=0.45)
        self.result_layout = BoxLayout(orientation='vertical', size_hint_y=None, spacing=5)
        self.result_layout.bind(minimum_height=self.result_layout.setter('height'))
        self.scroll.add_widget(self.result_layout)
        self.root.add_widget(self.scroll)

        # 底部按钮行
        btn_row = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=10)

        self.export_btn = Button(
            text="💾 导出CSV",
            font_size='16sp',
            disabled=True,
            background_color=(0.9, 0.6, 0.2, 1)
        )
        self.export_btn.bind(on_press=self.export_results)
        btn_row.add_widget(self.export_btn)

        self.clear_btn = Button(
            text="🗑️ 清空",
            font_size='16sp',
            background_color=(0.8, 0.3, 0.3, 1)
        )
        self.clear_btn.bind(on_press=self.clear_results)
        btn_row.add_widget(self.clear_btn)

        self.root.add_widget(btn_row)

        # 标定按钮
        self.calibrate_btn = Button(
            text="📐 标定设置",
            font_size='16sp',
            size_hint_y=0.07,
            background_color=(0.5, 0.5, 0.5, 1)
        )
        self.calibrate_btn.bind(on_press=self.open_calibration)
        self.root.add_widget(self.calibrate_btn)

        self.selected_video = None
        self.pixels_per_meter = 1000  # 默认标定系数

        return self.root

    def open_filechooser(self, instance):
        """打开文件选择器"""
        content = BoxLayout(orientation='vertical')
        filechooser = FileChooserListView(
            filters=['*.mp4', '*.avi', '*.mov', '*.mkv', '*.3gp'],
            path='/storage/emulated/0/'  # Android 默认路径
        )
        content.add_widget(filechooser)

        btn_row = BoxLayout(orientation='horizontal', size_hint_y=0.1)
        ok_btn = Button(text="确认")
        cancel_btn = Button(text="取消")
        btn_row.add_widget(ok_btn)
        btn_row.add_widget(cancel_btn)
        content.add_widget(btn_row)

        popup = Popup(title="选择视频文件", content=content, size_hint=(0.9, 0.9))

        def confirm(instance):
            if filechooser.selection:
                self.selected_video = filechooser.selection[0]
                self.status_label.text = f"已选择: {os.path.basename(self.selected_video)}"
                self.analyze_btn.disabled = False
            popup.dismiss()

        ok_btn.bind(on_press=confirm)
        cancel_btn.bind(on_press=popup.dismiss)
        popup.open()

    def open_calibration(self, instance):
        """标定设置弹窗"""
        content = BoxLayout(orientation='vertical', padding=10, spacing=10)
        content.add_widget(Label(
            text="📐 标定设置\n\n在视频中放入一个已知尺寸的物体\n（如杠铃片直径45cm）\n然后测量它在画面中的像素高度\n\n像素/米 = 像素高度 ÷ 实际高度(米)",
            font_size='14sp'
        ))

        from kivy.uix.textinput import TextInput
        input_field = TextInput(
            text=str(self.pixels_per_meter),
            multiline=False,
            input_type='number',
            size_hint_y=0.15
        )
        content.add_widget(input_field)

        btn = Button(text="保存", size_hint_y=0.15)
        content.add_widget(btn)

        popup = Popup(title="标定", content=content, size_hint=(0.8, 0.5))

        def save(instance):
            try:
                self.pixels_per_meter = float(input_field.text)
                self.analyzer.calculator.pixels_per_meter = self.pixels_per_meter
                self.status_label.text = f"标定已更新: {self.pixels_per_meter} px/m"
            except ValueError:
                self.status_label.text = "请输入有效数字"
            popup.dismiss()

        btn.bind(on_press=save)
        popup.open()

    def start_analysis(self, instance):
        """开始分析视频"""
        if not self.selected_video:
            self.status_label.text = "请先选择视频！"
            return

        self.analyze_btn.disabled = True
        self.status_label.text = "分析中..."
        self.result_layout.clear_widgets()

        # 在新线程中运行分析，避免卡住UI
        import threading
        thread = threading.Thread(target=self._run_analysis)
        thread.daemon = True
        thread.start()

    def _run_analysis(self):
        """后台运行分析"""
        self.analyzer.calculator.pixels_per_meter = self.pixels_per_meter

        def callback(data):
            if "error" in data:
                Clock.schedule_once(lambda dt: self._show_error(data["error"]))
            elif "progress" in data:
                Clock.schedule_once(lambda dt: self._update_progress(data["progress"]))
            elif "rep" in data:
                Clock.schedule_once(lambda dt: self._add_result(data))

        results = self.analyzer.analyze(self.selected_video, callback)

        Clock.schedule_once(lambda dt: self._analysis_done(results))

    def _update_progress(self, progress):
        self.status_label.text = f"分析中... {progress:.0f}%"

    def _show_error(self, msg):
        self.status_label.text = f"错误: {msg}"
        self.analyze_btn.disabled = False

    def _add_result(self, rep_data):
        """添加一组结果显示"""
        # 颜色区分速度等级
        if rep_data["mean_velocity"] >= 0.8:
            color = (0.2, 0.9, 0.2, 1)   # 绿色 - 高速度
        elif rep_data["mean_velocity"] >= 0.5:
            color = (0.9, 0.8, 0.2, 1)   # 黄色 - 中等
        else:
            color = (0.9, 0.3, 0.3, 1)   # 红色 - 低速

        text = (
            f"第{rep_data['rep']}组 | "
            f"峰值: {rep_data['peak_velocity']} m/s | "
            f"平均: {rep_data['mean_velocity']} m/s | "
            f"位移: {rep_data['displacement']}m"
        )
        lbl = Label(
            text=text,
            font_size='14sp',
            size_hint_y=None,
            height='40dp',
            color=color
        )
        self.result_layout.add_widget(lbl)
        self.results.append(rep_data)

    def _analysis_done(self, results):
        """分析完成"""
        self.analyze_btn.disabled = False
        if isinstance(results, dict) and "error" in results:
            self.status_label.text = f"错误: {results['error']}"
        else:
            self.status_label.text = f"✅ 分析完成！共 {len(results)} 组"
            self.export_btn.disabled = len(results) == 0

    def export_results(self, instance):
        """导出结果"""
        if not self.results:
            self.status_label.text = "没有可导出的数据"
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"BarbellVBT_{timestamp}.csv"

        # Android 存储路径
        export_dir = "/storage/emulated/0/Download/"
        if not os.path.exists(export_dir):
            export_dir = os.path.expanduser("~/")
            os.makedirs(export_dir, exist_ok=True)

        filepath = os.path.join(export_dir, filename)
        self.analyzer.export_csv(filepath)
        self.status_label.text = f"✅ 已导出: {filename}"

        # 弹出提示
        content = Label(text=f"文件已保存到:\n{filepath}", font_size='14sp')
        popup = Popup(title="导出成功", content=content, size_hint=(0.8, 0.3))
        popup.open()

    def clear_results(self, instance):
        """清空结果"""
        self.results = []
        self.result_layout.clear_widgets()
        self.status_label.text = "已清空"
        self.export_btn.disabled = True


if __name__ == "__main__":
    # 检测运行环境
    import platform
    if platform.system() == "Android":
        # 手机端直接启动
        BarbellVBTApp().run()
    else:
        # 电脑端：提供命令行模式 + GUI模式
        print("=" * 50)
        print("  BarbellVBT - 杠铃速度视频分析")
        print("=" * 50)
        print()
        print("选择运行模式:")
        print("  1. 手机App模式 (Kivy GUI)")
        print("  2. 电脑命令行模式 (分析视频，输出CSV)")
        print()
        choice = input("请输入 1 或 2: ").strip()

        if choice == "2":
            # 命令行模式
            video_path = input("请输入视频文件路径: ").strip().strip('"')
            if not os.path.exists(video_path):
                print("文件不存在！")
                exit(1)

            pixels_per_meter = input("请输入标定系数(像素/米, 默认1000): ").strip()
            ppm = float(pixels_per_meter) if pixels_per_meter else 1000

            analyzer = VideoAnalyzer()
            analyzer.calculator.pixels_per_meter = ppm

            print("\n开始分析...\n")
            results = analyzer.analyze(video_path)

            if isinstance(results, dict) and "error" in results:
                print(f"错误: {results['error']}")
            else:
                print(f"{'组数':<6}{'峰值速度':<12}{'平均速度':<12}{'位移':<10}{'时长':<8}")
                print("-" * 50)
                for r in results:
                    print(f"#{r['rep']:<5}{r['peak_velocity']:<12}{r['mean_velocity']:<12}"
                          f"{r['displacement']:<10}{r['duration']:<8}")

                # 自动导出CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_path = f"BarbellVBT_{timestamp}.csv"
                analyzer.export_csv(csv_path)
                print(f"\n✅ 结果已导出到: {csv_path}")

        else:
            # GUI模式
            BarbellVBTApp().run()
