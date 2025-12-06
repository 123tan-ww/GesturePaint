"""
坐标转换和映射工具模块
负责将摄像头坐标系中的手势坐标转换为画布坐标系中的绘画坐标
提供坐标平滑、透视校正和边界处理功能
"""

import numpy as np
import json
import os
from collections import deque
from typing import Tuple, List, Optional


class CoordinateMapper:
    """
    坐标映射器类
    处理摄像头坐标到画布坐标的转换和优化
    """

    def __init__(self, camera_width: int = 640, camera_height: int = 480,
                 canvas_width: int = 1200, canvas_height: int = 800):
        """
        初始化坐标映射器

        Args:
            camera_width: 摄像头宽度
            camera_height: 摄像头高度
            canvas_width: 画布宽度
            canvas_height: 画布高度
        """
        # 分辨率设置
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        # 映射参数
        self.mirror_x = True  # 前置摄像头需要镜像
        self.smoothing_enabled = True
        self.perspective_correction = True

        # 平滑滤波参数
        self.smoothing_window_size = 5
        self.smoothing_factor = 0.7
        self.coordinate_history = deque(maxlen=self.smoothing_window_size)

        # 校准数据
        self.calibration_data = {
            'workspace_bounds': None,  # 有效工作区域边界
            'scale_factor': 1.0,  # 缩放因子
            'offset_x': 0,  # X轴偏移
            'offset_y': 0  # Y轴偏移
        }

        # 性能统计
        self.performance_stats = {
            'total_points': 0,
            'smoothed_points': 0,
            'out_of_bounds': 0
        }

        # 加载校准数据（如果存在）
        self._load_calibration()

    def setup_resolutions(self, camera_res: Tuple[int, int], canvas_res: Tuple[int, int]):
        """
        设置摄像头和画布分辨率

        Args:
            camera_res: (width, height) 摄像头分辨率
            canvas_res: (width, height) 画布分辨率
        """
        self.camera_width, self.camera_height = camera_res
        self.canvas_width, self.canvas_height = canvas_res
        print(f"坐标映射器: 摄像头 {camera_res} -> 画布 {canvas_res}")

    def normalize_landmarks(self, landmarks) -> List[Tuple[float, float]]:
        """
        将MediaPipe的归一化坐标转换为摄像头像素坐标

        Args:
            landmarks: MediaPipe手部关键点

        Returns:
            像素坐标列表 [(x, y), ...]
        """
        pixel_coords = []
        if landmarks:
            for landmark in landmarks.landmark:
                # 将归一化坐标转换为像素坐标
                x_px = landmark.x * self.camera_width
                y_px = landmark.y * self.camera_height
                pixel_coords.append((x_px, y_px))

        return pixel_coords

    def map_to_canvas(self, camera_x: float, camera_y: float) -> Tuple[int, int]:
        """
        将摄像头坐标映射到画布坐标

        Args:
            camera_x: 摄像头X坐标
            camera_y: 摄像头Y坐标

        Returns:
            画布坐标 (x, y)
        """
        # 1. 应用镜像（前置摄像头）
        if self.mirror_x:
            camera_x = self.camera_width - camera_x

        # 2. 透视校正
        if self.perspective_correction:
            camera_x, camera_y = self._apply_perspective_correction(camera_x, camera_y)

        # 3. 应用校准数据
        calibrated_x, calibrated_y = self._apply_calibration(camera_x, camera_y)

        # 4. 坐标映射
        # 简单的线性映射
        canvas_x = int((calibrated_x / self.camera_width) * self.canvas_width)
        canvas_y = int((calibrated_y / self.camera_height) * self.canvas_height)

        # 5. 平滑滤波
        if self.smoothing_enabled:
            canvas_x, canvas_y = self.apply_smoothing(canvas_x, canvas_y)

        # 6. 边界约束
        canvas_x, canvas_y = self.constrain_to_canvas(canvas_x, canvas_y)

        # 更新性能统计
        self.performance_stats['total_points'] += 1

        return canvas_x, canvas_y

    def _apply_perspective_correction(self, x: float, y: float) -> Tuple[float, float]:
        """
        应用透视校正，补偿摄像头角度导致的坐标偏差

        Args:
            x: 原始X坐标
            y: 原始Y坐标

        Returns:
            校正后的坐标 (x, y)
        """
        # 简化的透视校正：假设摄像头在顶部中央，轻微向下倾斜
        # 实际项目中可能需要更复杂的相机标定

        # 归一化坐标
        norm_x = x / self.camera_width
        norm_y = y / self.camera_height

        # 轻微的非线性变换，补偿透视效果
        # 靠近边缘的坐标向内收缩
        center_x, center_y = 0.5, 0.5
        dist_x = norm_x - center_x
        dist_y = norm_y - center_y
        distance = np.sqrt(dist_x ** 2 + dist_y ** 2)

        # 距离中心的非线性缩放
        scale = 1.0 + 0.2 * distance  # 边缘稍微放大

        corrected_x = center_x + dist_x * scale
        corrected_y = center_y + dist_y * scale

        # 转换回像素坐标
        corrected_x = max(0, min(1, corrected_x)) * self.camera_width
        corrected_y = max(0, min(1, corrected_y)) * self.camera_height

        return corrected_x, corrected_y

    def _apply_calibration(self, x: float, y: float) -> Tuple[float, float]:
        """
        应用校准数据

        Args:
            x: 原始X坐标
            y: 原始Y坐标

        Returns:
            校准后的坐标 (x, y)
        """
        calibrated_x = x * self.calibration_data['scale_factor'] + self.calibration_data['offset_x']
        calibrated_y = y * self.calibration_data['scale_factor'] + self.calibration_data['offset_y']

        # 应用工作区域边界约束
        if self.calibration_data['workspace_bounds']:
            bounds = self.calibration_data['workspace_bounds']
            calibrated_x = max(bounds[0], min(bounds[2], calibrated_x))
            calibrated_y = max(bounds[1], min(bounds[3], calibrated_y))

        return calibrated_x, calibrated_y

    def apply_smoothing(self, x: int, y: int) -> Tuple[int, int]:
        """
        应用移动平均滤波平滑坐标

        Args:
            x: 当前X坐标
            y: 当前Y坐标

        Returns:
            平滑后的坐标 (x, y)
        """
        # 添加当前坐标到历史记录
        self.coordinate_history.append((x, y))

        # 如果历史记录不足，直接返回当前坐标
        if len(self.coordinate_history) < 2:
            return x, y

        # 计算加权移动平均
        smoothed_x, smoothed_y = 0, 0
        total_weight = 0

        for i, (hist_x, hist_y) in enumerate(self.coordinate_history):
            # 越新的点权重越高
            weight = self.smoothing_factor ** (len(self.coordinate_history) - i - 1)
            smoothed_x += hist_x * weight
            smoothed_y += hist_y * weight
            total_weight += weight

        smoothed_x = int(smoothed_x / total_weight)
        smoothed_y = int(smoothed_y / total_weight)

        self.performance_stats['smoothed_points'] += 1

        return smoothed_x, smoothed_y

    def constrain_to_canvas(self, x: int, y: int) -> Tuple[int, int]:
        """
        确保坐标在画布边界内

        Args:
            x: X坐标
            y: Y坐标

        Returns:
            约束后的坐标 (x, y)
        """
        constrained_x = max(0, min(self.canvas_width - 1, x))
        constrained_y = max(0, min(self.canvas_height - 1, y))

        if constrained_x != x or constrained_y != y:
            self.performance_stats['out_of_bounds'] += 1

        return constrained_x, constrained_y

    def calibrate_workspace(self, reference_points: List[Tuple[float, float]]):
        """
        使用参考点校准工作空间

        Args:
            reference_points: 参考点列表 [(x1, y1), (x2, y2), ...]
        """
        if len(reference_points) < 2:
            print("警告: 需要至少2个参考点进行校准")
            return

        # 计算参考点的边界框
        x_coords = [p[0] for p in reference_points]
        y_coords = [p[1] for p in reference_points]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # 设置工作区域边界
        self.calibration_data['workspace_bounds'] = (min_x, min_y, max_x, max_y)

        # 计算缩放因子，使工作区域填满画布
        workspace_width = max_x - min_x
        workspace_height = max_y - min_y

        if workspace_width > 0 and workspace_height > 0:
            scale_x = self.canvas_width / workspace_width
            scale_y = self.canvas_height / workspace_height
            self.calibration_data['scale_factor'] = min(scale_x, scale_y)

            # 计算偏移量，使工作区域居中
            self.calibration_data['offset_x'] = -min_x * self.calibration_data['scale_factor']
            self.calibration_data['offset_y'] = -min_y * self.calibration_data['scale_factor']

        print(f"工作空间校准完成: 边界({min_x}, {min_y}, {max_x}, {max_y})")
        self._save_calibration()

    def auto_calibrate(self, sample_frames: int = 30):
        """
        自动校准系统

        Args:
            sample_frames: 采样帧数
        """
        print(f"开始自动校准，采样 {sample_frames} 帧...")
        # 在实际实现中，这里会采集多帧数据并分析有效工作区域
        # 简化版本：使用默认校准
        self.calibration_data = {
            'workspace_bounds': None,
            'scale_factor': 1.0,
            'offset_x': 0,
            'offset_y': 0
        }
        self._save_calibration()
        print("自动校准完成")

    def _save_calibration(self):
        """保存校准数据到文件"""
        try:
            with open('calibration_data.json', 'w') as f:
                json.dump(self.calibration_data, f)
            print("校准数据已保存")
        except Exception as e:
            print(f"保存校准数据失败: {e}")

    def _load_calibration(self):
        """从文件加载校准数据"""
        try:
            if os.path.exists('calibration_data.json'):
                with open('calibration_data.json', 'r') as f:
                    self.calibration_data = json.load(f)
                print("校准数据已加载")
        except Exception as e:
            print(f"加载校准数据失败: {e}")

    def reset_smoothing(self):
        """重置平滑滤波器状态"""
        self.coordinate_history.clear()
        print("平滑滤波器已重置")

    def calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """
        计算两点间距离

        Args:
            x1, y1: 点1坐标
            x2, y2: 点2坐标

        Returns:
            两点间距离
        """
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def is_stable_position(self, x: int, y: int, threshold: float = 3.0) -> bool:
        """
        判断坐标是否稳定

        Args:
            x: X坐标
            y: Y坐标
            threshold: 稳定性阈值（像素）

        Returns:
            坐标是否稳定
        """
        if len(self.coordinate_history) < 2:
            return False

        # 检查最近几个点的变化是否小于阈值
        recent_points = list(self.coordinate_history)[-3:]  # 最近3个点
        if len(recent_points) < 2:
            return False

        max_distance = 0
        for i in range(len(recent_points) - 1):
            dist = self.calculate_distance(
                recent_points[i][0], recent_points[i][1],
                recent_points[i + 1][0], recent_points[i + 1][1]
            )
            max_distance = max(max_distance, dist)

        return max_distance <= threshold

    def get_movement_vector(self, prev_pos: Tuple[float, float], curr_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        获取移动方向向量

        Args:
            prev_pos: 前一个位置 (x, y)
            curr_pos: 当前位置 (x, y)

        Returns:
            移动向量 (dx, dy)
        """
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        return dx, dy

    def get_performance_stats(self) -> dict:
        """
        获取性能统计

        Returns:
            性能统计字典
        """
        return self.performance_stats.copy()

    def set_smoothing_parameters(self, window_size: int = None, factor: float = None):
        """
        设置平滑参数

        Args:
            window_size: 平滑窗口大小
            factor: 平滑因子
        """
        if window_size is not None:
            self.smoothing_window_size = max(1, window_size)
            self.coordinate_history = deque(maxlen=self.smoothing_window_size)

        if factor is not None:
            self.smoothing_factor = max(0.1, min(0.9, factor))

    def camera_to_canvas(self, x, y):
        """将摄像头坐标转换为画布坐标"""
        canvas_x = int(x * self.canvas_width)
        canvas_y = int(y * self.canvas_height)
        return canvas_x, canvas_y


# 工具函数
def calculate_trajectory_angle(points: List[Tuple[float, float]]) -> float:
    """
    计算轨迹角度

    Args:
        points: 轨迹点列表

    Returns:
        轨迹角度（弧度）
    """
    if len(points) < 2:
        return 0.0

    # 计算起始点和结束点的向量
    start_x, start_y = points[0]
    end_x, end_y = points[-1]

    dx = end_x - start_x
    dy = end_y - start_y

    # 计算角度
    angle = np.arctan2(dy, dx)
    return angle


def create_coordinate_mapper(camera_res: Tuple[int, int], canvas_res: Tuple[int, int]) -> CoordinateMapper:
    """
    创建坐标映射器的工厂函数

    Args:
        camera_res: 摄像头分辨率
        canvas_res: 画布分辨率

    Returns:
        配置好的CoordinateMapper实例
    """
    mapper = CoordinateMapper(camera_res[0], camera_res[1], canvas_res[0], canvas_res[1])
    return mapper


# 测试代码
if __name__ == "__main__":
    # 简单的测试示例
    mapper = CoordinateMapper(640, 480, 1200, 800)

    # 测试坐标映射
    test_points = [(100, 100), (320, 240), (500, 300)]

    print("坐标映射测试:")
    for cam_x, cam_y in test_points:
        canvas_x, canvas_y = mapper.map_to_canvas(cam_x, cam_y)
        print(f"摄像头({cam_x}, {cam_y}) -> 画布({canvas_x}, {canvas_y})")

    # 测试性能统计
    stats = mapper.get_performance_stats()
    print(f"\n性能统计: {stats}")