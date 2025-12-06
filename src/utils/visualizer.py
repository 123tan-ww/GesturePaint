import pygame
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any


class Visualizer:
    def __init__(self, screen: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font):
        """
        可视化器初始化

        Args:
            screen: Pygame显示表面
            font: 主字体
            small_font: 小字体
        """
        self.screen = screen
        self.font = font
        self.small_font = small_font

        # 颜色定义
        self.colors = {
            'background': (240, 240, 240),
            'panel': (220, 220, 220),
            'text': (50, 50, 50),
            'highlight': (70, 130, 180),
            'success': (0, 150, 0),
            'error': (200, 0, 0),
            'warning': (200, 150, 0),
            'landmark': (255, 0, 0),
            'connection': (0, 255, 0),
            'hand_left': (0, 100, 255),
            'hand_right': (255, 100, 0)
        }

        # 反馈消息队列
        self.feedback_messages = []
        self.feedback_duration = 3.0  # 消息显示时间（秒）

        # 调试信息
        self.debug_info = {}

        # 动画状态
        self.animation_timer = 0
        self.animation_duration = 0.5

    def draw_landmarks(self, image: np.ndarray, gesture_info) -> np.ndarray:
        """
        在图像上绘制手部关键点和连接线

        Args:
            image: 原始图像
            landmarks: 关键点列表

        Returns:
            绘制后的图像
        """
        if  gesture_info is None:
            return image

        h, w, _ = image.shape

        if gesture_info['landmarks']:
            for i, landmarks in enumerate(gesture_info['landmarks']):
        # 绘制关键点
                for landmark in landmarks:
                    x = int(landmark['x'] * w)
                    y = int(landmark['y'] * h)
                    cv2.circle(image, (x, y), 5, self.colors['landmark'], -1)

                # 绘制连接线
                connections = self._get_hand_connections()
                for connection in connections:
                    start_idx, end_idx = connection
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_point = (int(landmarks[start_idx]['x'] * w),
                                       int(landmarks[start_idx]['y'] * h))
                        end_point = (int(landmarks[end_idx]['x'] * w),
                                     int(landmarks[end_idx]['y'] * h))
                        cv2.line(image, start_point, end_point, self.colors['connection'], 2)

        return image

    def _get_hand_connections(self) -> list[Union[list[int], Any]]:
        """获取手部关键点连接关系"""
        return [
            # 拇指
            [0, 1], [1, 2], [2, 3], [3, 4],
            # 食指
            [0, 5], [5, 6], [6, 7], [7, 8],
            # 中指
            [0, 9], [9, 10], [10, 11], [11, 12],
            # 无名指
            [0, 13], [13, 14], [14, 15], [15, 16],
            # 小指
            [0, 17], [17, 18], [18, 19], [19, 20],
            # 手掌
            [5, 9], [9, 13], [13, 17]
        ]

    def draw_gesture_info(self, image: np.ndarray, gesture_info: Dict) -> np.ndarray:
        """
        在图像上绘制手势信息

        Args:
            image: 原始图像
            gesture_info: 手势信息字典

        Returns:
            绘制后的图像
        """
        if not gesture_info or not gesture_info['gesture']:
            return image

        y_offset = 30
        for i, gesture in enumerate(gesture_info['gesture']):
            if i < len(gesture_info.get('handedness', [])):
                hand = gesture_info['handedness'][i]

                # 根据左右手选择颜色
                hand_color = self.colors['hand_left'] if hand['label'] == 'Left' else self.colors['hand_right']

                # 绘制手势信息
                text = f"{hand['label']}: {gesture['category_name']} ({gesture['score']:.2f})"
                cv2.putText(image, text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
                y_offset += 30

                # 绘制手势边界框
                if gesture_info.get('landmarks') and i < len(gesture_info['landmarks']):
                    self._draw_hand_bbox(image, gesture_info['landmarks'][i], hand_color)

        return image

    def _draw_hand_bbox(self, image: np.ndarray, landmarks: List[Dict], color: Tuple[int, int, int]):
        """
        绘制手部边界框

        Args:
            image: 图像
            landmarks: 关键点列表
            color: 边界框颜色
        """
        if not landmarks:
            return

        h, w, _ = image.shape

        # 计算边界框
        xs = [landmark['x'] for landmark in landmarks]
        ys = [landmark['y'] for landmark in landmarks]

        x_min = int(min(xs) * w)
        y_min = int(min(ys) * h)
        x_max = int(max(xs) * w)
        y_max = int(max(ys) * h)

        # 绘制边界框
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # 绘制中心点
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        cv2.circle(image, (center_x, center_y), 5, color, -1)

    def show_feedback_message(self, message: str, message_type: str = "info"):
        """
        显示反馈消息

        Args:
            message: 消息内容
            message_type: 消息类型 (info, success, error, warning)
        """
        self.feedback_messages.append({
            'text': message,
            'type': message_type,
            'timestamp': pygame.time.get_ticks()
        })

        # 限制消息数量
        if len(self.feedback_messages) > 5:
            self.feedback_messages.pop(0)

    def draw_feedback_messages(self):
        """绘制反馈消息"""
        current_time = pygame.time.get_ticks()
        y_offset = 100

        # 过滤过期消息
        self.feedback_messages = [
            msg for msg in self.feedback_messages
            if current_time - msg['timestamp'] < self.feedback_duration * 1000
        ]

        # 绘制消息
        for i, message in enumerate(self.feedback_messages):
            # 计算透明度（淡入淡出效果）
            elapsed = current_time - message['timestamp']
            if elapsed < 500:  # 前0.5秒淡入
                alpha = int(255 * (elapsed / 500))
            elif elapsed > (self.feedback_duration - 0.5) * 1000:  # 最后0.5秒淡出
                alpha = int(255 * ((self.feedback_duration * 1000 - elapsed) / 500))
            else:
                alpha = 255

            # 根据消息类型选择颜色
            if message['type'] == 'success':
                color = self.colors['success']
            elif message['type'] == 'error':
                color = self.colors['error']
            elif message['type'] == 'warning':
                color = self.colors['warning']
            else:
                color = self.colors['text']

            # 创建带透明度的表面
            text_surface = self.small_font.render(message['text'], True, color)
            text_surface.set_alpha(alpha)

            # 绘制背景
            bg_rect = pygame.Rect(10, y_offset, text_surface.get_width() + 10, text_surface.get_height() + 5)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surface.fill((0, 0, 0, alpha // 3))  # 半透明黑色背景
            self.screen.blit(bg_surface, bg_rect)

            # 绘制文本
            self.screen.blit(text_surface, (15, y_offset + 2))
            y_offset += 30

    def draw_system_status(self, status_info: Dict):
        """
        绘制系统状态信息

        Args:
            status_info: 状态信息字典
        """
        # 状态面板位置
        panel_rect = pygame.Rect(10, 10, 300, 150)

        # 绘制面板背景
        pygame.draw.rect(self.screen, self.colors['panel'], panel_rect)
        pygame.draw.rect(self.screen, self.colors['text'], panel_rect, 2)

        # 绘制标题
        title = self.font.render("系统状态", True, self.colors['text'])
        self.screen.blit(title, (panel_rect.x + 10, panel_rect.y + 10))

        # 绘制状态信息
        y_offset = panel_rect.y + 40
        status_items = [
            f"绘画状态: {status_info.get('drawing_status', '未知')}",
            f"当前手势: {status_info.get('current_gesture', '无')}",
            f"笔刷颜色: {status_info.get('brush_color', '未知')}",
            f"笔刷大小: {status_info.get('brush_size', '未知')}",
            f"检测手数: {status_info.get('hand_count', 0)}",
            f"帧率: {status_info.get('fps', 0):.1f}"
        ]

        for item in status_items:
            text = self.small_font.render(item, True, self.colors['text'])
            self.screen.blit(text, (panel_rect.x + 15, y_offset))
            y_offset += 20

    def draw_brush_preview(self, brush_config, position: Tuple[int, int]):
        """
        绘制笔刷预览

        Args:
            brush_config: 笔刷配置对象
            position: 预览位置 (x, y)
        """
        x, y = position

        # 绘制标题
        title = self.small_font.render("当前笔刷", True, self.colors['text'])
        self.screen.blit(title, (x, y))

        # 绘制笔刷预览
        preview_radius = brush_config.size
        preview_rect = pygame.Rect(x, y + 25, preview_radius * 2 + 10, preview_radius * 2 + 10)

        # 绘制背景
        pygame.draw.rect(self.screen, (255, 255, 255), preview_rect)
        pygame.draw.rect(self.screen, self.colors['text'], preview_rect, 1)

        # 绘制笔刷点
        center_x = preview_rect.x + preview_rect.width // 2
        center_y = preview_rect.y + preview_rect.height // 2
        pygame.draw.circle(self.screen, brush_config.color, (center_x, center_y), preview_radius)

        # 绘制笔刷信息
        color_name = self._get_color_name(brush_config.color)
        info_text = f"颜色: {color_name}, 大小: {brush_config.size}"
        info_surface = self.small_font.render(info_text, True, self.colors['text'])
        self.screen.blit(info_surface, (x, y + preview_rect.height + 30))

    def _get_color_name(self, color: Tuple[int, int, int]) -> str:
        """获取颜色名称"""
        color_names = {
            (255, 0, 0): "红色",
            (0, 255, 0): "绿色",
            (0, 0, 255): "蓝色",
            (0, 0, 0): "黑色",
            (255, 255, 255): "白色"
        }
        return color_names.get(color, "自定义")

    def draw_gesture_help(self, position: Tuple[int, int]):
        """
        绘制手势帮助信息

        Args:
            position: 绘制位置 (x, y)
        """
        x, y = position

        # 绘制面板
        panel_width = 250
        panel_height = 200
        panel_rect = pygame.Rect(x, y, panel_width, panel_height)

        pygame.draw.rect(self.screen, self.colors['panel'], panel_rect)
        pygame.draw.rect(self.screen, self.colors['text'], panel_rect, 2)

        # 绘制标题
        title = self.font.render("手势说明", True, self.colors['text'])
        self.screen.blit(title, (x + 10, y + 10))

        # 绘制手势列表
        gestures = [
            "捏合手势: 开始/继续绘画",
            "张开手掌: 清空画布",
            "握拳: 撤销上一步",
            "胜利手势: 保存画布",
            "大拇指向上: 增大笔刷",
            "大拇指向下: 减小笔刷"
        ]

        y_offset = y + 40
        for gesture in gestures:
            text = self.small_font.render(gesture, True, self.colors['text'])
            self.screen.blit(text, (x + 15, y_offset))
            y_offset += 20

    def draw_debug_info(self, position: Tuple[int, int]):
        """
        绘制调试信息

        Args:
            position: 绘制位置 (x, y)
        """
        if not self.debug_info:
            return

        x, y = position

        # 绘制面板
        panel_width = 300
        panel_height = min(200, len(self.debug_info) * 20 + 30)
        panel_rect = pygame.Rect(x, y, panel_width, panel_height)

        pygame.draw.rect(self.screen, (200, 200, 200), panel_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), panel_rect, 1)

        # 绘制标题
        title = self.small_font.render("调试信息", True, (50, 50, 50))
        self.screen.blit(title, (x + 10, y + 5))

        # 绘制调试项
        y_offset = y + 25
        for key, value in self.debug_info.items():
            text = self.small_font.render(f"{key}: {value}", True, (50, 50, 50))
            self.screen.blit(text, (x + 10, y_offset))
            y_offset += 15

    def update_debug_info(self, key: str, value: str):
        """更新调试信息"""
        self.debug_info[key] = value

    def clear_debug_info(self):
        """清空调试信息"""
        self.debug_info.clear()

    def draw_loading_animation(self, message: str, position: Tuple[int, int]):
        """
        绘制加载动画

        Args:
            message: 加载消息
            position: 绘制位置 (x, y)
        """
        x, y = position

        # 计算动画进度
        progress = (pygame.time.get_ticks() % 1000) / 1000

        # 绘制加载圆圈
        radius = 15
        pygame.draw.circle(self.screen, self.colors['highlight'], (x + radius, y + radius), radius, 2)

        # 绘制进度弧
        start_angle = progress * 2 * 3.14159
        end_angle = start_angle + 1.5 * 3.14159
        pygame.draw.arc(self.screen, self.colors['highlight'],
                        (x, y, radius * 2, radius * 2), start_angle, end_angle, 2)

        # 绘制消息
        text = self.small_font.render(message, True, self.colors['text'])
        self.screen.blit(text, (x + radius * 2 + 10, y + radius - text.get_height() // 2))

    def draw_gesture_animation(self, gesture_name: str, position: Tuple[int, int], progress: float):
        """
        绘制手势识别动画

        Args:
            gesture_name: 手势名称
            position: 绘制位置 (x, y)
            progress: 动画进度 (0.0 - 1.0)
        """
        x, y = position

        # 绘制圆形背景
        radius = 30
        pygame.draw.circle(self.screen, self.colors['panel'], (x, y), radius)
        pygame.draw.circle(self.screen, self.colors['highlight'], (x, y), radius, 2)

        # 绘制进度环
        if progress > 0:
            end_angle = 2 * 3.14159 * progress
            pygame.draw.arc(self.screen, self.colors['success'],
                            (x - radius, y - radius, radius * 2, radius * 2),
                            -3.14159 / 2, -3.14159 / 2 + end_angle, 4)

        # 绘制手势图标（简化表示）
        if gesture_name == 'pinch':
            # 捏合手势图标
            pygame.draw.circle(self.screen, self.colors['highlight'], (x, y), 10, 2)
            pygame.draw.circle(self.screen, self.colors['highlight'], (x - 8, y - 8), 5, 2)
        elif gesture_name == 'Open_Palm':
            # 张开手掌图标
            pygame.draw.circle(self.screen, self.colors['highlight'], (x, y), 15, 2)
            for i in range(5):
                angle = 3.14159 / 2 + i * 3.14159 / 6
                end_x = x + int(20 * pygame.math.Vector2(pygame.math.Vector2(1, 0).rotate(angle * 180 / 3.14159)).x)
                end_y = y + int(20 * pygame.math.Vector2(pygame.math.Vector2(1, 0).rotate(angle * 180 / 3.14159)).y)
                pygame.draw.line(self.screen, self.colors['highlight'], (x, y), (end_x, end_y), 2)

        # 绘制手势名称
        text = self.small_font.render(gesture_name, True, self.colors['text'])
        self.screen.blit(text, (x - text.get_width() // 2, y + radius + 5))

    def draw_brush(self,color: Tuple[int, int, int],size,x,y,screen,image):

        h,w,_ =image.shape

        pygame.draw.circle(screen, color,(x*w,y*h), size, size)

        return image