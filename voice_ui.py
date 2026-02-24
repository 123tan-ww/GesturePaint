import pygame
import math
import random
import time
from typing import List, Dict, Optional, Tuple

class VoiceUIPanel:
    """语音界面面板 - 内部组件"""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        
        # 状态
        self.is_listening = False
        self.is_speaking = False
        
        # 动画效果
        self.animation_time = 0
        self.pulse_phase = 0
        
        # 粒子系统
        self.particles = []
        self.max_particles = 20
        self._init_particles()
        
        # 当前显示的文本
        self.current_text = ""
        self.text_display_time = 0
        self.text_max_display = 3.0  # 3秒
        
        # 命令历史
        self.command_history = []
        self.max_history = 5
        
        # 颜色主题
        self.colors = {
            'bg': (30, 30, 50, 200),
            'border': (80, 120, 200),
            'text': (255, 255, 255),
            'text_secondary': (200, 200, 220),
            'listening': (0, 255, 100),
            'speaking': (255, 200, 0),
            'inactive': (150, 150, 150),
        }
        
        # 字体
        try:
            self.font_title = pygame.font.SysFont('simhei', 20)
            self.font_normal = pygame.font.SysFont('simhei', 16)
            self.font_small = pygame.font.SysFont('simhei', 14)
        except:
            self.font_title = pygame.font.Font(None, 20)
            self.font_normal = pygame.font.Font(None, 16)
            self.font_small = pygame.font.Font(None, 14)
    
    def _init_particles(self):
        """初始化粒子系统"""
        for i in range(self.max_particles):
            self.particles.append({
                'x': random.randint(0, self.width),
                'y': random.randint(0, self.height),
                'size': random.randint(2, 5),
                'speed_x': random.uniform(-0.3, 0.3),
                'speed_y': random.uniform(-0.3, 0.3),
                'phase': random.uniform(0, 2 * math.pi),
                'color': (100, 200, 255)
            })
    
    def update(self, dt: float):
        """更新动画"""
        self.animation_time += dt
        self.pulse_phase = (self.pulse_phase + dt * 5) % (2 * math.pi)
        
        # 更新粒子
        if self.is_listening:
            for p in self.particles:
                p['x'] += p['speed_x']
                p['y'] += p['speed_y'] + math.sin(self.animation_time + p['phase']) * 0.2
                
                # 边界检查
                if p['x'] < 0:
                    p['x'] = self.width
                elif p['x'] > self.width:
                    p['x'] = 0
                
                if p['y'] < 0:
                    p['y'] = self.height
                elif p['y'] > self.height:
                    p['y'] = 0
        
        # 更新文本显示时间
        if self.current_text and self.text_display_time > 0:
            self.text_display_time -= dt
    
    def set_listening(self, listening: bool):
        """设置聆听状态"""
        self.is_listening = listening
    
    def set_speaking(self, speaking: bool):
        """设置说话状态"""
        self.is_speaking = speaking
    
    def add_command(self, text: str, command_type: str = "info"):
        """添加命令到历史"""
        self.current_text = text
        self.text_display_time = self.text_max_display
        
        self.command_history.append({
            'text': text,
            'type': command_type,
            'time': time.strftime("%H:%M:%S")
        })
        
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)
    
    def draw(self, screen: pygame.Surface):
        """绘制面板"""
        # 创建半透明表面
        panel_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # 绘制背景
        pygame.draw.rect(panel_surface, self.colors['bg'],
                        (0, 0, self.width, self.height), border_radius=10)
        
        # 绘制边框
        border_color = self._get_border_color()
        pygame.draw.rect(panel_surface, border_color,
                        (0, 0, self.width, self.height), 2, border_radius=10)
        
        # 绘制标题
        title = "🎤 语音助手"
        if self.is_listening:
            title += " (聆听中...)"
        
        title_surf = self.font_title.render(title, True, self.colors['text'])
        panel_surface.blit(title_surf, (10, 8))
        
        # 绘制状态指示器
        self._draw_status_indicator(panel_surface)
        
        # 绘制当前识别的文本
        self._draw_current_text(panel_surface)
        
        # 绘制粒子效果
        if self.is_listening:
            self._draw_particles(panel_surface)
        
        # 绘制命令历史
        self._draw_command_history(panel_surface)
        
        # 绘制到屏幕
        screen.blit(panel_surface, (self.x, self.y))
    
    def _get_border_color(self) -> Tuple[int, int, int]:
        """获取边框颜色"""
        if self.is_listening:
            # 呼吸效果
            brightness = 0.5 + 0.5 * math.sin(self.animation_time * 4)
            return tuple(int(c * brightness) for c in self.colors['listening'])
        else:
            return self.colors['border']
    
    def _draw_status_indicator(self, surface: pygame.Surface):
        """绘制状态指示器"""
        indicator_x = self.width - 30
        indicator_y = 15
        
        if self.is_listening:
            # 聆听状态
            radius = 6 + 2 * math.sin(self.animation_time * 6)
            color = self.colors['listening']
            
            # 外圈波纹
            for i in range(2):
                r = radius + i * 4
                alpha = 80 - i * 30
                circle_color = (*color, alpha)
                pygame.draw.circle(surface, circle_color,
                                 (indicator_x, indicator_y), int(r), 1)
            
            pygame.draw.circle(surface, color, (indicator_x, indicator_y), int(radius))
        else:
            # 空闲状态
            pygame.draw.circle(surface, self.colors['inactive'],
                             (indicator_x, indicator_y), 5)
    
    def _draw_current_text(self, surface: pygame.Surface):
        """绘制当前识别的文本"""
        if self.current_text and self.text_display_time > 0:
            # 计算透明度
            alpha = int(255 * self.text_display_time / self.text_max_display)
            
            # 文本
            text_surf = self.font_normal.render(f"“{self.current_text}”", True, (255, 255, 255))
            text_surf.set_alpha(alpha)
            
            # 居中显示
            text_rect = text_surf.get_rect(center=(self.width // 2, 45))
            surface.blit(text_surf, text_rect)
    
    def _draw_particles(self, surface: pygame.Surface):
        """绘制粒子"""
        for p in self.particles:
            alpha = 100 + int(100 * math.sin(self.animation_time * 2 + p['phase']))
            color = (*p['color'], alpha)
            
            size = p['size'] * (0.8 + 0.2 * math.sin(self.animation_time * 3 + p['phase']))
            
            pygame.draw.circle(surface, color,
                             (int(p['x']), int(p['y'])), int(size))
    
    def _draw_command_history(self, surface: pygame.Surface):
        """绘制命令历史"""
        y_offset = 70
        
        for cmd in self.command_history[-3:]:
            text = f"{cmd['time']} {cmd['text'][:15]}"
            text_surf = self.font_small.render(text, True, self.colors['text_secondary'])
            surface.blit(text_surf, (15, y_offset))
            y_offset += 18


class VoiceFeedbackOverlay:
    """语音反馈悬浮层"""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        self.feedback_items = []
        self.max_items = 3
        self.item_lifetime = 2.0
        
        try:
            self.font = pygame.font.SysFont('simhei', 18)
        except:
            self.font = pygame.font.Font(None, 18)
    
    def add_feedback(self, text: str, feedback_type: str = "info"):
        """添加反馈信息"""
        self.feedback_items.append({
            'text': text,
            'type': feedback_type,
            'time': self.item_lifetime,
            'y_offset': 0
        })
        
        if len(self.feedback_items) > self.max_items:
            self.feedback_items.pop(0)
    
    def update(self, dt: float):
        """更新反馈"""
        for item in self.feedback_items[:]:
            item['time'] -= dt
            if item['time'] <= 0:
                self.feedback_items.remove(item)
        
        # 计算Y轴偏移
        for i, item in enumerate(self.feedback_items):
            item['y_offset'] = i * 35
    
    def draw(self, screen: pygame.Surface):
        """绘制反馈"""
        base_x = self.screen_width - 350
        base_y = 100
        
        for item in self.feedback_items:
            # 根据类型选择颜色
            if item['type'] == "success":
                bg_color = (0, 100, 0, 200)
            elif item['type'] == "error":
                bg_color = (100, 0, 0, 200)
            elif item['type'] == "warning":
                bg_color = (100, 80, 0, 200)
            else:
                bg_color = (0, 0, 100, 200)
            
            # 计算透明度
            alpha = int(255 * min(1.0, item['time']))
            bg_color = (*bg_color[:3], alpha)
            
            # 创建背景
            item_surface = pygame.Surface((300, 30), pygame.SRCALPHA)
            pygame.draw.rect(item_surface, bg_color,
                           (0, 0, 300, 30), border_radius=5)
            
            # 绘制文本
            text_surf = self.font.render(item['text'], True, (255, 255, 255))
            text_surf.set_alpha(alpha)
            item_surface.blit(text_surf, (10, 6))
            
            # 绘制到屏幕
            screen.blit(item_surface, (base_x, base_y + item['y_offset']))


class VoiceUI:
    """语音界面主类 - 供main.py调用的主要接口"""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # 创建主面板（右下角）
        panel_width = 320
        panel_height = 360
        panel_x = screen_width - panel_width - 20
        panel_y = screen_height - panel_height - 20
        
        # 内部组件
        self._panel = VoiceUIPanel(panel_x, panel_y, panel_width, panel_height)
        self._feedback = VoiceFeedbackOverlay(screen_width, screen_height)
        
        # 全屏闪烁效果
        self.flash_alpha = 0
        self.flash_color = (255, 255, 255)
        self.flash_duration = 0.2
        self.flash_timer = 0
        
        # 是否启用
        self.enabled = True
        
        print("✅ VoiceUI 初始化完成")
    
    def update(self, dt: float):
        """更新UI"""
        if not self.enabled:
            return
        
        self._panel.update(dt)
        self._feedback.update(dt)
        
        # 更新闪烁效果
        if self.flash_timer > 0:
            self.flash_timer -= dt
            self.flash_alpha = int(255 * (self.flash_timer / self.flash_duration))
        else:
            self.flash_alpha = 0
    
    def set_listening(self, listening: bool):
        """设置聆听状态"""
        self._panel.set_listening(listening)
        if listening:
            self._feedback.add_feedback("🎤 语音识别已开启", "success")
        else:
            self._feedback.add_feedback("🔇 语音识别已关闭", "warning")
    
    def set_speaking(self, speaking: bool):
        """设置说话状态"""
        self._panel.set_speaking(speaking)
    
    def on_voice_result(self, text: str, is_final: bool = True):
        """语音识别结果回调"""
        # 判断命令类型
        cmd_type = "info"
        if is_final:
            if any(word in text for word in ["保存", "存"]):
                cmd_type = "save"
                self._feedback.add_feedback(f"✓ 保存命令: {text}", "success")
            elif any(word in text for word in ["清空", "清除"]):
                cmd_type = "clear"
                self._feedback.add_feedback(f"✓ 清空命令: {text}", "success")
            elif any(word in text for word in ["红色", "蓝色", "绿色", "黑色"]):
                cmd_type = "color"
                self._feedback.add_feedback(f"✓ 颜色命令: {text}", "success")
            elif any(word in text for word in ["大", "小", "粗", "细"]):
                cmd_type = "brush"
                self._feedback.add_feedback(f"✓ 笔刷命令: {text}", "success")
            elif any(word in text for word in ["暂停", "停止"]):
                cmd_type = "pause"
                self._feedback.add_feedback(f"✓ 暂停命令: {text}", "success")
            elif any(word in text for word in ["恢复", "继续"]):
                cmd_type = "resume"
                self._feedback.add_feedback(f"✓ 恢复命令: {text}", "success")
            elif any(word in text for word in ["撤销"]):
                cmd_type = "undo"
                self._feedback.add_feedback(f"✓ 撤销命令: {text}", "success")
            else:
                self._feedback.add_feedback(f"✓ {text}", "success")
        else:
            self._feedback.add_feedback(f"🎤 {text}", "info")
        
        self._panel.add_command(text, cmd_type if is_final else "partial")
    
    def trigger_flash(self, color: Tuple[int, int, int] = (255, 255, 255)):
        """触发闪烁效果"""
        self.flash_color = color
        self.flash_timer = self.flash_duration
    
    def draw(self, screen: pygame.Surface):
        """绘制UI"""
        if not self.enabled:
            return
        
        self._panel.draw(screen)
        self._feedback.draw(screen)
        
        # 绘制全屏闪烁
        if self.flash_alpha > 0:
            flash_surface = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
            flash_color = (*self.flash_color, self.flash_alpha)
            flash_surface.fill(flash_color)
            screen.blit(flash_surface, (0, 0))
    
    def toggle(self):
        """切换UI显示"""
        self.enabled = not self.enabled
        status = "开启" if self.enabled else "关闭"
        self._feedback.add_feedback(f"语音界面已{status}", "info")
        return self.enabled