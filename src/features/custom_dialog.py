import pygame
import os
from pathlib import Path


class CustomDialog:
    def __init__(self, rect, title, message, font_path=None, font_size=14):
        """
        自定义对话框类

        参数:
            rect: 对话框位置和大小 (pygame.Rect)
            title: 对话框标题
            message: 对话框消息内容
            font_path: 字体文件路径
            font_size: 字体大小
        """
        self.rect = rect
        self.title = title
        self.message = message
        self.visible = False
        self.result = None

        # 颜色定义
        self.colors = {
            'background': (255, 255, 255),
            'border': (0, 0, 0),
            'title_bar': (0, 120, 215),
            'title_text': (255, 255, 255),
            'message_text': (0, 0, 0),
            'button_normal': (200, 200, 200),
            'button_hover': (220, 220, 220),
            'button_text': (0, 0, 0)
        }

        # 加载字体
        self.load_fonts(font_path, font_size)

        # 创建按钮
        self.create_buttons()

        # 文本换行
        self.message_lines = self.wrap_text(message, self.message_font, self.rect.width - 40)

    def load_fonts(self, font_path, font_size):
        """加载字体"""
        try:
            if font_path and os.path.exists(font_path):
                self.title_font = pygame.font.Font(font_path, font_size + 2)  # 标题字体稍大
                self.message_font = pygame.font.Font(font_path, font_size+5)
                self.button_font = pygame.font.Font(font_path, font_size)
            else:
                self.title_font = pygame.font.SysFont(None, font_size + 2)
                self.message_font = pygame.font.SysFont(None, font_size)
                self.button_font = pygame.font.SysFont(None, font_size)
        except Exception as e:
            print(f"字体加载失败: {e}")
            # 使用默认字体
            self.title_font = pygame.font.SysFont(None, font_size + 2)
            self.message_font = pygame.font.SysFont(None, font_size)
            self.button_font = pygame.font.SysFont(None, font_size)

    def create_buttons(self):
        """创建对话框按钮"""
        button_width, button_height = 80, 30
        button_y = self.rect.y + self.rect.height - 40

        # 确认按钮
        self.ok_button_rect = pygame.Rect(
            self.rect.x + self.rect.width - 180,
            button_y,
            button_width,
            button_height
        )

        # 取消按钮
        self.cancel_button_rect = pygame.Rect(
            self.rect.x + self.rect.width - 90,
            button_y,
            button_width,
            button_height
        )

        # 按钮状态
        self.ok_hovered = False
        self.cancel_hovered = False

    def wrap_text(self, text, font, max_width):
        """将文本换行以适应最大宽度"""
        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            # 测试当前行加上新单词后的宽度
            test_line = ' '.join(current_line + [word])
            test_width, _ = font.size(test_line)

            if test_width <= max_width:
                current_line.append(word)
            else:
                # 如果当前行不为空，则添加到行列表中
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        # 添加最后一行
        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def show(self):
        """显示对话框"""
        self.visible = True
        self.result = None

    def hide(self):
        """隐藏对话框"""
        self.visible = False

    def draw(self, surface):
        """绘制对话框"""
        if not self.visible:
            return

        # 绘制对话框背景
        pygame.draw.rect(surface, self.colors['background'], self.rect)
        pygame.draw.rect(surface, self.colors['border'], self.rect, 2)

        # 绘制标题栏
        title_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, 30)
        pygame.draw.rect(surface, self.colors['title_bar'], title_rect)
        pygame.draw.rect(surface, self.colors['border'], title_rect, 1)

        # 绘制标题
        title_surf = self.title_font.render(self.title, True, self.colors['title_text'])
        title_pos = (self.rect.x + 10, self.rect.y + 5)
        surface.blit(title_surf, title_pos)

        # 绘制消息文本
        for i, line in enumerate(self.message_lines):
            msg_surf = self.message_font.render(line, True, self.colors['message_text'])
            msg_pos = (self.rect.x + 20, self.rect.y + 35 + i * 25)
            surface.blit(msg_surf, msg_pos)

        # 绘制按钮
        ok_color = self.colors['button_hover'] if self.ok_hovered else self.colors['button_normal']
        cancel_color = self.colors['button_hover'] if self.cancel_hovered else self.colors['button_normal']

        # 确认按钮
        pygame.draw.rect(surface, ok_color, self.ok_button_rect)
        pygame.draw.rect(surface, self.colors['border'], self.ok_button_rect, 2)
        ok_text = self.button_font.render('确认', True, self.colors['button_text'])
        ok_text_pos = ok_text.get_rect(center=self.ok_button_rect.center)
        surface.blit(ok_text, ok_text_pos)

        # 取消按钮
        pygame.draw.rect(surface, cancel_color, self.cancel_button_rect)
        pygame.draw.rect(surface, self.colors['border'], self.cancel_button_rect, 2)
        cancel_text = self.button_font.render('取消', True, self.colors['button_text'])
        cancel_text_pos = cancel_text.get_rect(center=self.cancel_button_rect.center)
        surface.blit(cancel_text, cancel_text_pos)

    def handle_event(self, event):
        """处理事件"""
        if not self.visible:
            return None

        # 更新按钮悬停状态
        mouse_pos = pygame.mouse.get_pos()
        self.ok_hovered = self.ok_button_rect.collidepoint(mouse_pos)
        self.cancel_hovered = self.cancel_button_rect.collidepoint(mouse_pos)

        # 处理鼠标点击事件
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.ok_button_rect.collidepoint(event.pos):
                self.result = "OK"
                self.hide()
                return "OK"
            elif self.cancel_button_rect.collidepoint(event.pos):
                self.result = "CANCEL"
                self.hide()
                return "CANCEL"

        return None

    def update(self):
        """更新对话框状态（可用于动画等）"""
        # 可以在这里添加动画效果
        pass


class OptionDialog(CustomDialog):
    """支持选项按钮的自定义对话框"""

    def __init__(self, rect, title, message, options, font_path=None, font_size=12, columns=2):
        """
        参数:
            rect: 对话框位置和大小
            title: 对话框标题
            message: 对话框消息
            options: 选项列表
            font_path: 字体路径
            font_size: 字体大小
            columns: 网格布局的列数
        """
        super().__init__(rect, title, message, font_path, font_size)

        self.options = options
        self.selected_option = None
        self.columns = columns
        self.option_rects = []

        # 创建选项按钮
        self.create_option_buttons()

    def create_option_buttons(self):
        """创建选项按钮 - 使用网格布局"""
        option_height = 35
        option_spacing = 10
        option_margin = 20

        # 计算每列的宽度
        column_width = (self.rect.width - 2 * option_margin) // self.columns

        # 计算行数
        rows = (len(self.options) + self.columns - 1) // self.columns

        # 计算起始Y坐标，确保内容在对话框内居中
        total_height = rows * option_height + (rows - 1) * option_spacing
        start_y = self.rect.y + 80 + (self.rect.height - 150 - total_height) // 2

        for i, option in enumerate(self.options):
            # 计算行和列
            row = i // self.columns
            col = i % self.columns

            # 计算位置
            x = self.rect.x + option_margin + col * column_width
            y = start_y + row * (option_height + option_spacing)

            # 创建选项按钮矩形
            option_rect = pygame.Rect(
                x + 5,  # 添加一点内边距
                y,
                column_width - 10,  # 减去内边距
                option_height
            )
            self.option_rects.append(option_rect)

    def draw(self, surface):
        """绘制选项对话框"""
        if not self.visible:
            return

        # 调用父类的绘制方法
        super().draw(surface)

        # 绘制选项按钮
        for i, (option, rect) in enumerate(zip(self.options, self.option_rects)):
            # 判断是否选中
            is_selected = (self.selected_option == i)
            option_color = self.colors['button_hover'] if is_selected else self.colors['button_normal']

            pygame.draw.rect(surface, option_color, rect)
            pygame.draw.rect(surface, self.colors['border'], rect, 2)

            # 绘制选项文本 - 自动换行
            wrapped_text = self.wrap_option_text(option, self.message_font, rect.width - 10)
            for j, line in enumerate(wrapped_text):
                option_surf = self.message_font.render(line, True, self.colors['button_text'])
                option_pos = option_surf.get_rect(center=(rect.centerx, rect.y + 10 + j * 15))
                surface.blit(option_surf, option_pos)

    def wrap_option_text(self, text, font, max_width):
        """将选项文本换行以适应按钮宽度"""
        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            # 测试当前行加上新单词后的宽度
            test_line = ' '.join(current_line + [word])
            test_width, _ = font.size(test_line)

            if test_width <= max_width:
                current_line.append(word)
            else:
                # 如果当前行不为空，则添加到行列表中
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        # 添加最后一行
        if current_line:
            lines.append(' '.join(current_line))

        # 如果行数超过2行，截断并添加省略号
        if len(lines) > 2:
            lines = lines[:2]
            if len(lines[1]) > 3:
                lines[1] = lines[1][:-3] + "..."
            else:
                lines[1] = "..."

        return lines

    def handle_event(self, event):
        """处理选项对话框事件"""
        if not self.visible:
            return None

        # 处理选项点击
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i, rect in enumerate(self.option_rects):
                if rect.collidepoint(event.pos):
                    self.selected_option = i

        # 处理按钮点击
        result = super().handle_event(event)
        if result == "OK" and self.selected_option is not None:
            self.result = self.options[self.selected_option]
        elif result == "OK" and self.selected_option is None:
            # 如果没有选择选项，不关闭对话框
            self.result = None
            return None

        return result
