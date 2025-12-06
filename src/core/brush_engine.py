class BrushConfig:
    def __init__(self):
        self.color = (0, 0, 0)  # 默认黑色
        self.size = 5  # 默认大小
        self.opacity = 255  # 不透明度


class BrushEngine:
    def __init__(self):
        self.brush = BrushConfig()
        self.colors = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'black': (0, 0, 0)
        }

    def change_color(self, color_name):
        """切换笔刷颜色"""
        if color_name in self.colors:
            self.brush.color = self.colors[color_name]

    def change_size(self, new_size):
        """调整笔刷大小"""
        self.brush.size = max(1, min(new_size, 50))  # 限制大小范围

    def get_current_brush(self):
        return self.brush