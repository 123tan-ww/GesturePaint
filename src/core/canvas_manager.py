import pygame


class CanvasManager:
    def __init__(self, width=1200, height=800):
        self.canvas = pygame.Surface((width, height))
        self.background_color = (255, 255, 255)
        self.clear_canvas()
        self.drawing_history = []

    def draw_point(self, x, y, brush_config):
        """在画布上绘制点"""
        pygame.draw.circle(self.canvas, brush_config.color, (x, y), brush_config.size)
        self.drawing_history.append((x, y, brush_config.size))

    def draw_line(self, start_pos, end_pos, brush_config):
        """在画布上绘制线段"""
        pygame.draw.line(self.canvas, brush_config.color, start_pos, end_pos, brush_config.size * 2)
        self.drawing_history.append((start_pos[0], start_pos[1], end_pos[0], end_pos[1], brush_config.size*2))

    def clear_canvas(self):
        """清空画布"""
        self.canvas.fill(self.background_color)
        self.drawing_history = []

    # def save_canvas(self, filename):
    #     """保存画布为图片"""
    #     pygame.image.save(self.canvas, f"assets/saved_drawings/{filename}.png")
    def save_canvas(self, filename):
        """保存画布为图片"""
        try:
            save_path = f"assets/saved_drawings/{filename}.png"
            pygame.image.save(self.canvas, save_path)
            print(f"画布已保存: {save_path}")
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False

    def undo(self):
        if len(self.drawing_history) == 0:
            return
        item=self.drawing_history.pop()
        if len(item) == 3:
            pygame.draw.circle(self.canvas,self.background_color, (item[0], item[1]), item[2])
        else:
            pygame.draw.line(self.canvas, self.background_color, (item[0], item[1]), (item[2], item[3]), item[4])




