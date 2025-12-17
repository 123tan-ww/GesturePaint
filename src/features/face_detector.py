import threading
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os


class FaceDetector:
    def __init__(self, model_path='/GesturePaint/models/blaze_face_short_range.tflite'):
        self.lock = threading.Lock()  # 线程锁

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.process_result,
            min_detection_confidence=0.5,
            min_suppression_threshold=0.5
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)
        self.last_result = None
        self.timestamp = 0

        # 预加载头像，避免每次读取文件
        self.avatar = None
        # 修正：删除 hasattr(self, 'avatar_path') 检查，直接指定路径
        avatar_path = '/GesturePaint/assets/avatar_sticker/avataaars.png'
        if os.path.exists(avatar_path):
            self.avatar = cv2.imread(avatar_path)
        else:
            print(f"警告: 头像文件不存在: {avatar_path}")


    def process_result(self, result: vision.FaceDetectorResult,
                       output_image: mp.Image, timestamp_ms: int):
        """异步回调函数"""
        with self.lock:  # 线程安全
            self.last_result = result

    def detect_face(self, image):
        """检测人脸"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # 异步检测
        self.face_detector.detect_async(mp_image, self.timestamp)
        self.timestamp += 1

        # 返回检测结果（可能需要等待一小段时间）
        time.sleep(0.01)  # 微小延迟，让回调有机会执行

    def draw_face(self, image, avatar_path=None):
        """绘制检测到的人脸"""
        if self.last_result is None or len(self.last_result.detections) == 0:
            return image

        # 加载头像（如果未预加载）
        if self.avatar is None and avatar_path:
            if os.path.exists(avatar_path):
                self.avatar = cv2.imread(avatar_path)
            else:
                print(f"头像文件不存在: {avatar_path}")
                return image
        elif self.avatar is None:
            return image

        with self.lock:  # 线程安全读取
            detections = self.last_result.detections

        for detection in detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            # 边界检查
            height, width = image.shape[:2]
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = min(w, width - x)
            h = min(h, height - y)

            if w <= 0 or h <= 0:
                continue

            # 调整头像大小
            avatar_resized = cv2.resize(self.avatar, (w, h))

            # 创建掩码（处理透明通道）
            if avatar_resized.shape[2] == 4:
                alpha = avatar_resized[:, :, 3] / 255.0
                for c in range(3):
                    image[y:y + h, x:x + w, c] = (
                            alpha * avatar_resized[:, :, c] +
                            (1 - alpha) * image[y:y + h, x:x + w, c]
                    )
            else:
                image[y:y + h, x:x + w] = avatar_resized

        return image

    def get_face_positions(self):
        """获取所有人脸位置"""
        if self.last_result is None:
            return []

        positions = []
        with self.lock:
            for detection in self.last_result.detections:
                bbox = detection.bounding_box
                positions.append({
                    'x': bbox.origin_x,
                    'y': bbox.origin_y,
                    'width': bbox.width,
                    'height': bbox.height,
                    'confidence': detection.categories[0].score
                })
        return positions

    def release(self):
        """释放资源"""
        self.face_detector.close()


