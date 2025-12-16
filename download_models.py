# download_models.py
import os
import urllib.request
# 设置HF_ENDPOINT环境变量为国内镜像站(未设置该变量时默认去huggingface官方站点下)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
cpu_count = os.cpu_count()

from huggingface_hub import snapshot_download

MODELS = {
    "blip-image-captioning-base": "Salesforce/blip-image-captioning-base",
    "sd-controlnet-scribble": "lllyasviel/sd-controlnet-scribble",
    "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
}

GESTURE_TASK_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

def download_gesture_recognizer():
    """下载MediaPipe手势识别模型"""
    print("正在下载 gesture_recognizer.task...")
    try:
        os.makedirs("models", exist_ok=True)
        output_path = "models/gesture_recognizer.task"
        if os.path.exists(output_path):
             print(f"✓ gesture_recognizer.task 已存在")
             return

        urllib.request.urlretrieve(GESTURE_TASK_URL, output_path)
        print(f"✓ gesture_recognizer.task 下载完成")
    except Exception as e:
        print(f"✗ gesture_recognizer.task 下载失败: {e}")
        print("请尝试手动下载: " + GESTURE_TASK_URL)
        print("并将其保存为 models/gesture_recognizer.task")


def download_all_models():
    """下载所有模型到本地models文件夹"""
    os.makedirs("models", exist_ok=True)
    
    # 下载手势识别模型
    download_gesture_recognizer()
    
    print(f"正在使用镜像站点: {os.environ.get('HF_ENDPOINT')}")

    for model_name, repo_id in MODELS.items():
        print(f"正在下载 {model_name}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=f"models/{model_name}",
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers= cpu_count  # 增加并发下载数
            )
            print(f"✓ {model_name} 下载完成")
        except Exception as e:
            print(f"✗ {model_name} 下载失败: {e}")
            print("建议：请检查网络连接，或者尝试手动下载。")


if __name__ == "__main__":
    download_all_models()
    print("所有模型下载完成！")