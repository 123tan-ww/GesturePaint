# download_models.py
import os
from huggingface_hub import snapshot_download

MODELS = {
    "blip-image-captioning-base": "Salesforce/blip-image-captioning-base",
    "sd-controlnet-scribble": "lllyasviel/sd-controlnet-scribble",
    "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
}


def download_all_models():
    """下载所有模型到本地models文件夹"""
    os.makedirs("models", exist_ok=True)

    for model_name, repo_id in MODELS.items():
        print(f"正在下载 {model_name}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=f"models/{model_name}",
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"✓ {model_name} 下载完成")
        except Exception as e:
            print(f"✗ {model_name} 下载失败: {e}")


if __name__ == "__main__":
    download_all_models()
    print("所有模型下载完成！")