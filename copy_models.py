# copy_models.py (放在项目根目录 GesturePaint/)
import os
import shutil
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def copy_model_from_cache(hf_model_name, project_model_dir):
    """
    从HuggingFace缓存复制模型到项目目录
    """
    # HuggingFace缓存路径
    cache_dir = Path("E:/huggingface_cache")  # 您的缓存目录
    cache_model_dir = cache_dir / f"models--{hf_model_name.replace('/', '--')}"

    if not cache_model_dir.exists():
        logger.error(f"在缓存中找不到模型: {hf_model_name}")
        logger.error(f"缓存路径: {cache_model_dir}")
        return False

    # 找到snapshot目录
    snapshots_dir = cache_model_dir / "snapshots"
    if not snapshots_dir.exists():
        logger.error(f"在 {cache_model_dir} 中找不到snapshots目录")
        return False

    # 获取最新的snapshot
    snapshot_dirs = list(snapshots_dir.iterdir())
    if not snapshot_dirs:
        logger.error(f"在 {snapshots_dir} 中找不到snapshot")
        return False

    # 找到最大的snapshot（按时间或数字）
    latest_snapshot = None
    for snapshot in snapshot_dirs:
        if latest_snapshot is None or snapshot.name > latest_snapshot.name:
            latest_snapshot = snapshot

    logger.info(f"找到snapshot: {latest_snapshot}")

    # 确保目标目录存在
    target_dir = Path(project_model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 复制文件
    logger.info(f"复制模型文件从 {latest_snapshot} 到 {target_dir}")

    try:
        # 先清空目标目录（如果存在）
        if target_dir.exists():
            for item in target_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

        # 复制整个目录内容
        for item in latest_snapshot.iterdir():
            dest = target_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        logger.info(f"✓ 成功复制 {hf_model_name} 到 {target_dir}")
        return True
    except Exception as e:
        logger.error(f"复制失败: {e}")
        return False


def main():
    """复制所有需要的模型到项目models目录"""
    # 获取当前脚本目录（项目根目录）
    current_dir = Path(__file__).parent
    print(f"项目根目录: {current_dir}")

    # 创建models目录
    models_dir = current_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # 定义模型映射
    models_to_copy = {
        "Salesforce/blip-image-captioning-base": models_dir / "blip-image-captioning-base",
        "lllyasviel/sd-controlnet-scribble": models_dir / "sd-controlnet-scribble",
        "runwayml/stable-diffusion-v1-5": models_dir / "stable-diffusion-v1-5"
    }

    success_count = 0
    for hf_name, local_path in models_to_copy.items():
        print(f"\n{'=' * 60}")
        print(f"处理模型: {hf_name}")
        print(f"目标路径: {local_path}")

        if copy_model_from_cache(hf_name, local_path):
            success_count += 1

    print(f"\n{'=' * 60}")
    print(f"复制完成！成功复制 {success_count}/{len(models_to_copy)} 个模型")

    if success_count == len(models_to_copy):
        print("✓ 所有模型已成功复制到项目目录")
        print(f"模型目录结构:")
        for model_path in models_dir.iterdir():
            if model_path.is_dir():
                size_mb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024 * 1024)
                print(f"  {model_path.name}: {size_mb:.1f} MB")
    else:
        print("⚠ 部分模型复制失败，请检查日志")

    print(f"\n现在可以运行: python src/features/doodle_to_art_system.py")


if __name__ == "__main__":
    main()