"""
上传 GitPulse 模型到 HuggingFace Hub

使用方法:
1. 安装 huggingface_hub: pip install huggingface_hub
2. 登录: huggingface-cli login
3. 运行此脚本: python upload_to_hf.py
"""

import os
from huggingface_hub import HfApi, create_repo, upload_folder

# ============== 配置 ==============
REPO_NAME = "GitPulse"  # 仓库名称
REPO_TYPE = "model"     # 类型: model, dataset, space
PRIVATE = False         # 是否私有
# ==================================


def main():
    print("=" * 60)
    print("GitPulse 模型上传到 HuggingFace Hub")
    print("=" * 60)
    
    # 获取当前目录
    local_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 初始化 API
    api = HfApi()
    
    # 获取用户信息
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"\n✓ 已登录: {username}")
    except Exception as e:
        print(f"\n✗ 未登录! 请先运行: huggingface-cli login")
        print(f"  错误: {e}")
        return
    
    # 完整仓库 ID
    repo_id = f"{username}/{REPO_NAME}"
    print(f"\n目标仓库: {repo_id}")
    
    # 创建仓库（如果不存在）
    try:
        create_repo(
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            private=PRIVATE,
            exist_ok=True
        )
        print(f"✓ 仓库已创建/存在: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"✗ 创建仓库失败: {e}")
        return
    
    # 要上传的文件
    files_to_upload = [
        "README.md",
        "model.py",
        "config.json",
        "gitpulse_weights.pt",
        "evaluation_results.json",
        "requirements.txt"
    ]
    
    print("\n待上传文件:")
    for f in files_to_upload:
        path = os.path.join(local_dir, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            if size > 1024 * 1024:
                print(f"  ✓ {f} ({size / 1024 / 1024:.1f} MB)")
            else:
                print(f"  ✓ {f} ({size / 1024:.1f} KB)")
        else:
            print(f"  ✗ {f} (不存在)")
    
    # 上传文件夹
    print("\n开始上传...")
    try:
        upload_folder(
            folder_path=local_dir,
            repo_id=repo_id,
            repo_type=REPO_TYPE,
            ignore_patterns=["*.pyc", "__pycache__", "upload_to_hf.py", ".git*"]
        )
        print(f"\n✓ 上传成功!")
        print(f"\n模型地址: https://huggingface.co/{repo_id}")
        print(f"使用方法: model = GitPulseModel.from_pretrained('{repo_id}')")
    except Exception as e:
        print(f"\n✗ 上传失败: {e}")


if __name__ == "__main__":
    main()





