"""
MaxKB自动上传功能使用示例
"""

from data_processor import DataProcessor
import os

# 示例1：使用环境变量配置（推荐）
def example_with_env():
    """使用环境变量配置MaxKB"""
    # 确保.env文件已配置好MaxKB相关参数
    processor = DataProcessor(
        json_file_path="microsoft_vscode_text_data_20251128_193435.json",
        enable_maxkb_upload=True  # 启用自动上传
    )
    processor.process_all()


# 示例2：在代码中直接配置
def example_with_config():
    """在代码中直接配置MaxKB参数"""
    processor = DataProcessor(
        json_file_path="microsoft_vscode_text_data_20251128_193435.json",
        enable_maxkb_upload=True,
        maxkb_config={
            'base_url': 'http://localhost:8080',
            'username': 'admin',
            'password': 'admin',
            'knowledge_id': '019ae417-c380-7790-92e6-2fc017ed1652',
            'chunk_size': 500
        }
    )
    processor.process_all()


# 示例3：单独测试上传功能
def example_manual_upload():
    """手动测试上传功能"""
    from maxkb_uploader import MaxKBUploader
    
    uploader = MaxKBUploader(
        base_url="http://localhost:8080",
        username="admin",
        password="admin",
        knowledge_id="019ae417-c380-7790-92e6-2fc017ed1652"
    )
    
    # 方法1：自动登录
    if uploader.login():
        uploader.upload_text_file("text_data_for_training.txt", chunk_size=500)
    
    # 方法2：手动设置token（如果自动登录失败）
    # 从浏览器F12获取token后：
    # uploader.set_token("你的token")
    # uploader.upload_text_file("text_data_for_training.txt", chunk_size=500)


if __name__ == "__main__":
    # 选择要运行的示例
    print("MaxKB自动上传功能示例")
    print("1. 使用环境变量配置")
    print("2. 在代码中直接配置")
    print("3. 手动测试上传")
    
    choice = input("请选择示例 (1/2/3): ")
    
    if choice == "1":
        example_with_env()
    elif choice == "2":
        example_with_config()
    elif choice == "3":
        example_manual_upload()
    else:
        print("无效选择")

