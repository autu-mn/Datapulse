"""
MaxKB真实环境测试脚本
测试真实的登录和上传功能
"""

import os
import sys
import tempfile
from dotenv import load_dotenv

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maxkb_uploader import MaxKBUploader

# 加载环境变量
load_dotenv()

def test_real_maxkb():
    """测试真实的MaxKB登录和上传"""
    
    print("=" * 60)
    print("MaxKB 真实环境测试")
    print("=" * 60)
    
    # 从环境变量读取配置
    base_url = os.getenv('MAXKB_URL', 'http://localhost:8080')
    username = os.getenv('MAXKB_USERNAME', 'admin')
    password = os.getenv('MAXKB_PASSWORD', 'admin')
    knowledge_id = os.getenv('MAXKB_KNOWLEDGE_ID')
    chunk_size = int(os.getenv('MAXKB_CHUNK_SIZE', '500'))
    
    print(f"\n配置信息:")
    print(f"  MaxKB地址: {base_url}")
    print(f"  用户名: {username}")
    print(f"  密码: {'*' * len(password)}")
    print(f"  知识库ID: {knowledge_id}")
    print(f"  分块大小: {chunk_size}")
    
    if not knowledge_id:
        print("\n[ERROR] 错误：未配置知识库ID（MAXKB_KNOWLEDGE_ID）")
        print("  请在.env文件中设置 MAXKB_KNOWLEDGE_ID")
        return False
    
    # 创建测试文件
    print(f"\n创建测试文件...")
    test_content = """# 测试文档

这是一个MaxKB上传功能测试文档。

## 测试内容

1. 测试登录功能
2. 测试文件上传功能
3. 验证文档是否正确保存到知识库

## 测试时间

测试时间：2025-12-03

## 测试结果

如果看到这个文档在MaxKB知识库中，说明上传功能正常工作。
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write(test_content)
        test_file = f.name
    
    print(f"  测试文件: {test_file}")
    
    try:
        # 创建上传器
        print(f"\n初始化MaxKB上传器...")
        uploader = MaxKBUploader(
            base_url=base_url,
            username=username,
            password=password,
            knowledge_id=knowledge_id
        )
        
        # 测试登录
        print(f"\n测试登录...")
        if uploader.login():
            print("[OK] 登录成功！")
        else:
            print("[ERROR] 登录失败！")
            print("  请检查：")
            print("  1. MaxKB服务是否运行")
            print("  2. 用户名和密码是否正确")
            print("  3. MaxKB地址是否正确")
            return False
        
        # 测试上传
        print(f"\n测试文件上传...")
        if uploader.upload_text_file(test_file, chunk_size=chunk_size):
            print("[OK] 文件上传成功！")
            print(f"\n请到MaxKB知识库中查看，应该能看到上传的测试文档。")
            return True
        else:
            print("[ERROR] 文件上传失败！")
            print("  请检查：")
            print("  1. 知识库ID是否正确")
            print("  2. 网络连接是否正常")
            print("  3. MaxKB服务日志")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] 测试过程中出错：{str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理测试文件
        if os.path.exists(test_file):
            os.unlink(test_file)
            print(f"\n已清理测试文件")


if __name__ == '__main__':
    success = test_real_maxkb()
    
    print("\n" + "=" * 60)
    if success:
        print("[OK] 所有测试通过！")
    else:
        print("[ERROR] 测试失败，请检查配置和MaxKB服务状态")
    print("=" * 60)
    
    sys.exit(0 if success else 1)

