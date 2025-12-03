"""
测试DataProcessor的真实MaxKB上传功能
需要有一个真实的JSON数据文件
"""

import os
import sys
import json
import tempfile
from dotenv import load_dotenv

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import DataProcessor

# 加载环境变量
load_dotenv()

def create_test_json():
    """创建测试用的JSON文件"""
    test_data = {
        "repo_info": {
            "full_name": "test/real-maxkb-test",
            "name": "real-maxkb-test",
            "description": "真实MaxKB上传测试"
        },
        "opendigger_metrics": {},
        "readme": {
            "content": "# 真实MaxKB上传测试\n\n这是一个测试文档，用于验证MaxKB自动上传功能。"
        },
        "issues": [
            {
                "title": "测试Issue 1",
                "body": "这是第一个测试Issue的内容",
                "state": "open",
                "created_at": "2025-12-01T00:00:00Z"
            }
        ],
        "pull_requests": [],
        "commits": [],
        "releases": []
    }
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
        return f.name

def test_real_processor_upload():
    """测试DataProcessor的真实上传功能"""
    
    print("=" * 60)
    print("DataProcessor 真实MaxKB上传测试")
    print("=" * 60)
    
    # 检查环境变量
    knowledge_id = os.getenv('MAXKB_KNOWLEDGE_ID')
    if not knowledge_id:
        print("\n[ERROR] 错误：未配置知识库ID（MAXKB_KNOWLEDGE_ID）")
        print("  请在.env文件中设置 MAXKB_KNOWLEDGE_ID")
        return False
    
    print(f"\n知识库ID: {knowledge_id}")
    
    # 创建测试JSON文件
    print("\n创建测试数据...")
    test_json = create_test_json()
    print(f"  测试文件: {test_json}")
    
    try:
        # 创建处理器并启用MaxKB上传
        print("\n初始化DataProcessor（启用MaxKB上传）...")
        processor = DataProcessor(
            json_file_path=test_json,
            enable_maxkb_upload=True  # 启用自动上传
        )
        
        # 处理数据（会自动上传）
        print("\n开始处理数据（会自动上传到MaxKB）...")
        processor.process_all()
        
        print("\n[OK] 处理完成！")
        print("  请到MaxKB知识库中查看，应该能看到上传的文档。")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] 测试过程中出错：{str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理测试文件
        if os.path.exists(test_json):
            os.unlink(test_json)
        # 清理输出目录
        output_dir = test_json.replace('.json', '_processed')
        if os.path.exists(output_dir):
            import shutil
            try:
                shutil.rmtree(output_dir)
            except:
                pass

if __name__ == '__main__':
    success = test_real_processor_upload()
    
    print("\n" + "=" * 60)
    if success:
        print("[OK] 测试完成！")
    else:
        print("[ERROR] 测试失败，请检查配置和MaxKB服务状态")
    print("=" * 60)
    
    sys.exit(0 if success else 1)

