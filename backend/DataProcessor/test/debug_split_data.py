"""
调试split返回的数据格式，检查段落数据
"""

import os
import sys
import tempfile
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maxkb_uploader import MaxKBUploader

load_dotenv()

def debug_split_data():
    """调试split返回的数据格式"""
    
    base_url = os.getenv('MAXKB_URL', 'http://localhost:8080')
    username = os.getenv('MAXKB_USERNAME', 'admin')
    password = os.getenv('MAXKB_PASSWORD', 'admin')
    knowledge_id = os.getenv('MAXKB_KNOWLEDGE_ID')
    workspace = "default"
    
    print("=" * 60)
    print("调试Split数据格式")
    print("=" * 60)
    
    uploader = MaxKBUploader(
        base_url=base_url,
        username=username,
        password=password,
        knowledge_id=knowledge_id
    )
    
    if not uploader.login():
        print("[ERROR] 登录失败")
        return
    
    # 创建测试文件
    test_content = """# 测试文档

这是第一段内容。

## 第二部分

这是第二段内容，包含更多文字。

### 第三部分

这是第三段内容，用于测试段落分割功能。
"""
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write(test_content)
        test_file = f.name
    
    print(f"\n测试文件: {test_file}")
    print(f"文件大小: {len(test_content)} 字符")
    
    try:
        # 调用split API
        split_url = f"{base_url}/admin/api/workspace/{workspace}/knowledge/{knowledge_id}/document/split"
        file_name = os.path.basename(test_file)
        
        with open(test_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        files = {'file': (file_name, file_content.encode('utf-8'), 'text/plain')}
        data = {'chunk_size': '500', 'chunk_overlap': '50'}
        
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Referer": f"{base_url}/admin/knowledge/document/upload/{workspace}?id={knowledge_id}",
            "Origin": base_url
        }
        if uploader.token:
            headers["Authorization"] = f"Bearer {uploader.token}"
        
        print(f"\n调用split API...")
        split_response = uploader.session.post(split_url, files=files, data=data, headers=headers, timeout=60)
        split_result = split_response.json()
        
        print(f"\nSplit响应:")
        print(f"  状态码: {split_response.status_code}")
        print(f"  响应code: {split_result.get('code', '未知')}")
        
        if isinstance(split_result, dict) and 'data' in split_result:
            split_data = split_result['data']
            if isinstance(split_data, list) and len(split_data) > 0:
                first_doc = split_data[0]
                print(f"\n文档数据:")
                print(f"  name: {first_doc.get('name', '未知')}")
                print(f"  source_file_id: {first_doc.get('source_file_id', '未知')}")
                print(f"  content类型: {type(first_doc.get('content', []))}")
                print(f"  content数量: {len(first_doc.get('content', []))}")
                
                content_list = first_doc.get('content', [])
                print(f"\n段落详情:")
                for i, para in enumerate(content_list[:5], 1):
                    print(f"  段落 {i}:")
                    if isinstance(para, dict):
                        print(f"    title: {para.get('title', '')}")
                        print(f"    content长度: {len(para.get('content', ''))}")
                        print(f"    content预览: {para.get('content', '')[:100]}...")
                    else:
                        print(f"    类型: {type(para)}")
                        print(f"    内容: {str(para)[:100]}...")
                
                # 打印完整的数据结构用于调试
                print(f"\n完整数据结构:")
                import json
                print(json.dumps(first_doc, ensure_ascii=False, indent=2)[:2000])
        
    except Exception as e:
        print(f"[ERROR] 调试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

if __name__ == '__main__':
    debug_split_data()

