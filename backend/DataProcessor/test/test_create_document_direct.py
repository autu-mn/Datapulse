"""
直接测试创建文档的API，尝试不同的参数格式
"""

import os
import sys
import tempfile
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maxkb_uploader import MaxKBUploader

load_dotenv()

def test_create_document():
    """测试创建文档的不同方式"""
    
    base_url = os.getenv('MAXKB_URL', 'http://localhost:8080')
    username = os.getenv('MAXKB_USERNAME', 'admin')
    password = os.getenv('MAXKB_PASSWORD', 'admin')
    knowledge_id = os.getenv('MAXKB_KNOWLEDGE_ID')
    workspace = "default"
    
    print("=" * 60)
    print("测试创建文档API")
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
    
    # 先调用split获取source_file_id
    print("\n1. 调用split API...")
    test_content = "测试文档内容\n这是用于调试的测试文档。"
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write(test_content)
        test_file = f.name
    
    try:
        # 调用split
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
        
        split_response = uploader.session.post(split_url, files=files, data=data, headers=headers, timeout=60)
        split_result = split_response.json()
        
        print(f"   Split响应: {split_result}")
        
        if isinstance(split_result, dict) and 'data' in split_result:
            split_data = split_result['data']
            if isinstance(split_data, list) and len(split_data) > 0:
                first_doc = split_data[0]
                source_file_id = first_doc.get('source_file_id')
                content_list = first_doc.get('content', [])
                
                print(f"\n2. source_file_id: {source_file_id}")
                print(f"   content数量: {len(content_list)}")
                
                # 尝试不同的创建文档方式
                create_url = f"{base_url}/admin/api/workspace/{workspace}/knowledge/{knowledge_id}/document"
                
                create_headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                if uploader.token:
                    create_headers["Authorization"] = f"Bearer {uploader.token}"
                
                # 方法A: 只传递source_file_id（最简单）
                print("\n3. 方法A: 只传递source_file_id")
                create_data_a = {'source_file_id': source_file_id}
                response_a = uploader.session.post(create_url, json=create_data_a, headers=create_headers, timeout=30)
                result_a = response_a.json()
                print(f"   响应: {result_a}")
                
                if result_a.get('code') == 200:
                    print("   [OK] 方法A成功！")
                    return
                
                # 方法B: 传递name和source_file_id
                print("\n4. 方法B: 传递name和source_file_id")
                create_data_b = {
                    'name': file_name,
                    'source_file_id': source_file_id
                }
                response_b = uploader.session.post(create_url, json=create_data_b, headers=create_headers, timeout=30)
                result_b = response_b.json()
                print(f"   响应: {result_b}")
                
                if result_b.get('code') == 200:
                    print("   [OK] 方法B成功！")
                    return
                
                # 方法C: 检查是否需要先创建段落
                print("\n5. 检查段落API...")
                paragraph_url = f"{base_url}/admin/api/workspace/{workspace}/knowledge/{knowledge_id}/paragraph"
                
                # 尝试创建段落
                for idx, content_item in enumerate(content_list):
                    if isinstance(content_item, dict):
                        para_data = {
                            'document_id': None,  # 可能需要先创建文档
                            'content': content_item.get('content', ''),
                            'title': content_item.get('title', ''),
                            'order': idx
                        }
                        print(f"   段落 {idx}: {para_data.get('content', '')[:50]}...")
                
                print("\n[INFO] 所有方法都尝试了，请检查上面的响应")
                
    except Exception as e:
        print(f"[ERROR] 测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

if __name__ == '__main__':
    test_create_document()

