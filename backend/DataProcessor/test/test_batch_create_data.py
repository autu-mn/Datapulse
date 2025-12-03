"""
测试batch_create API的数据格式
检查发送的数据和段落创建情况
"""

import os
import sys
import tempfile
import json
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maxkb_uploader import MaxKBUploader

load_dotenv()

def test_batch_create():
    """测试batch_create API的数据格式"""
    
    base_url = os.getenv('MAXKB_URL', 'http://localhost:8080')
    username = os.getenv('MAXKB_USERNAME', 'admin')
    password = os.getenv('MAXKB_PASSWORD', 'admin')
    knowledge_id = os.getenv('MAXKB_KNOWLEDGE_ID')
    workspace = "default"
    
    print("=" * 60)
    print("测试batch_create数据格式")
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
    test_content = """# 测试文档标题

这是第一段内容，包含一些文字用于测试。

## 第二部分

这是第二段内容，包含更多文字用于测试段落分割功能。

### 第三部分

这是第三段内容，用于测试段落数据是否正确传递到MaxKB。
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
        
        print(f"\n1. 调用split API...")
        split_response = uploader.session.post(split_url, files=files, data=data, headers=headers, timeout=60)
        split_result = split_response.json()
        
        print(f"   Split响应code: {split_result.get('code', '未知')}")
        
        if isinstance(split_result, dict) and 'data' in split_result:
            split_data = split_result['data']
            if isinstance(split_data, list) and len(split_data) > 0:
                first_doc = split_data[0]
                source_file_id = first_doc.get('source_file_id')
                content_list = first_doc.get('content', [])
                
                print(f"\n2. Split返回的数据:")
                print(f"   source_file_id: {source_file_id}")
                print(f"   content段落数: {len(content_list)}")
                
                if content_list:
                    total_chars = sum(len(p.get('content', '')) if isinstance(p, dict) else 0 for p in content_list)
                    print(f"   总字符数: {total_chars}")
                    print(f"\n   段落详情:")
                    for i, para in enumerate(content_list[:3], 1):
                        if isinstance(para, dict):
                            para_content = para.get('content', '')
                            print(f"     段落{i}: {len(para_content)} 字符")
                            print(f"       内容预览: {para_content[:80]}...")
                
                # 准备batch_create数据
                print(f"\n3. 准备batch_create数据...")
                first_doc_copy = first_doc.copy()
                first_doc_copy['name'] = file_name
                create_data = [first_doc_copy]
                
                print(f"   文档名称: {first_doc_copy.get('name')}")
                print(f"   content段落数: {len(first_doc_copy.get('content', []))}")
                
                # 打印完整的数据结构（前1000字符）
                data_json = json.dumps(create_data, ensure_ascii=False, indent=2)
                print(f"\n4. 发送给batch_create的数据结构（前1000字符）:")
                print(data_json[:1000])
                if len(data_json) > 1000:
                    print(f"   ... (总共 {len(data_json)} 字符)")
                
                # 调用batch_create
                create_url = f"{base_url}/admin/api/workspace/{workspace}/knowledge/{knowledge_id}/document/batch_create"
                create_headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
                if uploader.token:
                    create_headers["Authorization"] = f"Bearer {uploader.token}"
                
                print(f"\n5. 调用batch_create API...")
                create_response = uploader.session.put(
                    create_url,
                    json=create_data,
                    headers=create_headers,
                    timeout=60
                )
                
                create_result = create_response.json()
                print(f"   响应状态码: {create_response.status_code}")
                print(f"   响应code: {create_result.get('code', '未知')}")
                print(f"   响应message: {create_result.get('message', '未知')}")
                
                if create_result.get('code') == 200:
                    print(f"\n6. 文档创建成功，检查段落...")
                    import time
                    time.sleep(5)  # 等待段落处理
                    
                    # 检查段落列表
                    paragraph_url = f"{base_url}/admin/api/workspace/{workspace}/knowledge/{knowledge_id}/paragraph"
                    para_headers = {}
                    if uploader.token:
                        para_headers["Authorization"] = f"Bearer {uploader.token}"
                    
                    para_response = uploader.session.get(paragraph_url, headers=para_headers, timeout=10)
                    print(f"   段落API状态码: {para_response.status_code}")
                    if para_response.status_code == 200:
                        para_result = para_response.json()
                        print(f"   段落API响应类型: {type(para_result)}")
                        print(f"   段落API响应内容（前500字符）: {str(para_result)[:500]}")
                        
                        if isinstance(para_result, dict) and 'data' in para_result:
                            paragraphs = para_result['data']
                        elif isinstance(para_result, dict) and 'code' in para_result:
                            # MaxKB可能返回 {'code': 200, 'data': [...]}
                            paragraphs = para_result.get('data', [])
                        else:
                            paragraphs = para_result if isinstance(para_result, list) else []
                        
                        if isinstance(paragraphs, list):
                            print(f"   段落总数: {len(paragraphs)}")
                            # 查找刚创建的文档的段落
                            doc_id = None
                            if 'data' in create_result:
                                doc_data = create_result['data']
                                if isinstance(doc_data, dict):
                                    doc_id = doc_data.get('id')
                                elif isinstance(doc_data, list) and len(doc_data) > 0:
                                    doc_id = doc_data[0].get('id')
                            
                            if doc_id:
                                print(f"   文档ID: {doc_id}")
                                doc_paragraphs = [p for p in paragraphs if isinstance(p, dict) and p.get('document_id') == doc_id]
                                print(f"   该文档的段落数: {len(doc_paragraphs)}")
                                if doc_paragraphs:
                                    print(f"\n   段落内容:")
                                    for i, para in enumerate(doc_paragraphs[:3], 1):
                                        para_content = para.get('content', '')
                                        print(f"     段落{i}: {len(para_content)} 字符")
                                        print(f"       内容: {para_content[:100]}...")
                                else:
                                    print(f"   [WARN] 该文档没有段落！")
                                    # 打印所有段落看看格式
                                    if paragraphs:
                                        print(f"\n   所有段落（前3个）:")
                                        for i, para in enumerate(paragraphs[:3], 1):
                                            print(f"     段落{i}: {para}")
                            else:
                                print(f"   [WARN] 无法获取文档ID")
                        else:
                            print(f"   [WARN] 段落数据格式不正确: {type(paragraphs)}")
                    else:
                        print(f"   [ERROR] 获取段落列表失败: {para_response.text[:200]}")
        
    except Exception as e:
        print(f"[ERROR] 测试出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

if __name__ == '__main__':
    test_batch_create()

