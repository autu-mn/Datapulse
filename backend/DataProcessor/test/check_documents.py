"""
检查MaxKB知识库中的文档列表
"""

import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maxkb_uploader import MaxKBUploader

load_dotenv()

def check_documents():
    """检查知识库中的文档"""
    
    base_url = os.getenv('MAXKB_URL', 'http://localhost:8080')
    username = os.getenv('MAXKB_USERNAME', 'admin')
    password = os.getenv('MAXKB_PASSWORD', 'admin')
    knowledge_id = os.getenv('MAXKB_KNOWLEDGE_ID')
    workspace = "default"
    
    if not knowledge_id:
        print("[ERROR] 未配置知识库ID")
        return
    
    print("=" * 60)
    print("检查MaxKB知识库文档")
    print("=" * 60)
    print(f"知识库ID: {knowledge_id}")
    print(f"工作空间: {workspace}\n")
    
    uploader = MaxKBUploader(
        base_url=base_url,
        username=username,
        password=password,
        knowledge_id=knowledge_id
    )
    
    if not uploader.login():
        print("[ERROR] 登录失败")
        return
    
    # 检查文档列表
    print("1. 检查文档列表...")
    try:
        list_url = f"{base_url}/admin/api/workspace/{workspace}/knowledge/{knowledge_id}/document"
        headers = {}
        if uploader.token:
            headers["Authorization"] = f"Bearer {uploader.token}"
        
        response = uploader.session.get(list_url, headers=headers, timeout=10)
        print(f"   状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            # MaxKB API返回格式是 {'code': 200, 'message': '成功', 'data': [...]}
            if isinstance(result, dict) and 'data' in result:
                docs = result['data']
            else:
                docs = result if isinstance(result, list) else []
            
            if isinstance(docs, list):
                print(f"   文档总数: {len(docs)}")
                if docs:
                    print(f"\n   文档列表:")
                    for i, doc in enumerate(docs[:10], 1):
                        if isinstance(doc, dict):
                            name = doc.get('name', '未知')
                            doc_id = doc.get('id', '未知')
                            create_time = doc.get('create_time', doc.get('created_at', '未知'))
                            print(f"     {i}. {name}")
                            print(f"        ID: {doc_id}")
                            print(f"        创建时间: {create_time}")
                            print()
                else:
                    print("   [WARN] 文档列表为空")
            else:
                print(f"   响应格式: {type(result)}")
                print(f"   响应内容: {result}")
        else:
            print(f"   [ERROR] 获取文档列表失败")
            print(f"   响应: {response.text[:200]}")
    except Exception as e:
        print(f"   [ERROR] 检查文档列表出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 检查段落列表
    print("\n2. 检查段落列表...")
    try:
        paragraph_url = f"{base_url}/admin/api/workspace/{workspace}/knowledge/{knowledge_id}/paragraph"
        headers = {}
        if uploader.token:
            headers["Authorization"] = f"Bearer {uploader.token}"
        
        response = uploader.session.get(paragraph_url, headers=headers, timeout=10)
        print(f"   状态码: {response.status_code}")
        
        if response.status_code == 200:
            paragraphs = response.json()
            if isinstance(paragraphs, list):
                print(f"   段落总数: {len(paragraphs)}")
                if paragraphs:
                    print(f"\n   最新段落 (前5个):")
                    for i, para in enumerate(paragraphs[:5], 1):
                        if isinstance(para, dict):
                            content = para.get('content', '')[:100]
                            para_id = para.get('id', '未知')
                            doc_id = para.get('document_id', '未知')
                            print(f"     {i}. 段落ID: {para_id}")
                            print(f"        文档ID: {doc_id}")
                            print(f"        内容: {content}...")
                            print()
            else:
                print(f"   响应格式: {type(paragraphs)}")
                print(f"   响应内容: {paragraphs}")
        else:
            print(f"   [ERROR] 获取段落列表失败")
            print(f"   响应: {response.text[:200]}")
    except Exception as e:
        print(f"   [ERROR] 检查段落列表出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_documents()

