"""
调试MaxKB上传功能
检查上传响应和文档列表
"""

import os
import sys
import tempfile
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maxkb_uploader import MaxKBUploader
import requests

load_dotenv()

def debug_upload():
    """调试上传功能"""
    
    print("=" * 60)
    print("MaxKB上传调试")
    print("=" * 60)
    
    base_url = os.getenv('MAXKB_URL', 'http://localhost:8080')
    username = os.getenv('MAXKB_USERNAME', 'admin')
    password = os.getenv('MAXKB_PASSWORD', 'admin')
    knowledge_id = os.getenv('MAXKB_KNOWLEDGE_ID')
    workspace = "default"
    
    if not knowledge_id:
        print("[ERROR] 未配置知识库ID")
        return
    
    print(f"\n配置:")
    print(f"  地址: {base_url}")
    print(f"  知识库ID: {knowledge_id}")
    print(f"  工作空间: {workspace}")
    
    # 创建测试文件
    test_content = "测试文档内容\n这是用于调试的测试文档。"
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
        f.write(test_content)
        test_file = f.name
    
    print(f"\n测试文件: {test_file}")
    
    try:
        # 创建上传器
        uploader = MaxKBUploader(
            base_url=base_url,
            username=username,
            password=password,
            knowledge_id=knowledge_id
        )
        
        # 登录
        print("\n1. 测试登录...")
        if not uploader.login():
            print("[ERROR] 登录失败")
            return
        
        print(f"   Token: {uploader.token[:50] if uploader.token else 'None'}...")
        print(f"   Cookies: {len(uploader.session.cookies)} 个")
        
        # 上传前检查文档列表
        print("\n2. 上传前检查文档列表...")
        try:
            list_url = f"{base_url}/admin/api/workspace/{workspace}/knowledge/{knowledge_id}/document"
            headers = {}
            if uploader.token:
                headers["Authorization"] = f"Bearer {uploader.token}"
            
            list_response = uploader.session.get(list_url, headers=headers, timeout=10)
            print(f"   状态码: {list_response.status_code}")
            if list_response.status_code == 200:
                docs = list_response.json()
                print(f"   当前文档数: {len(docs) if isinstance(docs, list) else '未知'}")
                if isinstance(docs, list) and docs:
                    print(f"   最新文档: {docs[0].get('name', '未知') if isinstance(docs[0], dict) else '未知'}")
        except Exception as e:
            print(f"   获取文档列表失败: {e}")
        
        # 使用uploader的方法上传文件（会自动调用创建文档API）
        print("\n3. 上传文件（使用uploader方法）...")
        print(f"   文件: {test_file}")
        
        success = uploader.upload_text_file(test_file, chunk_size=500)
        
        print(f"\n4. 上传结果:")
        print(f"   成功: {success}")
        
        # 上传后再次检查文档列表
        print("\n5. 上传后检查文档列表...")
        try:
            list_url = f"{base_url}/admin/api/workspace/{workspace}/knowledge/{knowledge_id}/document"
            list_headers = {}
            if uploader.token:
                list_headers["Authorization"] = f"Bearer {uploader.token}"
            
            list_response = uploader.session.get(list_url, headers=list_headers, timeout=10)
            print(f"   状态码: {list_response.status_code}")
            if list_response.status_code == 200:
                result = list_response.json()
                # MaxKB API返回格式: {'code': 200, 'message': '成功', 'data': [...]}
                if isinstance(result, dict) and 'data' in result:
                    docs = result['data']
                else:
                    docs = result if isinstance(result, list) else []
                
                if isinstance(docs, list):
                    print(f"   当前文档数: {len(docs)}")
                    if docs:
                        print(f"   最新文档:")
                        for i, doc in enumerate(docs[:5], 1):
                            if isinstance(doc, dict):
                                name = doc.get('name', '未知')
                                doc_id = doc.get('id', '未知')
                                create_time = doc.get('create_time', '未知')
                                print(f"     {i}. {name}")
                                print(f"        ID: {doc_id}")
                                print(f"        创建时间: {create_time}")
                                print(f"        链接: {base_url}/admin/paragraph/{knowledge_id}/{doc_id}?from=workspace&isShared=false")
                                print()
        except Exception as e:
            print(f"   获取文档列表失败: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"\n[ERROR] 调试过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

if __name__ == '__main__':
    debug_upload()

