"""
MaxKB上传模块测试
"""

import unittest
import os
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maxkb_uploader import MaxKBUploader


class TestMaxKBUploader(unittest.TestCase):
    """MaxKB上传器测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.base_url = "http://localhost:8080"
        self.username = "admin"
        self.password = "admin"
        self.knowledge_id = "test-knowledge-id-12345"
        self.uploader = MaxKBUploader(
            base_url=self.base_url,
            username=self.username,
            password=self.password,
            knowledge_id=self.knowledge_id
        )
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.uploader.base_url, self.base_url)
        self.assertEqual(self.uploader.username, self.username)
        self.assertEqual(self.uploader.password, self.password)
        self.assertEqual(self.uploader.knowledge_id, self.knowledge_id)
        self.assertIsNone(self.uploader.token)
    
    def test_set_knowledge_id(self):
        """测试设置知识库ID"""
        new_id = "new-knowledge-id"
        self.uploader.set_knowledge_id(new_id)
        self.assertEqual(self.uploader.knowledge_id, new_id)
    
    def test_set_token(self):
        """测试手动设置token"""
        test_token = "test-token-12345"
        self.uploader.set_token(test_token)
        self.assertEqual(self.uploader.token, test_token)
        self.assertIn("Authorization", self.uploader.session.headers)
        self.assertEqual(
            self.uploader.session.headers["Authorization"],
            f"Bearer {test_token}"
        )
    
    @patch('maxkb_uploader.requests.Session.post')
    def test_login_success_with_token(self, mock_post):
        """测试登录成功（返回token）"""
        # 模拟登录响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"token": "test-token-12345"}
        mock_response.cookies = {}
        mock_post.return_value = mock_response
        
        result = self.uploader.login()
        
        self.assertTrue(result)
        self.assertEqual(self.uploader.token, "test-token-12345")
        mock_post.assert_called()
    
    @patch('maxkb_uploader.requests.Session.post')
    def test_login_success_with_cookie(self, mock_post):
        """测试登录成功（使用Cookie）"""
        from requests.cookies import RequestsCookieJar
        
        # 模拟登录响应（无token，有cookie）
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        
        # 创建真实的CookieJar对象
        cookie_jar = RequestsCookieJar()
        cookie_jar.set('session', 'cookie-value')
        mock_response.cookies = cookie_jar
        mock_post.return_value = mock_response
        
        result = self.uploader.login()
        
        self.assertTrue(result)
        self.assertIsNotNone(self.uploader.session.cookies)
    
    @patch('maxkb_uploader.requests.Session.post')
    def test_login_failure(self, mock_post):
        """测试登录失败"""
        # 模拟登录失败
        mock_response = Mock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response
        
        result = self.uploader.login()
        
        self.assertFalse(result)
    
    def test_upload_document_no_login(self):
        """测试未登录时上传失败"""
        # 确保未登录
        self.uploader.token = None
        self.uploader.session.cookies.clear()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            temp_file = f.name
        
        try:
            result = self.uploader.upload_document(temp_file)
            self.assertFalse(result)
        finally:
            os.unlink(temp_file)
    
    def test_upload_document_no_knowledge_id(self):
        """测试未设置知识库ID时上传失败"""
        self.uploader.knowledge_id = None
        self.uploader.token = "test-token"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            temp_file = f.name
        
        try:
            result = self.uploader.upload_document(temp_file)
            self.assertFalse(result)
        finally:
            os.unlink(temp_file)
    
    def test_upload_document_file_not_exist(self):
        """测试文件不存在时上传失败"""
        self.uploader.token = "test-token"
        
        result = self.uploader.upload_document("non_existent_file.txt")
        self.assertFalse(result)
    
    @patch('maxkb_uploader.requests.Session.post')
    def test_upload_document_success(self, mock_post):
        """测试上传成功"""
        # 设置token
        self.uploader.token = "test-token"
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("测试内容\nTest content")
            temp_file = f.name
        
        try:
            # 模拟上传响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"document_count": 10}
            mock_post.return_value = mock_response
            
            result = self.uploader.upload_document(temp_file, chunk_size=500)
            
            self.assertTrue(result)
            mock_post.assert_called_once()
            
            # 检查请求参数
            call_args = mock_post.call_args
            self.assertIn('files', call_args.kwargs)
            self.assertIn('data', call_args.kwargs)
            self.assertEqual(call_args.kwargs['data']['chunk_size'], '500')
            
        finally:
            os.unlink(temp_file)
    
    @patch('maxkb_uploader.requests.Session.post')
    def test_upload_document_failure(self, mock_post):
        """测试上传失败"""
        self.uploader.token = "test-token"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("test content")
            temp_file = f.name
        
        try:
            # 模拟上传失败响应
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Bad Request"
            mock_post.return_value = mock_response
            
            result = self.uploader.upload_document(temp_file)
            
            self.assertFalse(result)
        finally:
            os.unlink(temp_file)
    
    def test_upload_text_file(self):
        """测试upload_text_file方法（便捷方法）"""
        self.uploader.token = "test-token"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
            f.write("test content")
            temp_file = f.name
        
        try:
            with patch.object(self.uploader, 'upload_document') as mock_upload:
                mock_upload.return_value = True
                
                result = self.uploader.upload_text_file(temp_file, chunk_size=500)
                
                self.assertTrue(result)
                # upload_text_file调用upload_document时使用位置参数
                mock_upload.assert_called_once_with(temp_file, 500)
        finally:
            os.unlink(temp_file)


class TestMaxKBUploaderIntegration(unittest.TestCase):
    """MaxKB上传器集成测试（需要真实的MaxKB服务）"""
    
    @unittest.skip("需要真实的MaxKB服务，跳过集成测试")
    def test_real_login(self):
        """真实登录测试（需要MaxKB运行）"""
        uploader = MaxKBUploader(
            base_url="http://localhost:8080",
            username="admin",
            password="admin",
            knowledge_id="test-id"
        )
        result = uploader.login()
        # 根据实际情况判断
        # self.assertTrue(result)  # 如果MaxKB运行且密码正确
    
    @unittest.skip("需要真实的MaxKB服务，跳过集成测试")
    def test_real_upload(self):
        """真实上传测试（需要MaxKB运行）"""
        uploader = MaxKBUploader(
            base_url="http://localhost:8080",
            username="admin",
            password="admin",
            knowledge_id="test-id"
        )
        if uploader.login():
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as f:
                f.write("测试内容")
                temp_file = f.name
            try:
                result = uploader.upload_document(temp_file)
                # 根据实际情况判断
                # self.assertTrue(result)
            finally:
                os.unlink(temp_file)


if __name__ == '__main__':
    # 运行单元测试
    unittest.main(verbosity=2)

