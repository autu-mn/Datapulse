"""
DataProcessor MaxKB集成测试
"""

import unittest
import os
import sys
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processor import DataProcessor


class TestDataProcessorMaxKB(unittest.TestCase):
    """DataProcessor MaxKB集成测试"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时JSON文件
        self.temp_json = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8')
        self.test_data = {
            "repo_info": {
                "full_name": "test/example",
                "name": "example",
                "description": "Test repo"
            },
            "opendigger_metrics": {},
            "readme": {
                "content": "# Test README\n\nThis is a test."
            },
            "issues": [],
            "pull_requests": [],
            "commits": [],
            "releases": []
        }
        json.dump(self.test_data, self.temp_json, ensure_ascii=False)
        self.temp_json.close()
    
    def tearDown(self):
        """测试后清理"""
        if os.path.exists(self.temp_json.name):
            os.unlink(self.temp_json.name)
        # 清理输出目录
        output_dir = self.temp_json.name.replace('.json', '_processed')
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
    
    def test_init_without_maxkb(self):
        """测试不启用MaxKB时的初始化"""
        processor = DataProcessor(
            json_file_path=self.temp_json.name,
            enable_maxkb_upload=False
        )
        self.assertFalse(processor.enable_maxkb_upload)
    
    @patch.dict(os.environ, {
        'MAXKB_URL': 'http://localhost:8080',
        'MAXKB_USERNAME': 'admin',
        'MAXKB_PASSWORD': 'testpass',
        'MAXKB_KNOWLEDGE_ID': 'test-id-12345'
    })
    def test_init_with_maxkb_env(self):
        """测试使用环境变量配置MaxKB"""
        processor = DataProcessor(
            json_file_path=self.temp_json.name,
            enable_maxkb_upload=True
        )
        self.assertTrue(processor.enable_maxkb_upload)
        self.assertEqual(processor.maxkb_config['base_url'], 'http://localhost:8080')
        self.assertEqual(processor.maxkb_config['username'], 'admin')
        self.assertEqual(processor.maxkb_config['password'], 'testpass')
        self.assertEqual(processor.maxkb_config['knowledge_id'], 'test-id-12345')
    
    def test_init_with_maxkb_config(self):
        """测试使用代码配置MaxKB"""
        maxkb_config = {
            'base_url': 'http://localhost:8080',
            'username': 'admin',
            'password': 'testpass',
            'knowledge_id': 'test-id-12345',
            'chunk_size': 1000
        }
        processor = DataProcessor(
            json_file_path=self.temp_json.name,
            enable_maxkb_upload=True,
            maxkb_config=maxkb_config
        )
        self.assertTrue(processor.enable_maxkb_upload)
        self.assertEqual(processor.maxkb_config['chunk_size'], 1000)
    
    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_knowledge_id(self):
        """测试未提供知识库ID时的处理"""
        # 确保环境变量中没有MAXKB_KNOWLEDGE_ID
        processor = DataProcessor(
            json_file_path=self.temp_json.name,
            enable_maxkb_upload=True,
            maxkb_config={
                'base_url': 'http://localhost:8080',
                'username': 'admin',
                'password': 'testpass'
                # 没有knowledge_id
            }
        )
        # 应该自动禁用上传
        self.assertFalse(processor.enable_maxkb_upload)
    
    @patch('data_processor.MaxKBUploader')
    def test_upload_to_maxkb_success(self, mock_uploader_class):
        """测试成功上传到MaxKB"""
        # 模拟上传器
        mock_uploader = Mock()
        mock_uploader.login.return_value = True
        mock_uploader.upload_text_file.return_value = True
        mock_uploader_class.return_value = mock_uploader
        
        processor = DataProcessor(
            json_file_path=self.temp_json.name,
            enable_maxkb_upload=True,
            maxkb_config={
                'base_url': 'http://localhost:8080',
                'username': 'admin',
                'password': 'testpass',
                'knowledge_id': 'test-id-12345'
            }
        )
        
        # 处理数据（会触发上传）
        processor.process_all()
        
        # 验证上传器被调用
        mock_uploader_class.assert_called_once()
        mock_uploader.login.assert_called_once()
        mock_uploader.upload_text_file.assert_called_once()
    
    @patch('data_processor.MaxKBUploader')
    def test_upload_to_maxkb_login_failed(self, mock_uploader_class):
        """测试登录失败时的处理"""
        # 模拟登录失败
        mock_uploader = Mock()
        mock_uploader.login.return_value = False
        mock_uploader_class.return_value = mock_uploader
        
        processor = DataProcessor(
            json_file_path=self.temp_json.name,
            enable_maxkb_upload=True,
            maxkb_config={
                'base_url': 'http://localhost:8080',
                'username': 'admin',
                'password': 'testpass',
                'knowledge_id': 'test-id-12345'
            }
        )
        
        # 处理数据
        processor.process_all()
        
        # 验证登录被调用，但上传未被调用
        mock_uploader.login.assert_called_once()
        mock_uploader.upload_text_file.assert_not_called()
    
    @patch('data_processor.MAXKB_AVAILABLE', False)
    def test_upload_when_module_not_available(self):
        """测试MaxKB模块不可用时的处理"""
        processor = DataProcessor(
            json_file_path=self.temp_json.name,
            enable_maxkb_upload=True,
            maxkb_config={
                'base_url': 'http://localhost:8080',
                'username': 'admin',
                'password': 'testpass',
                'knowledge_id': 'test-id-12345'
            }
        )
        # 应该自动禁用上传
        self.assertFalse(processor.enable_maxkb_upload)


if __name__ == '__main__':
    unittest.main(verbosity=2)

