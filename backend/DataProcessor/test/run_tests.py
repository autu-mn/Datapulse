"""
运行所有测试

使用方法：
    cd Datapulse/backend/DataProcessor
    python -m test.run_tests
    或
    python test/run_tests.py
"""

import unittest
import sys
import os

# 添加父目录到路径，确保可以导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def run_all_tests():
    """运行所有测试"""
    # 切换到test目录
    original_dir = os.getcwd()
    try:
        os.chdir(current_dir)
        
        # 发现并加载所有测试
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # 添加测试模块（使用完整模块路径）
        suite.addTests(loader.loadTestsFromName('test.test_maxkb_uploader'))
        suite.addTests(loader.loadTestsFromName('test.test_data_processor_maxkb'))
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # 返回测试结果
        return result.wasSuccessful()
    finally:
        os.chdir(original_dir)


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

