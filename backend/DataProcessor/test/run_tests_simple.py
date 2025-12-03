"""
简单测试运行脚本（从DataProcessor目录运行）

使用方法：
    cd Datapulse/backend/DataProcessor
    python test/run_tests_simple.py
"""

import unittest
import sys
import os

# 确保可以导入测试模块
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

if __name__ == '__main__':
    # 发现并运行所有测试
    loader = unittest.TestLoader()
    start_dir = test_dir
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    sys.exit(0 if result.wasSuccessful() else 1)

