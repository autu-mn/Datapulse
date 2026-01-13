"""
统一的环境变量加载工具
确保 .env 文件在所有模块中都能正确加载
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 全局标记，避免重复加载
_env_loaded = False

def find_dotenv_file():
    """向上查找 .env 文件"""
    current = Path(__file__).resolve().parent
    for _ in range(6):  # 最多向上查找6层（utils -> backend -> 项目根目录）
        env_file = current / '.env'
        if env_file.exists():
            return str(env_file)
        current = current.parent
    return None

def ensure_env_loaded(silent: bool = False):
    """确保 .env 文件已加载（幂等操作）"""
    global _env_loaded
    
    if _env_loaded:
        return True
    
    # 优先从项目根目录加载（最可靠）
    # backend/utils/env_loader.py -> backend/utils -> backend -> 项目根目录
    project_root = Path(__file__).resolve().parent.parent.parent
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file, override=True)
        _env_loaded = True
        if not silent:
            print(f"[env_loader] 已加载 .env 文件: {env_file}")
        return True
    
    # 如果项目根目录没有，使用 find_dotenv_file 向上查找
    env_path = find_dotenv_file()
    if env_path:
        load_dotenv(env_path, override=True)
        _env_loaded = True
        if not silent:
            print(f"[env_loader] 已加载 .env 文件: {env_path}")
        return True
    
    # 最后尝试从当前工作目录加载
    load_dotenv(override=True)
    _env_loaded = True
    if not silent:
        print(f"[env_loader] 使用默认方式加载 .env（当前目录: {os.getcwd()}）")
    return True

# 模块导入时自动加载
ensure_env_loaded()

