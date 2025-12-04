"""
DeepSeek AI 客户端
用于调用 DeepSeek API 进行智能问答
"""
import os
import requests
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class DeepSeekClient:
    """DeepSeek AI 客户端"""
    
    def __init__(self, api_key: str = None):
        """
        初始化 DeepSeek 客户端
        
        Args:
            api_key: DeepSeek API Key，默认从环境变量 DEEPSEEK_KEY 读取
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_KEY')
        if not self.api_key:
            raise ValueError("未找到 DeepSeek API Key，请设置 DEEPSEEK_KEY 环境变量")
        
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat(self, messages: List[Dict[str, str]], 
             model: str = "deepseek-chat",
             temperature: float = 0.7,
             max_tokens: int = 2000) -> Dict:
        """
        调用 DeepSeek Chat API
        
        Args:
            messages: 消息列表，格式：[{"role": "user", "content": "..."}]
            model: 模型名称
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成token数
        
        Returns:
            API 响应字典
        """
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"[ERROR] DeepSeek API 调用失败: {response.status_code} {response.text}")
                return {"error": f"API调用失败: {response.status_code}"}
        
        except Exception as e:
            print(f"[ERROR] DeepSeek API 调用异常: {str(e)}")
            return {"error": str(e)}
    
    def ask(self, question: str, context: str = None) -> str:
        """
        简单的问答接口
        
        Args:
            question: 用户问题
            context: 上下文信息（可选）
        
        Returns:
            AI 回答
        """
        messages = []
        
        if context:
            messages.append({
                "role": "system",
                "content": f"你是一个GitHub仓库数据分析助手。以下是相关的项目数据：\n\n{context}\n\n请基于这些数据回答用户的问题。"
            })
        else:
            messages.append({
                "role": "system",
                "content": "你是一个GitHub仓库数据分析助手，帮助用户理解项目数据。"
            })
        
        messages.append({
            "role": "user",
            "content": question
        })
        
        result = self.chat(messages)
        
        if "error" in result:
            return f"抱歉，AI调用失败：{result['error']}"
        
        try:
            return result["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return "抱歉，无法解析AI响应。"

