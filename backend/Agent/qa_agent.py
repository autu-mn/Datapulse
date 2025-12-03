"""
问答Agent - 基于项目数据提供智能问答
"""
import os
import json
from typing import Dict, List, Optional


class QAAgent:
    """项目数据问答Agent"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化问答Agent
        
        Args:
            data_dir: 数据目录路径
        """
        if data_dir is None:
            # 默认数据目录
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(current_dir, 'DataProcessor', 'data')
        else:
            self.data_dir = data_dir
        
        self.project_cache = {}
    
    def load_project_data(self, project_name: str) -> Optional[Dict]:
        """
        加载项目数据
        
        Args:
            project_name: 项目名称（格式：owner_repo）
        
        Returns:
            项目数据字典，如果不存在返回None
        """
        if project_name in self.project_cache:
            return self.project_cache[project_name]
        
        project_path = os.path.join(self.data_dir, project_name)
        if not os.path.exists(project_path):
            return None
        
        # 查找最新的processed文件夹
        processed_folders = [
            f for f in os.listdir(project_path) 
            if os.path.isdir(os.path.join(project_path, f)) and '_processed' in f
        ]
        
        if not processed_folders:
            return None
        
        # 使用最新的文件夹
        latest_folder = sorted(processed_folders)[-1]
        processed_path = os.path.join(project_path, latest_folder)
        
        data = {}
        
        # 加载处理摘要
        summary_path = os.path.join(processed_path, 'processing_summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                data['summary'] = json.load(f)
        
        # 加载文本数据
        text_path = os.path.join(processed_path, 'text_data_structured.json')
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                data['text_data'] = json.load(f)
        
        # 加载时序数据
        timeseries_path = os.path.join(processed_path, 'timeseries_data.json')
        if os.path.exists(timeseries_path):
            with open(timeseries_path, 'r', encoding='utf-8') as f:
                data['timeseries'] = json.load(f)
        
        self.project_cache[project_name] = data
        return data
    
    def answer_question(self, question: str, project_name: str) -> Dict:
        """
        回答关于项目的问题
        
        Args:
            question: 用户问题
            project_name: 项目名称
        
        Returns:
            包含答案和来源的字典
        """
        data = self.load_project_data(project_name)
        if not data:
            return {
                'answer': f'抱歉，未找到项目 {project_name} 的数据。',
                'sources': [],
                'confidence': 0.0
            }
        
        question_lower = question.lower()
        
        # 项目基本信息问题
        if any(keyword in question_lower for keyword in ['什么', '介绍', '描述', '基本信息', '概况']):
            return self._answer_basic_info(data, question)
        
        # 统计信息问题
        if any(keyword in question_lower for keyword in ['多少', '数量', '统计', '总数']):
            return self._answer_statistics(data, question)
        
        # Issue相关问题
        if 'issue' in question_lower or '问题' in question_lower:
            return self._answer_issues(data, question)
        
        # 时序数据问题
        if any(keyword in question_lower for keyword in ['趋势', '变化', '增长', '下降', '时间']):
            return self._answer_timeseries(data, question)
        
        # 默认回答
        return self._answer_general(data, question)
    
    def _answer_basic_info(self, data: Dict, question: str) -> Dict:
        """回答基本信息问题"""
        summary = data.get('summary', {})
        text_data = data.get('text_data', [])
        
        # 查找repo_info
        repo_info = None
        for doc in text_data:
            if doc.get('type') == 'repo_info':
                repo_info = doc.get('content', '')
                break
        
        answer = "根据项目数据，"
        if repo_info:
            # 提取关键信息
            lines = repo_info.split('\n')
            key_info = []
            for line in lines[:10]:  # 前10行通常包含关键信息
                if ':' in line and any(keyword in line.lower() for keyword in ['仓库', '描述', '语言', 'star', 'fork']):
                    key_info.append(line.strip())
            
            if key_info:
                answer += "\n".join(key_info[:5])
            else:
                answer += repo_info[:200] + "..."
        else:
            answer += f"这是一个开源项目，已处理 {summary.get('text_documents_count', 0)} 个文档。"
        
        return {
            'answer': answer,
            'sources': ['项目基本信息'],
            'confidence': 0.8
        }
    
    def _answer_statistics(self, data: Dict, question: str) -> Dict:
        """回答统计问题"""
        summary = data.get('summary', {})
        
        stats = []
        if '文档' in question or 'document' in question.lower():
            stats.append(f"文档总数: {summary.get('text_documents_count', 0)}")
            by_type = summary.get('text_documents_by_type', {})
            for doc_type, count in by_type.items():
                stats.append(f"  - {doc_type}: {count}")
        
        if '指标' in question or 'metric' in question.lower():
            stats.append(f"时序指标数: {summary.get('timeseries_metrics_count', 0)}")
        
        answer = "项目统计信息：\n" + "\n".join(stats) if stats else "暂无相关统计信息。"
        
        return {
            'answer': answer,
            'sources': ['处理摘要'],
            'confidence': 0.9
        }
    
    def _answer_issues(self, data: Dict, question: str) -> Dict:
        """回答Issue相关问题"""
        text_data = data.get('text_data', [])
        issues = [doc for doc in text_data if doc.get('type') == 'issue']
        
        if not issues:
            return {
                'answer': '该项目暂无Issue数据。',
                'sources': [],
                'confidence': 0.7
            }
        
        # 统计open和closed的issue
        open_count = sum(1 for issue in issues if 'open' in issue.get('content', '').lower())
        closed_count = len(issues) - open_count
        
        answer = f"项目共有 {len(issues)} 个Issue，其中开放 {open_count} 个，已关闭 {closed_count} 个。"
        
        # 如果问最新的issue
        if '最新' in question or '最近' in question:
            latest = issues[0] if issues else None
            if latest:
                title = latest.get('title', '')
                answer += f"\n最新的Issue: {title}"
        
        return {
            'answer': answer,
            'sources': [f'Issue数据（共{len(issues)}条）'],
            'confidence': 0.85
        }
    
    def _answer_timeseries(self, data: Dict, question: str) -> Dict:
        """回答时序数据问题"""
        timeseries = data.get('timeseries', {})
        
        if not timeseries:
            return {
                'answer': '该项目暂无时序数据。',
                'sources': [],
                'confidence': 0.7
            }
        
        # 获取指标列表
        metrics = list(timeseries.keys())[:5]  # 前5个指标
        
        answer = f"项目包含 {len(timeseries)} 个时序指标。"
        if metrics:
            answer += f"\n主要指标包括：{', '.join(metrics)}"
        
        return {
            'answer': answer,
            'sources': ['时序数据'],
            'confidence': 0.8
        }
    
    def _answer_general(self, data: Dict, question: str) -> Dict:
        """通用回答"""
        summary = data.get('summary', {})
        
        answer = f"关于这个项目：\n"
        answer += f"- 已处理 {summary.get('text_documents_count', 0)} 个文档\n"
        answer += f"- 包含 {summary.get('timeseries_metrics_count', 0)} 个时序指标\n"
        answer += "\n您可以询问：\n"
        answer += "- 项目的基本信息\n"
        answer += "- 统计数据\n"
        answer += "- Issue情况\n"
        answer += "- 时序趋势"
        
        return {
            'answer': answer,
            'sources': ['项目数据'],
            'confidence': 0.6
        }
    
    def get_project_summary(self, project_name: str) -> Dict:
        """
        获取项目摘要
        
        Args:
            project_name: 项目名称
        
        Returns:
            项目摘要字典
        """
        data = self.load_project_data(project_name)
        if not data:
            return {
                'exists': False,
                'name': project_name
            }
        
        summary = data.get('summary', {})
        text_data = data.get('text_data', [])
        
        # 提取仓库信息
        repo_info = None
        for doc in text_data:
            if doc.get('type') == 'repo_info':
                content = doc.get('content', '')
                # 提取仓库名称
                for line in content.split('\n'):
                    if '仓库名称:' in line:
                        repo_info = line.split(':', 1)[1].strip()
                        break
                break
        
        return {
            'exists': True,
            'name': project_name,
            'repo': repo_info or project_name,
            'documents_count': summary.get('text_documents_count', 0),
            'metrics_count': summary.get('timeseries_metrics_count', 0),
            'processed_at': summary.get('processed_at', ''),
            'documents_by_type': summary.get('text_documents_by_type', {})
        }
