"""
数据完整性检测和断点续传工具
检测已爬取数据的完整性，识别缺失的部分，支持从断点继续爬取
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class DataCompletenessChecker:
    """数据完整性检测器"""
    
    def __init__(self, data_dir: str = None):
        """
        初始化检测器
        
        Args:
            data_dir: 数据目录路径，如果为None则自动检测
        """
        if data_dir is None:
            # 自动检测数据目录
            current_dir = os.path.dirname(__file__)
            data_dir_new = os.path.join(current_dir, 'data')
            data_dir_old = os.path.join(os.path.dirname(current_dir), 'Data')
            
            if os.path.exists(data_dir_new):
                self.data_dir = data_dir_new
            elif os.path.exists(data_dir_old):
                self.data_dir = data_dir_old
            else:
                self.data_dir = data_dir_new
        else:
            self.data_dir = data_dir
    
    def check_project_completeness(self, owner: str, repo: str) -> Dict:
        """
        检查项目的完整性
        
        Returns:
            {
                'is_complete': bool,  # 数据是否完整
                'completeness': float,  # 完整度百分比 (0-100)
                'missing_parts': List[str],  # 缺失的部分
                'existing_months': List[str],  # 已爬取的月份列表
                'missing_months': List[str],  # 缺失的月份列表
                'data_path': str,  # 数据路径
                'has_metrics': bool,  # 是否有指标数据
                'has_text': bool,  # 是否有文本数据
                'has_timeseries': bool,  # 是否有时序数据
                'total_months': int,  # 总月份数
                'crawled_months': int,  # 已爬取月份数
            }
        """
        project_name = f"{owner}_{repo}"
        project_dir = os.path.join(self.data_dir, project_name)
        
        result = {
            'is_complete': False,
            'completeness': 0.0,
            'missing_parts': [],
            'existing_months': [],
            'missing_months': [],
            'data_path': None,
            'has_metrics': False,
            'has_text': False,
            'has_timeseries': False,
            'total_months': 0,
            'crawled_months': 0,
        }
        
        if not os.path.exists(project_dir):
            result['missing_parts'].append('项目目录不存在')
            return result
        
        # 查找处理后的数据文件夹
        try:
            processed_folders = [
                f for f in os.listdir(project_dir)
                if os.path.isdir(os.path.join(project_dir, f)) and 
                ('monthly_data_' in f or '_processed' in f)
            ]
        except Exception as e:
            result['missing_parts'].append(f'无法读取项目目录: {e}')
            return result
        
        if not processed_folders:
            result['missing_parts'].append('没有找到处理后的数据文件夹')
            return result
        
        # 按时间戳排序，取最新的
        processed_folders.sort(reverse=True)
        latest_folder = processed_folders[0]
        folder_path = os.path.join(project_dir, latest_folder)
        result['data_path'] = folder_path
        
        # 检查指标数据（OpenDigger数据）
        timeseries_file = os.path.join(folder_path, 'timeseries_data.json')
        timeseries_for_model_dir = os.path.join(folder_path, 'timeseries_for_model')
        
        if os.path.exists(timeseries_file) and os.path.getsize(timeseries_file) > 0:
            result['has_metrics'] = True
            result['has_timeseries'] = True
        elif os.path.exists(timeseries_for_model_dir):
            try:
                json_files = [f for f in os.listdir(timeseries_for_model_dir) 
                             if f.endswith('.json') and f != 'all_months.json']
                if len(json_files) > 0:
                    # 检查文件是否有内容
                    for json_file in json_files[:3]:
                        file_path = os.path.join(timeseries_for_model_dir, json_file)
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            result['has_metrics'] = True
                            result['has_timeseries'] = True
                            break
            except Exception:
                pass
        
        if not result['has_metrics']:
            result['missing_parts'].append('指标数据缺失')
        
        # 检查文本数据
        metadata_file = os.path.join(folder_path, 'metadata.json')
        project_summary_file = os.path.join(folder_path, 'project_summary.json')
        maxkb_dir = os.path.join(folder_path, 'maxkb')
        
        if (os.path.exists(metadata_file) or 
            os.path.exists(project_summary_file) or 
            (os.path.exists(maxkb_dir) and os.listdir(maxkb_dir))):
            result['has_text'] = True
        else:
            result['missing_parts'].append('文本数据缺失')
        
        # 检查月份数据完整性
        if result['has_timeseries']:
            # 从 timeseries_for_model 目录读取月份列表
            if os.path.exists(timeseries_for_model_dir):
                try:
                    json_files = [f for f in os.listdir(timeseries_for_model_dir) 
                                 if f.endswith('.json') and f != 'all_months.json']
                    # 提取月份（格式：YYYY-MM.json）
                    existing_months = []
                    for json_file in json_files:
                        month = json_file.replace('.json', '')
                        if len(month) == 7 and month[4] == '-':  # YYYY-MM 格式
                            existing_months.append(month)
                    
                    existing_months.sort()
                    result['existing_months'] = existing_months
                    result['crawled_months'] = len(existing_months)
                    
                    # 尝试从 all_months.json 获取总月份数
                    all_months_file = os.path.join(timeseries_for_model_dir, 'all_months.json')
                    if os.path.exists(all_months_file):
                        try:
                            with open(all_months_file, 'r', encoding='utf-8') as f:
                                all_months_data = json.load(f)
                                if isinstance(all_months_data, dict):
                                    all_months = sorted(all_months_data.keys())
                                    result['total_months'] = len(all_months)
                                    
                                    # 找出缺失的月份
                                    missing_months = [m for m in all_months if m not in existing_months]
                                    result['missing_months'] = missing_months
                        except Exception:
                            pass
                except Exception as e:
                    result['missing_parts'].append(f'无法读取月份数据: {e}')
            
            # 如果无法从文件获取总月份数，尝试从 timeseries_data.json 获取
            if result['total_months'] == 0 and os.path.exists(timeseries_file):
                try:
                    with open(timeseries_file, 'r', encoding='utf-8') as f:
                        timeseries_data = json.load(f)
                        if isinstance(timeseries_data, dict):
                            # 尝试从任意一个指标获取月份列表
                            for metric_name, metric_data in timeseries_data.items():
                                if isinstance(metric_data, dict) and 'values' in metric_data:
                                    values = metric_data['values']
                                    if isinstance(values, list) and len(values) > 0:
                                        # 提取所有月份
                                        all_months = []
                                        for item in values:
                                            if isinstance(item, dict) and 'date' in item:
                                                date_str = item['date']
                                                if len(date_str) >= 7:
                                                    month = date_str[:7]  # YYYY-MM
                                                    if month not in all_months:
                                                        all_months.append(month)
                                        
                                        all_months.sort()
                                        result['total_months'] = len(all_months)
                                        
                                        # 找出缺失的月份
                                        missing_months = [m for m in all_months if m not in result['existing_months']]
                                        result['missing_months'] = missing_months
                                        break
                except Exception:
                    pass
        
        # 计算完整度
        completeness_score = 0.0
        total_checks = 3  # 指标数据、文本数据、月份数据
        
        if result['has_metrics']:
            completeness_score += 0.3
        if result['has_text']:
            completeness_score += 0.2
        
        if result['total_months'] > 0:
            month_completeness = result['crawled_months'] / result['total_months']
            completeness_score += 0.5 * month_completeness
        elif result['crawled_months'] > 0:
            # 如果无法确定总月份数，但已有部分月份数据，给予部分分数
            completeness_score += 0.3
        
        result['completeness'] = round(completeness_score * 100, 1)
        result['is_complete'] = result['completeness'] >= 95.0 and len(result['missing_parts']) == 0
        
        return result
    
    def get_resume_info(self, owner: str, repo: str) -> Dict:
        """
        获取续传信息（需要继续爬取的部分）
        
        Returns:
            {
                'needs_resume': bool,  # 是否需要续传
                'resume_type': str,  # 续传类型: 'full', 'months', 'text', 'metrics'
                'missing_months': List[str],  # 缺失的月份列表
                'data_path': str,  # 现有数据路径
            }
        """
        completeness = self.check_project_completeness(owner, repo)
        
        result = {
            'needs_resume': False,
            'resume_type': 'full',  # full, months, text, metrics
            'missing_months': [],
            'data_path': completeness.get('data_path'),
        }
        
        if completeness['completeness'] >= 95.0:
            return result  # 数据完整，无需续传
        
        # 判断续传类型
        if not completeness['has_metrics'] and not completeness['has_text']:
            result['needs_resume'] = True
            result['resume_type'] = 'full'  # 完全重新爬取
        elif not completeness['has_metrics']:
            result['needs_resume'] = True
            result['resume_type'] = 'metrics'  # 只爬取指标
        elif not completeness['has_text']:
            result['needs_resume'] = True
            result['resume_type'] = 'text'  # 只爬取文本
        elif completeness['missing_months']:
            result['needs_resume'] = True
            result['resume_type'] = 'months'  # 只爬取缺失的月份
            result['missing_months'] = completeness['missing_months']
        
        return result

