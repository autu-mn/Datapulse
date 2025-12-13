"""
OpenDigger MCP Server 客户端包装器
直接调用 OpenDigger API，提供与 MCP Server 相同的接口
"""
import requests
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict


class OpenDiggerMCPClient:
    """OpenDigger API 客户端（兼容 MCP Server 接口）"""
    
    def __init__(self):
        """初始化客户端"""
        self.base_url = "https://oss.open-digger.cn"
        self.cache = {}
        self.cache_ttl = int(os.getenv('CACHE_TTL_SECONDS', '300'))
    
    def _fetch_metric(self, owner: str, repo: str, metric_name: str, platform: str = 'GitHub') -> Dict[str, Any]:
        """获取单个指标数据"""
        cache_key = f"{platform}:{owner}:{repo}:{metric_name}"
        
        # 检查缓存
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_data
        
        # 调用 API
        platform_lower = platform.lower()
        url = f"{self.base_url}/{platform_lower}/{owner}/{repo}/{metric_name}.json"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # 缓存数据
            self.cache[cache_key] = (data, datetime.now())
            
            return {
                'success': True,
                'data': data,
                'metric': metric_name,
                'repository': f"{owner}/{repo}"
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'metric': metric_name,
                'repository': f"{owner}/{repo}"
            }
    
    def get_metric(self, owner: str, repo: str, metric_name: str, platform: str = 'GitHub') -> Dict[str, Any]:
        """
        获取单个仓库指标
        
        Args:
            owner: 仓库所有者
            repo: 仓库名称
            metric_name: 指标名称
            platform: 平台 (GitHub/Gitee)
            
        Returns:
            指标数据
        """
        return self._fetch_metric(owner, repo, metric_name, platform)
    
    def get_metrics_batch(self, owner: str, repo: str, metric_names: List[str], platform: str = 'GitHub') -> Dict[str, Any]:
        """
        批量获取多个指标
        
        Args:
            owner: 仓库所有者
            repo: 仓库名称
            metric_names: 指标名称列表
            platform: 平台 (GitHub/Gitee)
            
        Returns:
            批量指标数据
        """
        results = []
        for metric_name in metric_names:
            result = self._fetch_metric(owner, repo, metric_name, platform)
            results.append({
                'metric': metric_name,
                'success': result.get('success', False),
                'data': result.get('data') if result.get('success') else None,
                'error': result.get('error') if not result.get('success') else None
            })
        
        successful_count = sum(1 for r in results if r['success'])
        failed_count = len(results) - successful_count
        
        return {
            'success': True,
            'results': results,
            'summary': {
                'total': len(metric_names),
                'successful': successful_count,
                'failed': failed_count,
                'repository': f"{owner}/{repo}",
                'processingTime': datetime.now().isoformat()
            }
        }
    
    def compare_repositories(self, repos: List[Dict[str, str]], metrics: List[str], platform: str = 'GitHub') -> Dict[str, Any]:
        """
        对比多个仓库（增强版 - 添加分析洞察）
        
        Args:
            repos: 仓库列表，格式: [{'owner': 'owner1', 'repo': 'repo1'}, ...]
            metrics: 要对比的指标列表
            platform: 平台 (GitHub/Gitee)
            
        Returns:
            对比分析结果
        """
        # 获取所有数据
        comparison_results = []
        
        for repo_info in repos:
            owner = repo_info.get('owner')
            repo = repo_info.get('repo')
            if not owner or not repo:
                continue
            
            repo_metrics = []
            for metric_name in metrics:
                result = self._fetch_metric(owner, repo, metric_name, platform)
                repo_metrics.append({
                    'metric': metric_name,
                    'data': result.get('data') if result.get('success') else None,
                    'success': result.get('success', False),
                    'error': result.get('error') if not result.get('success') else None
                })
            
            comparison_results.append({
                'repository': f"{owner}/{repo}",
                'platform': platform,
                'metrics': repo_metrics
            })
        
        # 生成分析洞察
        analysis = self._generate_comparison_analysis(comparison_results, metrics)
        
        return {
            'success': True,
            'comparison': comparison_results,
            'analysis': analysis,
            'metadata': {
                'repositoryCount': len(repos),
                'metricsCompared': metrics,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _extract_latest_value(self, data: Any) -> float:
        """从指标数据中提取最新值"""
        if isinstance(data, (int, float)):
            return float(data)
        
        if isinstance(data, dict):
            # 找到所有 YYYY-MM 格式的键
            time_keys = [k for k in data.keys() if isinstance(k, str) and '-' in k and len(k) >= 7 and k.count('-') == 1]
            if time_keys:
                time_keys.sort()
                latest_key = time_keys[-1]
                value = data[latest_key]
                return float(value) if value is not None else 0.0
        
        return 0.0
    
    def _generate_comparison_analysis(self, results: List[Dict], metrics: List[str]) -> Dict[str, Any]:
        """生成对比分析洞察"""
        analysis = {
            'summary': {},
            'winners': {},
            'insights': [],
            'rankings': {},
            'healthScores': {}
        }
        
        # 分析每个指标
        for metric in metrics:
            metric_data = []
            for repo_result in results:
                repo_name = repo_result['repository']
                metric_result = next((m for m in repo_result['metrics'] if m['metric'] == metric), None)
                if metric_result and metric_result['success'] and metric_result['data']:
                    value = self._extract_latest_value(metric_result['data'])
                    if value > 0:
                        metric_data.append({'repo': repo_name, 'value': value})
            
            if metric_data:
                # 排序
                metric_data.sort(key=lambda x: x['value'], reverse=True)
                
                values = [item['value'] for item in metric_data]
                highest = max(values)
                lowest = min(values)
                average = sum(values) / len(values)
                
                winner = metric_data[0]['repo']
                
                analysis['winners'][metric] = winner
                analysis['summary'][metric] = {
                    'highest': round(highest, 2),
                    'average': round(average, 2),
                    'range': [round(lowest, 2), round(highest, 2)],
                    'winner': winner
                }
                
                # 排名
                analysis['rankings'][metric] = [
                    {'repo': item['repo'], 'value': round(item['value'], 2), 'rank': idx + 1}
                    for idx, item in enumerate(metric_data)
                ]
        
        # 计算健康评分（基于所有指标的归一化得分）
        for repo_result in results:
            repo_name = repo_result['repository']
            scores = []
            
            for metric in metrics:
                if metric in analysis['summary']:
                    metric_result = next((m for m in repo_result['metrics'] if m['metric'] == metric), None)
                    if metric_result and metric_result['success'] and metric_result['data']:
                        value = self._extract_latest_value(metric_result['data'])
                        max_value = analysis['summary'][metric]['highest']
                        if max_value > 0:
                            normalized_score = (value / max_value) * 100
                            scores.append(normalized_score)
            
            health_score = sum(scores) / len(scores) if scores else 0
            analysis['healthScores'][repo_name] = round(health_score, 1)
        
        # 生成洞察
        analysis['insights'].append(f"Repository comparison across {len(metrics)} metrics for {len(results)} repositories")
        
        # 找到总体最佳
        if analysis['healthScores']:
            top_performer = max(analysis['healthScores'].items(), key=lambda x: x[1])
            analysis['insights'].append(f"Top performer overall: {top_performer[0]} ({top_performer[1]}% health score)")
        
        # 找到最具竞争力的指标
        if analysis['summary']:
            most_competitive = max(analysis['summary'].items(), 
                                 key=lambda x: (x[1]['highest'] - x[1]['range'][0]))
            analysis['insights'].append(f"Most competitive metric: {most_competitive[0]}")
        
        # 每个指标的赢家
        for metric, winner in analysis['winners'].items():
            if metric in analysis['summary']:
                summary = analysis['summary'][metric]
                margin = summary['highest'] - summary['average']
                dominance = 'dominates' if margin > summary['average'] * 0.5 else 'leads'
                analysis['insights'].append(f"{winner} {dominance} in {metric} with {summary['highest']:.0f}")
        
        return analysis
    
    def analyze_trends(self, owner: str, repo: str, metric_name: str, 
                       start_date: Optional[str] = None, end_date: Optional[str] = None,
                       platform: str = 'GitHub') -> Dict[str, Any]:
        """
        分析趋势（增强版 - 类似 MCP Server 的 analysis.ts）
        
        Args:
            owner: 仓库所有者
            repo: 仓库名称
            metric_name: 指标名称
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            platform: 平台 (GitHub/Gitee)
            
        Returns:
            趋势分析结果
        """
        result = self._fetch_metric(owner, repo, metric_name, platform)
        
        if not result.get('success'):
            return result
        
        raw_data = result.get('data', {})
        
        # 提取时间序列数据（YYYY-MM 或 YYYY-MM-DD 格式）
        time_series = []
        for key, value in raw_data.items():
            if isinstance(key, str) and '-' in key and len(key) >= 7:
                # 只取月度数据（YYYY-MM）
                if key.count('-') == 1:
                    try:
                        time_series.append((key, float(value) if value is not None else 0))
                    except (ValueError, TypeError):
                        continue
        
        # 按日期排序
        time_series.sort(key=lambda x: x[0])
        
        if len(time_series) < 2:
            return {
                'success': True,
                'rawData': raw_data,
                'trendAnalysis': {
                    'dataPoints': len(time_series),
                    'timeRange': {},
                    'values': {'first': 0, 'last': 0, 'peak': 0, 'lowest': 0, 'average': 0, 'median': 0},
                    'trend': {
                        'direction': 'stable',
                        'totalGrowth': 0,
                        'growthRate': '0%',
                        'momentum': 'insufficient_data',
                        'volatility': 'low'
                    },
                    'patterns': {'hasSeasonality': False, 'growthPhases': []}
                },
                'metadata': {
                    'metric': metric_name,
                    'entity': f"{owner}/{repo}",
                    'platform': platform
                }
            }
        
        # 提取数值
        dates = [item[0] for item in time_series]
        values = [item[1] for item in time_series]
        
        # 基础统计
        first_value = values[0]
        last_value = values[-1]
        peak_value = max(values)
        lowest_value = min(values)
        average_value = sum(values) / len(values)
        sorted_values = sorted(values)
        median_value = sorted_values[len(sorted_values) // 2] if len(sorted_values) % 2 == 1 else (sorted_values[len(sorted_values) // 2 - 1] + sorted_values[len(sorted_values) // 2]) / 2
        
        # 增长率
        total_growth = last_value - first_value
        growth_rate = ((last_value - first_value) / first_value * 100) if first_value > 0 else 0
        
        # 趋势方向
        if abs(growth_rate) < 5:
            direction = 'stable'
        elif growth_rate > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # 波动性（标准差 / 平均值）
        variance = sum((v - average_value) ** 2 for v in values) / len(values)
        std_dev = variance ** 0.5
        coef_variation = std_dev / average_value if average_value > 0 else 0
        
        if coef_variation > 0.3:
            volatility = 'high'
            direction = 'volatile'
        elif coef_variation > 0.15:
            volatility = 'medium'
        else:
            volatility = 'low'
        
        # 动量（比较前后半段的增长率）
        midpoint = len(values) // 2
        first_half = values[:midpoint]
        second_half = values[midpoint:]
        
        first_half_growth = ((first_half[-1] - first_half[0]) / first_half[0]) if first_half and first_half[0] > 0 else 0
        second_half_growth = ((second_half[-1] - second_half[0]) / second_half[0]) if second_half and second_half[0] > 0 else 0
        
        momentum_diff = second_half_growth - first_half_growth
        if len(values) < 6:
            momentum = 'insufficient_data'
        elif abs(momentum_diff) < 0.05:
            momentum = 'stable'
        elif momentum_diff > 0:
            momentum = 'accelerating'
        else:
            momentum = 'decelerating'
        
        # 增长阶段检测
        growth_phases = []
        current_phase_start = 0
        current_phase_type = 'stable'
        
        for i in range(1, len(values)):
            prev_val = values[i - 1]
            curr_val = values[i]
            growth = curr_val - prev_val
            growth_rate_local = growth / prev_val if prev_val > 0 else 0
            
            if growth_rate_local > 0.05:
                phase_type = 'growth'
            elif growth_rate_local < -0.05:
                phase_type = 'decline'
            else:
                phase_type = 'stable'
            
            if phase_type != current_phase_type and i - current_phase_start >= 2:
                phase_growth = values[i - 1] - values[current_phase_start]
                growth_phases.append({
                    'phase': current_phase_type,
                    'startDate': dates[current_phase_start],
                    'endDate': dates[i - 1],
                    'growth': round(phase_growth, 2)
                })
                current_phase_start = i - 1
                current_phase_type = phase_type
        
        # 添加最后一个阶段
        if len(values) - current_phase_start >= 2:
            phase_growth = values[-1] - values[current_phase_start]
            growth_phases.append({
                'phase': current_phase_type,
                'startDate': dates[current_phase_start],
                'endDate': dates[-1],
                'growth': round(phase_growth, 2)
            })
        
        return {
            'success': True,
            'rawData': raw_data,
            'trendAnalysis': {
                'dataPoints': len(time_series),
                'timeRange': {
                    'start': dates[0],
                    'end': dates[-1]
                },
                'values': {
                    'first': round(first_value, 2),
                    'last': round(last_value, 2),
                    'peak': round(peak_value, 2),
                    'lowest': round(lowest_value, 2),
                    'average': round(average_value, 2),
                    'median': round(median_value, 2)
                },
                'trend': {
                    'direction': direction,
                    'totalGrowth': round(total_growth, 2),
                    'growthRate': f"{growth_rate:.2f}%",
                    'momentum': momentum,
                    'volatility': volatility
                },
                'patterns': {
                    'hasSeasonality': False,  # 简化版，不检测季节性
                    'growthPhases': growth_phases
                }
            },
            'metadata': {
                'metric': metric_name,
                'entity': f"{owner}/{repo}",
                'platform': platform,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def get_ecosystem_insights(self, owner: str, repo: str, platform: str = 'GitHub') -> Dict[str, Any]:
        """
        获取生态系统洞察
        
        Args:
            owner: 仓库所有者
            repo: 仓库名称
            platform: 平台 (GitHub/Gitee)
            
        Returns:
            生态系统洞察
        """
        # 获取多个关键指标
        key_metrics = ['openrank', 'stars', 'forks', 'contributors', 'activity']
        results = {}
        
        for metric_name in key_metrics:
            result = self._fetch_metric(owner, repo, metric_name, platform)
            if result.get('success'):
                results[metric_name] = result.get('data')
        
        return {
            'success': True,
            'insights': results,
            'repository': f"{owner}/{repo}",
            'metrics_analyzed': len(results)
        }
    
    def server_health(self) -> Dict[str, Any]:
        """
        获取服务器健康状态
        
        Returns:
            服务器健康信息
        """
        import time
        import psutil
        import os
        
        # 获取进程信息
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # 计算运行时间（秒）
        uptime = time.time() - process.create_time()
        
        return {
            'status': 'healthy',
            'version': '1.0.0',
            'uptime': uptime,
            'timestamp': datetime.now().isoformat(),
            'cache': {
                'size': len(self.cache),
                'ttl': self.cache_ttl,
                'entries': list(self.cache.keys())[:5]  # 显示前5个缓存键
            },
            'memory': {
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'percent': process.memory_percent()
            }
        }


# 全局客户端实例
_mcp_client: Optional[OpenDiggerMCPClient] = None


def get_mcp_client() -> OpenDiggerMCPClient:
    """获取 MCP 客户端单例"""
    global _mcp_client
    if _mcp_client is None:
        try:
            _mcp_client = OpenDiggerMCPClient()
        except Exception as e:
            print(f"警告: 无法初始化 MCP 客户端: {e}")
            raise
    return _mcp_client

