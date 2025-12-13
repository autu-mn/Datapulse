"""
使用GitHub API补充OpenDigger缺失的指标
优先级：OpenDigger > GitHub API > 0填充
"""

import os
import requests
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dotenv import load_dotenv
import time

load_dotenv()


class GitHubMetricsSupplement:
    """使用GitHub API补充缺失的指标"""
    
    def __init__(self):
        self.base_url = "https://api.github.com"
        
        # 支持多Token轮换（支持GITHUB_TOKEN和GITHUB_TOKEN_1到GITHUB_TOKEN_6）
        self.tokens = []
        self.current_token_index = 0
        
        # 加载主token
        token = os.getenv('GITHUB_TOKEN') or os.getenv('github_token')
        if token:
            self.tokens.append(token)
        
        # 加载GITHUB_TOKEN_1到GITHUB_TOKEN_6
        for i in range(1, 7):
            token_key = f'GITHUB_TOKEN_{i}'
            token_value = (os.getenv(token_key) or 
                          os.getenv(token_key.replace('GITHUB_TOKEN', 'GitHub_TOKEN')) or
                          os.getenv(token_key.lower()))
            if token_value:
                self.tokens.append(token_value)
        
        if not self.tokens:
            raise ValueError("未找到 GITHUB_TOKEN，请在 .env 文件中配置")
        
        self.token = self.tokens[0]
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
    
    def switch_token(self):
        """切换到下一个Token"""
        if len(self.tokens) > 1:
            self.current_token_index = (self.current_token_index + 1) % len(self.tokens)
            self.token = self.tokens[self.current_token_index]
            self.headers['Authorization'] = f'token {self.token}'
    
    def _safe_request(self, url, params=None, max_retries=3):
        """安全的API请求"""
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                if response.status_code == 200:
                    return response
                elif response.status_code == 403:
                    if len(self.tokens) > 1:
                        self.switch_token()
                        continue
                    else:
                        time.sleep(60)
                        continue
                elif response.status_code == 404:
                    return None
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    return None
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
        return None
    
    def get_commits_count_by_month(self, owner: str, repo: str, month: str) -> int:
        """
        获取指定月份的提交数
        month格式: 'YYYY-MM'
        """
        year, month_num = map(int, month.split('-'))
        start_date = datetime(year, month_num, 1, tzinfo=timezone.utc)
        if month_num == 12:
            end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end_date = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)
        
        url = f"{self.base_url}/repos/{owner}/{repo}/commits"
        params = {
            'since': start_date.isoformat(),
            'until': end_date.isoformat(),
            'per_page': 1  # 只需要总数，不需要详细数据
        }
        
        response = self._safe_request(url, params)
        if not response:
            return 0
        
        # 从Link头获取总数（如果支持）
        link_header = response.headers.get('Link', '')
        if 'rel="last"' in link_header:
            # 提取最后一页的页码
            import re
            match = re.search(r'page=(\d+)>; rel="last"', link_header)
            if match:
                last_page = int(match.group(1))
                # 获取最后一页的数据来计算总数
                params['per_page'] = 100
                params['page'] = last_page
                last_response = self._safe_request(url, params)
                if last_response:
                    last_data = last_response.json()
                    return (last_page - 1) * 100 + len(last_data)
        
        # 如果没有Link头，遍历所有页面（但限制最大页数）
        count = 0
        page = 1
        max_pages = 10  # 限制最多10页，避免太慢
        
        while page <= max_pages:
            params['page'] = page
            params['per_page'] = 100
            response = self._safe_request(url, params)
            if not response:
                break
            
            data = response.json()
            if not data:
                break
            
            count += len(data)
            if len(data) < 100:
                break
            
            page += 1
        
        return count
    
    def get_pr_declined_count_by_month(self, owner: str, repo: str, month: str) -> int:
        """
        获取指定月份被拒绝的PR数（closed但未merged）
        month格式: 'YYYY-MM'
        """
        year, month_num = map(int, month.split('-'))
        start_date = datetime(year, month_num, 1, tzinfo=timezone.utc)
        if month_num == 12:
            end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end_date = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)
        
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        params = {
            'state': 'closed',
            'per_page': 100,
            'sort': 'updated',
            'direction': 'desc'
        }
        
        count = 0
        page = 1
        max_pages = 5  # 限制最多5页
        
        while page <= max_pages:
            params['page'] = page
            response = self._safe_request(url, params)
            if not response:
                break
            
            prs = response.json()
            if not prs:
                break
            
            for pr in prs:
                updated_at = datetime.fromisoformat(pr['updated_at'].replace('Z', '+00:00'))
                
                if updated_at < start_date:
                    return count  # 已经过了这个月份
                
                if start_date <= updated_at < end_date:
                    # 检查是否被拒绝（closed但未merged）
                    if pr.get('merged_at') is None:
                        count += 1
            
            if len(prs) < 100:
                break
            
            page += 1
        
        return count
    
    def get_issue_response_time_by_month(self, owner: str, repo: str, month: str) -> Optional[float]:
        """
        计算指定月份的平均Issue响应时间（小时）
        响应时间 = 第一个评论时间 - Issue创建时间
        month格式: 'YYYY-MM'
        """
        year, month_num = map(int, month.split('-'))
        start_date = datetime(year, month_num, 1, tzinfo=timezone.utc)
        if month_num == 12:
            end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end_date = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)
        
        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        params = {
            'state': 'all',
            'per_page': 100,
            'sort': 'created',
            'direction': 'desc'
        }
        
        response_times = []
        page = 1
        max_pages = 3  # 限制最多3页，避免太慢
        
        while page <= max_pages:
            params['page'] = page
            response = self._safe_request(url, params)
            if not response:
                break
            
            issues = response.json()
            if not issues:
                break
            
            for issue in issues:
                # 跳过PR
                if 'pull_request' in issue:
                    continue
                
                created_at = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
                
                if created_at < start_date:
                    break  # 已经过了这个月份
                
                if start_date <= created_at < end_date:
                    # 获取第一个评论时间
                    comments_url = issue.get('comments_url')
                    if comments_url and issue.get('comments', 0) > 0:
                        comments_response = self._safe_request(comments_url, {'per_page': 1, 'sort': 'created', 'direction': 'asc'})
                        if comments_response:
                            comments = comments_response.json()
                            if comments:
                                first_comment_time = datetime.fromisoformat(comments[0]['created_at'].replace('Z', '+00:00'))
                                response_time = (first_comment_time - created_at).total_seconds() / 3600  # 转换为小时
                                response_times.append(response_time)
            
            page += 1
        
        if response_times:
            return sum(response_times) / len(response_times)
        return None
    
    def get_pr_response_time_by_month(self, owner: str, repo: str, month: str) -> Optional[float]:
        """
        计算指定月份的平均PR响应时间（小时）
        响应时间 = 第一个评论/审查时间 - PR创建时间
        month格式: 'YYYY-MM'
        """
        year, month_num = map(int, month.split('-'))
        start_date = datetime(year, month_num, 1, tzinfo=timezone.utc)
        if month_num == 12:
            end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end_date = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)
        
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        params = {
            'state': 'all',
            'per_page': 100,
            'sort': 'created',
            'direction': 'desc'
        }
        
        response_times = []
        page = 1
        max_pages = 3  # 限制最多3页，避免太慢
        
        while page <= max_pages:
            params['page'] = page
            response = self._safe_request(url, params)
            if not response:
                break
            
            prs = response.json()
            if not prs:
                break
            
            for pr in prs:
                created_at = datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00'))
                
                if created_at < start_date:
                    break  # 已经过了这个月份
                
                if start_date <= created_at < end_date:
                    # PR响应时间 = 第一个评论或审查的时间 - PR创建时间
                    # 优先使用审查时间（review），如果没有则使用评论时间（comment）
                    first_response_time = None
                    
                    pr_number = pr.get('number')
                    if not pr_number:
                        continue
                    
                    # 1. 尝试获取第一个审查（review）
                    reviews_url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
                    reviews_response = self._safe_request(reviews_url, {'per_page': 1})
                    if reviews_response:
                        reviews = reviews_response.json()
                        if reviews and len(reviews) > 0:
                            # 找到最早的审查时间
                            review_times = [datetime.fromisoformat(r['submitted_at'].replace('Z', '+00:00')) for r in reviews if r.get('submitted_at')]
                            if review_times:
                                first_response_time = min(review_times)
                    
                    # 2. 如果没有审查，尝试获取第一个评论（PR review comments）
                    if not first_response_time:
                        review_comments_url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
                        review_comments_response = self._safe_request(review_comments_url, {'per_page': 1, 'sort': 'created', 'direction': 'asc'})
                        if review_comments_response:
                            review_comments = review_comments_response.json()
                            if review_comments and len(review_comments) > 0:
                                first_response_time = datetime.fromisoformat(review_comments[0]['created_at'].replace('Z', '+00:00'))
                    
                    # 3. 如果都没有，尝试获取PR的issue评论（issue comments）
                    if not first_response_time:
                        issue_comments_url = pr.get('comments_url')
                        if issue_comments_url and pr.get('comments', 0) > 0:
                            issue_comments_response = self._safe_request(issue_comments_url, {'per_page': 1, 'sort': 'created', 'direction': 'asc'})
                            if issue_comments_response:
                                issue_comments = issue_comments_response.json()
                                if issue_comments and len(issue_comments) > 0:
                                    first_response_time = datetime.fromisoformat(issue_comments[0]['created_at'].replace('Z', '+00:00'))
                    
                    if first_response_time:
                        response_time = (first_response_time - created_at).total_seconds() / 3600  # 转换为小时
                        response_times.append(response_time)
            
            page += 1
        
        if response_times:
            return sum(response_times) / len(response_times)
        return None
    
    def get_issue_resolution_duration_by_month(self, owner: str, repo: str, month: str) -> Optional[float]:
        """
        计算指定月份的平均Issue解决时长（小时）
        解决时长 = Issue关闭时间 - Issue创建时间（仅计算已关闭的Issue）
        month格式: 'YYYY-MM'
        """
        year, month_num = map(int, month.split('-'))
        start_date = datetime(year, month_num, 1, tzinfo=timezone.utc)
        if month_num == 12:
            end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end_date = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)
        
        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        params = {
            'state': 'closed',  # 只获取已关闭的Issue
            'per_page': 100,
            'sort': 'created',
            'direction': 'desc'
        }
        
        durations = []
        page = 1
        max_pages = 3
        
        while page <= max_pages:
            params['page'] = page
            response = self._safe_request(url, params)
            if not response:
                break
            
            issues = response.json()
            if not issues:
                break
            
            for issue in issues:
                # 跳过PR
                if 'pull_request' in issue:
                    continue
                
                created_at = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
                
                if created_at < start_date:
                    break  # 已经过了这个月份
                
                if start_date <= created_at < end_date:
                    # 只计算已关闭的Issue
                    if issue.get('state') == 'closed' and issue.get('closed_at'):
                        closed_at = datetime.fromisoformat(issue['closed_at'].replace('Z', '+00:00'))
                        duration = (closed_at - created_at).total_seconds() / 3600  # 转换为小时
                        durations.append(duration)
            
            page += 1
        
        if durations:
            return sum(durations) / len(durations)
        return None
    
    def get_issue_age_by_month(self, owner: str, repo: str, month: str) -> Optional[float]:
        """
        计算指定月份的平均Issue存活时间（小时）
        存活时间 = 当前时间/最后更新时间 - Issue创建时间（未关闭），或关闭时间 - 创建时间（已关闭）
        month格式: 'YYYY-MM'
        """
        year, month_num = map(int, month.split('-'))
        start_date = datetime(year, month_num, 1, tzinfo=timezone.utc)
        if month_num == 12:
            end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end_date = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)
        
        url = f"{self.base_url}/repos/{owner}/{repo}/issues"
        params = {
            'state': 'all',
            'per_page': 100,
            'sort': 'created',
            'direction': 'desc'
        }
        
        ages = []
        page = 1
        max_pages = 3
        current_time = datetime.now(timezone.utc)
        
        while page <= max_pages:
            params['page'] = page
            response = self._safe_request(url, params)
            if not response:
                break
            
            issues = response.json()
            if not issues:
                break
            
            for issue in issues:
                # 跳过PR
                if 'pull_request' in issue:
                    continue
                
                created_at = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
                
                if created_at < start_date:
                    break  # 已经过了这个月份
                
                if start_date <= created_at < end_date:
                    # 计算存活时间
                    if issue.get('state') == 'closed' and issue.get('closed_at'):
                        # 已关闭：使用关闭时间
                        closed_at = datetime.fromisoformat(issue['closed_at'].replace('Z', '+00:00'))
                        age = (closed_at - created_at).total_seconds() / 3600
                    else:
                        # 未关闭：使用最后更新时间或当前时间
                        updated_at = datetime.fromisoformat(issue['updated_at'].replace('Z', '+00:00'))
                        # 如果更新时间太接近当前时间，说明可能还在活跃，使用当前时间
                        if (current_time - updated_at).total_seconds() < 86400:  # 24小时内
                            age = (current_time - created_at).total_seconds() / 3600
                        else:
                            age = (updated_at - created_at).total_seconds() / 3600
                    ages.append(age)
            
            page += 1
        
        if ages:
            return sum(ages) / len(ages)
        return None
    
    def get_pr_resolution_duration_by_month(self, owner: str, repo: str, month: str) -> Optional[float]:
        """
        计算指定月份的平均PR处理时长（小时）
        处理时长 = PR合并/关闭时间 - PR创建时间（仅计算已关闭/合并的PR）
        month格式: 'YYYY-MM'
        """
        year, month_num = map(int, month.split('-'))
        start_date = datetime(year, month_num, 1, tzinfo=timezone.utc)
        if month_num == 12:
            end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end_date = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)
        
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        params = {
            'state': 'closed',  # 只获取已关闭的PR
            'per_page': 100,
            'sort': 'created',
            'direction': 'desc'
        }
        
        durations = []
        page = 1
        max_pages = 3
        
        while page <= max_pages:
            params['page'] = page
            response = self._safe_request(url, params)
            if not response:
                break
            
            prs = response.json()
            if not prs:
                break
            
            for pr in prs:
                created_at = datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00'))
                
                if created_at < start_date:
                    break  # 已经过了这个月份
                
                if start_date <= created_at < end_date:
                    # 只计算已关闭/合并的PR
                    if pr.get('state') == 'closed':
                        # 优先使用合并时间，如果没有则使用关闭时间
                        merged_at = pr.get('merged_at')
                        if merged_at:
                            merged_time = datetime.fromisoformat(merged_at.replace('Z', '+00:00'))
                            duration = (merged_time - created_at).total_seconds() / 3600  # 转换为小时
                            durations.append(duration)
                        elif pr.get('closed_at'):
                            closed_at = datetime.fromisoformat(pr['closed_at'].replace('Z', '+00:00'))
                            duration = (closed_at - created_at).total_seconds() / 3600
                            durations.append(duration)
            
            page += 1
        
        if durations:
            return sum(durations) / len(durations)
        return None
    
    def get_pr_age_by_month(self, owner: str, repo: str, month: str) -> Optional[float]:
        """
        计算指定月份的平均PR存活时间（小时）
        存活时间 = 当前时间/最后更新时间 - PR创建时间（未关闭），或合并/关闭时间 - 创建时间（已关闭）
        month格式: 'YYYY-MM'
        """
        year, month_num = map(int, month.split('-'))
        start_date = datetime(year, month_num, 1, tzinfo=timezone.utc)
        if month_num == 12:
            end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end_date = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)
        
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        params = {
            'state': 'all',
            'per_page': 100,
            'sort': 'created',
            'direction': 'desc'
        }
        
        ages = []
        page = 1
        max_pages = 3
        current_time = datetime.now(timezone.utc)
        
        while page <= max_pages:
            params['page'] = page
            response = self._safe_request(url, params)
            if not response:
                break
            
            prs = response.json()
            if not prs:
                break
            
            for pr in prs:
                created_at = datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00'))
                
                if created_at < start_date:
                    break  # 已经过了这个月份
                
                if start_date <= created_at < end_date:
                    # 计算存活时间
                    if pr.get('state') == 'closed':
                        # 已关闭：优先使用合并时间，否则使用关闭时间
                        merged_at = pr.get('merged_at')
                        if merged_at:
                            merged_time = datetime.fromisoformat(merged_at.replace('Z', '+00:00'))
                            age = (merged_time - created_at).total_seconds() / 3600
                        elif pr.get('closed_at'):
                            closed_at = datetime.fromisoformat(pr['closed_at'].replace('Z', '+00:00'))
                            age = (closed_at - created_at).total_seconds() / 3600
                        else:
                            continue
                    else:
                        # 未关闭：使用最后更新时间或当前时间
                        updated_at = datetime.fromisoformat(pr['updated_at'].replace('Z', '+00:00'))
                        # 如果更新时间太接近当前时间，说明可能还在活跃，使用当前时间
                        if (current_time - updated_at).total_seconds() < 86400:  # 24小时内
                            age = (current_time - created_at).total_seconds() / 3600
                        else:
                            age = (updated_at - created_at).total_seconds() / 3600
                    ages.append(age)
            
            page += 1
        
        if ages:
            return sum(ages) / len(ages)
        return None
    
    def supplement_missing_metrics(self, owner: str, repo: str, opendigger_metrics: Dict, months: List[str]) -> Dict:
        """
        补充缺失的指标
        优先级：OpenDigger > GitHub API补充 > 0填充
        
        返回格式：{metric_name: {month: value, ...}, ...}
        """
        supplemented = {}
        
        # 定义需要补充的指标及其对应的补充函数
        supplement_functions = {
            'Issue响应时间': self.get_issue_response_time_by_month,
            'Issue解决时长': self.get_issue_resolution_duration_by_month,
            'Issue存活时间': self.get_issue_age_by_month,
            'PR响应时间': self.get_pr_response_time_by_month,
            'PR处理时长': self.get_pr_resolution_duration_by_month,
            'PR存活时间': self.get_pr_age_by_month,
            # 注意：OpenDigger没有"代码提交数"和"PR拒绝数"，这些指标不在25个标准指标中
            # 如果需要，可以添加，但需要确保指标名称与all_metrics_list一致
        }
        
        # 检查每个指标是否缺失（完全缺失或部分月份缺失）
        for metric_name, supplement_func in supplement_functions.items():
            if metric_name not in opendigger_metrics:
                # 完全缺失，需要补充
                print(f"    - 补充 {metric_name}（完全缺失）...")
                metric_data = {}
                
                for month in months:
                    try:
                        value = supplement_func(owner, repo, month)
                        if value is not None:
                            metric_data[month] = value
                        else:
                            # GitHub API无法获取，用0填充
                            metric_data[month] = 0.0
                    except Exception as e:
                        print(f"      ⚠ {month} 补充失败: {str(e)}")
                        # 补充失败，该月份用0填充
                        metric_data[month] = 0.0
                
                if metric_data:
                    supplemented[metric_name] = metric_data
                    print(f"      ✓ 已补充 {len(metric_data)} 个月的数据")
            else:
                # 部分缺失，补充缺失的月份
                existing_data = opendigger_metrics[metric_name]
                if isinstance(existing_data, dict):
                    # 检查是否所有月份的值都是0或None（全为0的情况）
                    all_zero = all(
                        existing_data.get(m, 0) == 0 or existing_data.get(m) is None 
                        for m in months
                    )
                    
                    # 找出缺失或为0的月份
                    missing_months = [
                        m for m in months 
                        if m not in existing_data or existing_data.get(m) == 0 or existing_data.get(m) is None
                    ]
                    
                    # 如果全为0或有很多缺失月份，则补充
                    if all_zero or missing_months:
                        if all_zero:
                            print(f"    - 补充 {metric_name}（所有月份值都为0，尝试用GitHub API补充）...")
                        else:
                            print(f"    - 补充 {metric_name}（{len(missing_months)} 个月缺失）...")
                        metric_data = existing_data.copy()
                        
                        # 如果全为0，补充所有月份；否则只补充缺失的月份
                        months_to_supplement = months if all_zero else missing_months
                        
                        for month in months_to_supplement:
                            try:
                                value = supplement_func(owner, repo, month)
                                if value is not None and value != 0:
                                    metric_data[month] = value
                                    print(f"      ✓ {month}: {value:.2f} 小时")
                                else:
                                    # GitHub API无法获取，保持原值（可能是0）
                                    pass
                            except Exception as e:
                                print(f"      ⚠ {month} 补充失败: {str(e)}")
                                # 补充失败，保持原值（可能是0）
                        
                        supplemented[metric_name] = metric_data
                        if all_zero:
                            print(f"      ✓ 已补充 {len([m for m in months if metric_data.get(m, 0) != 0])} 个月的数据（共尝试 {len(months)} 个月）")
                        else:
                            print(f"      ✓ 已补充 {len(missing_months)} 个月的数据")
        
        return supplemented

