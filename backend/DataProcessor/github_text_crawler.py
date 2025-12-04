import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import time
import json
import re

load_dotenv()

class OpenDiggerMetrics:
    """从 OpenDigger 获取基础指标（不消耗 GitHub Token）"""
    
    def __init__(self):
        self.base_url = "https://oss.open-digger.cn/github/"
    
    def get_metrics(self, owner, repo):
        """获取所有可用的 OpenDigger 指标"""
        metrics_config = {
            # 基础指标
            'activity': '活跃度',
            'openrank': '影响力',
            'stars': 'Star数',
            'participants': '参与者数',
            'technical_fork': 'Fork数',
            
            # Issue 相关
            'issues_new': '新增Issue',
            'issues_closed': '关闭Issue',
            'issue_response_time': 'Issue响应时间',
            'issue_resolution_duration': 'Issue解决时长',
            'issue_age': 'Issue存活时间',
            
            # PR 相关
            'change_requests_accepted': 'PR接受数',
            'change_requests_declined': 'PR拒绝数',
            'change_request_response_time': 'PR响应时间',
            'change_request_resolution_duration': 'PR处理时长',
            'change_request_age': 'PR存活时间',
            
            # 贡献者相关
            'new_contributors': '新增贡献者',
            'bus_factor': '总线因子',
            'inactive_contributors': '不活跃贡献者',
            
            # 其他
            'code_change_commits': '代码提交数',
        }
        
        result = {}
        success_count = 0
        missing_metrics = []
        
        print(f"\n正在从 OpenDigger 获取基础指标...")
        
        for metric_key, metric_name in metrics_config.items():
            url = f"{self.base_url}{owner}/{repo}/{metric_key}.json"
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result[metric_name] = data
                        success_count += 1
                    else:
                        missing_metrics.append(metric_name)
                else:
                    missing_metrics.append(metric_name)
            except:
                missing_metrics.append(metric_name)
        
        print(f"  成功获取 {success_count}/{len(metrics_config)} 个指标")
        
        if missing_metrics:
            print(f"  缺失 {len(missing_metrics)} 个指标: {', '.join(missing_metrics[:5])}{'...' if len(missing_metrics) > 5 else ''}")
        
        return result, missing_metrics

class GitHubTextCrawler:
    def __init__(self):
        # 支持多个 GitHub Token 轮换
        self.tokens = []
        self.current_token_index = 0
        
        # 尝试加载多个 token
        token = os.getenv('GITHUB_TOKEN')
        token_1 = os.getenv('GITHUB_TOKEN_1')
        token_2 = os.getenv('GITHUB_TOKEN_2')
        
        if token:
            self.tokens.append(token)
        if token_1:
            self.tokens.append(token_1)
        if token_2:
            self.tokens.append(token_2)
        
        if not self.tokens:
            raise ValueError("未找到 GITHUB_TOKEN，请在 .env 文件中配置 GITHUB_TOKEN、GITHUB_TOKEN_1 或 GITHUB_TOKEN_2")
        
        self.token = self.tokens[0]
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
        self.rate_limit_remaining = 5000
        
        print(f"✓ 已加载 {len(self.tokens)} 个 GitHub Token")
    
    def switch_token(self):
        """切换到下一个可用的 token"""
        if len(self.tokens) <= 1:
            print("⚠ 只有一个 token，无法切换")
            return False
        
        self.current_token_index = (self.current_token_index + 1) % len(self.tokens)
        self.token = self.tokens[self.current_token_index]
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        print(f"✓ 已切换到 Token #{self.current_token_index + 1}")
        return True
    
    def safe_request(self, url, max_retries=3, **kwargs):
        """带重试和 token 轮换的安全请求方法"""
        retries = 0
        last_error = None
        
        while retries < max_retries:
            try:
                # 添加超时设置
                if 'timeout' not in kwargs:
                    kwargs['timeout'] = 30
                
                response = requests.get(url, headers=self.headers, **kwargs)
                
                # 检查是否需要切换 token（rate limit）
                if response.status_code == 403:
                    rate_limit = response.headers.get('X-RateLimit-Remaining', '0')
                    if rate_limit == '0':
                        print(f"⚠ Token #{self.current_token_index + 1} 已达到 API 限制")
                        if self.switch_token():
                            retries += 1
                            time.sleep(2)
                            continue
                
                return response
                
            except (requests.exceptions.SSLError, 
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                last_error = e
                retries += 1
                print(f"⚠ 网络错误 (尝试 {retries}/{max_retries}): {type(e).__name__}")
                
                # 如果有多个 token，尝试切换
                if len(self.tokens) > 1 and retries < max_retries:
                    self.switch_token()
                
                # 等待后重试
                wait_time = min(2 ** retries, 10)  # 指数退避，最多10秒
                print(f"  等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            
            except Exception as e:
                print(f"✗ 请求失败: {str(e)}")
                raise
        
        # 所有重试都失败
        if last_error:
            print(f"✗ 请求失败，已重试 {max_retries} 次")
            print(f"  最后错误: {type(last_error).__name__}: {str(last_error)}")
        return None
    
    def calculate_fallback_metrics(self, owner, repo, issues_data, pulls_data, commits_data, repo_info):
        """当 OpenDigger 没有数据时，通过 GitHub API 计算基础指标"""
        print("\n  正在计算备用指标（消耗 Token）...")
        
        fallback_metrics = {}
        
        # 1. Stars 趋势（从仓库信息获取当前值）
        if repo_info:
            fallback_metrics['Star数_当前'] = {
                datetime.now().strftime('%Y-%m'): repo_info.get('stars', 0)
            }
            fallback_metrics['Fork数_当前'] = {
                datetime.now().strftime('%Y-%m'): repo_info.get('forks', 0)
            }
            fallback_metrics['Watchers数_当前'] = {
                datetime.now().strftime('%Y-%m'): repo_info.get('watchers', 0)
            }
        
        # 2. Issue 统计（按月聚合）
        if issues_data:
            issues_by_month = {}
            issues_closed_by_month = {}
            
            for issue in issues_data:
                created_date = issue.get('created_at', '')[:7]  # YYYY-MM
                if created_date:
                    issues_by_month[created_date] = issues_by_month.get(created_date, 0) + 1
                
                if issue.get('state') == 'closed' and issue.get('closed_at'):
                    closed_date = issue.get('closed_at', '')[:7]
                    if closed_date:
                        issues_closed_by_month[closed_date] = issues_closed_by_month.get(closed_date, 0) + 1
            
            if issues_by_month:
                fallback_metrics['新增Issue_计算'] = issues_by_month
            if issues_closed_by_month:
                fallback_metrics['关闭Issue_计算'] = issues_closed_by_month
        
        # 3. PR 统计（按月聚合）
        if pulls_data:
            prs_by_month = {}
            prs_merged_by_month = {}
            prs_closed_by_month = {}
            
            for pr in pulls_data:
                created_date = pr.get('created_at', '')[:7]
                if created_date:
                    prs_by_month[created_date] = prs_by_month.get(created_date, 0) + 1
                
                if pr.get('merged') and pr.get('merged_at'):
                    merged_date = pr.get('merged_at', '')[:7]
                    if merged_date:
                        prs_merged_by_month[merged_date] = prs_merged_by_month.get(merged_date, 0) + 1
                
                if pr.get('state') == 'closed' and not pr.get('merged') and pr.get('closed_at'):
                    closed_date = pr.get('closed_at', '')[:7]
                    if closed_date:
                        prs_closed_by_month[closed_date] = prs_closed_by_month.get(closed_date, 0) + 1
            
            if prs_by_month:
                fallback_metrics['新增PR_计算'] = prs_by_month
            if prs_merged_by_month:
                fallback_metrics['PR接受数_计算'] = prs_merged_by_month
            if prs_closed_by_month:
                fallback_metrics['PR拒绝数_计算'] = prs_closed_by_month
        
        # 4. Commit 统计（按月聚合）
        if commits_data:
            commits_by_month = {}
            
            for commit in commits_data:
                commit_date = commit.get('date', '')[:7]
                if commit_date:
                    commits_by_month[commit_date] = commits_by_month.get(commit_date, 0) + 1
            
            if commits_by_month:
                fallback_metrics['代码提交数_计算'] = commits_by_month
        
        print(f"  计算了 {len(fallback_metrics)} 个备用指标")
        
        return fallback_metrics
    
    def check_rate_limit(self):
        """检查 API 限流状态"""
        url = f"{self.base_url}/rate_limit"
        response = self.safe_request(url)
        if not response:
            return
        if response.status_code == 200:
            data = response.json()
            self.rate_limit_remaining = data['resources']['core']['remaining']
            reset_time = data['resources']['core']['reset']
            print(f"剩余请求次数: {self.rate_limit_remaining}")
            if self.rate_limit_remaining < 100:
                wait_time = reset_time - time.time()
                print(f"警告: 接近限流，{wait_time/60:.1f} 分钟后重置")
    
    def get_repo_info(self, owner, repo):
        """获取仓库基本信息"""
        url = f"{self.base_url}/repos/{owner}/{repo}"
        response = self.safe_request(url)
        if not response:
            return None
        
        if response.status_code == 200:
            data = response.json()
            return {
                'name': data['name'],
                'full_name': data['full_name'],
                'description': data.get('description', ''),
                'homepage': data.get('homepage', ''),
                'language': data.get('language', ''),
                'stars': data['stargazers_count'],
                'forks': data['forks_count'],
                'watchers': data['watchers_count'],
                'open_issues': data['open_issues_count'],
                'created_at': data['created_at'],
                'updated_at': data['updated_at'],
                'topics': data.get('topics', []),
                'license': data.get('license', {}).get('name', '') if data.get('license') else ''
            }
        else:
            print(f"获取仓库信息失败: {response.status_code}")
            return None
    
    def get_readme(self, owner, repo):
        """获取 README 内容"""
        url = f"{self.base_url}/repos/{owner}/{repo}/readme"
        response = self.safe_request(url)
        if not response:
            return None
        
        if response.status_code == 200:
            data = response.json()
            # 获取原始内容
            content_url = data['download_url']
            content_response = requests.get(content_url)
            if content_response.status_code == 200:
                return {
                    'name': data['name'],
                    'path': data['path'],
                    'size': data['size'],
                    'content': content_response.text
                }
        print(f"获取 README 失败: {response.status_code}")
        return None
    
    def get_issues(self, owner, repo, state='all', max_count=100):
        """获取 Issue 列表及内容"""
        issues_data = []
        page = 1
        per_page = 100
        
        print(f"\n正在获取 Issues (最多 {max_count} 个)...")
        
        while len(issues_data) < max_count:
            url = f"{self.base_url}/repos/{owner}/{repo}/issues"
            params = {
                'state': state,
                'per_page': per_page,
                'page': page,
                'sort': 'created',
                'direction': 'desc'
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"获取 Issues 失败: {response.status_code}")
                break
            
            data = response.json()
            if not data:
                break
            
            for issue in data:
                # 过滤掉 Pull Request (GitHub API 会把 PR 也返回在 issues 中)
                if 'pull_request' not in issue:
                    issues_data.append({
                        'number': issue.get('number', 0),
                        'title': issue.get('title', ''),
                        'body': issue.get('body', ''),
                        'state': issue.get('state', ''),
                        'labels': [label.get('name', '') for label in issue.get('labels', [])],
                        'user': issue.get('user', {}).get('login', ''),
                        'created_at': issue.get('created_at', ''),
                        'updated_at': issue.get('updated_at', ''),
                        'closed_at': issue.get('closed_at', ''),
                        'comments_count': issue.get('comments', 0),
                        'url': issue.get('html_url', '')
                    })
                    
                    if len(issues_data) >= max_count:
                        break
            
            print(f"  已获取 {len(issues_data)} 个 Issues")
            page += 1
            time.sleep(0.5)  # 避免请求过快
            
            if len(data) < per_page:
                break
        
        return issues_data
    
    def get_issue_comments(self, owner, repo, issue_number):
        """获取指定 Issue 的所有评论"""
        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{issue_number}/comments"
        response = self.safe_request(url)
        if not response:
            return []
        
        if response.status_code == 200:
            comments = response.json()
            return [{
                'user': comment.get('user', {}).get('login', ''),
                'body': comment.get('body', ''),
                'created_at': comment.get('created_at', ''),
                'updated_at': comment.get('updated_at', '')
            } for comment in comments]
        return []
    
    def get_pulls(self, owner, repo, state='all', max_count=100):
        """获取 Pull Request 列表及内容"""
        pulls_data = []
        page = 1
        per_page = 100
        
        print(f"\n正在获取 Pull Requests (最多 {max_count} 个)...")
        
        while len(pulls_data) < max_count:
            url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
            params = {
                'state': state,
                'per_page': per_page,
                'page': page,
                'sort': 'created',
                'direction': 'desc'
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"获取 Pull Requests 失败: {response.status_code}")
                break
            
            data = response.json()
            if not data:
                break
            
            for pr in data:
                pulls_data.append({
                    'number': pr.get('number', 0),
                    'title': pr.get('title', ''),
                    'body': pr.get('body', ''),
                    'state': pr.get('state', ''),
                    'user': pr.get('user', {}).get('login', ''),
                    'created_at': pr.get('created_at', ''),
                    'updated_at': pr.get('updated_at', ''),
                    'closed_at': pr.get('closed_at', ''),
                    'merged_at': pr.get('merged_at', ''),
                    'merged': pr.get('merged', False),
                    'comments_count': pr.get('comments', 0),
                    'commits_count': pr.get('commits', 0),
                    'additions': pr.get('additions', 0),
                    'deletions': pr.get('deletions', 0),
                    'changed_files': pr.get('changed_files', 0),
                    'url': pr.get('html_url', '')
                })
                
                if len(pulls_data) >= max_count:
                    break
            
            print(f"  已获取 {len(pulls_data)} 个 Pull Requests")
            page += 1
            time.sleep(0.5)
            
            if len(data) < per_page:
                break
        
        return pulls_data
    
    def get_pr_comments(self, owner, repo, pr_number):
        """获取 PR 的评论和 Review 评论"""
        comments = []
        
        # 获取普通评论
        url = f"{self.base_url}/repos/{owner}/{repo}/issues/{pr_number}/comments"
        response = self.safe_request(url)
        if response and response.status_code == 200:
            for comment in response.json():
                comments.append({
                    'type': 'comment',
                    'user': comment.get('user', {}).get('login', ''),
                    'body': comment.get('body', ''),
                    'created_at': comment.get('created_at', '')
                })
        
        # 获取 Review 评论
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
        response = self.safe_request(url)
        if response and response.status_code == 200:
            for review in response.json():
                if review.get('body'):
                    comments.append({
                        'type': 'review',
                        'user': review.get('user', {}).get('login', ''),
                        'body': review.get('body', ''),
                        'state': review.get('state', ''),
                        'created_at': review.get('submitted_at', '')
                    })
        
        return comments
    
    def get_labels(self, owner, repo):
        """获取仓库的所有标签"""
        url = f"{self.base_url}/repos/{owner}/{repo}/labels"
        response = self.safe_request(url)
        if not response:
            return []
        
        if response.status_code == 200:
            labels = response.json()
            return [{
                'name': label['name'],
                'description': label.get('description', ''),
                'color': label['color']
            } for label in labels]
        return []
    
    def get_commits(self, owner, repo, max_count=100):
        """获取提交历史"""
        commits_data = []
        page = 1
        per_page = 100
        
        print(f"\n正在获取 Commits (最多 {max_count} 个)...")
        
        while len(commits_data) < max_count:
            url = f"{self.base_url}/repos/{owner}/{repo}/commits"
            params = {
                'per_page': per_page,
                'page': page
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                print(f"获取 Commits 失败: {response.status_code}")
                break
            
            data = response.json()
            if not data:
                break
            
            for commit in data:
                commits_data.append({
                    'sha': commit.get('sha', ''),
                    'message': commit.get('commit', {}).get('message', ''),
                    'author': commit.get('commit', {}).get('author', {}).get('name', ''),
                    'author_email': commit.get('commit', {}).get('author', {}).get('email', ''),
                    'date': commit.get('commit', {}).get('author', {}).get('date', ''),
                    'url': commit.get('html_url', '')
                })
                
                if len(commits_data) >= max_count:
                    break
            
            print(f"  已获取 {len(commits_data)} 个 Commits")
            page += 1
            time.sleep(0.5)
            
            if len(data) < per_page:
                break
        
        return commits_data
    
    def get_contributors(self, owner, repo):
        """获取贡献者列表"""
        url = f"{self.base_url}/repos/{owner}/{repo}/contributors"
        response = self.safe_request(url)
        if not response:
            return []
        
        if response.status_code == 200:
            contributors = response.json()
            return [{
                'login': contributor.get('login', ''),
                'contributions': contributor.get('contributions', 0),
                'url': contributor.get('html_url', '')
            } for contributor in contributors[:50]]  # 只取前 50 个
        return []
    
    def get_releases(self, owner, repo):
        """获取发布版本信息"""
        url = f"{self.base_url}/repos/{owner}/{repo}/releases"
        response = self.safe_request(url)
        if not response:
            return []
        
        if response.status_code == 200:
            releases = response.json()
            return [{
                'tag_name': release.get('tag_name', ''),
                'name': release.get('name', ''),
                'body': release.get('body', ''),
                'created_at': release.get('created_at', ''),
                'published_at': release.get('published_at', ''),
                'author': release.get('author', {}).get('login', ''),
                'url': release.get('html_url', '')
            } for release in releases[:20]]  # 只取前 20 个
        return []
    
    def crawl_all(self, owner, repo, max_issues=100, max_prs=100, max_commits=100, include_opendigger=True):
        """爬取所有文本内容"""
        print(f"\n{'='*60}")
        print(f"开始爬取仓库: {owner}/{repo}")
        print(f"{'='*60}")
        
        self.check_rate_limit()
        
        all_data = {}
        missing_metrics = []
        
        # 0. OpenDigger 基础指标（不消耗 Token）
        if include_opendigger:
            print("\n[0/9] 获取 OpenDigger 基础指标（不消耗 Token）...")
            opendigger = OpenDiggerMetrics()
            opendigger_data, missing_metrics = opendigger.get_metrics(owner, repo)
            all_data['opendigger_metrics'] = opendigger_data
        
        # 1. 仓库核心信息
        print("\n[1/9] 获取仓库核心信息...")
        all_data['repo_info'] = self.get_repo_info(owner, repo)
        
        # 2. README
        print("\n[2/9] 获取 README...")
        all_data['readme'] = self.get_readme(owner, repo)
        
        # 3. Issues
        print("\n[3/9] 获取 Issues...")
        all_data['issues'] = self.get_issues(owner, repo, max_count=max_issues)
        
        # 4. Pull Requests
        print("\n[4/9] 获取 Pull Requests...")
        all_data['pulls'] = self.get_pulls(owner, repo, max_count=max_prs)
        
        # 5. Labels
        print("\n[5/9] 获取标签...")
        all_data['labels'] = self.get_labels(owner, repo)
        
        # 6. Commits
        print("\n[6/9] 获取提交历史...")
        all_data['commits'] = self.get_commits(owner, repo, max_count=max_commits)
        
        # 7. Contributors
        print("\n[7/9] 获取贡献者...")
        all_data['contributors'] = self.get_contributors(owner, repo)
        
        # 8. Releases
        print("\n[8/9] 获取发布版本...")
        all_data['releases'] = self.get_releases(owner, repo)
        
        # 9. 备用指标计算（如果 OpenDigger 缺失数据）
        if missing_metrics and include_opendigger:
            print("\n[9/9] OpenDigger 部分指标缺失，计算备用指标...")
            fallback_metrics = self.calculate_fallback_metrics(
                owner, repo,
                all_data.get('issues', []),
                all_data.get('pulls', []),
                all_data.get('commits', []),
                all_data.get('repo_info')
            )
            all_data['fallback_metrics'] = fallback_metrics
        
        print(f"\n{'='*60}")
        print("爬取完成！")
        print(f"{'='*60}")
        
        return all_data
    
    def _clean_excel_string(self, value):
        """
        清理Excel不允许的字符
        Excel不允许ASCII控制字符（0-31），除了换行符(10)、回车符(13)、制表符(9)
        """
        if not isinstance(value, str):
            return value
        
        # 移除或替换非法控制字符（保留换行符、回车符、制表符）
        # ASCII 0-8, 11-12, 14-31 都是非法字符
        cleaned = ''
        for char in value:
            code = ord(char)
            # 允许的字符：普通字符、换行符(10)、回车符(13)、制表符(9)
            if code >= 32 or code in [9, 10, 13]:
                cleaned += char
            else:
                # 替换为空格
                cleaned += ' '
        
        return cleaned
    
    def _clean_dataframe_for_excel(self, df):
        """清理DataFrame中的所有字符串，移除Excel不允许的字符"""
        if df is None or df.empty:
            return df
        
        df_cleaned = df.copy()
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':  # 字符串类型
                df_cleaned[col] = df_cleaned[col].apply(
                    lambda x: self._clean_excel_string(x) if isinstance(x, str) else x
                )
        return df_cleaned
    
    def save_to_excel(self, data, owner, repo):
        """保存数据到 Excel"""
        # 统一保存到 data 目录下的项目文件夹
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        project_dir = os.path.join(data_dir, f"{owner}_{repo}")
        os.makedirs(project_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(project_dir, f"{owner}_{repo}_text_data_{timestamp}.xlsx")
        
        print(f"\n正在保存到 Excel: {filename}")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # OpenDigger 基础指标
            if data.get('opendigger_metrics'):
                metrics_data = []
                for metric_name, metric_values in data['opendigger_metrics'].items():
                    if isinstance(metric_values, dict):
                        for date, value in metric_values.items():
                            metrics_data.append({
                                '来源': 'OpenDigger',
                                '指标': metric_name,
                                '日期': date,
                                '数值': value
                            })
                if metrics_data:
                    df = pd.DataFrame(metrics_data)
                    df = self._clean_dataframe_for_excel(df)
                    df.to_excel(writer, sheet_name='OpenDigger指标', index=False)
            
            # 备用指标（从 GitHub API 计算）
            if data.get('fallback_metrics'):
                fallback_data = []
                for metric_name, metric_values in data['fallback_metrics'].items():
                    if isinstance(metric_values, dict):
                        for date, value in metric_values.items():
                            fallback_data.append({
                                '来源': 'GitHub API计算',
                                '指标': metric_name,
                                '日期': date,
                                '数值': value
                            })
                if fallback_data:
                    df = pd.DataFrame(fallback_data)
                    df = self._clean_dataframe_for_excel(df)
                    df.to_excel(writer, sheet_name='备用指标', index=False)
            
            # 仓库信息
            if data.get('repo_info'):
                df = pd.DataFrame([data['repo_info']])
                df = self._clean_dataframe_for_excel(df)
                df.to_excel(writer, sheet_name='仓库信息', index=False)
            
            # README
            if data.get('readme'):
                df = pd.DataFrame([data['readme']])
                df = self._clean_dataframe_for_excel(df)
                df.to_excel(writer, sheet_name='README', index=False)
            
            # Issues
            if data.get('issues'):
                df = pd.DataFrame(data['issues'])
                df = self._clean_dataframe_for_excel(df)
                df.to_excel(writer, sheet_name='Issues', index=False)
            
            # Pull Requests
            if data.get('pulls'):
                df = pd.DataFrame(data['pulls'])
                df = self._clean_dataframe_for_excel(df)
                df.to_excel(writer, sheet_name='Pull Requests', index=False)
            
            # Labels
            if data.get('labels'):
                df = pd.DataFrame(data['labels'])
                df = self._clean_dataframe_for_excel(df)
                df.to_excel(writer, sheet_name='标签', index=False)
            
            # Commits
            if data.get('commits'):
                df = pd.DataFrame(data['commits'])
                df = self._clean_dataframe_for_excel(df)
                df.to_excel(writer, sheet_name='提交历史', index=False)
            
            # Contributors
            if data.get('contributors'):
                df = pd.DataFrame(data['contributors'])
                df = self._clean_dataframe_for_excel(df)
                df.to_excel(writer, sheet_name='贡献者', index=False)
            
            # Releases
            if data.get('releases'):
                df = pd.DataFrame(data['releases'])
                df = self._clean_dataframe_for_excel(df)
                df.to_excel(writer, sheet_name='发布版本', index=False)
        
        print(f"已保存: {filename}")
        return filename
    
    def save_to_json(self, data, owner, repo):
        """保存原始数据到 JSON（便于后续处理）"""
        # 统一保存到 data 目录下的项目文件夹
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        project_dir = os.path.join(data_dir, f"{owner}_{repo}")
        os.makedirs(project_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(project_dir, f"{owner}_{repo}_text_data_{timestamp}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存 JSON: {filename}")
        return filename
    
    def process_data(self, json_file_path, enable_maxkb_upload=None):
        """
        处理数据：分离时序数据和文本数据
        
        Args:
            json_file_path: JSON数据文件路径
            enable_maxkb_upload: 是否启用MaxKB自动上传
                - None: 自动检测（如果.env中配置了MAXKB_KNOWLEDGE_ID则启用）
                - True: 强制启用
                - False: 强制禁用
        """
        try:
            from data_processor import DataProcessor
            
            # 自动检测是否启用MaxKB上传
            if enable_maxkb_upload is None:
                # 检查环境变量中是否配置了知识库ID
                maxkb_knowledge_id = os.getenv('MAXKB_KNOWLEDGE_ID')
                enable_maxkb_upload = bool(maxkb_knowledge_id)
                if enable_maxkb_upload:
                    print("\n检测到MaxKB配置，将自动上传到知识库...")
            
            processor = DataProcessor(
                json_file_path=json_file_path,
                enable_maxkb_upload=enable_maxkb_upload
            )
            processor.process_all()
        except ImportError:
            print("警告: 未找到 data_processor 模块，跳过数据处理")
        except Exception as e:
            print(f"数据处理失败: {e}")
            import traceback
            traceback.print_exc()


def main():
    print("="*60)
    print("GitHub 仓库文本内容爬取工具")
    print("="*60)
    print("\n说明:")
    print("- 需要在 .env 文件中配置 GITHUB_TOKEN")
    print("- 可以爬取 Issues、PRs、README、Commits 等文本内容")
    print("- 数据会保存为 Excel 和 JSON 两种格式")
    print()
    
    try:
        crawler = GitHubTextCrawler()
        
        repo_name = input("请输入仓库名 (例如: apache/echarts): ").strip()
        
        if not repo_name or '/' not in repo_name:
            print("错误: 仓库名格式不正确")
            return
        
        owner, repo = repo_name.split('/', 1)
        
        # 询问爬取数量
        max_issues = input("最多爬取多少个 Issues? (默认 100): ").strip()
        max_issues = int(max_issues) if max_issues.isdigit() else 100
        
        max_prs = input("最多爬取多少个 PRs? (默认 100): ").strip()
        max_prs = int(max_prs) if max_prs.isdigit() else 100
        
        max_commits = input("最多爬取多少个 Commits? (默认 100): ").strip()
        max_commits = int(max_commits) if max_commits.isdigit() else 100
        
        # 开始爬取
        data = crawler.crawl_all(owner, repo, max_issues, max_prs, max_commits)
        
        # 保存数据
        crawler.save_to_excel(data, owner, repo)
        json_file = crawler.save_to_json(data, owner, repo)
        
        # 自动处理数据：分离时序数据和文本数据
        print("\n开始处理数据...")
        crawler.process_data(json_file)
        
        # 统计信息
        print("\n" + "="*60)
        print("爬取统计:")
        if data.get('opendigger_metrics'):
            print(f"  OpenDigger 指标: {len(data.get('opendigger_metrics', {}))} 个")
        if data.get('fallback_metrics'):
            print(f"  备用指标（API计算）: {len(data.get('fallback_metrics', {}))} 个")
        print(f"  Issues: {len(data.get('issues', []))} 个")
        print(f"  Pull Requests: {len(data.get('pulls', []))} 个")
        print(f"  Commits: {len(data.get('commits', []))} 个")
        print(f"  Contributors: {len(data.get('contributors', []))} 个")
        print(f"  Labels: {len(data.get('labels', []))} 个")
        print(f"  Releases: {len(data.get('releases', []))} 个")
        print("="*60)
        
    except ValueError as e:
        print(f"\n错误: {e}")
        print("\n请确保:")
        print("1. 在项目根目录创建 .env 文件")
        print("2. 在 .env 文件中添加: GITHUB_TOKEN=your_token_here")
        print("3. Token 需要有 repo 权限")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

