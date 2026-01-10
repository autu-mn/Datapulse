"""
GitHub 仓库数据转换脚本 v3 - 滑动窗口版

将 DataPulse 爬取的 GitHub 仓库时序数据转换为 GitPulse 多变量格式

**重要改进**: 使用滑动窗口从每个仓库生成多个样本，大幅增加数据量

输入维度: 16 个指标
输出格式: 多个样本，每个样本包含 [hist_len, 16] 历史 + [pred_len, 16] 预测

滑动窗口参数:
- hist_len: 48 个月（4年历史）
- pred_len: 12 个月（1年预测）
- stride: 6 个月（每次滑动半年）

过滤规则:
1. 至少有 hist_len + pred_len 个月的数据
2. 核心指标（OpenRank, 活跃度）非零率 > 50%
"""

import os
import json
import csv
import glob
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


# 16 个输入指标（按顺序）
METRICS = [
    "OpenRank",       # 0: 项目影响力指数
    "活跃度",         # 1: 综合活跃度评分
    "Star数",         # 2: 当月新增 Star
    "Fork数",         # 3: 当月新增 Fork
    "关注度",         # 4: 关注者增量
    "参与者数",       # 5: 活跃参与者
    "新增贡献者",     # 6: 新加入的贡献者
    "贡献者",         # 7: 活跃贡献者
    "不活跃贡献者",   # 8: 流失贡献者
    "总线因子",       # 9: 项目风险指标
    "新增Issue",      # 10: 新开 Issue
    "关闭Issue",      # 11: 关闭的 Issue
    "Issue评论",      # 12: Issue 讨论数
    "变更请求",       # 13: PR 数量
    "PR接受数",       # 14: 合并的 PR
    "PR审查",         # 15: PR 审查数
]

# 核心指标索引（用于质量检查）
CORE_METRICS = [0, 1]  # OpenRank, 活跃度

# ============ 滑动窗口参数 ============
HIST_LEN = 48    # 历史长度：48个月（4年）
PRED_LEN = 12    # 预测长度：12个月（1年）
STRIDE = 6       # 滑动步长：6个月

# 最小月份数
MIN_MONTHS = HIST_LEN + PRED_LEN  # 至少需要 60 个月

# 核心指标非零率阈值
MIN_NONZERO_RATIO = 0.5


def find_all_repos(data_dir: str) -> List[str]:
    """查找所有仓库的数据目录"""
    repos = []
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and '_' in item:
            monthly_dirs = glob.glob(os.path.join(item_path, 'monthly_data_*'))
            if monthly_dirs:
                monthly_dirs.sort(reverse=True)
                repos.append(monthly_dirs[0])
    
    return repos


def load_all_months(repo_dir: str) -> Optional[Dict]:
    """加载仓库的 all_months.json"""
    all_months_path = os.path.join(repo_dir, 'timeseries_for_model', 'all_months.json')
    
    if not os.path.exists(all_months_path):
        print(f"  [!] not found: {all_months_path}")
        return None
    
    try:
        # 检查文件大小
        file_size = os.path.getsize(all_months_path)
        if file_size == 0:
            print(f"  [!] Empty file: {all_months_path}")
            return None
        
        with open(all_months_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                print(f"  [!] Empty content: {all_months_path}")
                return None
            
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  [!] Invalid JSON in {all_months_path}: {e}")
        print(f"      File size: {file_size} bytes")
        # 尝试读取前100个字符用于调试
        try:
            with open(all_months_path, 'r', encoding='utf-8') as f:
                preview = f.read(100)
                print(f"      Preview: {repr(preview)}")
        except:
            pass
        return None
    except Exception as e:
        print(f"  [!] Error loading {all_months_path}: {e}")
        return None


def load_project_summary(repo_dir: str) -> Optional[Dict]:
    """加载仓库的 project_summary.json"""
    summary_path = os.path.join(repo_dir, 'project_summary.json')
    
    if not os.path.exists(summary_path):
        return None
    
    try:
        # 检查文件大小
        file_size = os.path.getsize(summary_path)
        if file_size == 0:
            return None
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return None
            
            return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  [!] Invalid JSON in {summary_path}: {e}")
        return None
    except Exception as e:
        print(f"  [!] Error loading {summary_path}: {e}")
        return None


def extract_multivar_timeseries(all_months: Dict) -> Tuple[List[str], np.ndarray]:
    """
    提取多变量时序数据
    
    Returns:
        months: 月份列表
        data: [T, 16] 的多变量时序矩阵
    """
    months = sorted(all_months.keys())
    T = len(months)
    D = len(METRICS)
    
    data = np.zeros((T, D), dtype=np.float32)
    
    for t, month in enumerate(months):
        month_data = all_months[month]
        metrics = month_data.get('opendigger_metrics', {})
        
        for d, metric_name in enumerate(METRICS):
            value = metrics.get(metric_name, 0.0)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = 0.0
            data[t, d] = float(value)
    
    return months, data


def check_data_quality(data: np.ndarray, repo_name: str, min_months: int = MIN_MONTHS) -> Tuple[bool, str]:
    """
    检查数据质量
    
    Returns:
        (是否通过, 原因)
    """
    T, D = data.shape
    
    # 检查月份数
    if T < min_months:
        return False, f"months too few: {T} < {min_months}"
    
    # 检查核心指标非零率
    for idx in CORE_METRICS:
        nonzero_ratio = np.sum(data[:, idx] != 0) / T
        if nonzero_ratio < MIN_NONZERO_RATIO:
            metric_name = METRICS[idx]
            return False, f"{metric_name} nonzero ratio too low: {nonzero_ratio:.1%} < {MIN_NONZERO_RATIO:.0%}"
    
    return True, "OK"


def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score 标准化（按列）
    
    Returns:
        normalized: 标准化后的数据
        mean: 均值
        std: 标准差
    """
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True)
    std[std == 0] = 1  # 避免除零
    normalized = (data - mean) / std
    return normalized, mean.flatten(), std.flatten()


def compute_trend(values: List[float], recent_n: int = 6) -> str:
    """
    计算趋势方向
    
    Args:
        values: 时序数据
        recent_n: 最近N个月用于判断趋势
    
    Returns:
        趋势描述: "rising", "falling", "stable", "volatile"
    """
    if len(values) < 2:
        return "stable"
    
    recent = values[-recent_n:] if len(values) >= recent_n else values
    
    if len(recent) < 2:
        return "stable"
    
    # 计算变化率
    first_half = np.mean(recent[:len(recent)//2])
    second_half = np.mean(recent[len(recent)//2:])
    
    if first_half == 0:
        if second_half > 0:
            return "rising"
        return "stable"
    
    change_rate = (second_half - first_half) / (abs(first_half) + 1e-6)
    
    # 计算波动性
    std_ratio = np.std(recent) / (np.mean(np.abs(recent)) + 1e-6)
    
    if std_ratio > 0.5:
        return "volatile"
    elif change_rate > 0.15:
        return "rising"
    elif change_rate < -0.15:
        return "falling"
    else:
        return "stable"


def extract_issue_titles_and_labels(all_months: Dict, window_months: List[str], max_issues: int = 15) -> Tuple[List[str], Dict[str, int]]:
    """
    从 issue_classification 提取 issue 标题和标签统计
    
    Returns:
        titles: issue 标题列表
        label_counts: 标签计数
    """
    titles = []
    label_counts = {}
    
    # 只取最近的月份（更相关）
    recent_months = window_months[-12:] if len(window_months) > 12 else window_months
    
    for month in recent_months:
        month_data = all_months.get(month, {})
        issue_class = month_data.get('issue_classification', {})
        
        for category in ['feature', 'bug', 'question', 'other']:
            cat_data = issue_class.get(category, {})
            issues = cat_data.get('issues', [])
            
            for issue in issues:
                # 提取标题
                title = issue.get('title', '').strip()
                if title and len(titles) < max_issues:
                    # 清理标题
                    title = title.replace('\n', ' ').replace('\r', ' ')
                    if len(title) > 100:
                        title = title[:100] + "..."
                    titles.append(f"[{category}] {title}")
                
                # 统计标签
                labels = issue.get('labels', [])
                for label in labels:
                    label = label.strip().lower()
                    if label and label not in ['feature', 'bug', 'question', 'enhancement']:
                        label_counts[label] = label_counts.get(label, 0) + 1
    
    return titles, label_counts


def extract_commit_summaries(all_months: Dict, window_months: List[str], max_commits: int = 10) -> List[str]:
    """
    从 text_data 提取 commit 摘要（支持多种数据格式）
    
    Returns:
        summaries: commit 摘要列表
    """
    summaries = []
    
    # 只取最近的月份
    recent_months = window_months[-6:] if len(window_months) > 6 else window_months
    
    for month in recent_months:
        month_data = all_months.get(month, {})
        text_data = month_data.get('text_data', {})
        
        # 尝试多种数据格式
        commits_text = text_data.get('commits_text', '')
        if not commits_text:
            # 尝试从 breakdown 中获取
            breakdown = text_data.get('breakdown', {})
            commits_text = breakdown.get('commits_text', '')
        
        if not commits_text:
            continue
        
        # 解析 commit 消息
        commits = commits_text.split('---\n')
        for commit in commits:
            if len(summaries) >= max_commits:
                break
            
            # 提取 commit 标题（第一行）
            lines = commit.strip().split('\n')
            if not lines:
                continue
            
            first_line = lines[0].strip()
            
            # 清理：移除 "Commit xxxx:" 前缀
            if first_line.startswith('Commit '):
                parts = first_line.split(':', 1)
                if len(parts) > 1:
                    first_line = parts[1].strip()
            
            # 过滤掉太短或无意义的
            if len(first_line) > 10 and len(first_line) < 200:
                summaries.append(first_line)
    
    return summaries


def extract_issues_from_text(all_months: Dict, window_months: List[str], max_issues: int = 10) -> List[str]:
    """
    从 text_data.breakdown.issues_text 提取 issue 标题（备用方法）
    
    Returns:
        titles: issue 标题列表
    """
    titles = []
    
    # 只取最近的月份
    recent_months = window_months[-12:] if len(window_months) > 12 else window_months
    
    for month in recent_months:
        month_data = all_months.get(month, {})
        text_data = month_data.get('text_data', {})
        
        # 尝试从 breakdown 获取
        breakdown = text_data.get('breakdown', {})
        issues_text = breakdown.get('issues_text', '')
        
        if not issues_text:
            continue
        
        # 解析 issue 文本，按 "Issue #" 分割
        parts = issues_text.split('Issue #')
        for part in parts[1:]:  # 跳过第一个空部分
            if len(titles) >= max_issues:
                break
            
            lines = part.strip().split('\n')
            if not lines:
                continue
            
            # 第一行格式: "12345: Issue Title"
            first_line = lines[0].strip()
            if ':' in first_line:
                title_part = first_line.split(':', 1)
                if len(title_part) > 1:
                    title = title_part[1].strip()
                    # 清理和截断
                    title = title.replace('\n', ' ').replace('\r', ' ')
                    if len(title) > 100:
                        title = title[:100] + "..."
                    if len(title) > 10:
                        titles.append(title)
    
    return titles


def get_top_labels(label_counts: Dict[str, int], top_n: int = 10) -> List[str]:
    """获取出现最多的标签"""
    sorted_labels = sorted(label_counts.items(), key=lambda x: -x[1])
    return [label for label, count in sorted_labels[:top_n]]


def generate_context(repo_name: str, all_months: Dict, 
                    window_months: List[str], project_summary: Optional[Dict] = None) -> str:
    """
    生成结构化文本上下文（高信息密度版 + 语义内容）
    
    生成格式:
    - 项目基本信息
    - 活跃度趋势（基于时序数据）
    - Issue 统计和分类
    - 代码贡献统计
    - 关键 Issue 标题（语义信息）
    - 主要 Commit 摘要（语义信息）
    - 热门标签（主题信息）
    """
    lines = []
    
    # === 1. 项目标识 ===
    lines.append(f"Project: {repo_name}")
    lines.append(f"Period: {window_months[0]} to {window_months[-1]} ({len(window_months)} months)")
    
    # === 2. 收集所有月份的指标 ===
    metrics_series = {metric: [] for metric in METRICS}
    
    for month in window_months:
        month_data = all_months.get(month, {})
        metrics = month_data.get('opendigger_metrics', {})
        
        for metric_name in METRICS:
            value = metrics.get(metric_name, 0.0)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = 0.0
            metrics_series[metric_name].append(float(value))
    
    # === 3. 活跃度趋势 ===
    lines.append("")
    lines.append("Activity Trends:")
    
    # OpenRank 趋势
    openrank_values = metrics_series.get('OpenRank', [])
    if openrank_values:
        openrank_current = openrank_values[-1]
        openrank_trend = compute_trend(openrank_values)
        lines.append(f"- OpenRank: {openrank_current:.1f} ({openrank_trend})")
    
    # 活跃度趋势
    activity_values = metrics_series.get('活跃度', [])
    if activity_values:
        activity_current = activity_values[-1]
        activity_trend = compute_trend(activity_values)
        lines.append(f"- Activity score: {activity_current:.1f} ({activity_trend})")
    
    # Star/Fork 增长
    star_values = metrics_series.get('Star数', [])
    fork_values = metrics_series.get('Fork数', [])
    if star_values or fork_values:
        total_stars = sum(star_values) if star_values else 0
        total_forks = sum(fork_values) if fork_values else 0
        recent_stars = sum(star_values[-6:]) if len(star_values) >= 6 else sum(star_values)
        lines.append(f"- Growth: +{int(total_stars)} stars, +{int(total_forks)} forks (recent 6mo: +{int(recent_stars)} stars)")
    
    # === 4. 贡献者健康 ===
    lines.append("")
    lines.append("Contributors:")
    
    contributor_values = metrics_series.get('贡献者', [])
    new_contributor_values = metrics_series.get('新增贡献者', [])
    inactive_values = metrics_series.get('不活跃贡献者', [])
    
    if contributor_values:
        avg_contributors = np.mean(contributor_values[-6:]) if len(contributor_values) >= 6 else np.mean(contributor_values)
        contributor_trend = compute_trend(contributor_values)
        lines.append(f"- Active: {avg_contributors:.0f}/month ({contributor_trend})")
    
    if new_contributor_values:
        total_new = sum(new_contributor_values)
        recent_new = sum(new_contributor_values[-6:]) if len(new_contributor_values) >= 6 else sum(new_contributor_values)
        lines.append(f"- New contributors: +{int(total_new)} total, +{int(recent_new)} recent")
    
    if inactive_values and contributor_values:
        avg_inactive = np.mean(inactive_values[-6:]) if len(inactive_values) >= 6 else np.mean(inactive_values)
        avg_active = np.mean(contributor_values[-6:]) if len(contributor_values) >= 6 else np.mean(contributor_values)
        if avg_active > 0:
            churn_rate = avg_inactive / avg_active
            health = "healthy" if churn_rate < 0.3 else ("moderate" if churn_rate < 0.6 else "concerning")
            lines.append(f"- Churn rate: {churn_rate:.1%} ({health})")
    
    # 总线因子（项目风险）
    bus_factor_values = metrics_series.get('总线因子', [])
    if bus_factor_values:
        bus_factor = bus_factor_values[-1]
        risk = "low risk" if bus_factor >= 3 else ("moderate risk" if bus_factor >= 2 else "high risk")
        lines.append(f"- Bus factor: {bus_factor:.1f} ({risk})")
    
    # === 5. Issue 统计 ===
    lines.append("")
    lines.append("Issues:")
    
    new_issue_values = metrics_series.get('新增Issue', [])
    closed_issue_values = metrics_series.get('关闭Issue', [])
    comment_values = metrics_series.get('Issue评论', [])
    
    if new_issue_values or closed_issue_values:
        total_new = sum(new_issue_values) if new_issue_values else 0
        total_closed = sum(closed_issue_values) if closed_issue_values else 0
        
        # 计算解决率
        resolution_rate = total_closed / total_new if total_new > 0 else 0
        resolution_status = "excellent" if resolution_rate >= 0.9 else ("good" if resolution_rate >= 0.7 else "needs attention")
        
        lines.append(f"- Volume: {int(total_new)} opened, {int(total_closed)} closed")
        lines.append(f"- Resolution rate: {resolution_rate:.0%} ({resolution_status})")
        
        # Issue 趋势
        issue_trend = compute_trend(new_issue_values) if new_issue_values else "stable"
        lines.append(f"- Trend: {issue_trend}")
    
    if comment_values:
        total_comments = sum(comment_values)
        avg_comments = total_comments / len(window_months)
        engagement = "high" if avg_comments > 50 else ("moderate" if avg_comments > 20 else "low")
        lines.append(f"- Discussion: {int(total_comments)} comments ({engagement} engagement)")
    
    # === 6. 代码贡献统计 ===
    lines.append("")
    lines.append("Code:")
    
    pr_values = metrics_series.get('变更请求', [])
    merged_values = metrics_series.get('PR接受数', [])
    review_values = metrics_series.get('PR审查', [])
    
    if pr_values or merged_values:
        total_prs = sum(pr_values) if pr_values else 0
        total_merged = sum(merged_values) if merged_values else 0
        
        merge_rate = total_merged / total_prs if total_prs > 0 else 0
        
        lines.append(f"- PRs: {int(total_prs)} submitted, {int(total_merged)} merged ({merge_rate:.0%} rate)")
        
        # PR 趋势
        pr_trend = compute_trend(merged_values) if merged_values else "stable"
        lines.append(f"- Development pace: {pr_trend}")
    
    if review_values:
        total_reviews = sum(review_values)
        if total_prs > 0:
            review_ratio = total_reviews / total_prs
            review_culture = "strong" if review_ratio >= 2 else ("moderate" if review_ratio >= 1 else "needs improvement")
            lines.append(f"- Review culture: {review_ratio:.1f} reviews/PR ({review_culture})")
    
    # === 7. Issue 分类统计（如果有）===
    total_feature = 0
    total_bug = 0
    total_question = 0
    
    for month in window_months[-12:]:  # 只统计最近12个月
        month_data = all_months.get(month, {})
        issue_class = month_data.get('issue_classification', {})
        total_feature += issue_class.get('feature', {}).get('count', 0)
        total_bug += issue_class.get('bug', {}).get('count', 0)
        total_question += issue_class.get('question', {}).get('count', 0)
    
    if total_feature + total_bug + total_question > 0:
        lines.append("")
        lines.append("Issue Types (recent 12mo):")
        total = total_feature + total_bug + total_question
        if total_bug > 0:
            lines.append(f"- Bugs: {int(total_bug)} ({total_bug/total:.0%})")
        if total_feature > 0:
            lines.append(f"- Features: {int(total_feature)} ({total_feature/total:.0%})")
        if total_question > 0:
            lines.append(f"- Questions: {int(total_question)} ({total_question/total:.0%})")
    
    # === 8. 项目描述（如果有）===
    if project_summary:
        description = project_summary.get('description', '')
        if description and len(description) > 10:
            # 清理描述，只保留有意义的部分
            description = description.strip()
            if len(description) > 150:
                description = description[:150] + "..."
            lines.append("")
            lines.append(f"Description: {description}")
    
    # === 9. 关键 Issue 标题（语义信息）===
    issue_titles, label_counts = extract_issue_titles_and_labels(all_months, window_months, max_issues=10)
    
    # 如果没有从 issue_classification 获取到，尝试从 text_data 获取
    if not issue_titles:
        issue_titles_alt = extract_issues_from_text(all_months, window_months, max_issues=10)
        if issue_titles_alt:
            issue_titles = issue_titles_alt
    
    if issue_titles:
        lines.append("")
        lines.append("Recent Issues:")
        for title in issue_titles[:8]:  # 最多显示8个
            lines.append(f"  - {title}")
    
    # === 10. 热门标签/主题 ===
    top_labels = get_top_labels(label_counts, top_n=8)
    if top_labels:
        lines.append("")
        lines.append(f"Hot Topics: {', '.join(top_labels)}")
    
    # === 11. 主要 Commit 摘要（语义信息）===
    commit_summaries = extract_commit_summaries(all_months, window_months, max_commits=6)
    
    if commit_summaries:
        lines.append("")
        lines.append("Recent Commits:")
        for summary in commit_summaries[:5]:  # 最多显示5个
            # 清理和截断
            summary = summary.replace('\n', ' ').strip()
            if len(summary) > 80:
                summary = summary[:80] + "..."
            lines.append(f"  - {summary}")
    
    # 组合文本
    full_text = "\n".join(lines)
    
    # 限制总长度（避免过长）
    if len(full_text) > 2500:
        full_text = full_text[:2500] + "\n..."
    
    return full_text


def generate_sliding_windows(data: np.ndarray, months: List[str], 
                            all_months: Dict, repo_name: str,
                            project_summary: Optional[Dict],
                            hist_len: int = HIST_LEN, 
                            pred_len: int = PRED_LEN, 
                            stride: int = STRIDE) -> List[Dict]:
    """
    使用滑动窗口生成多个样本
    
    Args:
        data: [T, D] 原始多变量时序
        months: 月份列表
        all_months: 原始月份数据
        repo_name: 仓库名称
        project_summary: 项目摘要
        hist_len: 历史窗口长度
        pred_len: 预测窗口长度
        stride: 滑动步长
    
    Returns:
        samples: 样本列表
    """
    T = data.shape[0]
    window_size = hist_len + pred_len
    
    if T < window_size:
        return []
    
    samples = []
    
    # 整体标准化（使用全部数据的统计量）
    normalized, mean, std = normalize_data(data)
    
    # 滑动窗口
    start = 0
    while start + window_size <= T:
        # 提取窗口数据
        window_data = normalized[start:start + window_size]
        window_months = months[start:start + window_size]
        
        hist = window_data[:hist_len]
        pred = window_data[hist_len:]
        
        # 生成上下文（基于历史窗口）
        hist_months = window_months[:hist_len]
        context = generate_context(repo_name, all_months, hist_months, project_summary)
        
        samples.append({
            'Repo': repo_name,
            'WindowStart': window_months[0],
            'WindowEnd': window_months[-1],
            'HistLen': hist_len,
            'PredLen': pred_len,
            'Hist': hist.tolist(),
            'Pred': pred.tolist(),
            'Text': context,
            'NormMean': mean.tolist(),
            'NormStd': std.tolist()
        })
        
        start += stride
    
    return samples


def convert_repo_data(repo_dir: str, output_rows: List[Dict], 
                     hist_len: int, pred_len: int, stride: int, min_months: int) -> int:
    """
    转换单个仓库的多变量数据（滑动窗口版）
    
    Returns:
        生成的样本数
    """
    # 从路径提取仓库名
    parent_dir = os.path.dirname(repo_dir)
    repo_key = os.path.basename(parent_dir)
    repo_name = repo_key.replace('_', '/')
    
    print(f"  Processing: {repo_name}")
    
    try:
        # 加载数据
        all_months = load_all_months(repo_dir)
        if not all_months:
            return 0
        
        project_summary = load_project_summary(repo_dir)
        
        # 提取多变量时序
        months, data = extract_multivar_timeseries(all_months)
        
        # 质量检查
        passed, reason = check_data_quality(data, repo_name, min_months)
        if not passed:
            print(f"    [SKIP] {reason}")
            return 0
        
        # 使用滑动窗口生成样本
        samples = generate_sliding_windows(
            data, months, all_months, repo_name, project_summary,
            hist_len=hist_len, pred_len=pred_len, stride=stride
        )
        
        if not samples:
            print(f"    [SKIP] Cannot generate windows (T={len(months)} < {min_months})")
            return 0
        
        output_rows.extend(samples)
        print(f"    [OK] {len(months)} months -> {len(samples)} samples (stride={stride})")
        
        return len(samples)
    except Exception as e:
        print(f"    [ERROR] Failed to process {repo_name}: {e}")
        import traceback
        print(f"    Traceback: {traceback.format_exc()}")
        return 0


def save_dataset(output_rows: List[Dict], output_dir: str, hist_len: int, pred_len: int, stride: int):
    """保存数据集"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为 JSON
    json_path = os.path.join(output_dir, 'github_multivar.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': METRICS,
            'n_dims': len(METRICS),
            'hist_len': hist_len,
            'pred_len': pred_len,
            'stride': stride,
            'samples': output_rows
        }, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON saved: {json_path}")
    
    # 保存简化版 CSV
    csv_path = os.path.join(output_dir, 'github_multivar_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Repo', 'WindowStart', 'WindowEnd', 'HistLen', 'PredLen'])
        writer.writeheader()
        for row in output_rows:
            writer.writerow({
                'Repo': row['Repo'],
                'WindowStart': row['WindowStart'],
                'WindowEnd': row['WindowEnd'],
                'HistLen': row['HistLen'],
                'PredLen': row['PredLen']
            })
    print(f"[OK] CSV summary saved: {csv_path}")


def convert_all_repos(data_dir: str, output_dir: str, hist_len: int, pred_len: int, stride: int):
    """转换所有仓库的数据"""
    min_months = hist_len + pred_len
    
    print("=" * 70)
    print("GitPulse Data Converter v3 - Sliding Window")
    print("=" * 70)
    print(f"Input dimensions: {len(METRICS)}")
    print(f"Sliding window: hist_len={hist_len}, pred_len={pred_len}, stride={stride}")
    print(f"Min months: {min_months}")
    print(f"Min nonzero ratio: {MIN_NONZERO_RATIO:.0%}")
    print("=" * 70)
    
    # 查找所有仓库
    repos = find_all_repos(data_dir)
    print(f"\nFound {len(repos)} repos\n")
    
    if not repos:
        print("No repos found!")
        return
    
    # 转换数据
    output_rows = []
    repo_sample_counts = {}
    
    for repo_dir in repos:
        parent_dir = os.path.dirname(repo_dir)
        repo_key = os.path.basename(parent_dir).replace('_', '/')
        count = convert_repo_data(repo_dir, output_rows, hist_len, pred_len, stride, min_months)
        if count > 0:
            repo_sample_counts[repo_key] = count
    
    print(f"\n" + "=" * 70)
    print(f"Conversion complete!")
    print("=" * 70)
    
    if output_rows:
        save_dataset(output_rows, output_dir, hist_len, pred_len, stride)
        
        # 统计信息
        print("\n" + "=" * 70)
        print("Dataset Statistics")
        print("=" * 70)
        print(f"Total samples: {len(output_rows)}")
        print(f"Total repos: {len(repo_sample_counts)}")
        print(f"Dimensions: {len(METRICS)}")
        print(f"Hist length: {hist_len}")
        print(f"Pred length: {pred_len}")
        
        avg_samples = len(output_rows) / len(repo_sample_counts) if repo_sample_counts else 0
        print(f"Avg samples per repo: {avg_samples:.1f}")
        
        print("\nSamples per repo:")
        for repo, count in sorted(repo_sample_counts.items(), key=lambda x: -x[1]):
            print(f"  - {repo}: {count} samples")
        
        # 数据量对比
        print("\n" + "-" * 50)
        print("数据量增益（相比单样本/仓库）:")
        print(f"  旧方法: {len(repo_sample_counts)} 样本")
        print(f"  滑动窗口: {len(output_rows)} 样本")
        print(f"  增益: {len(output_rows) / len(repo_sample_counts):.1f}x")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert GitHub data to GitPulse format (sliding window)')
    parser.add_argument('--data-dir', type=str, 
                       default='../backend/DataProcessor/data',
                       help='Data directory path')
    parser.add_argument('--output-dir', type=str,
                       default='./Pretrain-data',
                       help='Output directory path')
    parser.add_argument('--hist-len', type=int, default=HIST_LEN,
                       help=f'History length (default: {HIST_LEN})')
    parser.add_argument('--pred-len', type=int, default=PRED_LEN,
                       help=f'Prediction length (default: {PRED_LEN})')
    parser.add_argument('--stride', type=int, default=STRIDE,
                       help=f'Sliding stride (default: {STRIDE})')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 处理相对路径
    data_dir = os.path.join(script_dir, args.data_dir) if not os.path.isabs(args.data_dir) else args.data_dir
    output_dir = os.path.join(script_dir, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    
    convert_all_repos(data_dir, output_dir, args.hist_len, args.pred_len, args.stride)


if __name__ == '__main__':
    main()
