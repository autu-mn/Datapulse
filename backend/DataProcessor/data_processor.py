import json
import os
import pandas as pd
from datetime import datetime
from collections import defaultdict
import re
import requests
from dotenv import load_dotenv

# 尝试导入MaxKB上传模块（可选）
try:
    from .maxkb_uploader import MaxKBUploader
    MAXKB_AVAILABLE = True
except ImportError:
    try:
        from maxkb_uploader import MaxKBUploader
        MAXKB_AVAILABLE = True
    except ImportError:
        MAXKB_AVAILABLE = False
        print("⚠ MaxKB上传模块未找到，将跳过自动上传功能")

# 加载环境变量
load_dotenv()

class DataProcessor:
    """处理爬取的GitHub数据，分离时序数据和文本数据"""
    
    def __init__(self, json_file_path, enable_maxkb_upload: bool = False,
                 maxkb_config: dict = None):
        """
        初始化，加载JSON数据
        
        Args:
            json_file_path: JSON数据文件路径
            enable_maxkb_upload: 是否启用MaxKB自动上传，默认False
            maxkb_config: MaxKB配置字典，包含：
                - base_url: MaxKB服务地址（默认从环境变量MAXKB_URL读取，或http://localhost:8080）
                - username: 登录用户名（默认从环境变量MAXKB_USERNAME读取，或admin）
                - password: 登录密码（默认从环境变量MAXKB_PASSWORD读取，或admin）
                - knowledge_id: 知识库ID（默认从环境变量MAXKB_KNOWLEDGE_ID读取）
                - chunk_size: 文档分块大小（默认500）
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 从文件名提取仓库信息
        basename = os.path.basename(json_file_path)
        parts = basename.replace('_text_data_', '_').replace('.json', '').split('_')
        if len(parts) >= 2:
            self.owner = parts[0]
            self.repo = parts[1]
        else:
            self.owner = self.data.get('repo_info', {}).get('full_name', 'unknown').split('/')[0]
            self.repo = self.data.get('repo_info', {}).get('full_name', 'unknown').split('/')[-1]
        
        # 统一保存到 data 目录下的项目文件夹
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        project_dir = os.path.join(data_dir, f"{self.owner}_{self.repo}")
        os.makedirs(project_dir, exist_ok=True)
        
        # 设置输出目录（在项目文件夹下创建_processed文件夹）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(project_dir, f"{self.owner}_{self.repo}_text_data_{timestamp}_processed")
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"处理数据: {self.owner}/{self.repo}")
        print(f"输出目录: {self.output_dir}")
        
        # MaxKB上传配置
        self.enable_maxkb_upload = enable_maxkb_upload and MAXKB_AVAILABLE
        if self.enable_maxkb_upload:
            self.maxkb_config = maxkb_config or {}
            self.maxkb_config.setdefault('base_url', os.getenv('MAXKB_URL', 'http://localhost:8080'))
            self.maxkb_config.setdefault('username', os.getenv('MAXKB_USERNAME', 'admin'))
            self.maxkb_config.setdefault('password', os.getenv('MAXKB_PASSWORD', 'admin'))
            self.maxkb_config.setdefault('knowledge_id', os.getenv('MAXKB_KNOWLEDGE_ID'))
            self.maxkb_config.setdefault('chunk_size', int(os.getenv('MAXKB_CHUNK_SIZE', '500')))
            
            if not self.maxkb_config.get('knowledge_id'):
                print("⚠ MaxKB知识库ID未配置，将跳过自动上传")
                print("  请设置环境变量 MAXKB_KNOWLEDGE_ID 或在代码中提供 knowledge_id")
                self.enable_maxkb_upload = False
        
        # 如果JSON中没有OpenDigger数据，自动获取
        if not self.data.get('opendigger_metrics'):
            print("\n检测到JSON中没有OpenDigger数据，正在从API获取...")
            self._fetch_opendigger_metrics()
    
    def _fetch_opendigger_metrics(self):
        """从OpenDigger API获取指标"""
        base_url = "https://oss.open-digger.cn/github/"
        metrics_config = {
            'activity': '活跃度', 'openrank': '影响力', 'stars': 'Star数', 
            'participants': '参与者数', 'technical_fork': 'Fork数',
            'issues_new': '新增Issue', 'issues_closed': '关闭Issue',
            'issue_response_time': 'Issue响应时间', 
            'issue_resolution_duration': 'Issue解决时长', 'issue_age': 'Issue存活时间',
            'change_requests_accepted': 'PR接受数', 'change_requests_declined': 'PR拒绝数',
            'change_request_response_time': 'PR响应时间',
            'change_request_resolution_duration': 'PR处理时长', 'change_request_age': 'PR存活时间',
            'new_contributors': '新增贡献者', 'bus_factor': '总线因子',
            'inactive_contributors': '不活跃贡献者', 'code_change_commits': '代码提交数',
        }
        
        result = {}
        for metric_key, metric_name in metrics_config.items():
            try:
                url = f"{base_url}{self.owner}/{self.repo}/{metric_key}.json"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        result[metric_name] = data
            except:
                pass
        
        if result:
            self.data['opendigger_metrics'] = result
            print(f"  成功获取 {len(result)} 个OpenDigger指标")
    
    def get_quarter(self, date_str):
        """从日期字符串获取季度"""
        if not date_str or len(date_str) < 7:
            return None
        try:
            year, month = map(int, date_str[:7].split('-'))
            quarter = (month - 1) // 3 + 1
            return f"{year}Q{quarter}"
        except:
            return None
    
    def get_year(self, date_str):
        """从日期字符串获取年份"""
        if not date_str or len(date_str) < 4:
            return None
        try:
            return date_str[:4]
        except:
            return None
    
    def _process_metric(self, metric_values, prefix=''):
        """处理单个指标，按年、季度分组"""
        if not isinstance(metric_values, dict):
            return None
        
        by_year = defaultdict(dict)
        by_quarter = defaultdict(dict)
        
        for date_str, value in metric_values.items():
            year = self.get_year(date_str)
            quarter = self.get_quarter(date_str)
            if year:
                by_year[year][date_str] = value
            if quarter:
                by_quarter[quarter][date_str] = value
        
        return {
            'by_year': dict(by_year),
            'by_quarter': dict(by_quarter),
            'raw': metric_values
        }
    
    def process_timeseries_data(self):
        """处理时序数据，按年、季度分组"""
        print("\n处理时序数据...")
        
        timeseries_data = {}
        
        # OpenDigger指标
        if self.data.get('opendigger_metrics'):
            for metric_name, metric_values in self.data['opendigger_metrics'].items():
                processed = self._process_metric(metric_values)
                if processed:
                    timeseries_data[f'opendigger_{metric_name}'] = processed
        
        # 备用指标
        if self.data.get('fallback_metrics'):
            for metric_name, metric_values in self.data['fallback_metrics'].items():
                processed = self._process_metric(metric_values)
                if processed:
                    timeseries_data[f'fallback_{metric_name}'] = processed
        
        # 3. 保存时序数据
        # JSON格式
        json_path = os.path.join(self.output_dir, 'timeseries_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(timeseries_data, f, ensure_ascii=False, indent=2)
        print(f"  已保存: timeseries_data.json")
        
        # Excel格式 - 按年、季度
        if timeseries_data:
            self._save_timeseries_excel(timeseries_data, 'year', '年份')
            self._save_timeseries_excel(timeseries_data, 'quarter', '季度')
        
        return timeseries_data
    
    def _save_timeseries_excel(self, timeseries_data, period_type, period_label):
        """保存时序数据到Excel"""
        excel_path = os.path.join(self.output_dir, f'timeseries_by_{period_type}.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for metric_name, metric_data in timeseries_data.items():
                period_key = f'by_{period_type}'
                data_list = []
                for period, period_values in metric_data.get(period_key, {}).items():
                    for date_str, value in sorted(period_values.items()):
                        data_list.append({period_label: period, '日期': date_str, '数值': value})
                
                if data_list:
                    df = pd.DataFrame(data_list)
                    sheet_name = metric_name[:31] if len(metric_name) <= 31 else metric_name[:28] + '...'
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"  已保存: timeseries_by_{period_type}.xlsx ({len(timeseries_data)} 个指标)")
        
        return timeseries_data
    
    def clean_text(self, text):
        """清理文本，保留必要格式，移除控制字符"""
        if not text:
            return ""
        # 移除控制字符（但保留换行符）
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # 规范化换行符
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        # 移除多余的空行（保留最多2个连续换行）
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除行首尾空白
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()
    
    def _format_doc(self, doc_type, title, content, metadata):
        """格式化单个文档"""
        return {
            'type': doc_type,
            'title': title,
            'content': self.clean_text(content),
            'metadata': metadata
        }
    
    def format_text_for_training(self):
        """格式化文本数据，适合训练大模型"""
        print("\n处理文本数据...")
        
        text_documents = []
        
        # 仓库基本信息
        if self.data.get('repo_info'):
            r = self.data['repo_info']
            doc = f"""# 仓库基本信息

仓库名称: {r.get('full_name', '')}
描述: {r.get('description', '')}
主页: {r.get('homepage', '')}
编程语言: {r.get('language', '')}
Star数: {r.get('stars', 0)}
Fork数: {r.get('forks', 0)}
Watcher数: {r.get('watchers', 0)}
开放Issue数: {r.get('open_issues', 0)}
创建时间: {r.get('created_at', '')}
更新时间: {r.get('updated_at', '')}
许可证: {r.get('license', '')}
标签: {', '.join(r.get('topics', []))}
"""
            text_documents.append(self._format_doc('repo_info', '仓库基本信息', doc, 
                {'source': 'repo_info', 'repo': r.get('full_name', '')}))
        
        # README
        if self.data.get('readme'):
            r = self.data['readme']
            doc = f"""# README文档

文件路径: {r.get('path', '')}
文件大小: {r.get('size', 0)} 字节

内容:
{r.get('content', '')}
"""
            text_documents.append(self._format_doc('readme', f"README: {r.get('name', 'README.md')}", doc,
                {'source': 'readme', 'path': r.get('path', '')}))
        
        # Issues
        for issue in self.data.get('issues', []):
            labels_str = ', '.join(issue.get('labels', []))
            doc = f"""# Issue #{issue.get('number', '')}

标题: {issue.get('title', '')}
状态: {issue.get('state', '')}
创建者: {issue.get('user', '')}
创建时间: {issue.get('created_at', '')}
更新时间: {issue.get('updated_at', '')}
关闭时间: {issue.get('closed_at', '')}
标签: {labels_str}
评论数: {issue.get('comments_count', 0)}
链接: {issue.get('url', '')}

内容:
{issue.get('body', '')}
"""
            text_documents.append(self._format_doc('issue', 
                f"Issue #{issue.get('number', '')}: {issue.get('title', '')}", doc,
                {'source': 'issue', 'number': issue.get('number', ''), 
                 'state': issue.get('state', ''), 'url': issue.get('url', '')}))
        
        # Pull Requests
        for pr in self.data.get('pulls', []):
            merged_status = "已合并" if pr.get('merged', False) else "未合并"
            doc = f"""# Pull Request #{pr.get('number', '')}

标题: {pr.get('title', '')}
状态: {pr.get('state', '')}
合并状态: {merged_status}
创建者: {pr.get('user', '')}
创建时间: {pr.get('created_at', '')}
更新时间: {pr.get('updated_at', '')}
关闭时间: {pr.get('closed_at', '')}
合并时间: {pr.get('merged_at', '')}
评论数: {pr.get('comments_count', 0)}
提交数: {pr.get('commits_count', 0)}
新增代码行数: {pr.get('additions', 0)}
删除代码行数: {pr.get('deletions', 0)}
修改文件数: {pr.get('changed_files', 0)}
链接: {pr.get('url', '')}

内容:
{pr.get('body', '')}
"""
            text_documents.append(self._format_doc('pull_request', 
                f"PR #{pr.get('number', '')}: {pr.get('title', '')}", doc,
                {'source': 'pull_request', 'number': pr.get('number', ''), 
                 'state': pr.get('state', ''), 'merged': pr.get('merged', False), 
                 'url': pr.get('url', '')}))
        
        # Commits
        for commit in self.data.get('commits', []):
            doc = f"""# Commit {commit.get('sha', '')[:8]}

提交信息: {commit.get('message', '')}
作者: {commit.get('author', '')}
作者邮箱: {commit.get('author_email', '')}
提交时间: {commit.get('date', '')}
链接: {commit.get('url', '')}
"""
            text_documents.append(self._format_doc('commit', 
                f"Commit {commit.get('sha', '')[:8]}: {commit.get('message', '')[:50]}", doc,
                {'source': 'commit', 'sha': commit.get('sha', ''), 
                 'author': commit.get('author', ''), 'url': commit.get('url', '')}))
        
        # Releases
        for release in self.data.get('releases', []):
            doc = f"""# Release {release.get('tag_name', '')}

版本名称: {release.get('name', '')}
标签: {release.get('tag_name', '')}
创建时间: {release.get('created_at', '')}
发布时间: {release.get('published_at', '')}
发布者: {release.get('author', '')}
链接: {release.get('url', '')}

发布说明:
{release.get('body', '')}
"""
            text_documents.append(self._format_doc('release', 
                f"Release {release.get('tag_name', '')}: {release.get('name', '')}", doc,
                {'source': 'release', 'tag': release.get('tag_name', ''), 
                 'author': release.get('author', ''), 'url': release.get('url', '')}))
        
        # 保存文本数据
        json_path = os.path.join(self.output_dir, 'text_data_structured.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(text_documents, f, ensure_ascii=False, indent=2)
        print(f"  已保存: text_data_structured.json ({len(text_documents)} 个文档)")
        
        txt_path = os.path.join(self.output_dir, 'text_data_for_training.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(text_documents, 1):
                f.write(f"\n{'='*80}\n文档 {i}: {doc['title']}\n类型: {doc['type']}\n{'='*80}\n\n")
                f.write(doc['content'] + "\n\n")
        print(f"  已保存: text_data_for_training.txt")
        
        # 自动上传到MaxKB（如果启用）
        if self.enable_maxkb_upload:
            self._upload_to_maxkb(txt_path)
        
        if text_documents:
            df = pd.DataFrame([{
                '类型': d['type'], '标题': d['title'], '内容长度': len(d['content']),
                '内容': d['content'][:500] + '...' if len(d['content']) > 500 else d['content']
            } for d in text_documents])
            df.to_excel(os.path.join(self.output_dir, 'text_data_overview.xlsx'), 
                       index=False, engine='openpyxl')
            print(f"  已保存: text_data_overview.xlsx")
        
        return text_documents
    
    def _upload_to_maxkb(self, file_path: str):
        """
        上传文件到MaxKB知识库
        
        Args:
            file_path: 要上传的文件路径
        """
        if not MAXKB_AVAILABLE:
            print("  [WARN] MaxKB上传模块不可用，跳过上传")
            return
        
        try:
            print(f"\n{'='*60}\n开始上传到MaxKB知识库\n{'='*60}")
            
            uploader = MaxKBUploader(
                base_url=self.maxkb_config['base_url'],
                username=self.maxkb_config['username'],
                password=self.maxkb_config['password'],
                knowledge_id=self.maxkb_config['knowledge_id']
            )
            
            if uploader.login():
                # 生成文档名称：仓库名_时间戳
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                document_name = f"{self.owner}_{self.repo}_text_data_{timestamp}.txt"
                
                success = uploader.upload_text_file(
                    file_path,
                    chunk_size=self.maxkb_config['chunk_size'],
                    document_name=document_name
                )
                if success:
                    print(f"[OK] 文件已成功上传到MaxKB知识库")
                else:
                    print(f"[ERROR] 文件上传失败，请检查日志")
            else:
                print(f"[ERROR] MaxKB登录失败，无法上传文件")
                
        except Exception as e:
            print(f"[ERROR] 上传到MaxKB时出错：{str(e)}")
            print(f"  请检查MaxKB配置和网络连接")
    
    def process_all(self):
        """处理所有数据"""
        print(f"\n{'='*60}\n开始处理数据: {self.owner}/{self.repo}\n{'='*60}")
        
        timeseries_data = self.process_timeseries_data()
        text_data = self.format_text_for_training()
        
        # 生成摘要
        summary = {
            'repo': f"{self.owner}/{self.repo}",
            'processed_at': datetime.now().isoformat(),
            'timeseries_metrics_count': len(timeseries_data),
            'text_documents_count': len(text_data),
            'text_documents_by_type': {}
        }
        for doc in text_data:
            doc_type = doc['type']
            summary['text_documents_by_type'][doc_type] = summary['text_documents_by_type'].get(doc_type, 0) + 1
        
        with open(os.path.join(self.output_dir, 'processing_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}\n处理完成！\n{'='*60}")
        print(f"输出目录: {self.output_dir}")
        print(f"时序指标数: {summary['timeseries_metrics_count']}")
        print(f"文本文档数: {summary['text_documents_count']}")
        print(f"文档类型分布: {summary['text_documents_by_type']}\n{'='*60}")
        
        return summary


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python data_processor.py <json_file_path>")
        print("示例: python data_processor.py X-lab2017_open-digger_text_data_20251125_154401.json")
        return
    
    json_file_path = sys.argv[1]
    
    if not os.path.exists(json_file_path):
        print(f"错误: 文件不存在: {json_file_path}")
        return
    
    try:
        processor = DataProcessor(json_file_path)
        processor.process_all()
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

