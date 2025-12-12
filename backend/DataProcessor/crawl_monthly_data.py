"""
按月爬取GitHub仓库数据的主入口
整合所有数据源，按月组织，分离数据用于MaxKB和双塔模型

执行顺序：
1. 爬取指标数据（数字指标）
2. 爬取描述文本（预处理后，上传到知识库）
3. 爬取issue等时序文本
4. 时序文本+时序指标，按照月份时序对齐
"""

import os
import sys
import json
import argparse
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.DataProcessor.monthly_crawler import MonthlyCrawler
from backend.DataProcessor.monthly_data_processor import MonthlyDataProcessor
from backend.DataProcessor.github_text_crawler import OpenDiggerMetrics, GitHubTextCrawler


def crawl_project_monthly(owner: str, repo: str, max_per_month: int = 3, enable_llm_summary: bool = True):
    """
    爬取项目的月度数据
    
    Args:
        owner: 仓库所有者
        repo: 仓库名称
        max_per_month: 每月最多爬取的数量
        enable_llm_summary: 是否启用LLM摘要生成
    """
    print(f"\n{'='*80}")
    print(f"开始爬取项目: {owner}/{repo}")
    print(f"{'='*80}\n")
    
    # 初始化爬虫和处理器
    monthly_crawler = MonthlyCrawler()
    text_crawler = GitHubTextCrawler()
    
    # ========== 步骤1: 爬取指标数据（数字指标）==========
    print("[1/4] 爬取指标数据（OpenDigger数字指标）...")
    opendigger = OpenDiggerMetrics()
    opendigger_data, missing_metrics = opendigger.get_metrics(owner, repo)
    print(f"  ✓ 获取了 {len(opendigger_data)} 个OpenDigger指标")
    if missing_metrics:
        print(f"  ⚠ 缺失指标: {', '.join(missing_metrics[:5])}{'...' if len(missing_metrics) > 5 else ''}")
    
    # ========== 步骤2: 爬取描述文本（预处理后，上传到知识库）==========
    print("\n[2/4] 爬取描述文本（README、LICENSE、文档等）...")
    static_docs = {
        'repo_info': text_crawler.get_repo_info(owner, repo),
        'readme': text_crawler.get_readme(owner, repo),
        'license': text_crawler.get_license_file(owner, repo),
        'docs_files': text_crawler.get_docs_files(owner, repo, max_files=30, max_depth=2),
        'important_md_files': text_crawler.get_important_md_files(owner, repo, max_files=10),
        'all_doc_files': text_crawler.get_all_markdown_files(owner, repo, max_files=50, max_depth=3),
        'config_files': text_crawler.get_config_files(owner, repo)
    }
    print(f"  ✓ 获取了静态文档")
    
    # 初始化LLM客户端（用于摘要生成）
    llm_client = None
    if enable_llm_summary:
        try:
            from backend.Agent.deepseek_client import DeepSeekClient
            llm_client = DeepSeekClient()
            print("  ✓ LLM客户端已初始化（用于摘要生成）")
        except ImportError:
            try:
                from Agent.deepseek_client import DeepSeekClient
                llm_client = DeepSeekClient()
                print("  ✓ LLM客户端已初始化（用于摘要生成）")
            except Exception as e:
                print(f"  ⚠ LLM客户端初始化失败: {str(e)}")
                print("  ℹ 将跳过LLM摘要生成")
        except Exception as e:
            print(f"  ⚠ LLM客户端初始化失败: {str(e)}")
            print("  ℹ 将跳过LLM摘要生成")
    
    processor = MonthlyDataProcessor(llm_client=llm_client)
    
    # 提取静态文本
    static_texts = processor.extract_static_texts(static_docs)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(__file__),
        'data',
        f"{owner}_{repo}",
        f"monthly_data_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存并上传到MaxKB
    print("\n  → 保存描述文本并上传到MaxKB...")
    maxkb_dir = processor.save_for_maxkb(static_texts, output_dir)
    processor.upload_to_maxkb(maxkb_dir, owner, repo)
    
    # ========== 步骤3: 爬取issue等时序文本 ==========
    print("\n[3/4] 爬取Issue/PR/Commit/Release时序文本...")
    monthly_data_result = monthly_crawler.crawl_all_months(
        owner, repo, 
        max_per_month=max_per_month,
        progress_callback=lambda idx, title, desc, progress: print(f"  [{idx+1}] {title}: {desc}")
    )
    
    monthly_data = monthly_data_result['monthly_data']
    repo_info = monthly_data_result['repo_info']
    
    print(f"  ✓ 爬取了 {len(monthly_data)} 个月的数据")
    
    # ========== 步骤4: 时序文本+时序指标，按照月份时序对齐 ==========
    print("\n[4/4] 时序对齐：合并时序文本和时序指标...")
    processed_data = processor.process_monthly_data_for_model(monthly_data, opendigger_data)
    print(f"  ✓ 已完成时序对齐，共 {len(processed_data)} 个月的数据")
    
    # 保存所有数据
    print("\n  → 保存数据...")
    
    # 保存原始月度数据
    raw_data_file = os.path.join(output_dir, 'raw_monthly_data.json')
    with open(raw_data_file, 'w', encoding='utf-8') as f:
        json.dump({
            'repo_info': repo_info,
            'monthly_data': monthly_data,
            'opendigger_metrics': opendigger_data
        }, f, ensure_ascii=False, indent=2)
    
    # 保存用于双塔模型的数据（时序对齐后的数据）
    processor.save_for_model(processed_data, output_dir)
    
    # 保存元数据
    metadata = {
        'owner': owner,
        'repo': repo,
        'crawl_time': datetime.now().isoformat(),
        'months_count': len(monthly_data),
        'max_per_month': max_per_month,
        'llm_summary_enabled': enable_llm_summary,
        'opendigger_metrics_count': len(opendigger_data),
        'missing_opendigger_metrics': missing_metrics
    }
    
    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print("数据爬取和处理完成！")
    print(f"{'='*80}")
    print(f"\n输出目录: {output_dir}")
    print(f"  - 原始数据: raw_monthly_data.json")
    print(f"  - MaxKB文本: text_for_maxkb/")
    print(f"  - 双塔模型数据（时序对齐）: timeseries_for_model/")
    print(f"  - 元数据: metadata.json")
    print()
    
    return output_dir


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='按月爬取GitHub仓库数据')
    parser.add_argument('owner', help='仓库所有者')
    parser.add_argument('repo', help='仓库名称')
    parser.add_argument('--max-per-month', type=int, default=3, help='每月最多爬取的数量（默认3，即top 3）')
    parser.add_argument('--no-llm-summary', action='store_true', help='禁用LLM摘要生成')
    
    args = parser.parse_args()
    
    try:
        output_dir = crawl_project_monthly(
            args.owner,
            args.repo,
            max_per_month=args.max_per_month,
            enable_llm_summary=not args.no_llm_summary
        )
        print(f"\n✓ 成功完成！数据保存在: {output_dir}")
    except Exception as e:
        print(f"\n✗ 爬取失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

