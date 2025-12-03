"""
DataPulse 后端 API
GitHub 仓库生态画像分析平台 - 时序数据可视化与归因分析
从真实数据文件读取，动态确定时间范围
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
from datetime import datetime
from collections import defaultdict
import re
from data_service import DataService
from Agent.qa_agent import QAAgent

app = Flask(__name__)
CORS(app)

# 数据服务实例
data_service = DataService()

# AI Agent实例
qa_agent = QAAgent()


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


@app.route('/api/repos', methods=['GET'])
def get_repos():
    """获取已加载的仓库列表"""
    repos = data_service.get_loaded_repos()
    summaries = [data_service.get_repo_summary(repo) for repo in repos]
    return jsonify({
        'repos': repos,
        'summaries': summaries
    })


@app.route('/api/repo/<path:repo_key>/summary', methods=['GET'])
def get_repo_summary(repo_key):
    """获取仓库摘要"""
    try:
        # 支持两种格式：owner/repo 或 owner_repo
        # 如果是 owner_repo 格式，转换为 owner/repo
        if '_' in repo_key and '/' not in repo_key:
            repo_key = repo_key.replace('_', '/')
        
        summary = data_service.get_repo_summary(repo_key)
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/load', methods=['POST'])
def load_data():
    """加载数据文件"""
    data = request.json
    file_path = data.get('file_path')
    
    if not file_path:
        return jsonify({'error': '请提供数据文件路径'}), 400
    
    try:
        result = data_service.load_data(file_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/timeseries/grouped/<path:repo_key>', methods=['GET'])
def get_grouped_timeseries(repo_key):
    """
    获取分组时序数据 - 所有 OpenDigger 指标按类型分组
    动态确定时间范围，标记缺失值
    """
    try:
        # 支持两种格式：owner/repo 或 owner_repo
        if '_' in repo_key and '/' not in repo_key:
            repo_key = repo_key.replace('_', '/')
        
        grouped = data_service.get_grouped_timeseries(repo_key)
        return jsonify(grouped)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/issues/<path:repo_key>', methods=['GET'])
def get_issues_by_month(repo_key):
    """
    获取按月对齐的 Issue 数据
    包含：标签分类、高频关键词、重大事件
    """
    month = request.args.get('month')  # 可选参数，获取特定月份
    
    try:
        # 支持两种格式：owner/repo 或 owner_repo
        if '_' in repo_key and '/' not in repo_key:
            repo_key = repo_key.replace('_', '/')
        
        issues_data = data_service.get_aligned_issues(repo_key, month)
        return jsonify(issues_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analysis/<path:repo_key>', methods=['GET'])
def get_wave_analysis(repo_key):
    """
    波动归因分析
    识别指标的显著变化，并关联对应月份的 Issue 文本
    """
    try:
        # 支持两种格式：owner/repo 或 owner_repo
        if '_' in repo_key and '/' not in repo_key:
            repo_key = repo_key.replace('_', '/')
        
        analysis = data_service.analyze_waves(repo_key)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/keywords/<path:repo_key>/<month>', methods=['GET'])
def get_keywords(repo_key, month):
    """获取指定月份的关键词"""
    try:
        keywords = data_service.get_month_keywords(repo_key, month)
        return jsonify(keywords)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/events/<path:repo_key>', methods=['GET'])
def get_events(repo_key):
    """获取重大事件列表"""
    try:
        events = data_service.get_major_events(repo_key)
        return jsonify(events)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo', methods=['GET'])
def get_demo_data():
    """获取演示数据 - 优先使用真实数据"""
    return jsonify(data_service.get_demo_data())


@app.route('/api/metric-groups', methods=['GET'])
def get_metric_groups():
    """获取指标分组配置"""
    return jsonify(data_service.metric_groups)


@app.route('/api/projects', methods=['GET'])
def get_projects():
    """获取所有可用项目列表"""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), 'DataProcessor', 'data')
        if not os.path.exists(data_dir):
            return jsonify({'projects': [], 'default': None})
        
        projects = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                # 检查是否有processed文件夹
                has_processed = any(
                    '_processed' in f and os.path.isdir(os.path.join(item_path, f))
                    for f in os.listdir(item_path)
                )
                if has_processed:
                    summary = qa_agent.get_project_summary(item)
                    projects.append(summary)
        
        # 默认项目
        default_project = 'X-lab2017_open-digger'
        
        return jsonify({
            'projects': projects,
            'default': default_project
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/search', methods=['GET'])
def search_projects():
    """搜索项目"""
    try:
        query = request.args.get('q', '').strip().lower()
        if not query:
            return jsonify({'projects': []})
        
        data_dir = os.path.join(os.path.dirname(__file__), 'DataProcessor', 'data')
        if not os.path.exists(data_dir):
            return jsonify({'projects': []})
        
        results = []
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                # 检查是否有processed文件夹
                has_processed = any(
                    '_processed' in f and os.path.isdir(os.path.join(item_path, f))
                    for f in os.listdir(item_path)
                )
                if has_processed:
                    # 简单的名称匹配
                    if query in item.lower():
                        summary = qa_agent.get_project_summary(item)
                        results.append(summary)
        
        return jsonify({'projects': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/project/<path:project_name>/summary', methods=['GET'])
def get_project_summary(project_name):
    """获取项目摘要"""
    try:
        summary = qa_agent.get_project_summary(project_name)
        return jsonify(summary)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/qa', methods=['POST'])
def ask_question():
    """AI问答接口"""
    try:
        data = request.json
        question = data.get('question', '').strip()
        project_name = data.get('project', '')
        
        if not question:
            return jsonify({'error': '请提供问题'}), 400
        
        if not project_name:
            return jsonify({'error': '请指定项目'}), 400
        
        result = qa_agent.answer_question(question, project_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("DataPulse 后端服务启动")
    print("="*60)
    
    repos = data_service.get_loaded_repos()
    if repos:
        print(f"\n已加载 {len(repos)} 个仓库的数据:")
        for repo in repos:
            summary = data_service.get_repo_summary(repo)
            time_range = summary.get('timeRange', {})
            print(f"  - {repo}: {time_range.get('start', '?')} ~ {time_range.get('end', '?')} ({time_range.get('months', 0)} 个月)")
    else:
        print("\n警告: 没有找到数据文件")
        print("请将处理后的数据放入 backend/Data 目录")
    
    print("\n" + "="*60)
    
    app.run(debug=True, port=5000, host='0.0.0.0')
