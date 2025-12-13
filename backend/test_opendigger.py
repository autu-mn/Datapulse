"""
OpenDigger MCP Client 测试脚本
测试所有 6 个功能是否正常工作
"""
import json
import sys
import io
from mcp_client import OpenDiggerMCPClient

# 设置输出编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_single_metric():
    """测试单个指标获取"""
    print_section("测试 1: 单个指标获取")
    
    client = OpenDiggerMCPClient()
    result = client.get_metric('microsoft', 'vscode', 'openrank')
    
    if result.get('success'):
        print("[OK] 成功获取指标")
        print(f"   仓库: {result.get('repository')}")
        print(f"   指标: {result.get('metric')}")
        data_keys = list(result.get('data', {}).keys())[:5]
        print(f"   数据键（前5个）: {data_keys}")
    else:
        print(f"[FAIL] 失败: {result.get('error')}")

def test_batch_metrics():
    """测试批量指标获取"""
    print_section("测试 2: 批量指标获取")
    
    client = OpenDiggerMCPClient()
    result = client.get_metrics_batch('microsoft', 'vscode', ['openrank', 'stars', 'contributors'])
    
    if result.get('success'):
        print("[OK] 成功批量获取指标")
        summary = result.get('summary', {})
        print(f"   总数: {summary.get('total')}")
        print(f"   成功: {summary.get('successful')}")
        print(f"   失败: {summary.get('failed')}")
        print(f"   仓库: {summary.get('repository')}")
        
        print("\n   结果列表:")
        for item in result.get('results', []):
            status = "[OK]" if item.get('success') else "[FAIL]"
            print(f"   {status} {item.get('metric')}")
    else:
        print(f"[FAIL] 失败: {result.get('error')}")

def test_compare_repos():
    """测试仓库对比"""
    print_section("测试 3: 仓库对比")
    
    client = OpenDiggerMCPClient()
    repos = [
        {'owner': 'microsoft', 'repo': 'vscode'},
        {'owner': 'facebook', 'repo': 'react'}
    ]
    result = client.compare_repositories(repos, ['openrank', 'stars'])
    
    if result.get('success'):
        print("[OK] 成功对比仓库")
        metadata = result.get('metadata', {})
        print(f"   仓库数: {metadata.get('repositoryCount')}")
        print(f"   对比指标: {metadata.get('metricsCompared')}")
        
        analysis = result.get('analysis', {})
        if analysis:
            print("\n   分析洞察:")
            for insight in analysis.get('insights', [])[:3]:
                print(f"   - {insight}")
            
            print("\n   健康评分:")
            for repo, score in analysis.get('healthScores', {}).items():
                print(f"   - {repo}: {score}%")
    else:
        print(f"[FAIL] 失败: {result.get('error')}")

def test_trends():
    """测试趋势分析"""
    print_section("测试 4: 趋势分析")
    
    client = OpenDiggerMCPClient()
    result = client.analyze_trends('microsoft', 'vscode', 'openrank')
    
    if result.get('success'):
        print("[OK] 成功分析趋势")
        trend_analysis = result.get('trendAnalysis', {})
        
        print(f"   数据点数: {trend_analysis.get('dataPoints')}")
        
        time_range = trend_analysis.get('timeRange', {})
        print(f"   时间范围: {time_range.get('start')} -> {time_range.get('end')}")
        
        trend = trend_analysis.get('trend', {})
        print(f"   趋势方向: {trend.get('direction')}")
        print(f"   增长率: {trend.get('growthRate')}")
        print(f"   动量: {trend.get('momentum')}")
        print(f"   波动性: {trend.get('volatility')}")
        
        values = trend_analysis.get('values', {})
        print(f"\n   数值统计:")
        print(f"   - 首值: {values.get('first')}")
        print(f"   - 末值: {values.get('last')}")
        print(f"   - 峰值: {values.get('peak')}")
        print(f"   - 平均: {values.get('average')}")
        
        patterns = trend_analysis.get('patterns', {})
        growth_phases = patterns.get('growthPhases', [])
        if growth_phases:
            print(f"\n   增长阶段: {len(growth_phases)} 个")
            for phase in growth_phases[:2]:
                print(f"   - {phase.get('phase')}: {phase.get('startDate')} -> {phase.get('endDate')} ({phase.get('growth'):+.2f})")
    else:
        print(f"[FAIL] 失败: {result.get('error')}")

def test_ecosystem():
    """测试生态系统洞察"""
    print_section("测试 5: 生态系统洞察")
    
    client = OpenDiggerMCPClient()
    result = client.get_ecosystem_insights('microsoft', 'vscode')
    
    if result.get('success'):
        print("[OK] 成功获取生态洞察")
        print(f"   仓库: {result.get('repository')}")
        print(f"   分析指标数: {result.get('metrics_analyzed')}")
        
        insights = result.get('insights', {})
        print(f"\n   关键指标:")
        for metric, data in list(insights.items())[:3]:
            if isinstance(data, dict):
                keys = list(data.keys())[:3]
                print(f"   - {metric}: {len(data)} 个数据点（{keys}...）")
    else:
        print(f"[FAIL] 失败: {result.get('error')}")

def test_health():
    """测试服务健康"""
    print_section("测试 6: 服务健康")
    
    client = OpenDiggerMCPClient()
    result = client.server_health()
    
    print("[OK] 服务健康检查")
    print(f"   状态: {result.get('status')}")
    print(f"   缓存大小: {result.get('cache_size')}")
    print(f"   缓存TTL: {result.get('cache_ttl')} 秒")
    print(f"   时间戳: {result.get('timestamp')}")

def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("  OpenDigger MCP Client 功能测试")
    print("=" * 60)
    
    try:
        test_single_metric()
        test_batch_metrics()
        test_compare_repos()
        test_trends()
        test_ecosystem()
        test_health()
        
        print("\n" + "=" * 60)
        print("  所有测试完成！")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

