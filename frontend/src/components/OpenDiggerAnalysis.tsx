import { useState } from 'react'
import { TrendingUp, BarChart3, GitCompare, Zap, Loader2, Activity, Users, Star, GitFork, Code, AlertCircle } from 'lucide-react'
import { Line, Bar, Radar } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'

// 注册 Chart.js 组件
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  RadialLinearScale,
  Title,
  Tooltip,
  Legend,
  Filler
)

interface OpenDiggerAnalysisProps {
  owner: string
  repo: string
}

export default function OpenDiggerAnalysis({ owner, repo }: OpenDiggerAnalysisProps) {
  const [loading, setLoading] = useState(false)
  const [activeView, setActiveView] = useState<'single' | 'batch' | 'compare' | 'trends' | 'ecosystem' | 'health'>('single')
  
  // 单个指标
  const [metricData, setMetricData] = useState<any>(null)
  const [selectedMetric, setSelectedMetric] = useState('openrank')
  
  // 批量指标
  const [batchData, setBatchData] = useState<any>(null)
  const [selectedBatchMetrics, setSelectedBatchMetrics] = useState<string[]>(['openrank', 'stars', 'contributors'])
  
  // 对比
  const [comparisonData, setComparisonData] = useState<any>(null)
  const [compareRepos, setCompareRepos] = useState<string>('')
  const [compareMetrics, setCompareMetrics] = useState<string[]>(['openrank', 'stars', 'contributors'])
  
  // 趋势
  const [trendData, setTrendData] = useState<any>(null)
  const [trendMetric, setTrendMetric] = useState('openrank')
  
  // 生态系统
  const [ecosystemData, setEcosystemData] = useState<any>(null)
  
  // 健康状态
  const [healthData, setHealthData] = useState<any>(null)

  const allMetrics = [
    { value: 'openrank', label: 'OpenRank (影响力)', icon: <Zap className="w-4 h-4" /> },
    { value: 'stars', label: 'Stars', icon: <Star className="w-4 h-4" /> },
    { value: 'forks', label: 'Forks', icon: <GitFork className="w-4 h-4" /> },
    { value: 'contributors', label: 'Contributors', icon: <Users className="w-4 h-4" /> },
    { value: 'activity', label: 'Activity (活跃度)', icon: <Activity className="w-4 h-4" /> },
    { value: 'participants', label: 'Participants', icon: <Users className="w-4 h-4" /> },
    { value: 'issues_new', label: 'New Issues', icon: <AlertCircle className="w-4 h-4" /> },
    { value: 'issues_closed', label: 'Closed Issues', icon: <AlertCircle className="w-4 h-4" /> },
    { value: 'pull_requests', label: 'Pull Requests', icon: <Code className="w-4 h-4" /> },
    { value: 'commits', label: 'Commits', icon: <Code className="w-4 h-4" /> },
  ]

  // 获取单个指标
  const fetchMetric = async (metric: string) => {
    setLoading(true)
    try {
      const response = await fetch(
        `/api/opendigger/metric?owner=${owner}&repo=${repo}&metric=${metric}`
      )
      const data = await response.json()
      setMetricData(data)
    } catch (error) {
      console.error('获取指标失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 批量获取指标
  const fetchBatchMetrics = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/opendigger/metrics/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          owner,
          repo,
          metrics: selectedBatchMetrics,
          platform: 'GitHub'
        })
      })
      const data = await response.json()
      setBatchData(data)
    } catch (error) {
      console.error('批量获取指标失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 对比仓库
  const fetchComparison = async () => {
    if (!compareRepos.trim()) {
      alert('请输入要对比的仓库（格式：owner/repo，多个用逗号分隔）')
      return
    }
    
    setLoading(true)
    try {
      const repos = compareRepos.split(',').map(r => {
        const [o, rp] = r.trim().split('/')
        return { owner: o, repo: rp }
      }).filter(r => r.owner && r.repo)
      
      // 添加当前仓库
      repos.unshift({ owner, repo })
      
      const response = await fetch('/api/opendigger/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repos,
          metrics: compareMetrics,
          platform: 'GitHub'
        })
      })
      const data = await response.json()
      setComparisonData(data)
    } catch (error) {
      console.error('对比失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 趋势分析
  const fetchTrends = async () => {
    setLoading(true)
    try {
      const response = await fetch(
        `/api/opendigger/trends?owner=${owner}&repo=${repo}&metric=${trendMetric}`
      )
      const data = await response.json()
      setTrendData(data)
    } catch (error) {
      console.error('获取趋势失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 生态系统洞察
  const fetchEcosystem = async () => {
    setLoading(true)
    try {
      const response = await fetch(
        `/api/opendigger/ecosystem?owner=${owner}&repo=${repo}`
      )
      const data = await response.json()
      setEcosystemData(data)
    } catch (error) {
      console.error('获取生态系统洞察失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 服务器健康状态
  const fetchHealth = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/opendigger/health')
      const data = await response.json()
      setHealthData(data)
    } catch (error) {
      console.error('获取健康状态失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 切换批量指标选择
  const toggleBatchMetric = (metric: string) => {
    setSelectedBatchMetrics(prev =>
      prev.includes(metric)
        ? prev.filter(m => m !== metric)
        : [...prev, metric]
    )
  }

  // 切换对比指标选择
  const toggleCompareMetric = (metric: string) => {
    setCompareMetrics(prev =>
      prev.includes(metric)
        ? prev.filter(m => m !== metric)
        : [...prev, metric]
    )
  }

  return (
    <div className="space-y-6">
      {/* 视图切换导航 */}
      <div className="bg-cyber-card/50 rounded-xl border border-cyber-border p-4">
        <div className="flex flex-wrap gap-2">
          <ViewButton
            active={activeView === 'single'}
            onClick={() => setActiveView('single')}
            icon={<BarChart3 className="w-4 h-4" />}
            label="单个指标"
          />
          <ViewButton
            active={activeView === 'batch'}
            onClick={() => setActiveView('batch')}
            icon={<Zap className="w-4 h-4" />}
            label="批量指标"
          />
          <ViewButton
            active={activeView === 'compare'}
            onClick={() => setActiveView('compare')}
            icon={<GitCompare className="w-4 h-4" />}
            label="仓库对比"
          />
          <ViewButton
            active={activeView === 'trends'}
            onClick={() => setActiveView('trends')}
            icon={<TrendingUp className="w-4 h-4" />}
            label="趋势分析"
          />
          <ViewButton
            active={activeView === 'ecosystem'}
            onClick={() => setActiveView('ecosystem')}
            icon={<Activity className="w-4 h-4" />}
            label="生态洞察"
          />
          <ViewButton
            active={activeView === 'health'}
            onClick={() => setActiveView('health')}
            icon={<AlertCircle className="w-4 h-4" />}
            label="服务健康"
          />
        </div>
      </div>

      {/* 单个指标视图 */}
      {activeView === 'single' && (
        <SingleMetricView
          metrics={allMetrics}
          selectedMetric={selectedMetric}
          setSelectedMetric={setSelectedMetric}
          loading={loading}
          fetchMetric={fetchMetric}
          metricData={metricData}
        />
      )}

      {/* 批量指标视图 */}
      {activeView === 'batch' && (
        <BatchMetricsView
          metrics={allMetrics}
          selectedMetrics={selectedBatchMetrics}
          toggleMetric={toggleBatchMetric}
          loading={loading}
          fetchBatch={fetchBatchMetrics}
          batchData={batchData}
        />
      )}

      {/* 仓库对比视图 */}
      {activeView === 'compare' && (
        <CompareView
          currentRepo={`${owner}/${repo}`}
          compareRepos={compareRepos}
          setCompareRepos={setCompareRepos}
          metrics={allMetrics}
          selectedMetrics={compareMetrics}
          toggleMetric={toggleCompareMetric}
          loading={loading}
          fetchComparison={fetchComparison}
          comparisonData={comparisonData}
        />
      )}

      {/* 趋势分析视图 */}
      {activeView === 'trends' && (
        <TrendsView
          metrics={allMetrics}
          selectedMetric={trendMetric}
          setSelectedMetric={setTrendMetric}
          loading={loading}
          fetchTrends={fetchTrends}
          trendData={trendData}
        />
      )}

      {/* 生态系统洞察视图 */}
      {activeView === 'ecosystem' && (
        <EcosystemView
          owner={owner}
          repo={repo}
          loading={loading}
          fetchEcosystem={fetchEcosystem}
          ecosystemData={ecosystemData}
        />
      )}

      {/* 服务健康视图 */}
      {activeView === 'health' && (
        <HealthView
          loading={loading}
          fetchHealth={fetchHealth}
          healthData={healthData}
        />
      )}
    </div>
  )
}

// 视图按钮组件
function ViewButton({ active, onClick, icon, label }: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
        active
          ? 'bg-cyber-primary text-white shadow-lg shadow-cyber-primary/30'
          : 'bg-cyber-card/50 text-cyber-muted hover:bg-cyber-card hover:text-cyber-text'
      }`}
    >
      {icon}
      <span className="font-medium">{label}</span>
    </button>
  )
}

// 单个指标视图组件
function SingleMetricView({ metrics, selectedMetric, setSelectedMetric, loading, fetchMetric, metricData }: any) {
  return (
    <div className="bg-cyber-card/50 rounded-xl border border-cyber-border p-6">
      <h3 className="text-lg font-display font-bold text-cyber-primary mb-4 flex items-center gap-2">
        <BarChart3 className="w-5 h-5" />
        获取单个指标
      </h3>

      <div className="mb-4">
        <label className="block text-sm text-cyber-muted mb-2">选择指标</label>
        <select
          value={selectedMetric}
          onChange={(e) => setSelectedMetric(e.target.value)}
          className="w-full px-4 py-2 bg-cyber-card border border-cyber-border rounded-lg text-cyber-text focus:outline-none focus:border-cyber-primary"
        >
          {metrics.map((m: any) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      <button
        onClick={() => fetchMetric(selectedMetric)}
        disabled={loading}
        className="flex items-center gap-2 px-6 py-3 bg-cyber-primary text-white rounded-lg hover:bg-cyber-primary/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <BarChart3 className="w-4 h-4" />}
        获取指标数据
      </button>

      {metricData && (
        <div className="mt-6 p-4 bg-cyber-card/30 rounded-lg border border-cyber-border">
          <h4 className="font-bold text-cyber-primary mb-3">指标数据</h4>
          <pre className="text-xs text-cyber-muted overflow-auto max-h-96 p-3 bg-black/20 rounded">
            {JSON.stringify(metricData, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

// 批量指标视图组件
function BatchMetricsView({ metrics, selectedMetrics, toggleMetric, loading, fetchBatch, batchData }: any) {
  return (
    <div className="bg-cyber-card/50 rounded-xl border border-cyber-border p-6">
      <h3 className="text-lg font-display font-bold text-cyber-primary mb-4 flex items-center gap-2">
        <Zap className="w-5 h-5" />
        批量获取指标
      </h3>

      <div className="mb-4">
        <label className="block text-sm text-cyber-muted mb-3">选择要获取的指标（可多选）</label>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
          {metrics.map((m: any) => (
            <button
              key={m.value}
              onClick={() => toggleMetric(m.value)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border transition-all ${
                selectedMetrics.includes(m.value)
                  ? 'bg-cyber-primary/20 border-cyber-primary text-cyber-primary'
                  : 'bg-cyber-card/30 border-cyber-border text-cyber-muted hover:border-cyber-primary/50'
              }`}
            >
              {m.icon}
              <span className="text-sm">{m.label}</span>
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={fetchBatch}
        disabled={loading || selectedMetrics.length === 0}
        className="flex items-center gap-2 px-6 py-3 bg-cyber-primary text-white rounded-lg hover:bg-cyber-primary/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
        批量获取 ({selectedMetrics.length} 个指标)
      </button>

      {batchData && (
        <div className="mt-6 space-y-4">
          <h4 className="font-bold text-cyber-primary">批量数据结果</h4>
          {batchData.results && batchData.results.map((result: any, idx: number) => (
            <div key={idx} className="p-4 bg-cyber-card/30 rounded-lg border border-cyber-border">
              <div className="flex items-center justify-between mb-2">
                <span className="font-semibold text-cyber-text">{result.metric}</span>
                <span className={`text-xs px-2 py-1 rounded ${result.success ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                  {result.success ? '成功' : '失败'}
                </span>
              </div>
              {result.success && result.data && (
                <pre className="text-xs text-cyber-muted overflow-auto max-h-40 p-2 bg-black/20 rounded">
                  {JSON.stringify(result.data, null, 2)}
                </pre>
              )}
              {!result.success && result.error && (
                <p className="text-xs text-red-400">{result.error}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// 仓库对比视图组件
function CompareView({ currentRepo, compareRepos, setCompareRepos, metrics, selectedMetrics, toggleMetric, loading, fetchComparison, comparisonData }: any) {
  return (
    <div className="bg-cyber-card/50 rounded-xl border border-cyber-border p-6">
      <h3 className="text-lg font-display font-bold text-cyber-primary mb-4 flex items-center gap-2">
        <GitCompare className="w-5 h-5" />
        仓库对比分析
      </h3>

      <div className="mb-4">
        <label className="block text-sm text-cyber-muted mb-2">
          当前仓库: <span className="text-cyber-primary font-semibold">{currentRepo}</span>
        </label>
        <label className="block text-sm text-cyber-muted mb-2">
          对比仓库（格式：owner/repo，多个用逗号分隔）
        </label>
        <input
          type="text"
          value={compareRepos}
          onChange={(e) => setCompareRepos(e.target.value)}
          placeholder="例如: facebook/react, vuejs/vue"
          className="w-full px-4 py-2 bg-cyber-card border border-cyber-border rounded-lg text-cyber-text focus:outline-none focus:border-cyber-primary"
        />
      </div>

      <div className="mb-4">
        <label className="block text-sm text-cyber-muted mb-3">选择对比指标</label>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
          {metrics.slice(0, 6).map((m: any) => (
            <button
              key={m.value}
              onClick={() => toggleMetric(m.value)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg border transition-all ${
                selectedMetrics.includes(m.value)
                  ? 'bg-cyber-primary/20 border-cyber-primary text-cyber-primary'
                  : 'bg-cyber-card/30 border-cyber-border text-cyber-muted hover:border-cyber-primary/50'
              }`}
            >
              {m.icon}
              <span className="text-sm">{m.label}</span>
            </button>
          ))}
        </div>
      </div>

      <button
        onClick={fetchComparison}
        disabled={loading || !compareRepos.trim()}
        className="flex items-center gap-2 px-6 py-3 bg-cyber-primary text-white rounded-lg hover:bg-cyber-primary/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <GitCompare className="w-4 h-4" />}
        开始对比
      </button>

      {comparisonData && comparisonData.comparison && (
        <div className="mt-6 space-y-4">
          <h4 className="font-bold text-cyber-primary">对比结果</h4>
          {comparisonData.comparison.map((repo: any, idx: number) => (
            <div key={idx} className="p-4 bg-cyber-card/30 rounded-lg border border-cyber-border">
              <h5 className="font-semibold text-cyber-text mb-3">{repo.repository}</h5>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {repo.metrics.map((metric: any, midx: number) => (
                  <div key={midx} className="p-2 bg-black/20 rounded">
                    <div className="text-xs text-cyber-muted mb-1">{metric.metric}</div>
                    {metric.success ? (
                      <div className="text-sm font-semibold text-cyber-primary">
                        {typeof metric.data === 'object' ? '查看详情 ↓' : metric.data}
                      </div>
                    ) : (
                      <div className="text-xs text-red-400">失败</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
          
          {comparisonData.analysis && (
            <div className="p-4 bg-cyber-primary/10 rounded-lg border border-cyber-primary/30">
              <h5 className="font-semibold text-cyber-primary mb-2">分析洞察</h5>
              {comparisonData.analysis.insights && comparisonData.analysis.insights.map((insight: string, idx: number) => (
                <p key={idx} className="text-sm text-cyber-text mb-1">• {insight}</p>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// 趋势分析视图组件
function TrendsView({ metrics, selectedMetric, setSelectedMetric, loading, fetchTrends, trendData }: any) {
  return (
    <div className="bg-cyber-card/50 rounded-xl border border-cyber-border p-6">
      <h3 className="text-lg font-display font-bold text-cyber-primary mb-4 flex items-center gap-2">
        <TrendingUp className="w-5 h-5" />
        趋势分析
      </h3>

      <div className="mb-4">
        <label className="block text-sm text-cyber-muted mb-2">选择指标</label>
        <select
          value={selectedMetric}
          onChange={(e) => setSelectedMetric(e.target.value)}
          className="w-full px-4 py-2 bg-cyber-card border border-cyber-border rounded-lg text-cyber-text focus:outline-none focus:border-cyber-primary"
        >
          {metrics.slice(0, 5).map((m: any) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      <button
        onClick={fetchTrends}
        disabled={loading}
        className="flex items-center gap-2 px-6 py-3 bg-cyber-primary text-white rounded-lg hover:bg-cyber-primary/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <TrendingUp className="w-4 h-4" />}
        分析趋势
      </button>

      {trendData && trendData.trendAnalysis && (
        <div className="mt-6 space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard label="数据点数" value={trendData.trendAnalysis.dataPoints} />
            <StatCard label="趋势方向" value={trendData.trendAnalysis.trend?.direction || 'N/A'} />
            <StatCard label="增长率" value={trendData.trendAnalysis.trend?.growthRate || 'N/A'} />
            <StatCard label="波动性" value={trendData.trendAnalysis.trend?.volatility || 'N/A'} />
          </div>

          <div className="p-4 bg-cyber-card/30 rounded-lg border border-cyber-border">
            <h5 className="font-semibold text-cyber-primary mb-3">数值统计</h5>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
              <div><span className="text-cyber-muted">首值:</span> <span className="text-cyber-text font-semibold">{trendData.trendAnalysis.values?.first || 0}</span></div>
              <div><span className="text-cyber-muted">末值:</span> <span className="text-cyber-text font-semibold">{trendData.trendAnalysis.values?.last || 0}</span></div>
              <div><span className="text-cyber-muted">峰值:</span> <span className="text-cyber-text font-semibold">{trendData.trendAnalysis.values?.peak || 0}</span></div>
              <div><span className="text-cyber-muted">最低:</span> <span className="text-cyber-text font-semibold">{trendData.trendAnalysis.values?.lowest || 0}</span></div>
              <div><span className="text-cyber-muted">平均:</span> <span className="text-cyber-text font-semibold">{trendData.trendAnalysis.values?.average?.toFixed(2) || 0}</span></div>
              <div><span className="text-cyber-muted">中位数:</span> <span className="text-cyber-text font-semibold">{trendData.trendAnalysis.values?.median?.toFixed(2) || 0}</span></div>
            </div>
          </div>

          {trendData.trendAnalysis.patterns?.growthPhases && trendData.trendAnalysis.patterns.growthPhases.length > 0 && (
            <div className="p-4 bg-cyber-card/30 rounded-lg border border-cyber-border">
              <h5 className="font-semibold text-cyber-primary mb-3">增长阶段</h5>
              <div className="space-y-2">
                {trendData.trendAnalysis.patterns.growthPhases.map((phase: any, idx: number) => (
                  <div key={idx} className="flex items-center justify-between text-sm p-2 bg-black/20 rounded">
                    <span className="text-cyber-muted">{phase.startDate} → {phase.endDate}</span>
                    <span className={`font-semibold ${phase.growth > 0 ? 'text-green-400' : phase.growth < 0 ? 'text-red-400' : 'text-cyber-muted'}`}>
                      {phase.phase} ({phase.growth > 0 ? '+' : ''}{phase.growth})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// 生态系统洞察视图组件
function EcosystemView({ owner, repo, loading, fetchEcosystem, ecosystemData }: any) {
  // 提取最新值
  const extractLatestValue = (data: any): number => {
    if (typeof data === 'number') return data
    if (typeof data === 'object' && data !== null) {
      const monthKeys = Object.keys(data).filter(k => k.match(/^\d{4}-\d{2}$/)).sort()
      if (monthKeys.length > 0) {
        const latestKey = monthKeys[monthKeys.length - 1]
        return data[latestKey] || 0
      }
    }
    return 0
  }

  return (
    <div className="bg-cyber-card/50 rounded-xl border border-cyber-border p-6">
      <h3 className="text-lg font-display font-bold text-cyber-primary mb-4 flex items-center gap-2">
        <Activity className="w-5 h-5" />
        生态系统洞察
      </h3>

      <p className="text-sm text-cyber-muted mb-4">
        获取 <span className="text-cyber-primary font-semibold">{owner}/{repo}</span> 的生态系统综合分析
      </p>

      <button
        onClick={fetchEcosystem}
        disabled={loading}
        className="flex items-center gap-2 px-6 py-3 bg-cyber-primary text-white rounded-lg hover:bg-cyber-primary/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />}
        获取生态洞察
      </button>

      {ecosystemData && ecosystemData.insights && (
        <div className="mt-6 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-bold text-cyber-primary">生态系统分析</h4>
            <span className="text-sm text-cyber-muted">
              分析了 {ecosystemData.metrics_analyzed || 0} 个关键指标
            </span>
          </div>

          {/* 关键指标卡片 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(ecosystemData.insights).map(([metric, data]: [string, any]) => {
              const latestValue = extractLatestValue(data)
              const dataPoints = typeof data === 'object' ? Object.keys(data).filter(k => k.match(/^\d{4}-\d{2}$/)).length : 0
              
              return (
                <div key={metric} className="p-4 bg-cyber-card/30 rounded-lg border border-cyber-border">
                  <div className="text-xs text-cyber-muted mb-1 capitalize">{metric}</div>
                  <div className="text-2xl font-bold text-cyber-primary mb-1">
                    {latestValue.toFixed(0)}
                  </div>
                  <div className="text-xs text-cyber-muted">
                    {dataPoints} 个数据点
                  </div>
                </div>
              )
            })}
          </div>

          {/* 详细数据（可折叠） */}
          <details className="p-4 bg-cyber-card/30 rounded-lg border border-cyber-border">
            <summary className="cursor-pointer font-semibold text-cyber-primary mb-2">
              查看详细数据
            </summary>
            <pre className="text-xs text-cyber-muted overflow-auto max-h-96 p-3 bg-black/20 rounded mt-2">
              {JSON.stringify(ecosystemData, null, 2)}
            </pre>
          </details>
        </div>
      )}

      {ecosystemData && !ecosystemData.insights && (
        <div className="mt-6 p-4 bg-yellow-500/10 rounded-lg border border-yellow-500/30">
          <p className="text-sm text-yellow-400">
            未获取到生态系统数据。可能该仓库在 OpenDigger 中没有足够的数据。
          </p>
        </div>
      )}
    </div>
  )
}

// 服务健康视图组件
function HealthView({ loading, fetchHealth, healthData }: any) {
  return (
    <div className="bg-cyber-card/50 rounded-xl border border-cyber-border p-6">
      <h3 className="text-lg font-display font-bold text-cyber-primary mb-4 flex items-center gap-2">
        <AlertCircle className="w-5 h-5" />
        服务器健康状态
      </h3>

      <button
        onClick={fetchHealth}
        disabled={loading}
        className="flex items-center gap-2 px-6 py-3 bg-cyber-primary text-white rounded-lg hover:bg-cyber-primary/80 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <AlertCircle className="w-4 h-4" />}
        检查健康状态
      </button>

      {healthData && (
        <div className="mt-6 space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <StatCard 
              label="状态" 
              value={healthData.status || 'unknown'} 
              valueColor={healthData.status === 'healthy' ? 'text-green-400' : 'text-red-400'} 
            />
            <StatCard 
              label="版本" 
              value={healthData.version || 'N/A'} 
            />
            <StatCard 
              label="运行时间" 
              value={healthData.uptime ? `${(healthData.uptime / 60).toFixed(1)} 分钟` : 'N/A'} 
            />
            <StatCard 
              label="缓存条目" 
              value={healthData.cache?.size || 0} 
            />
          </div>

          {healthData.cache && (
            <div className="p-4 bg-cyber-card/30 rounded-lg border border-cyber-border">
              <h5 className="font-semibold text-cyber-primary mb-3">缓存统计</h5>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                <div>
                  <span className="text-cyber-muted">缓存大小:</span>{' '}
                  <span className="text-cyber-text font-semibold">{healthData.cache.size || 0}</span>
                </div>
                <div>
                  <span className="text-cyber-muted">TTL:</span>{' '}
                  <span className="text-cyber-text font-semibold">{healthData.cache.ttl || 0} 秒</span>
                </div>
                <div>
                  <span className="text-cyber-muted">条目数:</span>{' '}
                  <span className="text-cyber-text font-semibold">{healthData.cache.entries?.length || 0}</span>
                </div>
              </div>
              {healthData.cache.entries && healthData.cache.entries.length > 0 && (
                <div className="mt-3">
                  <div className="text-xs text-cyber-muted mb-1">最近缓存的键（前5个）:</div>
                  <div className="flex flex-wrap gap-2">
                    {healthData.cache.entries.map((key: string, idx: number) => (
                      <span key={idx} className="text-xs px-2 py-1 bg-cyber-primary/10 text-cyber-primary rounded">
                        {key}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {healthData.memory && (
            <div className="p-4 bg-cyber-card/30 rounded-lg border border-cyber-border">
              <h5 className="font-semibold text-cyber-primary mb-3">内存使用</h5>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                <div>
                  <span className="text-cyber-muted">RSS:</span>{' '}
                  <span className="text-cyber-text font-semibold">
                    {(healthData.memory.rss / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
                <div>
                  <span className="text-cyber-muted">VMS:</span>{' '}
                  <span className="text-cyber-text font-semibold">
                    {(healthData.memory.vms / 1024 / 1024).toFixed(2)} MB
                  </span>
                </div>
                <div>
                  <span className="text-cyber-muted">使用率:</span>{' '}
                  <span className="text-cyber-text font-semibold">
                    {healthData.memory.percent?.toFixed(2) || 0}%
                  </span>
                </div>
              </div>
            </div>
          )}

          <div className="p-4 bg-cyber-card/30 rounded-lg border border-cyber-border">
            <h5 className="font-semibold text-cyber-primary mb-2">时间戳</h5>
            <p className="text-sm text-cyber-muted">{healthData.timestamp}</p>
          </div>
        </div>
      )}

      {healthData && !healthData.status && (
        <div className="mt-6 p-4 bg-red-500/10 rounded-lg border border-red-500/30">
          <p className="text-sm text-red-400">
            无法获取健康状态数据。请检查后端服务是否正常运行。
          </p>
        </div>
      )}
    </div>
  )
}

// 统计卡片组件
function StatCard({ label, value, valueColor = 'text-cyber-primary' }: { label: string; value: any; valueColor?: string }) {
  return (
    <div className="p-3 bg-cyber-card/30 rounded-lg border border-cyber-border">
      <div className="text-xs text-cyber-muted mb-1">{label}</div>
      <div className={`text-lg font-bold ${valueColor}`}>{value}</div>
    </div>
  )
}

