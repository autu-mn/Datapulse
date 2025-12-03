import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  TrendingUp, TrendingDown, AlertTriangle, ChevronDown, ChevronUp,
  Calendar, Tag, ExternalLink, Zap
} from 'lucide-react'
import type { WaveData } from '../types'

interface WaveAnalysisProps {
  waves: WaveData[]
  onWaveClick: (wave: WaveData) => void
}

export default function WaveAnalysis({ waves, onWaveClick }: WaveAnalysisProps) {
  const [expandedWave, setExpandedWave] = useState<number | null>(null)
  const [filterTrend, setFilterTrend] = useState<'all' | '上升' | '下降'>('all')
  const [filterMetric, setFilterMetric] = useState<string>('all')

  // 获取所有指标类型
  const metrics = [...new Set(waves?.map(w => w.metric) || [])]

  // 过滤波动
  const filteredWaves = waves?.filter(wave => {
    if (filterTrend !== 'all' && wave.trend !== filterTrend) return false
    if (filterMetric !== 'all' && wave.metric !== filterMetric) return false
    return true
  }) || []

  // 统计
  const stats = {
    total: waves?.length || 0,
    rising: waves?.filter(w => w.trend === '上升').length || 0,
    falling: waves?.filter(w => w.trend === '下降').length || 0,
    avgChange: waves?.length 
      ? (waves.reduce((sum, w) => sum + Math.abs(w.changeRate), 0) / waves.length).toFixed(1)
      : 0
  }

  const toggleExpand = (index: number) => {
    setExpandedWave(expandedWave === index ? null : index)
  }

  return (
    <div className="space-y-6">
      {/* 统计概览 */}
      <motion.div
        className="grid grid-cols-2 md:grid-cols-4 gap-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <StatBox 
          label="总波动数" 
          value={stats.total} 
          icon={<AlertTriangle className="w-5 h-5" />}
          color="warning"
        />
        <StatBox 
          label="上升波动" 
          value={stats.rising} 
          icon={<TrendingUp className="w-5 h-5" />}
          color="success"
        />
        <StatBox 
          label="下降波动" 
          value={stats.falling} 
          icon={<TrendingDown className="w-5 h-5" />}
          color="accent"
        />
        <StatBox 
          label="平均变化率" 
          value={`${stats.avgChange}%`} 
          icon={<Zap className="w-5 h-5" />}
          color="primary"
        />
      </motion.div>

      {/* 过滤器 */}
      <motion.div
        className="flex flex-wrap gap-4 p-4 bg-cyber-card/50 rounded-xl border border-cyber-border"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
      >
        <div className="flex items-center gap-2">
          <span className="text-sm text-cyber-muted font-chinese">趋势:</span>
          <div className="flex bg-cyber-bg rounded-lg p-1">
            {(['all', '上升', '下降'] as const).map(trend => (
              <button
                key={trend}
                onClick={() => setFilterTrend(trend)}
                className={`
                  px-3 py-1 rounded text-sm transition-all font-chinese
                  ${filterTrend === trend 
                    ? 'bg-cyber-primary/20 text-cyber-primary' 
                    : 'text-cyber-muted hover:text-cyber-text'
                  }
                `}
              >
                {trend === 'all' ? '全部' : trend}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-cyber-muted font-chinese">指标:</span>
          <select
            value={filterMetric}
            onChange={(e) => setFilterMetric(e.target.value)}
            className="bg-cyber-bg border border-cyber-border rounded-lg px-3 py-1.5 text-sm
                     text-cyber-text focus:outline-none focus:border-cyber-primary"
          >
            <option value="all">全部指标</option>
            {metrics.map(metric => (
              <option key={metric} value={metric}>{metric}</option>
            ))}
          </select>
        </div>

        <div className="ml-auto text-sm text-cyber-muted font-chinese">
          显示 {filteredWaves.length} / {waves?.length || 0} 个波动
        </div>
      </motion.div>

      {/* 波动列表 */}
      <motion.div
        className="bg-cyber-card/50 backdrop-blur-sm rounded-xl border border-cyber-border overflow-hidden"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="px-6 py-4 border-b border-cyber-border">
          <h2 className="text-xl font-display font-bold text-cyber-text">
            波动归因分析
          </h2>
          <p className="text-sm text-cyber-muted font-chinese mt-1">
            识别指标的显著变化，关联对应月份的 Issue 文本进行归因
          </p>
        </div>

        <div className="divide-y divide-cyber-border">
          <AnimatePresence>
            {filteredWaves.map((wave, index) => (
              <motion.div
                key={`${wave.month}-${wave.metric}-${index}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ delay: index * 0.03 }}
                className="hover:bg-cyber-surface/30 transition-colors"
              >
                {/* 波动主要信息 */}
                <button
                  onClick={() => toggleExpand(index)}
                  className="w-full px-6 py-4 flex items-center gap-4 text-left"
                >
                  {/* 趋势指示器 */}
                  <div className={`
                    w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0
                    ${wave.trend === '上升' 
                      ? 'bg-cyber-success/20 text-cyber-success' 
                      : 'bg-cyber-accent/20 text-cyber-accent'
                    }
                  `}>
                    {wave.trend === '上升' 
                      ? <TrendingUp className="w-6 h-6" />
                      : <TrendingDown className="w-6 h-6" />
                    }
                  </div>

                  {/* 信息 */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-3 mb-1">
                      <span className="font-mono text-cyber-text">{wave.month}</span>
                      {wave.group && (
                        <span className="px-2 py-0.5 bg-cyber-secondary/20 rounded text-xs text-cyber-secondary">
                          {wave.group}
                        </span>
                      )}
                      <span className="px-2 py-0.5 bg-cyber-bg rounded text-xs text-cyber-muted">
                        {wave.metric}
                      </span>
                      <span className={`
                        font-mono font-semibold
                        ${wave.trend === '上升' ? 'text-cyber-success' : 'text-cyber-accent'}
                      `}>
                        {wave.changeRate > 0 ? '+' : ''}{wave.changeRate}%
                      </span>
                    </div>
                    <div className="text-sm text-cyber-muted truncate font-chinese">
                      {wave.previousValue.toLocaleString()} → {wave.currentValue.toLocaleString()}
                    </div>
                  </div>

                  {/* 关键词预览 */}
                  <div className="hidden md:flex items-center gap-2 flex-shrink-0">
                    {wave.keywords?.slice(0, 3).map((kw, idx) => (
                      <span 
                        key={idx}
                        className="px-2 py-1 bg-cyber-bg rounded text-xs text-cyber-muted"
                      >
                        {kw.word}
                      </span>
                    ))}
                  </div>

                  {/* 展开指示器 */}
                  <div className="text-cyber-muted">
                    {expandedWave === index 
                      ? <ChevronUp className="w-5 h-5" />
                      : <ChevronDown className="w-5 h-5" />
                    }
                  </div>
                </button>

                {/* 展开详情 */}
                <AnimatePresence>
                  {expandedWave === index && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: 'auto', opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="overflow-hidden"
                    >
                      <div className="px-6 pb-6 pt-2 ml-16 space-y-4">
                        {/* 归因解释 */}
                        <div className="p-4 bg-cyber-bg/50 rounded-lg border border-cyber-border">
                          <h4 className="text-sm font-semibold text-cyber-primary mb-2 font-chinese flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4" />
                            归因分析
                          </h4>
                          <p className="text-cyber-text font-chinese text-sm leading-relaxed">
                            {wave.explanation}
                          </p>
                        </div>

                        {/* 关键词详情 */}
                        {wave.keywords && wave.keywords.length > 0 && (
                          <div>
                            <h4 className="text-sm font-semibold text-cyber-muted mb-2 font-chinese flex items-center gap-2">
                              <Tag className="w-4 h-4" />
                              当月高频关键词
                            </h4>
                            <div className="flex flex-wrap gap-2">
                              {wave.keywords.map((kw, idx) => (
                                <span
                                  key={idx}
                                  className="px-3 py-1.5 bg-cyber-surface rounded-lg text-sm
                                           border border-cyber-border hover:border-cyber-primary
                                           transition-colors cursor-default"
                                  style={{
                                    opacity: 0.5 + kw.weight * 0.5
                                  }}
                                >
                                  <span className="text-cyber-text">{kw.word}</span>
                                  <span className="ml-2 text-cyber-muted text-xs">
                                    {(kw.weight * 100).toFixed(0)}%
                                  </span>
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* 重大事件 */}
                        {wave.events && wave.events.length > 0 && (
                          <div>
                            <h4 className="text-sm font-semibold text-cyber-muted mb-2 font-chinese flex items-center gap-2">
                              <Calendar className="w-4 h-4" />
                              重大事件
                            </h4>
                            <div className="space-y-2">
                              {wave.events.map((event, idx) => (
                                <div 
                                  key={idx}
                                  className="flex items-center gap-3 p-3 bg-cyber-surface rounded-lg
                                           border border-cyber-border hover:border-cyber-primary transition-colors"
                                >
                                  <span className="text-cyber-primary font-mono">
                                    #{event.number}
                                  </span>
                                  <span className="text-cyber-text text-sm flex-1 truncate">
                                    {event.title}
                                  </span>
                                  {event.url && (
                                    <a 
                                      href={event.url} 
                                      target="_blank" 
                                      rel="noopener noreferrer"
                                      className="text-cyber-muted hover:text-cyber-primary"
                                    >
                                      <ExternalLink className="w-4 h-4" />
                                    </a>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* 操作按钮 */}
                        <div className="flex gap-3 pt-2">
                          <button
                            onClick={() => onWaveClick(wave)}
                            className="px-4 py-2 bg-cyber-primary/20 text-cyber-primary rounded-lg
                                     hover:bg-cyber-primary/30 transition-colors text-sm font-chinese"
                          >
                            查看该月 Issue 详情
                          </button>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        {filteredWaves.length === 0 && (
          <div className="py-12 text-center">
            <AlertTriangle className="w-12 h-12 text-cyber-muted mx-auto mb-4" />
            <p className="text-cyber-muted font-chinese">没有符合条件的波动数据</p>
          </div>
        )}
      </motion.div>
    </div>
  )
}

// 统计卡片组件
function StatBox({ label, value, icon, color }: {
  label: string
  value: number | string
  icon: React.ReactNode
  color: 'primary' | 'success' | 'accent' | 'warning'
}) {
  const colorClasses = {
    primary: 'text-cyber-primary bg-cyber-primary/10 border-cyber-primary/30',
    success: 'text-cyber-success bg-cyber-success/10 border-cyber-success/30',
    accent: 'text-cyber-accent bg-cyber-accent/10 border-cyber-accent/30',
    warning: 'text-cyber-warning bg-cyber-warning/10 border-cyber-warning/30'
  }

  return (
    <div className={`
      p-4 rounded-xl border backdrop-blur-sm
      ${colorClasses[color]}
    `}>
      <div className="flex items-center gap-3">
        <div className="opacity-80">{icon}</div>
        <div>
          <div className="text-2xl font-display font-bold">{value}</div>
          <div className="text-sm opacity-70 font-chinese">{label}</div>
        </div>
      </div>
    </div>
  )
}

