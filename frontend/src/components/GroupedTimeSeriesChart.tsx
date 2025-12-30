import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Brush, ComposedChart
} from 'recharts'
import { Eye, EyeOff, ChevronDown, ChevronUp, Calendar, Maximize2, Minimize2, AlertCircle, TrendingUp } from 'lucide-react'
import type { GroupedTimeSeriesData, MetricGroupData } from '../types'
import PredictionChart from './PredictionChart'

interface GroupedTimeSeriesChartProps {
  data: GroupedTimeSeriesData
  onMonthClick: (month: string) => void
  repoKey?: string
}

// 分组图标配置
const GROUP_ICONS: Record<string, string> = {
  popularity: '⭐',
  development: '💻',
  issues: '📋',
  contributors: '👥',
  issue_response: '⏱️',
  pr_response: '🔄'
}

export default function GroupedTimeSeriesChart({ data, onMonthClick, repoKey }: GroupedTimeSeriesChartProps) {
  // 初始化时展开所有分组
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(() => {
    if (data?.groups) {
      return new Set(Object.keys(data.groups))
    }
    return new Set()
  })
  const [hiddenMetrics, setHiddenMetrics] = useState<Set<string>>(new Set())
  const [focusedGroup, setFocusedGroup] = useState<string | null>(null)
  const [predictionMetric, setPredictionMetric] = useState<{
    groupKey: string
    metricKey: string
    metricName: string
  } | null>(null)

  // 当数据变化时，确保所有分组都展开
  useEffect(() => {
    if (data?.groups) {
      const allGroupKeys = Object.keys(data.groups)
      setExpandedGroups(prev => {
        const newSet = new Set(prev)
        allGroupKeys.forEach(key => newSet.add(key))
        return newSet
      })
    }
  }, [data])

  const toggleGroup = (groupKey: string) => {
    setExpandedGroups(prev => {
      const newSet = new Set(prev)
      if (newSet.has(groupKey)) {
        newSet.delete(groupKey)
      } else {
        newSet.add(groupKey)
      }
      return newSet
    })
  }

  const toggleMetric = (metricKey: string) => {
    setHiddenMetrics(prev => {
      const newSet = new Set(prev)
      if (newSet.has(metricKey)) {
        newSet.delete(metricKey)
      } else {
        newSet.add(metricKey)
      }
      return newSet
    })
  }

  const focusGroup = (groupKey: string | null) => {
    setFocusedGroup(groupKey)
  }

  // 渲染单个分组的图表
  const renderGroupChart = (groupKey: string, groupData: MetricGroupData) => {
    const isExpanded = expandedGroups.has(groupKey)
    const isFocused = focusedGroup === groupKey
    
    // 转换数据格式用于 Recharts
    const chartData = data.timeAxis.map((month, index) => {
      const point: Record<string, string | number | null> = {
        month,
        displayMonth: month.slice(2), // 简化显示：20-01
        index
      }
      
      Object.entries(groupData.metrics).forEach(([metricKey, metricInfo]) => {
        // 原始数据（可能包含 null）
        point[metricKey] = metricInfo.data[index]
        // 插值数据（用于显示缺失点的位置）
        if (metricInfo.interpolated) {
          point[`${metricKey}_interpolated`] = metricInfo.interpolated[index]
        }
        // 标记是否为缺失值
        if (metricInfo.missingIndices?.includes(index)) {
          point[`${metricKey}_missing`] = 1  // 使用数字而非boolean
        }
      })
      
      return point
    })
    
    // 获取当前组的单位
    const firstMetric = Object.values(groupData.metrics)[0]
    const unit = firstMetric?.unit || ''
    
    // 统计缺失值数量
    const totalMissing = Object.values(groupData.metrics).reduce(
      (sum, m) => sum + (m.missingIndices?.length || 0),
      0
    )
    
    // 计算当前组所有可见指标的数据范围（排除null值）
    const calculateYAxisDomain = () => {
      const visibleMetrics = Object.entries(groupData.metrics).filter(
        ([metricKey]) => !hiddenMetrics.has(`${groupKey}-${metricKey}`)
      )
      
      if (visibleMetrics.length === 0) return ['auto', 'auto']
      
      let minValue = Infinity
      let maxValue = -Infinity
      
      visibleMetrics.forEach(([_, metricInfo]) => {
        metricInfo.data.forEach((value) => {
          if (value !== null && value !== undefined && typeof value === 'number') {
            const numValue = Number(value)
            if (!isNaN(numValue) && isFinite(numValue)) {
              minValue = Math.min(minValue, numValue)
              maxValue = Math.max(maxValue, numValue)
            }
          }
        })
      })
      
      // 如果没有有效数据，使用自动范围
      if (minValue === Infinity || maxValue === -Infinity) {
        return ['auto', 'auto']
      }
      
      // 判断所有数据是否都 >= 0
      const allNonNegative = minValue >= 0
      
      // 如果所有数据都是同一个值，使用合理的范围
      if (minValue === maxValue) {
        const singleValue = minValue
        if (singleValue === 0) {
          return [0, 1] // 如果值为0，显示0到1
        } else if (singleValue > 0) {
          return [0, singleValue * 1.2] // 正数：从0到值的120%
        } else {
          return [singleValue * 1.2, 0] // 负数：从值的120%到0
        }
      }
      
      // 计算数据范围
      const range = maxValue - minValue
      
      // 如果数据范围非常小（接近0），使用自动范围避免精度问题
      if (range < Number.EPSILON * 1000) {
        return ['auto', 'auto']
      }
      
      // 计算padding，使用相对较小的比例（5%），但确保不会过大
      // 对于很小的数据范围，使用固定的小padding
      let padding: number
      if (range < 1) {
        padding = Math.max(range * 0.1, 0.01) // 小范围数据使用10% padding，最小0.01
      } else if (range < 10) {
        padding = range * 0.05 // 中等范围使用5% padding
      } else {
        padding = range * 0.05 // 大范围使用5% padding
      }
      
      // 计算新的最小值和最大值
      let newMin = minValue - padding
      let newMax = maxValue + padding
      
      // 如果所有数据都 >= 0，确保最小值至少为0
      if (allNonNegative) {
        newMin = Math.max(0, newMin)
      }
      
      // 处理浮点数精度问题，四舍五入到合理的小数位数
      // 根据数据大小决定精度
      const getPrecision = (value: number) => {
        if (value >= 1000) return 0
        if (value >= 100) return 1
        if (value >= 10) return 2
        if (value >= 1) return 3
        return 4
      }
      
      const minPrecision = getPrecision(Math.abs(newMin))
      const maxPrecision = getPrecision(Math.abs(newMax))
      const precision = Math.max(minPrecision, maxPrecision)
      
      newMin = Number.parseFloat(newMin.toFixed(precision))
      newMax = Number.parseFloat(newMax.toFixed(precision))
      
      // 确保最小值小于最大值
      if (newMin >= newMax) {
        const mid = (minValue + maxValue) / 2
        // 使用数据范围的一半作为padding，但至少保证有合理的范围
        const halfRange = Math.max(range / 2, range * 0.1, 0.1)
        newMin = allNonNegative ? Math.max(0, mid - halfRange) : mid - halfRange
        newMax = mid + halfRange
        
        // 再次处理精度
        const fallbackPrecision = getPrecision(Math.abs(Math.max(Math.abs(newMin), Math.abs(newMax))))
        newMin = Number.parseFloat(newMin.toFixed(fallbackPrecision))
        newMax = Number.parseFloat(newMax.toFixed(fallbackPrecision))
        
        // 再次确保如果所有数据都 >= 0，最小值至少为0
        if (allNonNegative) {
          newMin = Math.max(0, newMin)
        }
      }
      
      return [newMin, newMax]
    }
    
    const yAxisDomain = calculateYAxisDomain()

    return (
      <motion.div
        key={groupKey}
        className={`bg-cyber-card/50 backdrop-blur-sm rounded-xl border border-cyber-border overflow-hidden transition-all duration-300 ${isFocused ? 'col-span-full' : ''}`}
        layout
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        {/* 分组头部 */}
        <div 
          className="px-4 py-3 border-b border-cyber-border flex items-center justify-between cursor-pointer hover:bg-cyber-surface/30 transition-colors"
          onClick={() => toggleGroup(groupKey)}
        >
          <div className="flex items-center gap-3">
            <span className="text-xl">{GROUP_ICONS[groupKey] || '📊'}</span>
            <div>
              <h3 className="text-lg font-display font-bold text-cyber-text">
                {groupData.name}
              </h3>
              <p className="text-xs text-cyber-muted font-chinese">
                {groupData.description}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            {/* 缺失值提示 */}
            {totalMissing > 0 && (
              <div className="flex items-center gap-1 px-2 py-1 bg-white/10 rounded text-xs text-white/70">
                <AlertCircle className="w-3 h-3" />
                <span>{totalMissing} 缺失</span>
              </div>
            )}
            
            {/* 全屏切换 */}
            <button
              onClick={(e) => {
                e.stopPropagation()
                focusGroup(isFocused ? null : groupKey)
              }}
              className="p-2 text-cyber-muted hover:text-cyber-primary transition-colors"
              title={isFocused ? '退出全屏' : '全屏查看'}
            >
              {isFocused ? (
                <Minimize2 className="w-4 h-4" />
              ) : (
                <Maximize2 className="w-4 h-4" />
              )}
            </button>
            
            {/* 展开/收起 */}
            <button className="p-2 text-cyber-muted hover:text-cyber-primary transition-colors">
              {isExpanded ? (
                <ChevronUp className="w-5 h-5" />
              ) : (
                <ChevronDown className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>

        {/* 图表内容 */}
        <AnimatePresence>
          {isExpanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              {/* 指标切换按钮 */}
              <div className="px-4 py-2 border-b border-cyber-border/50 flex flex-wrap gap-2">
                {Object.entries(groupData.metrics).map(([metricKey, metricInfo]) => {
                  const isHidden = hiddenMetrics.has(`${groupKey}-${metricKey}`)
                  const missingCount = metricInfo.missingIndices?.length || 0
                  
                  return (
                    <div key={metricKey} className="flex items-center gap-2">
                      <button
                        onClick={() => toggleMetric(`${groupKey}-${metricKey}`)}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-all ${
                          !isHidden
                            ? 'bg-cyber-surface border-2'
                            : 'bg-cyber-bg/50 border border-cyber-border opacity-50'
                        }`}
                        style={{
                          borderColor: !isHidden ? metricInfo.color : undefined
                        }}
                      >
                        {!isHidden ? (
                          <Eye className="w-3 h-3" />
                        ) : (
                          <EyeOff className="w-3 h-3" />
                        )}
                        <span className="font-chinese">{metricInfo.name}</span>
                        {metricInfo.unit && (
                          <span className="text-cyber-muted text-xs">({metricInfo.unit})</span>
                        )}
                        <div 
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: metricInfo.color }}
                        />
                        {missingCount > 0 && (
                          <span className="text-xs text-white/50 ml-1">
                            ({missingCount}缺失)
                          </span>
                        )}
                      </button>
                      {repoKey && (
                        <button
                          onClick={() =>
                            setPredictionMetric({
                              groupKey,
                              metricKey,
                              metricName: metricInfo.name
                            })
                          }
                          className="px-2 py-1.5 bg-yellow-500/20 hover:bg-yellow-500/30 border border-yellow-500/50 rounded-lg text-yellow-400 transition-colors"
                          title="预测未来趋势"
                        >
                          <TrendingUp className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  )
                })}
              </div>

              {/* 图表 */}
              <div className="p-4">
                <div className={isFocused ? 'h-[500px]' : 'h-[300px]'}>
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart
                      data={chartData}
                      margin={{ top: 10, right: 30, left: 10, bottom: 40 }}
                    >
                      <defs>
                        <filter id={`glow-${groupKey}`}>
                          <feGaussianBlur stdDeviation="2" result="coloredBlur" />
                          <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                          </feMerge>
                        </filter>
                      </defs>
                      
                      <CartesianGrid 
                        strokeDasharray="3 3" 
                        stroke="rgba(45, 58, 79, 0.5)"
                        vertical={false}
                      />
                      
                      <XAxis 
                        dataKey="displayMonth"
                        stroke="#8b97a8"
                        tick={{ fill: '#8b97a8', fontSize: 10 }}
                        tickLine={{ stroke: '#2d3a4f' }}
                        axisLine={{ stroke: '#2d3a4f' }}
                        interval="preserveStartEnd"
                        tickFormatter={(value) => {
                          const month = value.slice(3)
                          return month === '01' ? `20${value.slice(0, 2)}` : ''
                        }}
                      />
                      
                      <YAxis 
                        type="number"
                        domain={yAxisDomain}
                        stroke="#8b97a8"
                        tick={{ fill: '#8b97a8', fontSize: 10 }}
                        tickLine={{ stroke: '#2d3a4f' }}
                        axisLine={{ stroke: '#2d3a4f' }}
                        tickFormatter={(value) => {
                          // 处理NaN和Infinity
                          if (!isFinite(value)) return ''
                          
                          // 大数值使用k/M单位
                          if (Math.abs(value) >= 1000000) {
                            return `${(value / 1000000).toFixed(1)}M`
                          }
                          if (Math.abs(value) >= 1000) {
                            return `${(value / 1000).toFixed(1)}k`
                          }
                          
                          // 小数值根据精度格式化
                          if (Math.abs(value) < 1 && value !== 0) {
                            // 对于小于1的数值，保留最多3位有效数字
                            const absValue = Math.abs(value)
                            if (absValue >= 0.1) {
                              return value.toFixed(2)
                            } else if (absValue >= 0.01) {
                              return value.toFixed(3)
                            } else {
                              return value.toExponential(1)
                            }
                          }
                          
                          // 整数或接近整数的值
                          if (Math.abs(value - Math.round(value)) < 0.001) {
                            return Math.round(value).toString()
                          }
                          
                          // 其他情况保留最多2位小数
                          return value.toFixed(2)
                        }}
                        allowDataOverflow={false}
                      />
                      
                      <Tooltip 
                        content={({ active, payload, label }) => {
                          if (!active || !payload) return null
                          
                          return (
                            <div className="bg-cyber-card/95 backdrop-blur-md border border-cyber-border rounded-lg p-3 shadow-2xl">
                              <div className="flex items-center gap-2 mb-2 pb-2 border-b border-cyber-border">
                                <Calendar className="w-4 h-4 text-cyber-primary" />
                                <span className="text-cyber-text font-mono text-sm">20{label}</span>
                              </div>
                              <div className="space-y-1">
                                {payload
                                  .filter((entry: any) => !String(entry.dataKey || '').endsWith('_interpolated'))
                                  .map((entry: any, index: number) => {
                                    const isMissing = entry.value === null
                                    return (
                                      <div key={index} className="flex items-center justify-between gap-4">
                                        <div className="flex items-center gap-2">
                                          <div 
                                            className="w-2 h-2 rounded-full"
                                            style={{ backgroundColor: isMissing ? '#ffffff' : entry.color }}
                                          />
                                          <span className="text-cyber-muted text-xs font-chinese">
                                            {entry.name}
                                          </span>
                                        </div>
                                        <span
                                          className={`font-mono text-sm ${
                                            isMissing ? 'text-white/50 italic' : 'text-cyber-text'
                                          }`}
                                        >
                                          {isMissing
                                            ? '缺失'
                                            : typeof entry.value === 'number'
                                              ? entry.value.toLocaleString()
                                              : entry.value}
                                          {!isMissing && unit && (
                                            <span className="text-cyber-muted ml-1">{unit}</span>
                                          )}
                                        </span>
                                      </div>
                                    )
                                  })}
                              </div>
                              <button
                                onClick={() => onMonthClick(`20${label}`)}
                                className="mt-2 w-full py-1.5 bg-cyber-primary/20 hover:bg-cyber-primary/30 text-cyber-primary text-xs rounded transition-colors font-chinese"
                              >
                                查看 Issue 详情
                              </button>
                            </div>
                          )
                        }}
                      />
                      
                      <Legend 
                        wrapperStyle={{ paddingTop: '10px' }}
                        formatter={(value) => (
                          <span className="text-cyber-text text-xs font-chinese">{value}</span>
                        )}
                      />

                      {/* 渲染每个指标的线条 */}
                      {Object.entries(groupData.metrics).map(([metricKey, metricInfo]) => {
                        const isHidden = hiddenMetrics.has(`${groupKey}-${metricKey}`)
                        if (isHidden) return null
                        
                        return (
                          <Line
                            key={metricKey}
                            type="monotone"
                            dataKey={metricKey}
                            name={metricInfo.name}
                            stroke={metricInfo.color}
                            strokeWidth={2}
                            dot={(props: { cx?: number; cy?: number; index?: number; payload?: Record<string, unknown> }) => {
                              const { cx, cy, payload } = props
                              if (!cx || !cy) return <circle key={`empty-${metricKey}`} />
                              
                              // 检查是否为缺失值点
                              const isMissing = payload?.[`${metricKey}_missing`]
                              
                              if (isMissing) {
                                // 缺失值：显示白色空心圆点
                                return (
                                  <g key={`missing-${metricKey}-${props.index}`}>
                                    <circle
                                      cx={cx}
                                      cy={cy}
                                      r={6}
                                      fill="transparent"
                                      stroke="#ffffff"
                                      strokeWidth={2}
                                      strokeDasharray="3 2"
                                    />
                                    <circle
                                      cx={cx}
                                      cy={cy}
                                      r={3}
                                      fill="#ffffff"
                                      opacity={0.5}
                                    />
                                  </g>
                                )
                              }
                              
                              // 正常点：不显示（除了 hover 时）
                              return <circle key={`normal-${metricKey}-${props.index}`} />
                            }}
                            activeDot={{ 
                              r: 5, 
                              fill: metricInfo.color, 
                              filter: `url(#glow-${groupKey})` 
                            }}
                            connectNulls={false}
                          />
                        )
                      })}

                      {isFocused && (
                        <Brush
                          dataKey="displayMonth"
                          height={25}
                          stroke="#2d3a4f"
                          fill="#111827"
                          tickFormatter={() => ''}
                        />
                      )}
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
                
                {/* 图例说明 */}
                <div className="mt-2 flex items-center gap-4 text-xs text-cyber-muted">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full border-2 border-dashed border-white bg-white/20" />
                    <span>缺失数据点（位置为前后值平均）</span>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    )
  }

  // 如果没有数据
  if (!data?.groups || Object.keys(data.groups).length === 0) {
    return (
      <div className="bg-cyber-card/50 rounded-xl border border-cyber-border p-8 text-center">
        <AlertCircle className="w-12 h-12 text-cyber-muted mx-auto mb-4" />
        <p className="text-cyber-muted font-chinese">暂无时序数据</p>
        <p className="text-cyber-muted font-chinese text-sm mt-2">
          请确保 backend/Data 目录下有处理后的数据文件
        </p>
      </div>
    )
  }

  // 如果有聚焦的分组，只显示该分组
  if (focusedGroup && data.groups[focusedGroup]) {
    return (
      <div className="space-y-4">
        {renderGroupChart(focusedGroup, data.groups[focusedGroup])}
      </div>
    )
  }

  // 获取预测所需的历史数据
  const getHistoricalDataForPrediction = (groupKey: string, metricKey: string): Record<string, number> => {
    if (!data?.groups?.[groupKey]?.metrics?.[metricKey]) return {}
    
    const metricData = data.groups[groupKey].metrics[metricKey]
    const historicalData: Record<string, number> = {}
    
    data.timeAxis.forEach((month, index) => {
      if (metricData.data[index] !== null && metricData.data[index] !== undefined) {
        historicalData[month] = metricData.data[index] as number
      }
    })
    
    return historicalData
  }

  // 正常显示所有分组
  return (
    <div className="space-y-6">
      {/* 预测图表弹窗 */}
      <AnimatePresence>
        {predictionMetric && repoKey && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm"
            onClick={() => setPredictionMetric(null)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="max-w-4xl w-full max-h-[90vh] overflow-y-auto"
            >
              <PredictionChart
                repoKey={repoKey}
                metricName={predictionMetric.metricName}
                historicalData={getHistoricalDataForPrediction(
                  predictionMetric.groupKey,
                  predictionMetric.metricKey
                )}
                onClose={() => setPredictionMetric(null)}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 分组总览 */}
      <motion.div
        className="bg-cyber-card/50 backdrop-blur-sm rounded-xl border border-cyber-border p-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h2 className="text-xl font-display font-bold text-cyber-text mb-2">
          时序指标分组视图
        </h2>
        <p className="text-sm text-cyber-muted font-chinese mb-4">
          数据范围：{data.startMonth || data.timeAxis[0]} 至{' '}
          {data.endMonth || data.timeAxis[data.timeAxis.length - 1]} · 共{' '}
          {data.timeAxis.length} 个月 · {Object.keys(data.groups).length} 个分组
        </p>
        
        {/* 分组快捷入口 */}
        <div className="flex flex-wrap gap-2">
          {Object.entries(data.groups).map(([groupKey, groupData]) => {
            // 统计该组的缺失值
            const missingCount = Object.values(groupData.metrics).reduce(
              (sum, m) => sum + (m.missingIndices?.length || 0),
              0
            )
            
            return (
              <button
                key={groupKey}
                onClick={() => {
                  setExpandedGroups(prev => new Set([...prev, groupKey]))
                  document.getElementById(`group-${groupKey}`)?.scrollIntoView({ behavior: 'smooth' })
                }}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-all border border-cyber-border hover:border-cyber-primary/50 ${
                  expandedGroups.has(groupKey)
                    ? 'bg-cyber-primary/10 text-cyber-primary'
                    : 'bg-cyber-bg text-cyber-muted'
                }`}
              >
                <span>{GROUP_ICONS[groupKey] || '📊'}</span>
                <span className="font-chinese">{groupData.name}</span>
                <span className="text-xs opacity-60">
                  ({Object.keys(groupData.metrics).length}指标)
                </span>
                {missingCount > 0 && (
                  <span className="text-xs text-white/50">
                    {missingCount}缺失
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </motion.div>

      {/* 各分组图表 - 使用网格布局 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {Object.entries(data.groups).map(([groupKey, groupData]) => (
          <div key={groupKey} id={`group-${groupKey}`}>
            {renderGroupChart(groupKey, groupData)}
          </div>
        ))}
      </div>
    </div>
  )
}
