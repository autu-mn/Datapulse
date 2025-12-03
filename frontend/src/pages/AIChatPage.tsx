import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, ArrowLeft } from 'lucide-react'
import { motion } from 'framer-motion'
import { useNavigate, useSearchParams } from 'react-router-dom'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  confidence?: number
}

export default function AIChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [projectName, setProjectName] = useState<string>('')
  const [projects, setProjects] = useState<Array<{name: string, repo?: string}>>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()

  useEffect(() => {
    // 从URL参数获取项目名称
    const project = searchParams.get('project') || 'X-lab2017_open-digger'
    setProjectName(project)
    
    // 加载项目列表
    fetchProjects()
    
    // 初始化欢迎消息
    setMessages([{
      role: 'assistant',
      content: `你好！我是项目数据分析助手。我可以帮你了解 ${project.replace('_', '/')} 项目的相关信息。\n\n你可以问我：\n- 项目的基本信息\n- 统计数据\n- Issue情况\n- 时序趋势`,
    }])
  }, [searchParams])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const fetchProjects = async () => {
    try {
      const response = await fetch('/api/projects')
      const data = await response.json()
      setProjects(data.projects || [])
    } catch (error) {
      console.error('Failed to fetch projects:', error)
    }
  }

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleSend = async () => {
    if (!input.trim() || loading || !projectName) return

    const userMessage: Message = {
      role: 'user',
      content: input.trim()
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch('/api/qa', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userMessage.content,
          project: projectName
        })
      })

      const data = await response.json()

      if (data.error) {
        throw new Error(data.error)
      }

      const assistantMessage: Message = {
        role: 'assistant',
        content: data.answer || '抱歉，我无法回答这个问题。',
        sources: data.sources || [],
        confidence: data.confidence || 0
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: `抱歉，发生了错误：${error instanceof Error ? error.message : '未知错误'}`
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleProjectChange = (project: string) => {
    setProjectName(project)
    // 更新URL参数
    navigate(`/ai?project=${encodeURIComponent(project)}`)
    // 重置对话
    setMessages([{
      role: 'assistant',
      content: `已切换到 ${project.replace('_', '/')} 项目。我可以帮你了解这个项目的相关信息。\n\n你可以问我：\n- 项目的基本信息\n- 统计数据\n- Issue情况\n- 时序趋势`,
    }])
  }

  return (
    <div className="min-h-screen bg-cyber-bg bg-cyber-grid">
      {/* 背景发光效果 */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyber-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyber-secondary/5 rounded-full blur-3xl" />
      </div>

      {/* 顶部导航栏 */}
      <header className="relative border-b border-cyber-border bg-cyber-surface/80 backdrop-blur-xl">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <button
              onClick={() => navigate('/')}
              className="flex items-center gap-2 px-4 py-2 text-cyber-muted hover:text-cyber-primary transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span className="font-chinese">返回主页</span>
            </button>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <Bot className="w-6 h-6 text-cyber-primary" />
                <h1 className="text-xl font-display font-bold text-cyber-text">AI 数据分析助手</h1>
              </div>
              
              {/* 项目选择 */}
              <select
                value={projectName}
                onChange={(e) => handleProjectChange(e.target.value)}
                className="px-4 py-2 bg-cyber-card border border-cyber-border rounded-lg
                         text-cyber-text focus:outline-none focus:border-cyber-primary
                         font-chinese text-sm"
              >
                {projects.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.repo || p.name.replace('_', '/')}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </header>

      {/* 主内容区 */}
      <main className="relative z-10 container mx-auto px-4 py-8 max-w-4xl">
        <motion.div
          className="bg-cyber-card/30 rounded-lg border border-cyber-border shadow-xl"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          {/* 消息列表 */}
          <div className="h-[calc(100vh-280px)] overflow-y-auto p-6 space-y-4">
            {messages.map((message, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                {message.role === 'assistant' && (
                  <div className="w-10 h-10 rounded-full bg-cyber-primary/20 flex items-center justify-center flex-shrink-0">
                    <Bot className="w-5 h-5 text-cyber-primary" />
                  </div>
                )}
                
                <div
                  className={`max-w-[80%] rounded-lg p-4 ${
                    message.role === 'user'
                      ? 'bg-cyber-primary/20 text-cyber-text'
                      : 'bg-cyber-card border border-cyber-border text-cyber-text'
                  }`}
                >
                  <div className="whitespace-pre-wrap font-chinese text-sm leading-relaxed">
                    {message.content}
                  </div>
                  
                  {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-cyber-border">
                      <div className="text-xs text-cyber-muted">
                        来源: {message.sources.join(', ')}
                        {message.confidence !== undefined && (
                          <span className="ml-2">
                            置信度: {Math.round(message.confidence * 100)}%
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {message.role === 'user' && (
                  <div className="w-10 h-10 rounded-full bg-cyber-secondary/20 flex items-center justify-center flex-shrink-0">
                    <User className="w-5 h-5 text-cyber-secondary" />
                  </div>
                )}
              </motion.div>
            ))}

            {loading && (
              <div className="flex gap-4 justify-start">
                <div className="w-10 h-10 rounded-full bg-cyber-primary/20 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-cyber-primary" />
                </div>
                <div className="bg-cyber-card border border-cyber-border rounded-lg p-4">
                  <Loader2 className="w-5 h-5 text-cyber-primary animate-spin" />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* 输入框 */}
          <div className="p-6 border-t border-cyber-border">
            <div className="flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="输入你的问题..."
                className="flex-1 px-4 py-3 bg-cyber-bg border border-cyber-border rounded-lg
                         text-cyber-text placeholder-cyber-muted focus:outline-none focus:border-cyber-primary
                         font-chinese"
                disabled={loading || !projectName}
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || loading || !projectName}
                className="px-6 py-3 bg-cyber-primary/20 text-cyber-primary rounded-lg
                         hover:bg-cyber-primary/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed
                         flex items-center gap-2 font-chinese"
              >
                <Send className="w-5 h-5" />
                <span>发送</span>
              </button>
            </div>
          </div>
        </motion.div>
      </main>
    </div>
  )
}

