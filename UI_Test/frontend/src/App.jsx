import { useState } from 'react'
import DropZone from './components/DropZone'
import ChatWindow from './components/ChatWindow'
import IndexStats from './components/IndexStats'
import EvalPanel from './components/EvalPanel'

export default function App() {
  const [indexInfo, setIndexInfo] = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const [messages, setMessages]   = useState([])
  const [thinking, setThinking]   = useState(false)
  const [showEval, setShowEval]   = useState(false)

  const handleIndexReady = (info) => {
    setIndexInfo(info)
    setMessages([{
      role: 'assistant',
      text: `Index ready! Loaded **${info.filename}** — **${info.chunk_count.toLocaleString()} chunks** across all ${info.columns.length} columns.\n\nAsk me anything about your data.`,
      meta: null,
    }])
    setError(null)
  }

  const handleReset = async () => {
    await fetch('/reset', { method: 'DELETE' })
    setIndexInfo(null)
    setMessages([])
    setError(null)
    setShowEval(false)
  }

  const handleSend = async (question) => {
    if (!question.trim() || thinking) return

    setMessages(prev => [...prev, { role: 'user', text: question, meta: null }])
    setThinking(true)

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, k: 15, fast: true, max_context_tokens: 6000 }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Chat error')
      }
      const data = await res.json()
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: data.answer,
        meta: {
          confidence:        data.confidence,
          retrieval_ms:      data.retrieval_ms,
          llm_ms:            data.llm_ms,
          total_ms:          data.total_ms,
          chunks_retrieved:  data.chunks_retrieved,
          chunks_in_context: data.chunks_in_context,
          chunks_dropped:    data.chunks_dropped,
          context_tokens:    data.context_tokens,
          sources:           data.sources,
        },
      }])
    } catch (e) {
      setMessages(prev => [...prev, { role: 'error', text: e.message, meta: null }])
    } finally {
      setThinking(false)
    }
  }

  return (
    <div className="flex flex-col h-screen max-h-screen overflow-hidden">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-3 border-b border-zinc-800 bg-zinc-900/80 backdrop-blur shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded-md bg-violet-600 flex items-center justify-center text-xs font-bold tracking-tight">K</div>
          <span className="font-semibold text-sm text-zinc-100">KSelect Chat</span>
          {indexInfo && (
            <span className="text-xs text-zinc-500 ml-1">— {indexInfo.filename}</span>
          )}
        </div>

        {indexInfo && (
          <div className="flex items-center gap-4">
            <button
              onClick={() => setShowEval(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-xs font-semibold text-zinc-300 transition-colors"
            >
              <span className="text-amber-400">⚡</span> Benchmark
            </button>
            <button
              onClick={handleReset}
              className="text-xs text-zinc-500 hover:text-red-400 transition-colors"
            >
              Reset index
            </button>
          </div>
        )}
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Left sidebar */}
        {indexInfo && (
          <aside className="w-64 shrink-0 border-r border-zinc-800 bg-zinc-900/40 overflow-y-auto">
            <IndexStats info={indexInfo} />
          </aside>
        )}

        {/* Main area */}
        <main className="flex flex-1 flex-col overflow-hidden">
          {!indexInfo ? (
            <div className="flex flex-1 items-center justify-center p-8">
              <DropZone
                onReady={handleIndexReady}
                loading={loading}
                setLoading={setLoading}
                error={error}
                setError={setError}
              />
            </div>
          ) : (
            <ChatWindow
              messages={messages}
              thinking={thinking}
              onSend={handleSend}
            />
          )}
        </main>
      </div>

      {/* Eval modal */}
      {showEval && <EvalPanel onClose={() => setShowEval(false)} />}
    </div>
  )
}
