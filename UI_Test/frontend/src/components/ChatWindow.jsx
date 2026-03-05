import { useRef, useEffect, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import MetaBadges from './MetaBadges'

export default function ChatWindow({ messages, thinking, onSend }) {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinking])

  const submit = () => {
    if (!input.trim() || thinking) return
    onSend(input.trim())
    setInput('')
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      {/* Message list */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4 chat-scroll">
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} />
        ))}
        {thinking && <ThinkingBubble />}
        <div ref={bottomRef} />
      </div>

      {/* Input bar */}
      <div className="shrink-0 border-t border-zinc-800 bg-zinc-900/80 px-4 py-3">
        <div className="flex gap-2 items-end">
          <textarea
            rows={1}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Ask a question about your data…"
            className="flex-1 resize-none bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-3 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:ring-1 focus:ring-violet-500 leading-relaxed"
          />
          <button
            onClick={submit}
            disabled={thinking || !input.trim()}
            className="shrink-0 h-11 px-4 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed text-sm font-semibold text-white transition-colors"
          >
            Send
          </button>
        </div>
        <p className="text-[10px] text-zinc-700 mt-1.5 ml-1">Enter to send · Shift+Enter for newline</p>
      </div>
    </div>
  )
}

function Message({ msg }) {
  const isUser  = msg.role === 'user'
  const isError = msg.role === 'error'

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] space-y-1.5 ${isUser ? 'items-end' : 'items-start'} flex flex-col`}>
        {/* Bubble */}
        <div
          className={`
            rounded-2xl px-4 py-3 text-sm leading-relaxed
            ${isUser  ? 'bg-violet-600 text-white rounded-br-sm' : ''}
            ${isError ? 'bg-red-950 border border-red-800 text-red-300 rounded-bl-sm' : ''}
            ${!isUser && !isError ? 'bg-zinc-800/80 text-zinc-100 rounded-bl-sm' : ''}
          `}
        >
          {isUser || isError ? (
            <p className="whitespace-pre-wrap">{msg.text}</p>
          ) : (
            <ReactMarkdown
              components={{
                p: ({children}) => <p className="mb-2 last:mb-0">{children}</p>,
                ul: ({children}) => <ul className="list-disc ml-4 mb-2 space-y-0.5">{children}</ul>,
                ol: ({children}) => <ol className="list-decimal ml-4 mb-2 space-y-0.5">{children}</ol>,
                strong: ({children}) => <strong className="font-semibold text-zinc-50">{children}</strong>,
                code: ({children}) => <code className="font-mono text-[11px] bg-zinc-700 px-1 py-0.5 rounded">{children}</code>,
              }}
            >
              {msg.text}
            </ReactMarkdown>
          )}
        </div>

        {/* Meta badges (assistant only) */}
        {msg.meta && <MetaBadges meta={msg.meta} />}
      </div>
    </div>
  )
}

function ThinkingBubble() {
  return (
    <div className="flex justify-start">
      <div className="bg-zinc-800/80 rounded-2xl rounded-bl-sm px-4 py-3">
        <div className="flex gap-1.5 items-center h-4">
          {[0, 1, 2].map(i => (
            <span
              key={i}
              className="w-1.5 h-1.5 rounded-full bg-zinc-500 animate-bounce"
              style={{ animationDelay: `${i * 0.15}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
