import { useState, useRef } from 'react'

const DEFAULT_QUESTIONS = [
  "Who paid the highest fare and what class were they in?",
  "What was the fare paid by the youngest passenger?",
  "List passengers who survived and were in first class.",
  "What were the ages of passengers who did not survive?",
  "Which passengers embarked from Cherbourg?",
]

const METRIC_LABELS = {
  faithfulness:      { label: "Faithfulness",       desc: "Are answers grounded in retrieved context? (LLM-judged)" },
  answer_relevancy:  { label: "Answer Relevancy",   desc: "Is the answer on-topic with the question? (embedding cosine)" },
  context_recall:    { label: "Context Recall",     desc: "Does retrieved context cover the question? (embedding cosine)" },
  context_precision: { label: "Context Precision",  desc: "Signal-to-noise of retrieved chunks (higher = less noise)" },
  latency_ms:        { label: "Avg Latency",        desc: "End-to-end wall-clock time per query", unit: "ms", raw: true },
  tokens_used:       { label: "Avg Tokens",         desc: "Context tokens sent to LLM per query", raw: true },
  overall:           { label: "Overall Score",      desc: "Mean of the four semantic metrics", accent: true },
}

export default function EvalPanel({ onClose }) {
  const [questions, setQuestions]     = useState(DEFAULT_QUESTIONS.join('\n'))
  const [running, setRunning]         = useState(false)
  const [phase, setPhase]             = useState(null)    // 'kselect' | 'langchain'
  const [progress, setProgress]       = useState({})      // { kselect: [...], langchain: [...] }
  const [summary, setSummary]         = useState(null)
  const [warning, setWarning]         = useState(null)
  const [error, setError]             = useState(null)
  const [activeQ, setActiveQ]         = useState(null)
  const abortRef                      = useRef(null)

  const handleRun = async () => {
    const qs = questions.split('\n').map(s => s.trim()).filter(Boolean)
    if (!qs.length) return
    setRunning(true)
    setPhase(null)
    setProgress({})
    setSummary(null)
    setWarning(null)
    setError(null)
    setActiveQ(null)

    const ctrl = new AbortController()
    abortRef.current = ctrl

    try {
      const res = await fetch('/eval/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ questions: qs, k: 15 }),
        signal: ctrl.signal,
      })

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const blocks = buf.split('\n\n')
        buf = blocks.pop()
        for (const block of blocks) {
          const eventLine = block.split('\n').find(l => l.startsWith('event:'))
          const dataLine  = block.split('\n').find(l => l.startsWith('data:'))
          if (!eventLine || !dataLine) continue
          const event = eventLine.replace('event:', '').trim()
          const data  = JSON.parse(dataLine.replace('data:', '').trim())

          if (event === 'phase') {
            setPhase(data.phase)
          } else if (event === 'progress') {
            setActiveQ({ system: data.system, question: data.question, index: data.index })
          } else if (event === 'result') {
            const { system, index, ...rest } = data
            setProgress(prev => ({
              ...prev,
              [system]: [...(prev[system] || []), { index, ...rest }],
            }))
          } else if (event === 'summary') {
            setSummary(data)
          } else if (event === 'warning') {
            setWarning(data.message)
          } else if (event === 'error') {
            setError(data.message)
          }
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') setError(e.message)
    } finally {
      setRunning(false)
      setActiveQ(null)
    }
  }

  const handleStop = () => {
    abortRef.current?.abort()
    setRunning(false)
  }

  const qs = questions.split('\n').map(s => s.trim()).filter(Boolean)
  const ksRows = progress.kselect || []
  const lcRows = progress.langchain || []

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/70 backdrop-blur-sm overflow-y-auto py-8 px-4">
      <div className="w-full max-w-5xl bg-zinc-900 border border-zinc-700 rounded-2xl shadow-2xl">

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-zinc-800">
          <div>
            <h2 className="text-lg font-bold text-zinc-100">RAG Benchmark</h2>
            <p className="text-xs text-zinc-500 mt-0.5">KSelect vs vanilla LangChain — same data, same questions, same LLM</p>
          </div>
          <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300 text-xl leading-none">✕</button>
        </div>

        <div className="p-6 space-y-6">

          {/* Question editor */}
          {!summary && (
            <div className="space-y-2">
              <label className="text-xs text-zinc-400 font-medium">Eval questions <span className="text-zinc-600">(one per line)</span></label>
              <textarea
                rows={6}
                value={questions}
                onChange={e => setQuestions(e.target.value)}
                disabled={running}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-xl px-4 py-3 text-sm text-zinc-200 font-mono focus:outline-none focus:ring-1 focus:ring-violet-500 resize-none"
              />
              <p className="text-[11px] text-zinc-600">
                {qs.length} question{qs.length !== 1 ? 's' : ''} · each runs through KSelect AND LangChain · LLM-as-judge via Claude Haiku
              </p>
            </div>
          )}

          {/* Run / Stop */}
          {!summary && (
            <div className="flex gap-3">
              {running ? (
                <button onClick={handleStop} className="px-5 py-2.5 rounded-xl bg-red-700 hover:bg-red-600 text-sm font-semibold text-white transition-colors">
                  Stop
                </button>
              ) : (
                <button onClick={handleRun} disabled={!qs.length} className="px-5 py-2.5 rounded-xl bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-sm font-semibold text-white transition-colors">
                  Run benchmark
                </button>
              )}
            </div>
          )}

          {/* Live progress */}
          {(running || ksRows.length > 0 || lcRows.length > 0) && !summary && (
            <ProgressFeed
              phase={phase}
              activeQ={activeQ}
              ksRows={ksRows}
              lcRows={lcRows}
              total={qs.length}
            />
          )}

          {warning && (
            <div className="rounded-lg bg-amber-950/50 border border-amber-800 px-4 py-3 text-sm text-amber-300">{warning}</div>
          )}
          {error && (
            <div className="rounded-lg bg-red-950/50 border border-red-800 px-4 py-3 text-sm text-red-300">{error}</div>
          )}

          {/* Summary dashboard */}
          {summary && (
            <SummaryDashboard
              summary={summary}
              ksRows={ksRows}
              lcRows={lcRows}
              questions={qs}
              onRerun={() => { setSummary(null); setProgress({}) }}
            />
          )}
        </div>
      </div>
    </div>
  )
}


// ── Live progress feed ─────────────────────────────────────────────────────────

function ProgressFeed({ phase, activeQ, ksRows, lcRows, total }) {
  return (
    <div className="space-y-3">
      <PhaseBar label="KSelect" rows={ksRows} total={total} active={phase === 'kselect'} color="violet" />
      <PhaseBar label="LangChain" rows={lcRows} total={total} active={phase === 'langchain'} color="emerald" />
      {activeQ && (
        <p className="text-[11px] text-zinc-500 animate-pulse">
          ⏳ [{activeQ.system}] Q{activeQ.index + 1}: {activeQ.question}
        </p>
      )}
    </div>
  )
}

function PhaseBar({ label, rows, total, active, color }) {
  const pct = total > 0 ? Math.round((rows.length / total) * 100) : 0
  const barColor = color === 'violet' ? 'bg-violet-500' : 'bg-emerald-500'
  const textColor = color === 'violet' ? 'text-violet-400' : 'text-emerald-400'
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[11px]">
        <span className={`font-medium ${active ? textColor : 'text-zinc-600'}`}>
          {label} {active && <span className="animate-pulse">●</span>}
        </span>
        <span className="text-zinc-600">{rows.length}/{total}</span>
      </div>
      <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all duration-300 ${barColor}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}


// ── Summary dashboard ──────────────────────────────────────────────────────────

function SummaryDashboard({ summary, ksRows, lcRows, questions, onRerun }) {
  const [selectedQ, setSelectedQ] = useState(null)
  const ks = summary.kselect || {}
  const lc = summary.langchain || {}
  const hasBoth = ks.overall !== undefined && lc.overall !== undefined

  return (
    <div className="space-y-6">
      {/* Top metric cards */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {['faithfulness', 'answer_relevancy', 'context_recall', 'overall'].map(m => (
          <MetricCard
            key={m}
            label={METRIC_LABELS[m].label}
            desc={METRIC_LABELS[m].desc}
            ksVal={ks[m]}
            lcVal={lc[m]}
            hasBoth={hasBoth}
            isScore
            accent={m === 'overall'}
          />
        ))}
      </div>

      {/* Perf + token row */}
      <div className="grid grid-cols-2 gap-4">
        <MetricCard
          label="Avg Latency"
          desc="End-to-end per query"
          ksVal={ks.latency_ms !== undefined ? `${Math.round(ks.latency_ms)} ms` : undefined}
          lcVal={lc.latency_ms !== undefined ? `${Math.round(lc.latency_ms)} ms` : undefined}
          hasBoth={hasBoth}
          lowerIsBetter
        />
        <MetricCard
          label="Avg Tokens to LLM"
          desc="Context tokens per query"
          ksVal={ks.tokens_used}
          lcVal={lc.tokens_used}
          hasBoth={hasBoth}
          lowerIsBetter
        />
      </div>

      {/* Per-question table */}
      <div>
        <h3 className="text-xs font-semibold uppercase tracking-widest text-zinc-500 mb-3">Per-question breakdown</h3>
        <div className="overflow-x-auto rounded-xl border border-zinc-800">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-zinc-800 text-zinc-500">
                <th className="text-left px-4 py-2.5 font-medium">Question</th>
                <th className="px-3 py-2.5 font-medium">System</th>
                <th className="px-3 py-2.5 font-medium">Faithful</th>
                <th className="px-3 py-2.5 font-medium">Relevancy</th>
                <th className="px-3 py-2.5 font-medium">Recall</th>
                <th className="px-3 py-2.5 font-medium">Precision</th>
                <th className="px-3 py-2.5 font-medium">Latency</th>
                <th className="px-3 py-2.5 font-medium">Overall</th>
              </tr>
            </thead>
            <tbody>
              {questions.map((q, i) => {
                const ksR = ksRows.find(r => r.index === i)
                const lcR = lcRows.find(r => r.index === i)
                const rows = [
                  ksR ? { ...ksR, sys: 'KSelect', color: 'violet' } : null,
                  lcR ? { ...lcR, sys: 'LangChain', color: 'emerald' } : null,
                ].filter(Boolean)
                return rows.map((row, ri) => (
                  <tr
                    key={`${i}-${ri}`}
                    className={`border-b border-zinc-800/50 hover:bg-zinc-800/30 cursor-pointer transition-colors ${selectedQ === `${i}-${ri}` ? 'bg-zinc-800/40' : ''}`}
                    onClick={() => setSelectedQ(selectedQ === `${i}-${ri}` ? null : `${i}-${ri}`)}
                  >
                    {ri === 0 && (
                      <td rowSpan={rows.length} className="px-4 py-3 text-zinc-400 max-w-[200px] align-top">
                        <p className="truncate">{q}</p>
                      </td>
                    )}
                    <td className="px-3 py-3">
                      <span className={`font-semibold ${row.color === 'violet' ? 'text-violet-400' : 'text-emerald-400'}`}>{row.sys}</span>
                    </td>
                    <ScoreCell v={row.faithfulness} />
                    <ScoreCell v={row.answer_relevancy} />
                    <ScoreCell v={row.context_recall} />
                    <ScoreCell v={row.context_precision} />
                    <td className="px-3 py-3 text-center text-zinc-400">{Math.round(row.latency_ms)}ms</td>
                    <ScoreCell v={row.overall} bold />
                  </tr>
                ))
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Answer viewer */}
      {selectedQ !== null && (() => {
        const [qi, ri] = selectedQ.split('-').map(Number)
        const rows = [
          ksRows.find(r => r.index === qi),
          lcRows.find(r => r.index === qi),
        ].filter(Boolean)
        const row = rows[ri]
        if (!row) return null
        return (
          <div className="rounded-xl border border-zinc-800 bg-zinc-800/30 p-4 space-y-3">
            <p className="text-xs font-semibold text-zinc-400">Q{qi + 1}: {questions[qi]}</p>
            <div>
              <p className="text-[10px] text-zinc-600 uppercase mb-1">Answer</p>
              <p className="text-sm text-zinc-200 leading-relaxed">{row.answer}</p>
            </div>
            {row.context_snippets?.length > 0 && (
              <div>
                <p className="text-[10px] text-zinc-600 uppercase mb-1">Top retrieved chunks</p>
                {row.context_snippets.map((s, j) => (
                  <p key={j} className="text-[11px] text-zinc-500 font-mono bg-zinc-900 rounded px-2 py-1 mb-1 truncate">{s}</p>
                ))}
              </div>
            )}
          </div>
        )
      })()}

      <button onClick={onRerun} className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors">← Run again with different questions</button>
    </div>
  )
}

function MetricCard({ label, desc, ksVal, lcVal, hasBoth, isScore, accent, lowerIsBetter }) {
  const fmt = v => {
    if (v === undefined || v === null) return '—'
    if (isScore) return `${Math.round(v * 100)}%`
    return String(v)
  }
  const ksWins = hasBoth && isScore && ksVal !== undefined && lcVal !== undefined &&
    (lowerIsBetter ? ksVal < lcVal : ksVal > lcVal)
  const lcWins = hasBoth && isScore && ksVal !== undefined && lcVal !== undefined &&
    (lowerIsBetter ? lcVal < ksVal : lcVal > ksVal)

  return (
    <div className={`rounded-xl border p-4 space-y-3 ${accent ? 'border-violet-700 bg-violet-950/30' : 'border-zinc-800 bg-zinc-800/30'}`}>
      <div>
        <p className={`text-[10px] uppercase tracking-wider font-semibold ${accent ? 'text-violet-400' : 'text-zinc-500'}`}>{label}</p>
        <p className="text-[10px] text-zinc-700 mt-0.5 leading-snug">{desc}</p>
      </div>
      <div className="flex gap-4">
        <SystemVal label="KSelect" val={fmt(ksVal)} wins={ksWins} color="violet" />
        {hasBoth && <SystemVal label="LangChain" val={fmt(lcVal)} wins={lcWins} color="emerald" />}
      </div>
    </div>
  )
}

function SystemVal({ label, val, wins, color }) {
  const textColor = color === 'violet' ? 'text-violet-400' : 'text-emerald-400'
  return (
    <div>
      <p className={`text-[10px] ${textColor} font-medium`}>{label}</p>
      <p className={`text-xl font-bold mt-0.5 ${wins ? (color === 'violet' ? 'text-violet-300' : 'text-emerald-300') : 'text-zinc-300'}`}>
        {val}
        {wins && <span className="text-[10px] ml-1">✓</span>}
      </p>
    </div>
  )
}

function ScoreCell({ v, bold }) {
  if (v === undefined || v === null) return <td className="px-3 py-3 text-center text-zinc-700">—</td>
  const pct = Math.round(v * 100)
  const color = pct >= 70 ? 'text-emerald-400' : pct >= 45 ? 'text-amber-400' : 'text-red-400'
  return (
    <td className={`px-3 py-3 text-center ${color} ${bold ? 'font-bold' : ''}`}>{pct}%</td>
  )
}
