import { useState } from 'react'

/**
 * Small metadata strip below each assistant message.
 * Shows: confidence bar, timing breakdown, chunk counts, sources toggle.
 */
export default function MetaBadges({ meta }) {
  const [showSources, setShowSources] = useState(false)
  const conf = meta.confidence   // 0–1

  return (
    <div className="text-[11px] text-zinc-500 space-y-1.5 pl-1 max-w-[80%]">
      {/* Row 1: confidence + timing */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <ConfBar value={conf} />
        <Chip label="retrieval" value={`${meta.retrieval_ms} ms`} />
        <Chip label="llm" value={`${(meta.llm_ms / 1000).toFixed(2)} s`} />
        <Chip label="total" value={`${(meta.total_ms / 1000).toFixed(2)} s`} color="text-zinc-400" />
      </div>

      {/* Row 2: chunk counts + context tokens */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <Chip label="retrieved" value={meta.chunks_retrieved} />
        <Chip label="in context" value={meta.chunks_in_context} />
        <Chip label="dropped" value={meta.chunks_dropped} color={meta.chunks_dropped > 0 ? 'text-amber-500' : undefined} />
        <Chip label="ctx tokens" value={`${meta.context_tokens.toLocaleString()}`} />
      </div>

      {/* Sources toggle */}
      {meta.sources?.length > 0 && (
        <div>
          <button
            onClick={() => setShowSources(v => !v)}
            className="text-zinc-600 hover:text-zinc-400 transition-colors"
          >
            {showSources ? '▾ hide' : '▸ show'} {meta.sources.length} source{meta.sources.length !== 1 ? 's' : ''}
          </button>
          {showSources && (
            <div className="mt-1.5 space-y-1.5 max-h-48 overflow-y-auto pr-1">
              {meta.sources.map((s, i) => (
                <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-lg p-2 space-y-0.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-zinc-400 font-mono truncate max-w-[70%]">{s.doc_id || s.chunk_id}</span>
                    <ScorePill score={s.score} />
                  </div>
                  <p className="text-zinc-500 leading-snug line-clamp-2">{s.snippet}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ConfBar({ value }) {
  const pct = Math.round(value * 100)
  const color = pct >= 70 ? 'bg-emerald-500' : pct >= 40 ? 'bg-amber-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-zinc-600">confidence</span>
      <div className="w-16 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`font-mono ${pct >= 70 ? 'text-emerald-400' : pct >= 40 ? 'text-amber-400' : 'text-red-400'}`}>
        {pct}%
      </span>
    </div>
  )
}

function Chip({ label, value, color }) {
  return (
    <span>
      <span className="text-zinc-700">{label} </span>
      <span className={color || 'text-zinc-500'}>{value}</span>
    </span>
  )
}

function ScorePill({ score }) {
  const pct = Math.round(score * 100)
  return (
    <span className="shrink-0 font-mono text-[10px] text-zinc-600 bg-zinc-800 rounded px-1.5 py-0.5">
      {score.toFixed(4)}
    </span>
  )
}
