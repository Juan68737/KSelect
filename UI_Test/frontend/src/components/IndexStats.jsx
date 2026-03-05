/**
 * Left sidebar panel showing live index diagnostics.
 */
export default function IndexStats({ info }) {
  return (
    <div className="p-4 space-y-5">
      <h2 className="text-[11px] font-semibold uppercase tracking-widest text-zinc-600">Index Info</h2>

      <StatBlock label="File" value={info.filename} mono truncate />
      <StatBlock label="Indexed as" value="all columns fused" mono />
      <StatBlock label="Chunks indexed" value={info.chunk_count?.toLocaleString()} accent />
      <StatBlock label="Index build time" value={`${(info.index_ms / 1000).toFixed(2)} s`} />

      <div className="border-t border-zinc-800" />

      <h2 className="text-[11px] font-semibold uppercase tracking-widest text-zinc-600">Columns</h2>
      <div className="flex flex-wrap gap-1">
        {info.columns?.map(c => (
          <span
            key={c}
            className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
              c === info.text_col
                ? 'bg-violet-600/30 text-violet-300 border border-violet-700'
                : 'bg-zinc-800 text-zinc-500'
            }`}
          >
            {c}
          </span>
        ))}
      </div>

      <div className="border-t border-zinc-800" />

      <p className="text-[10px] text-zinc-700 leading-relaxed">
        Powered by <span className="text-zinc-500">KSelect</span> · FAISS + BM25 hybrid · Claude Haiku
      </p>
    </div>
  )
}

function StatBlock({ label, value, mono, accent, truncate }) {
  return (
    <div className="space-y-0.5">
      <p className="text-[10px] text-zinc-600 uppercase tracking-wide">{label}</p>
      <p className={`text-xs ${accent ? 'text-violet-300 font-semibold' : 'text-zinc-300'} ${mono ? 'font-mono' : ''} ${truncate ? 'truncate' : ''}`}>
        {value ?? '—'}
      </p>
    </div>
  )
}
