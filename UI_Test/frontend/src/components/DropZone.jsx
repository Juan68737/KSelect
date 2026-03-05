import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

export default function DropZone({ onReady, loading, setLoading, error, setError }) {
  const [file, setFile]                   = useState(null)
  const [columns, setColumns]             = useState([])
  const [chunkSize, setChunkSize]         = useState(200)
  const [chunkOverlap, setChunkOverlap]   = useState(20)
  const [bm25, setBm25]                   = useState(true)
  const [stage, setStage]                 = useState('drop')  // drop | config | indexing

  // Parse CSV header client-side just to show the user what columns exist
  const parseColumns = useCallback((f) => {
    const reader = new FileReader()
    reader.onload = (e) => {
      const firstLine = e.target.result.split('\n')[0]
      const cols = firstLine.split(',').map(c => c.replace(/^"|"$/g, '').trim())
      setColumns(cols)
      setStage('config')
    }
    reader.readAsText(f)
  }, [])

  const onDrop = useCallback((accepted) => {
    if (!accepted.length) return
    setFile(accepted[0])
    setError(null)
    parseColumns(accepted[0])
  }, [parseColumns, setError])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    multiple: false,
    disabled: loading,
  })

  const handleBuild = async () => {
    if (!file) return
    setLoading(true)
    setStage('indexing')
    setError(null)

    const form = new FormData()
    form.append('file', file)
    form.append('chunk_size', chunkSize)
    form.append('chunk_overlap', chunkOverlap)
    form.append('bm25', bm25)

    try {
      const res = await fetch('/upload', { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      onReady(data)
    } catch (e) {
      setError(e.message)
      setStage('config')
    } finally {
      setLoading(false)
    }
  }

  const handleBack = () => {
    setFile(null)
    setColumns([])
    setStage('drop')
    setError(null)
  }

  return (
    <div className="w-full max-w-lg space-y-5">
      <div className="text-center space-y-1">
        <h1 className="text-2xl font-bold text-zinc-100">KSelect CSV Chat</h1>
        <p className="text-sm text-zinc-500">Drop any CSV — every column gets indexed so you can ask anything</p>
      </div>

      {/* Drop target */}
      {stage === 'drop' && (
        <div
          {...getRootProps()}
          className={`
            relative border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all
            ${isDragActive
              ? 'border-violet-500 bg-violet-500/10'
              : 'border-zinc-700 hover:border-zinc-500 bg-zinc-900/60'}
          `}
        >
          <input {...getInputProps()} />
          <div className="text-4xl mb-3">📂</div>
          <p className="text-sm font-medium text-zinc-300">
            {isDragActive ? 'Drop it!' : 'Drag & drop a CSV here'}
          </p>
          <p className="text-xs text-zinc-600 mt-1">or click to browse</p>
        </div>
      )}

      {/* Config panel */}
      {(stage === 'config' || stage === 'indexing') && (
        <div className="bg-zinc-900/70 border border-zinc-800 rounded-xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-zinc-200 truncate">{file?.name}</p>
            <button onClick={handleBack} className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors ml-3 shrink-0">← Back</button>
          </div>

          {/* Column preview — read-only, all get indexed */}
          <div>
            <p className="text-xs text-zinc-400 mb-1.5">
              Columns detected — <span className="text-violet-400">all will be indexed</span>
            </p>
            <div className="flex flex-wrap gap-1 max-h-20 overflow-y-auto">
              {columns.map(c => (
                <span key={c} className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700">
                  {c}
                </span>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-zinc-400 mb-1">Chunk size</label>
              <input
                type="number" min={50} max={1000} step={50}
                value={chunkSize}
                onChange={e => setChunkSize(+e.target.value)}
                disabled={loading}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-1 focus:ring-violet-500"
              />
            </div>
            <div>
              <label className="block text-xs text-zinc-400 mb-1">Chunk overlap</label>
              <input
                type="number" min={0} max={200} step={10}
                value={chunkOverlap}
                onChange={e => setChunkOverlap(+e.target.value)}
                disabled={loading}
                className="w-full bg-zinc-800 border border-zinc-700 rounded-lg px-3 py-2 text-sm text-zinc-100 focus:outline-none focus:ring-1 focus:ring-violet-500"
              />
            </div>
          </div>

          <label className="flex items-center gap-2 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={bm25}
              onChange={e => setBm25(e.target.checked)}
              disabled={loading}
              className="rounded border-zinc-600 bg-zinc-800 text-violet-500 focus:ring-violet-500"
            />
            <span className="text-xs text-zinc-400">Enable BM25 hybrid search</span>
          </label>

          {stage === 'indexing' ? (
            <div className="flex items-center gap-3 py-2">
              <Spinner />
              <span className="text-sm text-zinc-400">Building index across all columns…</span>
            </div>
          ) : (
            <button
              onClick={handleBuild}
              className="w-full py-2.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-sm font-semibold text-white transition-colors"
            >
              Build index &amp; start chatting
            </button>
          )}
        </div>
      )}

      {error && (
        <div className="rounded-lg bg-red-950/60 border border-red-800 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}
    </div>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4 text-violet-400" viewBox="0 0 24 24" fill="none">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
    </svg>
  )
}
