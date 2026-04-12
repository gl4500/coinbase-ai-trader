import React, { useCallback, useEffect, useRef, useState } from 'react'

// ── Types ──────────────────────────────────────────────────────────────────────

interface LogEntry {
  time:    string
  level:   'WARNING' | 'ERROR' | 'CRITICAL'
  logger:  string
  message: string
}

interface LogResponse {
  total:   number
  entries: LogEntry[]
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function levelColor(level: string) {
  switch (level) {
    case 'CRITICAL': return 'text-red-300 bg-red-950 border-red-700'
    case 'ERROR':    return 'text-red-400 bg-red-950/60 border-red-800'
    case 'WARNING':  return 'text-amber-400 bg-amber-950/40 border-amber-800'
    default:         return 'text-gray-400 bg-gray-800 border-gray-700'
  }
}

function levelBadge(level: string) {
  switch (level) {
    case 'CRITICAL': return 'bg-red-700 text-white'
    case 'ERROR':    return 'bg-red-900 text-red-300 border border-red-700'
    case 'WARNING':  return 'bg-amber-900 text-amber-300 border border-amber-700'
    default:         return 'bg-gray-700 text-gray-300'
  }
}

function shortLogger(name: string) {
  const parts = name.split('.')
  return parts[parts.length - 1]
}

// ── Component ──────────────────────────────────────────────────────────────────

export default function LogViewer() {
  const [entries,   setEntries]   = useState<LogEntry[]>([])
  const [total,     setTotal]     = useState(0)
  const [filter,    setFilter]    = useState<'WARNING' | 'ERROR'>('WARNING')
  const [search,    setSearch]    = useState('')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [clearing,  setClearing]  = useState(false)
  const [lastFetch, setLastFetch] = useState<Date | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const listRef   = useRef<HTMLDivElement>(null)

  const fetchLogs = useCallback(async () => {
    try {
      const r = await fetch(`/api/logs?level=${filter}&limit=400`)
      if (!r.ok) return
      const data: LogResponse = await r.json()
      setEntries(data.entries)
      setTotal(data.total)
      setLastFetch(new Date())
    } catch {}
  }, [filter])

  useEffect(() => {
    fetchLogs()
  }, [fetchLogs])

  useEffect(() => {
    if (!autoRefresh) return
    const id = setInterval(fetchLogs, 5_000)
    return () => clearInterval(id)
  }, [autoRefresh, fetchLogs])

  const handleClear = async () => {
    setClearing(true)
    try {
      // Read key from status then clear
      const s = await fetch('/api/status').then(r => r.json()).catch(() => ({}))
      await fetch('/api/logs', {
        method: 'DELETE',
        headers: { 'X-API-Key': s.app_api_key ?? '' },
      })
      setEntries([])
      setTotal(0)
    } finally {
      setClearing(false)
    }
  }

  const displayed = search
    ? entries.filter(e =>
        e.message.toLowerCase().includes(search.toLowerCase()) ||
        e.logger.toLowerCase().includes(search.toLowerCase())
      )
    : entries

  const errorCount   = entries.filter(e => e.level === 'ERROR' || e.level === 'CRITICAL').length
  const warningCount = entries.filter(e => e.level === 'WARNING').length

  return (
    <div className="space-y-4">

      {/* ── Header / Controls ── */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-bold text-white">Logs</h2>

          {/* Summary badges */}
          <span className="text-xs px-2 py-0.5 rounded border bg-red-900/50 text-red-400 border-red-800">
            {errorCount} error{errorCount !== 1 ? 's' : ''}
          </span>
          <span className="text-xs px-2 py-0.5 rounded border bg-amber-900/50 text-amber-400 border-amber-800">
            {warningCount} warning{warningCount !== 1 ? 's' : ''}
          </span>
          {lastFetch && (
            <span className="text-xs text-gray-600">
              updated {lastFetch.toLocaleTimeString()}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Level filter */}
          <div className="flex rounded border border-gray-700 overflow-hidden text-xs">
            {(['WARNING', 'ERROR'] as const).map(lvl => (
              <button
                key={lvl}
                onClick={() => setFilter(lvl)}
                className={`px-3 py-1.5 transition-colors ${
                  filter === lvl
                    ? lvl === 'ERROR'
                      ? 'bg-red-900 text-red-300'
                      : 'bg-amber-900 text-amber-300'
                    : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                }`}
              >
                {lvl}+
              </button>
            ))}
          </div>

          {/* Search */}
          <input
            type="text"
            placeholder="Search…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-xs text-white placeholder-gray-600 w-40 focus:outline-none focus:border-gray-500"
          />

          {/* Auto-refresh toggle */}
          <button
            onClick={() => setAutoRefresh(v => !v)}
            className={`text-xs px-3 py-1.5 rounded border transition-colors ${
              autoRefresh
                ? 'bg-green-900/50 border-green-700 text-green-400'
                : 'bg-gray-800 border-gray-700 text-gray-400'
            }`}
          >
            {autoRefresh ? '⏵ Live' : '⏸ Paused'}
          </button>

          <button
            onClick={fetchLogs}
            className="btn-secondary text-xs px-3 py-1.5"
          >
            ↺ Refresh
          </button>

          <button
            onClick={handleClear}
            disabled={clearing || entries.length === 0}
            className="text-xs px-3 py-1.5 rounded border border-gray-700 text-gray-400 hover:text-red-400 hover:border-red-800 transition-colors disabled:opacity-40"
          >
            🗑 Clear
          </button>
        </div>
      </div>

      {/* ── Log count ── */}
      <div className="text-xs text-gray-600">
        Showing {displayed.length} of {total} entries (last 500 captured, newest first)
      </div>

      {/* ── Log entries ── */}
      {displayed.length === 0 ? (
        <div className="card text-center py-16">
          <p className="text-gray-500 text-sm">
            {entries.length === 0
              ? 'No warnings or errors captured yet — the app is running clean'
              : 'No entries match your search'}
          </p>
        </div>
      ) : (
        <div ref={listRef} className="space-y-1.5 max-h-[75vh] overflow-y-auto pr-1">
          {displayed.map((e, i) => (
            <div
              key={i}
              className={`rounded border px-3 py-2 font-mono text-xs ${levelColor(e.level)}`}
            >
              <div className="flex items-start gap-2">
                {/* Level badge */}
                <span className={`flex-shrink-0 text-xs font-bold px-1.5 py-0.5 rounded ${levelBadge(e.level)}`}>
                  {e.level === 'CRITICAL' ? 'CRIT' : e.level === 'WARNING' ? 'WARN' : e.level}
                </span>

                {/* Time */}
                <span className="flex-shrink-0 text-gray-500 text-xs pt-0.5">
                  {new Date(e.time).toLocaleTimeString()}
                </span>

                {/* Logger */}
                <span className="flex-shrink-0 text-gray-500 text-xs pt-0.5 hidden sm:block">
                  [{shortLogger(e.logger)}]
                </span>

                {/* Message */}
                <span className="break-all leading-relaxed whitespace-pre-wrap">{e.message}</span>
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  )
}
