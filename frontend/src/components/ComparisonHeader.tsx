import React, { useEffect, useState } from 'react'

interface BackendState {
  balance: number
  realized_pnl: number
  open_positions: number
  unrealized_pnl_est: number
}

interface CompareResponse {
  v3: BackendState | null
  v45: BackendState | null
  delta_realized: number
}

const fmt = (n: number) =>
  `${n >= 0 ? '+' : ''}$${n.toFixed(2)}`

const pnlColor = (n: number) =>
  n > 0 ? 'text-green-400' : n < 0 ? 'text-red-400' : 'text-gray-400'

export default function ComparisonHeader() {
  const [data, setData] = useState<CompareResponse | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const poll = async () => {
      try {
        const r = await fetch('/api/compare')
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const j = await r.json()
        if (!cancelled) { setData(j); setErr(null) }
      } catch (e) {
        if (!cancelled) setErr((e as Error).message)
      }
    }
    poll()
    const id = setInterval(poll, 30_000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  if (err && !data) return (
    <div className="sticky top-0 z-50 bg-gray-900 border-b border-gray-800 px-4 py-2 text-xs text-red-400">
      Shadow-week monitor: {err}
    </div>
  )

  if (!data) return (
    <div className="sticky top-0 z-50 bg-gray-900 border-b border-gray-800 px-4 py-2 text-xs text-gray-500">
      Loading shadow-week monitor…
    </div>
  )

  const renderSide = (label: string, badge: string, s: BackendState | null) => {
    if (!s) return (
      <div className="flex items-center gap-3 text-xs text-gray-500">
        <span className="font-mono px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700">{badge}</span>
        <span>(not running)</span>
      </div>
    )
    return (
      <div className="flex items-center gap-3 text-xs font-mono">
        <span className="px-1.5 py-0.5 rounded bg-gray-800 border border-gray-700 text-gray-300">{badge}</span>
        <span className="text-gray-400">bal <span className="text-gray-100">${s.balance.toFixed(2)}</span></span>
        <span className="text-gray-400">realized <span className={pnlColor(s.realized_pnl)}>{fmt(s.realized_pnl)}</span></span>
        <span className="text-gray-400">open <span className="text-gray-100">{s.open_positions}</span></span>
        <span className="text-gray-400">unreal <span className={pnlColor(s.unrealized_pnl_est)}>{fmt(s.unrealized_pnl_est)}</span></span>
      </div>
    )
  }

  return (
    <div className="sticky top-0 z-50 bg-gray-900 border-b border-gray-800 px-4 py-2 flex items-center justify-between gap-6 flex-wrap">
      {renderSide('8001', '8001 v3', data.v3)}
      {renderSide('8002', '8002 v45', data.v45)}
      <div className="flex items-center gap-2 text-xs font-mono">
        <span className="text-gray-500">Δ realized:</span>
        <span className={`${pnlColor(data.delta_realized)} font-semibold`}>{fmt(data.delta_realized)}</span>
        <span className="text-gray-600">(v45 − v3)</span>
      </div>
    </div>
  )
}
