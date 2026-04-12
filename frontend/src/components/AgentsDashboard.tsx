import React, { useCallback, useEffect, useMemo, useState } from 'react'

// ── Types ──────────────────────────────────────────────────────────────────────

interface AgentPosition {
  size:            number
  avg_price:       number
  current_price:   number | null
  unrealized_pnl:  number | null
  pct_pnl:         number | null
  high_water?:     number   // Momentum only
}

interface SubAgentStatus {
  agent:          string
  balance:        number
  realized_pnl:   number
  open_positions: number
  positions:      Record<string, AgentPosition>
  scan_count:     number
  signals_buy:    number
  signals_sell:   number
  last_scan_at:   number | null
  trailing_stops?: number
}

interface Trade {
  id:            number
  agent:         string
  product_id:    string
  entry_price:   number
  exit_price:    number | null
  size:          number
  usd_open:      number
  usd_close:     number | null
  pnl:           number | null
  pct_pnl:       number | null
  hold_secs:     number | null
  trigger_open:  string
  trigger_close: string | null
  balance_after: number
  opened_at:     string
  closed_at:     string | null
}

interface AgentDecision {
  id:         number
  agent:      string
  product_id: string
  side:       string
  confidence: number
  price:      number
  score:      number | null
  reasoning:  string | null
  balance:    number | null
  pnl:        number | null
  created_at: string
}

type SortKey = 'agent' | 'product_id' | 'side' | 'confidence' | 'score' | 'pnl' | 'balance' | 'created_at'
type AgentFilter = 'ALL' | 'TECH' | 'MOMENTUM' | 'CNN' | 'SCALP'
type SideFilter  = 'ALL' | 'BUY' | 'SELL' | 'HOLD'

// ── Helpers ────────────────────────────────────────────────────────────────────

function timeAgo(unix: number | null): string {
  if (!unix) return '—'
  const secs = Math.floor(Date.now() / 1000 - unix)
  if (secs < 60)   return `${secs}s ago`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`
  return `${Math.floor(secs / 3600)}h ago`
}

/** Parse indicator pills from reasoning string (Tech/Momentum format) */
function parseIndicators(reasoning: string | null): Record<string, string> {
  if (!reasoning) return {}
  const out: Record<string, string> = {}
  const pairs: [RegExp, string][] = [
    [/RSI[=:\s]+([\d.]+)/i,        'RSI'],
    [/ADX[=:\s]+([\d.]+)/i,        'ADX'],
    [/MFI[=:\s]+([\d.]+)/i,        'MFI'],
    [/BB[=:\s]+([+-]?[\d.]+)/i,    'BB'],
    [/MACD[=:\s]+([+-]?[\d.]+)/i,  'MACD'],
    [/stoch[=:\s]+([\d.]+)/i,      'Stoch'],
    [/ROC[=:\s]+([+-]?[\d.]+)/i,   'ROC'],
    [/mom[=:\s]+([+-]?[\d.]+)/i,   'Mom'],
    [/vwm[=:\s]+([+-]?[\d.]+)/i,   'VWMom'],
  ]
  for (const [re, label] of pairs) {
    const m = reasoning.match(re)
    if (m) out[label] = m[1]
  }
  return out
}

// ── Sub-components ─────────────────────────────────────────────────────────────

function StatCard({ label, value, sub, color = 'text-white' }: {
  label: string; value: React.ReactNode; sub?: string; color?: string
}) {
  return (
    <div className="card p-4 flex flex-col gap-1">
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`text-xl font-bold font-mono ${color}`}>{value}</div>
      {sub && <div className="text-xs text-gray-600">{sub}</div>}
    </div>
  )
}

function ConfBar({ value, max = 1, color }: { value: number | null; max?: number; color: string }) {
  if (value === null) return <span className="text-gray-600 text-xs">—</span>
  const pct = Math.min(Math.round((value / max) * 100), 100)
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 bg-gray-700 rounded-full h-1.5 flex-shrink-0">
        <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-gray-300">{Math.round(value * 100)}%</span>
    </div>
  )
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function AgentsDashboard() {
  const [agentStatus,    setAgentStatus]    = useState<{ tech: SubAgentStatus | null; momentum: SubAgentStatus | null; cnn: SubAgentStatus | null; scalp: SubAgentStatus | null }>({ tech: null, momentum: null, cnn: null, scalp: null })
  const [signals,        setSignals]        = useState<AgentDecision[]>([])   // BUY/SELL only
  const [decisions,      setDecisions]      = useState<AgentDecision[]>([])   // all (for table)
  const [trades,         setTrades]         = useState<Trade[]>([])
  const [tradeView,      setTradeView]      = useState<'ALL' | 'OPEN' | 'CLOSED'>('ALL')
  const [search,         setSearch]         = useState('')
  const [agentFilter,    setAgentFilter]    = useState<AgentFilter>('ALL')
  const [sideFilter,     setSideFilter]     = useState<SideFilter>('ALL')
  const [sigOnly,        setSigOnly]        = useState(false)
  const [sortKey,        setSortKey]        = useState<SortKey>('created_at')
  const [sortAsc,        setSortAsc]        = useState(false)
  const [showTable,      setShowTable]      = useState(true)

  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch('/api/agents/status')
      if (r.ok) {
        const d = await r.json()
        setAgentStatus({ tech: d.tech ?? null, momentum: d.momentum ?? null, cnn: d.cnn ?? null, scalp: d.scalp ?? null })
      }
    } catch {}
  }, [])

  const fetchSignals = useCallback(async () => {
    try {
      const r = await fetch('/api/agents/decisions?signals_only=true&limit=200')
      if (!r.ok) return
      const data: AgentDecision[] = await r.json()
      // Only update if something actually changed — prevents blinking on poll
      setSignals(prev =>
        prev.length === data.length && prev[0]?.id === data[0]?.id ? prev : data
      )
    } catch {}
  }, [])

  const fetchDecisions = useCallback(async () => {
    try {
      const r = await fetch('/api/agents/decisions?limit=500')
      if (!r.ok) return
      const data: AgentDecision[] = await r.json()
      setDecisions(prev =>
        prev.length === data.length && prev[0]?.id === data[0]?.id ? prev : data
      )
    } catch {}
  }, [])

  const fetchTrades = useCallback(async () => {
    try {
      const r = await fetch('/api/trades?limit=500')
      if (!r.ok) return
      const data: Trade[] = await r.json()
      setTrades(prev =>
        prev.length === data.length && prev[0]?.id === data[0]?.id ? prev : data
      )
    } catch {}
  }, [])

  useEffect(() => {
    fetchStatus()
    fetchSignals()
    fetchDecisions()
    fetchTrades()
    const id = setInterval(() => { fetchStatus(); fetchSignals(); fetchDecisions(); fetchTrades() }, 15_000)
    return () => clearInterval(id)
  }, [fetchStatus, fetchSignals, fetchDecisions, fetchTrades])

  // ── Tech signals for left column, Mom for right (signals feed is already BUY/SELL only) ──
  const techSignals = useMemo(() => signals.filter(d => d.agent === 'TECH'),     [signals])
  const momSignals  = useMemo(() => signals.filter(d => d.agent === 'MOMENTUM'), [signals])

  // ── Confidence table rows ───────────────────────────────────────────────────
  const filteredRows = useMemo(() => {
    let rows = decisions
    if (search.trim()) {
      const q = search.trim().toLowerCase()
      rows = rows.filter(d => d.product_id.toLowerCase().includes(q) || d.agent.toLowerCase().includes(q))
    }
    if (agentFilter !== 'ALL') rows = rows.filter(d => d.agent === agentFilter)
    if (sideFilter  !== 'ALL') rows = rows.filter(d => d.side === sideFilter)
    if (sigOnly)               rows = rows.filter(d => d.side !== 'HOLD')

    return [...rows].sort((a, b) => {
      const av: any = a[sortKey] ?? (sortKey === 'product_id' || sortKey === 'agent' || sortKey === 'side' || sortKey === 'created_at' ? '' : -9999)
      const bv: any = b[sortKey] ?? (sortKey === 'product_id' || sortKey === 'agent' || sortKey === 'side' || sortKey === 'created_at' ? '' : -9999)
      if (av < bv) return sortAsc ? -1 : 1
      if (av > bv) return sortAsc ?  1 : -1
      return 0
    })
  }, [decisions, search, agentFilter, sideFilter, sigOnly, sortKey, sortAsc])

  const sortTh = (key: SortKey, label: string) => (
    <th
      key={key}
      onClick={() => { if (sortKey === key) setSortAsc(v => !v); else { setSortKey(key); setSortAsc(false) } }}
      className="px-3 py-2 text-left cursor-pointer hover:text-white select-none whitespace-nowrap"
    >
      {label} {sortKey === key ? (sortAsc ? '↑' : '↓') : ''}
    </th>
  )

  // ── Aggregate stats ─────────────────────────────────────────────────────────
  const techAg  = agentStatus.tech
  const momAg   = agentStatus.momentum
  const cnnAg   = agentStatus.cnn
  const scalpAg = agentStatus.scalp

  const totalBuy  = (techAg?.signals_buy  ?? 0) + (momAg?.signals_buy  ?? 0)
  const totalSell = (techAg?.signals_sell ?? 0) + (momAg?.signals_sell ?? 0)
  const totalPnl  = (techAg?.realized_pnl ?? 0) + (momAg?.realized_pnl ?? 0) + (cnnAg?.realized_pnl ?? 0) + (scalpAg?.realized_pnl ?? 0)

  return (
    <div className="space-y-6">

      {/* ── Combined stat row ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard
          label="Combined Signals"
          value={totalBuy + totalSell}
          sub={`${totalBuy} buy · ${totalSell} sell`}
        />
        <StatCard
          label="Combined PnL"
          value={`${totalPnl >= 0 ? '+' : ''}$${totalPnl.toFixed(2)}`}
          sub="Tech + Momentum + CNN + Scalp"
          color={totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <StatCard
          label="Open Positions"
          value={(techAg?.open_positions ?? 0) + (momAg?.open_positions ?? 0) + (cnnAg?.open_positions ?? 0) + (scalpAg?.open_positions ?? 0)}
          sub="across all agents"
        />
        <StatCard
          label="Total Signals"
          value={signals.length}
          sub={`${techSignals.length} tech · ${momSignals.length} mom`}
        />
      </div>

      {/* ── Per-agent stat cards ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {(['tech', 'momentum', 'cnn', 'scalp'] as const).map(key => {
          const ag    = agentStatus[key]
          const label = key === 'tech' ? 'TechAgent' : key === 'momentum' ? 'MomentumAgent' : key === 'cnn' ? 'CNN Agent' : 'ScalpAgent'
          const color = key === 'tech' ? 'text-purple-400' : key === 'momentum' ? 'text-blue-400' : key === 'cnn' ? 'text-yellow-400' : 'text-emerald-400'
          const borderClass = key === 'tech'
            ? 'border border-purple-900/50'
            : key === 'momentum'
            ? 'border border-blue-900/50'
            : key === 'cnn'
            ? 'border border-yellow-900/50'
            : 'border border-emerald-900/50'
          const pnlColor = !ag ? 'text-gray-500' : ag.realized_pnl >= 0 ? 'text-green-400' : 'text-red-400'

          return (
            <div key={key} className={`card p-4 ${borderClass}`}>
              <div className="flex items-center justify-between mb-3">
                <span className={`text-sm font-semibold ${color}`}>{label}</span>
                <span className="text-xs text-gray-600">
                  {ag?.last_scan_at ? timeAgo(ag.last_scan_at) : 'not scanned yet'}
                </span>
              </div>
              {/* Summary row */}
              <div className="grid grid-cols-3 gap-3 text-xs mb-3">
                <div>
                  <div className="text-gray-500 mb-0.5">Balance</div>
                  <div className="font-mono text-white text-sm">${ag?.balance?.toFixed(2) ?? '1000.00'}</div>
                </div>
                <div>
                  <div className="text-gray-500 mb-0.5">Realized PnL</div>
                  <div className={`font-mono text-sm ${pnlColor}`}>
                    {ag ? `${ag.realized_pnl >= 0 ? '+' : ''}$${ag.realized_pnl.toFixed(2)}` : '—'}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500 mb-0.5">Signals</div>
                  <div className="font-mono text-gray-300 text-sm">
                    {ag ? `${ag.signals_buy}↑ ${ag.signals_sell}↓` : '—'}
                  </div>
                </div>
                <div>
                  <div className="text-gray-500 mb-0.5">Scans</div>
                  <div className="font-mono text-gray-400">{ag?.scan_count ?? '—'}</div>
                </div>
                <div>
                  <div className="text-gray-500 mb-0.5">Open</div>
                  <div className={`font-mono ${(ag?.open_positions ?? 0) > 0 ? 'text-amber-400' : 'text-gray-500'}`}>
                    {ag?.open_positions ?? 0}
                  </div>
                </div>
                {key === 'momentum' && ag?.trailing_stops != null && (
                  <div>
                    <div className="text-gray-500 mb-0.5">Trailing stops</div>
                    <div className="font-mono text-amber-400">{ag.trailing_stops}</div>
                  </div>
                )}
              </div>

              {/* Open positions table */}
              {ag && Object.keys(ag.positions ?? {}).length > 0 && (
                <div className="mt-2 border-t border-gray-700 pt-2">
                  <div className="text-xs text-gray-500 mb-1.5 font-semibold">Open Positions</div>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-gray-600 border-b border-gray-800">
                        <th className="text-left pb-1">Symbol</th>
                        <th className="text-right pb-1">Size</th>
                        <th className="text-right pb-1">Entry</th>
                        <th className="text-right pb-1">Current</th>
                        <th className="text-right pb-1">Unreal. PnL</th>
                        {key === 'momentum' && <th className="text-right pb-1">Stop</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(ag.positions).map(([pid, pos]) => {
                        const pnlClr = pos.unrealized_pnl == null ? 'text-gray-500'
                          : pos.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'
                        return (
                          <tr key={pid} className="border-b border-gray-800/40">
                            <td className="py-1 font-mono font-bold text-white">
                              {pid.replace('-USD', '')}
                            </td>
                            <td className="py-1 text-right font-mono text-gray-300">
                              {pos.size.toFixed(4)}
                            </td>
                            <td className="py-1 text-right font-mono text-gray-400">
                              ${pos.avg_price >= 1000
                                ? pos.avg_price.toLocaleString('en-US', { maximumFractionDigits: 2 })
                                : pos.avg_price.toFixed(4)}
                            </td>
                            <td className="py-1 text-right font-mono text-gray-300">
                              {pos.current_price != null
                                ? `$${pos.current_price >= 1000
                                    ? pos.current_price.toLocaleString('en-US', { maximumFractionDigits: 2 })
                                    : pos.current_price.toFixed(4)}`
                                : '—'}
                            </td>
                            <td className={`py-1 text-right font-mono ${pnlClr}`}>
                              {pos.unrealized_pnl != null
                                ? `${pos.unrealized_pnl >= 0 ? '+' : ''}$${pos.unrealized_pnl.toFixed(2)}`
                                : '—'}
                              {pos.pct_pnl != null && (
                                <span className="text-gray-600 ml-1">
                                  ({pos.pct_pnl >= 0 ? '+' : ''}{pos.pct_pnl.toFixed(1)}%)
                                </span>
                              )}
                            </td>
                            {key === 'momentum' && (
                              <td className="py-1 text-right font-mono text-amber-500 text-xs">
                                {pos.high_water
                                  ? `$${pos.high_water >= 1000
                                      ? pos.high_water.toLocaleString('en-US', { maximumFractionDigits: 2 })
                                      : pos.high_water.toFixed(4)}`
                                  : '—'}
                              </td>
                            )}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {ag && Object.keys(ag.positions ?? {}).length === 0 && (
                <div className="text-xs text-gray-700 italic mt-1">No open positions</div>
              )}
            </div>
          )
        })}
      </div>

      {/* ── Signal feeds: Tech left / Momentum right ── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">

        {/* Tech signals */}
        <div>
          <h2 className="text-base font-bold text-white mb-3">
            <span className="text-purple-400">Tech</span> Signals
            <span className="text-sm text-gray-500 font-normal ml-2">({techSignals.length})</span>
          </h2>
          {techSignals.length === 0 ? (
            <div className="card text-center py-10">
              <p className="text-gray-500 text-sm">No Tech signals yet — agents start ~30s after backend</p>
            </div>
          ) : (
            <div className="space-y-3 max-h-[70vh] overflow-y-auto pr-1">
              {techSignals.map(d => {
                const isBuy  = d.side === 'BUY'
                const ind    = parseIndicators(d.reasoning)
                return (
                  <div key={d.id} className="card p-3 border border-purple-900/30">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                          isBuy
                            ? 'bg-green-900/50 text-green-400 border border-green-800'
                            : 'bg-red-900/50 text-red-400 border border-red-800'
                        }`}>{d.side}</span>
                        <span className="font-bold text-white text-sm">{d.product_id.replace('-USD', '')}</span>
                      </div>
                      <div className="text-right">
                        <div className="font-mono text-white text-sm">
                          ${d.price >= 1000
                            ? d.price.toLocaleString('en-US', { maximumFractionDigits: 2 })
                            : d.price.toFixed(4)}
                        </div>
                        <div className="text-xs text-gray-600">
                          {new Date(d.created_at).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>

                    {/* Confidence bar */}
                    <div className="mb-2">
                      <div className="text-xs text-gray-500 mb-0.5">Confidence</div>
                      <ConfBar value={d.confidence} color={isBuy ? 'bg-purple-500' : 'bg-orange-500'} />
                    </div>

                    {/* Indicator pills */}
                    <div className="flex flex-wrap gap-1.5 text-xs">
                      {Object.entries(ind).map(([k, v]) => {
                        const fv = parseFloat(v)
                        let cls = 'text-gray-400 border-gray-700 bg-gray-800'
                        if (k === 'RSI') cls = fv < 30 ? 'text-green-300 border-green-800 bg-green-900/30' : fv > 70 ? 'text-red-300 border-red-800 bg-red-900/30' : cls
                        if (k === 'ADX') cls = fv >= 25 ? 'text-amber-300 border-amber-800 bg-amber-900/20' : cls
                        if (k === 'MFI') cls = fv < 20 ? 'text-green-300 border-green-800 bg-green-900/30' : fv > 80 ? 'text-red-300 border-red-800 bg-red-900/30' : cls
                        return (
                          <span key={k} className={`px-2 py-0.5 rounded border ${cls}`}>
                            {k} {v}
                          </span>
                        )
                      })}
                      {d.score != null && (
                        <span className="px-2 py-0.5 rounded border text-purple-300 border-purple-800 bg-purple-900/20">
                          score {d.score.toFixed(2)}
                        </span>
                      )}
                      {d.pnl != null && (
                        <span className={`px-2 py-0.5 rounded border ${d.pnl >= 0 ? 'text-green-300 border-green-800 bg-green-900/20' : 'text-red-300 border-red-800 bg-red-900/20'}`}>
                          PnL {d.pnl >= 0 ? '+' : ''}${d.pnl.toFixed(2)}
                        </span>
                      )}
                    </div>

                    {d.reasoning && (
                      <div className="mt-2 text-xs text-gray-600 truncate" title={d.reasoning}>{d.reasoning}</div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* Momentum signals */}
        <div>
          <h2 className="text-base font-bold text-white mb-3">
            <span className="text-blue-400">Momentum</span> Signals
            <span className="text-sm text-gray-500 font-normal ml-2">({momSignals.length})</span>
          </h2>
          {momSignals.length === 0 ? (
            <div className="card text-center py-10">
              <p className="text-gray-500 text-sm">No Momentum signals yet — agents start ~60s after backend</p>
            </div>
          ) : (
            <div className="space-y-3 max-h-[70vh] overflow-y-auto pr-1">
              {momSignals.map(d => {
                const isBuy  = d.side === 'BUY'
                const ind    = parseIndicators(d.reasoning)
                return (
                  <div key={d.id} className="card p-3 border border-blue-900/30">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                          isBuy
                            ? 'bg-green-900/50 text-green-400 border border-green-800'
                            : 'bg-red-900/50 text-red-400 border border-red-800'
                        }`}>{d.side}</span>
                        <span className="font-bold text-white text-sm">{d.product_id.replace('-USD', '')}</span>
                      </div>
                      <div className="text-right">
                        <div className="font-mono text-white text-sm">
                          ${d.price >= 1000
                            ? d.price.toLocaleString('en-US', { maximumFractionDigits: 2 })
                            : d.price.toFixed(4)}
                        </div>
                        <div className="text-xs text-gray-600">
                          {new Date(d.created_at).toLocaleTimeString()}
                        </div>
                      </div>
                    </div>

                    <div className="mb-2">
                      <div className="text-xs text-gray-500 mb-0.5">Confidence</div>
                      <ConfBar value={d.confidence} color={isBuy ? 'bg-blue-500' : 'bg-orange-500'} />
                    </div>

                    <div className="flex flex-wrap gap-1.5 text-xs">
                      {Object.entries(ind).map(([k, v]) => {
                        const fv = parseFloat(v)
                        let cls = 'text-gray-400 border-gray-700 bg-gray-800'
                        if (k === 'ROC')   cls = fv > 0 ? 'text-green-300 border-green-800 bg-green-900/30' : 'text-red-300 border-red-800 bg-red-900/30'
                        if (k === 'Mom')   cls = fv > 0 ? 'text-green-300 border-green-800 bg-green-900/30' : 'text-red-300 border-red-800 bg-red-900/30'
                        if (k === 'VWMom') cls = fv > 0 ? 'text-green-300 border-green-800 bg-green-900/30' : 'text-red-300 border-red-800 bg-red-900/30'
                        return (
                          <span key={k} className={`px-2 py-0.5 rounded border ${cls}`}>
                            {k} {v}
                          </span>
                        )
                      })}
                      {d.score != null && (
                        <span className="px-2 py-0.5 rounded border text-blue-300 border-blue-800 bg-blue-900/20">
                          score {d.score.toFixed(2)}
                        </span>
                      )}
                      {d.pnl != null && (
                        <span className={`px-2 py-0.5 rounded border ${d.pnl >= 0 ? 'text-green-300 border-green-800 bg-green-900/20' : 'text-red-300 border-red-800 bg-red-900/20'}`}>
                          PnL {d.pnl >= 0 ? '+' : ''}${d.pnl.toFixed(2)}
                        </span>
                      )}
                    </div>

                    {d.reasoning && (
                      <div className="mt-2 text-xs text-gray-600 truncate" title={d.reasoning}>{d.reasoning}</div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* ── Trade Ledger ── */}
      {(() => {
        const filteredTrades = trades.filter(t =>
          tradeView === 'OPEN'   ? t.closed_at === null :
          tradeView === 'CLOSED' ? t.closed_at !== null : true
        )
        const openTrades   = trades.filter(t => t.closed_at === null)
        const closedTrades = trades.filter(t => t.closed_at !== null)
        const totalPnl     = closedTrades.reduce((s, t) => s + (t.pnl ?? 0), 0)

        const fmtDuration = (secs: number | null) => {
          if (secs === null) return '—'
          if (secs < 60)   return `${secs}s`
          if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`
          return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`
        }

        const triggerBadge = (t: string | null) => {
          if (!t) return null
          const cls =
            t === 'TICK_STOP'   ? 'bg-red-900/50 text-red-300 border-red-800' :
            t === 'TICK_PROFIT' ? 'bg-green-900/50 text-green-300 border-green-800' :
            t === 'TICK_SIGNAL' ? 'bg-blue-900/50 text-blue-300 border-blue-800' :
            t === 'TP'          ? 'bg-green-900/50 text-green-300 border-green-800' :
            t === 'SL'          ? 'bg-red-900/50 text-red-300 border-red-800' :
            t === 'TRAIL'       ? 'bg-orange-900/50 text-orange-300 border-orange-800' :
            t === 'TIME'        ? 'bg-gray-700 text-gray-300 border-gray-600' :
            t === 'SCALP'       ? 'bg-emerald-900/50 text-emerald-300 border-emerald-800' :
            t.startsWith('SCAN') ? 'bg-gray-800 text-gray-400 border-gray-700' :
                                   'bg-gray-800 text-gray-500 border-gray-700'
          return <span className={`px-1.5 py-0.5 rounded text-xs border ${cls}`}>{t}</span>
        }

        return (
          <div className="card overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
              <div className="flex items-center gap-3">
                <span className="text-sm font-semibold text-white">Trade Ledger</span>
                <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
                  {openTrades.length} open · {closedTrades.length} closed
                </span>
                {closedTrades.length > 0 && (
                  <span className={`text-xs px-2 py-0.5 rounded border ${
                    totalPnl >= 0
                      ? 'text-green-400 bg-green-900/20 border-green-800'
                      : 'text-red-400 bg-red-900/20 border-red-800'
                  }`}>
                    Total PnL {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
                  </span>
                )}
              </div>
              {/* View toggle */}
              <div className="flex rounded border border-gray-700 overflow-hidden text-xs">
                {(['ALL', 'OPEN', 'CLOSED'] as const).map(v => (
                  <button key={v}
                    onClick={() => setTradeView(v)}
                    className={`px-2.5 py-1.5 ${tradeView === v ? 'bg-blue-700 text-white' : 'text-gray-400 hover:bg-gray-700'}`}
                  >{v}</button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto max-h-[55vh] overflow-y-auto">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-gray-900 z-10">
                  <tr className="text-gray-500 border-b border-gray-800">
                    <th className="px-3 py-2 text-left">Agent</th>
                    <th className="px-3 py-2 text-left">Symbol</th>
                    <th className="px-3 py-2 text-left">Status</th>
                    <th className="px-3 py-2 text-right">Size</th>
                    <th className="px-3 py-2 text-right">Entry $</th>
                    <th className="px-3 py-2 text-right">Exit $</th>
                    <th className="px-3 py-2 text-right">USD In</th>
                    <th className="px-3 py-2 text-right">USD Out</th>
                    <th className="px-3 py-2 text-right">PnL</th>
                    <th className="px-3 py-2 text-right">PnL %</th>
                    <th className="px-3 py-2 text-left">Held</th>
                    <th className="px-3 py-2 text-left">Open trigger</th>
                    <th className="px-3 py-2 text-left">Close trigger</th>
                    <th className="px-3 py-2 text-left">Balance</th>
                    <th className="px-3 py-2 text-left">Opened</th>
                    <th className="px-3 py-2 text-left">Closed</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTrades.length === 0 ? (
                    <tr>
                      <td colSpan={16} className="text-center py-8 text-gray-600">
                        {trades.length === 0
                          ? 'No trades yet — waiting for first BUY signal'
                          : `No ${tradeView.toLowerCase()} trades`}
                      </td>
                    </tr>
                  ) : filteredTrades.map(t => {
                    const isOpen   = t.closed_at === null
                    const pnlColor = t.pnl == null ? 'text-gray-600'
                      : t.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                    return (
                      <tr key={t.id} className={`border-b border-gray-800/50 hover:bg-gray-800/20 ${
                        isOpen ? 'bg-amber-900/5' : t.pnl != null && t.pnl >= 0 ? 'bg-green-900/5' : 'bg-red-900/5'
                      }`}>
                        <td className="px-3 py-2">
                          <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${
                            t.agent === 'TECH'
                              ? 'bg-purple-900/40 text-purple-300 border border-purple-800'
                              : t.agent === 'CNN'
                              ? 'bg-yellow-900/40 text-yellow-300 border border-yellow-800'
                              : t.agent === 'SCALP'
                              ? 'bg-emerald-900/40 text-emerald-300 border border-emerald-800'
                              : 'bg-blue-900/40 text-blue-300 border border-blue-800'
                          }`}>{t.agent}</span>
                        </td>
                        <td className="px-3 py-2 font-mono font-bold text-white">
                          {t.product_id.replace('-USD', '')}
                        </td>
                        <td className="px-3 py-2">
                          {isOpen
                            ? <span className="text-amber-400 font-semibold">OPEN</span>
                            : <span className="text-gray-400">CLOSED</span>}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-gray-300">
                          {t.size.toFixed(5)}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-gray-300">
                          ${t.entry_price >= 1000
                            ? t.entry_price.toLocaleString('en-US', { maximumFractionDigits: 2 })
                            : t.entry_price.toFixed(4)}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-gray-400">
                          {t.exit_price != null
                            ? `$${t.exit_price >= 1000
                                ? t.exit_price.toLocaleString('en-US', { maximumFractionDigits: 2 })
                                : t.exit_price.toFixed(4)}`
                            : '—'}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-gray-400">
                          ${t.usd_open.toFixed(2)}
                        </td>
                        <td className="px-3 py-2 text-right font-mono text-gray-400">
                          {t.usd_close != null ? `$${t.usd_close.toFixed(2)}` : '—'}
                        </td>
                        <td className={`px-3 py-2 text-right font-mono font-semibold ${pnlColor}`}>
                          {t.pnl != null ? `${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)}` : '—'}
                        </td>
                        <td className={`px-3 py-2 text-right font-mono ${pnlColor}`}>
                          {t.pct_pnl != null ? `${t.pct_pnl >= 0 ? '+' : ''}${t.pct_pnl.toFixed(2)}%` : '—'}
                        </td>
                        <td className="px-3 py-2 font-mono text-gray-500">
                          {fmtDuration(t.hold_secs)}
                        </td>
                        <td className="px-3 py-2">{triggerBadge(t.trigger_open)}</td>
                        <td className="px-3 py-2">{triggerBadge(t.trigger_close) ?? <span className="text-gray-700">—</span>}</td>
                        <td className="px-3 py-2 font-mono text-gray-500">
                          ${t.balance_after.toFixed(0)}
                        </td>
                        <td className="px-3 py-2 text-gray-600 whitespace-nowrap">
                          {new Date(t.opened_at).toLocaleTimeString()}
                        </td>
                        <td className="px-3 py-2 text-gray-600 whitespace-nowrap">
                          {t.closed_at ? new Date(t.closed_at).toLocaleTimeString() : '—'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )
      })()}

      {/* ── All-Decisions Confidence Table ── */}
      <div className="card overflow-hidden">
        <button
          onClick={() => setShowTable(v => !v)}
          className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-800/50 transition-colors"
        >
          <div className="flex items-center gap-3">
            <span className="text-sm font-semibold text-white">Agent Decision History</span>
            <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
              {decisions.length} total
            </span>
            {signals.length > 0 && (
              <span className="text-xs text-green-400 bg-green-900/30 border border-green-800 px-2 py-0.5 rounded">
                {signals.length} signals
              </span>
            )}
          </div>
          <span className="text-gray-500 text-sm">{showTable ? '▲' : '▼'}</span>
        </button>

        {showTable && (
          <div className="border-t border-gray-800">
            {/* Filters */}
            <div className="flex flex-wrap items-center gap-2 px-4 py-3 bg-gray-900/50">
              <input
                type="text"
                placeholder="Search symbol…"
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-xs text-white w-36 focus:outline-none focus:border-blue-600"
              />

              {/* Agent filter */}
              <div className="flex rounded border border-gray-700 overflow-hidden text-xs">
                {(['ALL', 'TECH', 'MOMENTUM', 'CNN', 'SCALP'] as AgentFilter[]).map(f => (
                  <button key={f}
                    onClick={() => setAgentFilter(f)}
                    className={`px-2.5 py-1.5 ${agentFilter === f ? 'bg-blue-700 text-white' : 'text-gray-400 hover:bg-gray-700'}`}
                  >{f}</button>
                ))}
              </div>

              {/* Side filter */}
              <div className="flex rounded border border-gray-700 overflow-hidden text-xs">
                {(['ALL', 'BUY', 'SELL', 'HOLD'] as SideFilter[]).map(f => (
                  <button key={f}
                    onClick={() => setSideFilter(f)}
                    className={`px-2.5 py-1.5 ${sideFilter === f ? 'bg-blue-700 text-white' : 'text-gray-400 hover:bg-gray-700'}`}
                  >{f}</button>
                ))}
              </div>

              {/* Signals only toggle */}
              <button
                onClick={() => setSigOnly(v => !v)}
                className={`text-xs px-2.5 py-1.5 rounded border transition-colors ${
                  sigOnly ? 'bg-green-800 border-green-600 text-white' : 'border-gray-700 text-gray-400 hover:bg-gray-700'
                }`}
              >
                Signals only
              </button>

              <button onClick={() => { fetchSignals(); fetchDecisions() }} className="text-xs text-gray-400 hover:text-white px-2 py-1.5 border border-gray-700 rounded hover:border-gray-500">
                ↺ Refresh
              </button>

              <span className="text-xs text-gray-600 ml-auto">{filteredRows.length} rows</span>
            </div>

            {/* Table */}
            <div className="overflow-x-auto max-h-[60vh] overflow-y-auto">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-gray-900 z-10">
                  <tr className="text-gray-500 border-b border-gray-800">
                    {sortTh('agent',      'Agent')}
                    {sortTh('product_id', 'Symbol')}
                    {sortTh('side',       'Side')}
                    {sortTh('confidence', 'Conf')}
                    <th className="px-3 py-2 text-left whitespace-nowrap">Conf bar</th>
                    {sortTh('score',      'Score')}
                    {sortTh('pnl',        'PnL')}
                    {sortTh('balance',    'Balance')}
                    {sortTh('created_at', 'Time')}
                    <th className="px-3 py-2 text-left">Reasoning</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRows.length === 0 ? (
                    <tr>
                      <td colSpan={10} className="text-center py-8 text-gray-600">
                        No decisions yet — agents start scanning 30–60s after backend starts
                      </td>
                    </tr>
                  ) : filteredRows.map(d => {
                    const isBuy  = d.side === 'BUY'
                    const isSell = d.side === 'SELL'
                    const isTech = d.agent === 'TECH'
                    const isCNN  = d.agent === 'CNN'
                    return (
                      <tr key={d.id} className={`border-b border-gray-800/50 hover:bg-gray-800/30 ${
                        isBuy ? 'bg-green-900/5' : isSell ? 'bg-red-900/5' : ''
                      }`}>
                        <td className="px-3 py-2">
                          <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${
                            isTech
                              ? 'bg-purple-900/40 text-purple-300 border border-purple-800'
                              : isCNN
                              ? 'bg-yellow-900/40 text-yellow-300 border border-yellow-800'
                              : 'bg-blue-900/40 text-blue-300 border border-blue-800'
                          }`}>{d.agent}</span>
                        </td>
                        <td className="px-3 py-2 font-mono font-bold text-white">
                          {d.product_id.replace('-USD', '')}
                        </td>
                        <td className="px-3 py-2">
                          <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${
                            isBuy  ? 'bg-green-900/50 text-green-400 border border-green-800' :
                            isSell ? 'bg-red-900/50   text-red-400   border border-red-800'   :
                                     'bg-gray-800     text-gray-500  border border-gray-700'
                          }`}>{d.side}</span>
                        </td>
                        <td className={`px-3 py-2 font-mono ${isBuy ? 'text-green-400' : isSell ? 'text-red-400' : 'text-gray-500'}`}>
                          {Math.round(d.confidence * 100)}%
                        </td>
                        <td className="px-3 py-2">
                          <ConfBar
                            value={d.side !== 'HOLD' ? d.confidence : null}
                            color={isTech
                              ? (isBuy ? 'bg-purple-500' : 'bg-orange-500')
                              : isCNN
                              ? (isBuy ? 'bg-yellow-500' : 'bg-orange-500')
                              : (isBuy ? 'bg-blue-500'   : 'bg-orange-400')}
                          />
                        </td>
                        <td className="px-3 py-2 font-mono text-gray-400">
                          {d.score != null ? d.score.toFixed(2) : '—'}
                        </td>
                        <td className={`px-3 py-2 font-mono ${
                          d.pnl == null ? 'text-gray-600' : d.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {d.pnl != null ? `${d.pnl >= 0 ? '+' : ''}$${d.pnl.toFixed(2)}` : '—'}
                        </td>
                        <td className="px-3 py-2 font-mono text-gray-400">
                          {d.balance != null ? `$${d.balance.toFixed(0)}` : '—'}
                        </td>
                        <td className="px-3 py-2 text-gray-600 whitespace-nowrap">
                          {new Date(d.created_at).toLocaleTimeString()}
                        </td>
                        <td className="px-3 py-2 text-gray-500 max-w-xs truncate" title={d.reasoning ?? ''}>
                          {d.reasoning ?? '—'}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
