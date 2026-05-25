import React, { useCallback, useEffect, useMemo, useState } from 'react'
import PerformanceDashboard from './PerformanceDashboard'
import FiringCounter from './FiringCounter'

// ── Types ──────────────────────────────────────────────────────────────────────

interface AgentPosition {
  size:            number
  avg_price:       number
  current_price:   number | null
  unrealized_pnl:  number | null
  pct_pnl:         number | null
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
  // TechAgent retired #311-refactor-c — tech field kept in shape for API back-compat but ignored
  const [agentStatus, setAgentStatus] = useState<{ tech: SubAgentStatus | null; cnn: SubAgentStatus | null }>({ tech: null, cnn: null })
  const [signals,     setSignals]     = useState<AgentDecision[]>([])
  const [view,        setView]        = useState<'Live' | 'Performance'>('Live')

  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch('/api/agents/status')
      if (r.ok) {
        const d = await r.json()
        setAgentStatus({ tech: d.tech ?? null, cnn: d.cnn ?? null })
      }
    } catch {}
  }, [])

  const fetchSignals = useCallback(async () => {
    try {
      const r = await fetch('/api/agents/decisions?signals_only=true&limit=200')
      if (!r.ok) return
      const data: AgentDecision[] = await r.json()
      setSignals(prev =>
        prev.length === data.length && prev[0]?.id === data[0]?.id ? prev : data
      )
    } catch {}
  }, [])

  useEffect(() => {
    fetchStatus()
    fetchSignals()
    const id = setInterval(() => { fetchStatus(); fetchSignals() }, 15_000)
    return () => clearInterval(id)
  }, [fetchStatus, fetchSignals])

  // TechAgent retired #311-refactor-c — TECH signals (if any historical rows
  // leak into the API response) are filtered out here so the live UI only
  // shows CNN. PerformanceDashboard's TECH filter still surfaces TECH history.
  const cnnSignals  = useMemo(() => signals.filter(d => d.agent === 'CNN'),  [signals])

  // ── Aggregate stats ─────────────────────────────────────────────────────────
  const cnnAg   = agentStatus.cnn

  const totalBuy  = (cnnAg?.signals_buy  ?? 0)
  const totalSell = (cnnAg?.signals_sell ?? 0)
  const totalPnl  = (cnnAg?.realized_pnl ?? 0)

  return (
    <div className="space-y-4">
      {/* ── Inner sub-tabs: live agent view / performance ── */}
      <div className="flex gap-0.5 border-b border-gray-800 pb-2">
        {(['Live', 'Performance'] as const).map(v => (
          <button
            key={v}
            onClick={() => setView(v)}
            className={`tab ${view === v ? 'tab-active' : 'tab-inactive'}`}
          >
            {v}
          </button>
        ))}
      </div>

      {view === 'Performance' ? (
        <div className="space-y-6">
          <PerformanceDashboard />
          <FiringCounter />
        </div>
      ) : (
      <div className="space-y-6">

      {/* ── Combined stat row ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard
          label="Signals"
          value={totalBuy + totalSell}
          sub={`${totalBuy} buy · ${totalSell} sell`}
        />
        <StatCard
          label="Realized PnL"
          value={`${totalPnl >= 0 ? '+' : ''}$${totalPnl.toFixed(2)}`}
          sub="XGB (live)"
          color={totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <StatCard
          label="Open Positions"
          value={cnnAg?.open_positions ?? 0}
          sub="XGB"
        />
        <StatCard
          label="Total Signals"
          value={cnnSignals.length}
          sub="XGB only"
        />
      </div>

      {/* ── Per-agent columns: XGB only (TechAgent retired #311-refactor-c) ── */}
      <div className="grid grid-cols-1 gap-6 items-start">
        {(['cnn'] as const).map(key => {
          const ag    = agentStatus[key]
          const label = 'XGB Agent'
          const color = 'text-yellow-400'
          const borderClass = 'border border-yellow-900/50'
          const pnlColor = !ag ? 'text-gray-500' : ag.realized_pnl >= 0 ? 'text-green-400' : 'text-red-400'

          return (
            <div key={key} className={`card p-5 ${borderClass}`}>
              <div className="flex items-center justify-between mb-4">
                <span className={`text-base font-semibold ${color}`}>{label}</span>
                <span className="text-xs text-gray-600">
                  {ag?.last_scan_at ? timeAgo(ag.last_scan_at) : 'not scanned yet'}
                </span>
              </div>
              {/* Summary row */}
              <div className="grid grid-cols-5 gap-4 text-xs mb-4">
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
              </div>

              {/* Open positions table */}
              {ag && Object.keys(ag.positions ?? {}).length > 0 && (
                <div className="mt-2 border-t border-gray-700 pt-2 max-h-48 overflow-y-auto">
                  <div className="text-xs text-gray-500 mb-1.5 font-semibold">Open Positions</div>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-gray-600 border-b border-gray-800">
                        <th className="text-left pb-1">Symbol</th>
                        <th className="text-right pb-1">Size</th>
                        <th className="text-right pb-1">Entry</th>
                        <th className="text-right pb-1">Current</th>
                        <th className="text-right pb-1">Unreal. PnL</th>
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

      {/* TechAgent retired #311-refactor-c (2026-05-16) — fully removed from
          the UI as of dashboard cleanup. Historical TECH trades still live in
          the trades table but are filtered out of every view. */}

      </div>
      )}
    </div>
  )
}
