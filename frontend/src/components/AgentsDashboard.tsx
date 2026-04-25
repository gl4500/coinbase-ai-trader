import React, { useCallback, useEffect, useMemo, useState } from 'react'

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
  const [agentStatus, setAgentStatus] = useState<{ tech: SubAgentStatus | null; cnn: SubAgentStatus | null }>({ tech: null, cnn: null })
  const [signals,     setSignals]     = useState<AgentDecision[]>([])

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

  const techSignals = useMemo(() => signals.filter(d => d.agent === 'TECH'), [signals])

  // ── Aggregate stats ─────────────────────────────────────────────────────────
  const techAg  = agentStatus.tech
  const cnnAg   = agentStatus.cnn

  const totalBuy  = (techAg?.signals_buy  ?? 0)
  const totalSell = (techAg?.signals_sell ?? 0)
  const totalPnl  = (techAg?.realized_pnl ?? 0) + (cnnAg?.realized_pnl ?? 0)

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
          sub="Tech + CNN"
          color={totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <StatCard
          label="Open Positions"
          value={(techAg?.open_positions ?? 0) + (cnnAg?.open_positions ?? 0)}
          sub="across all agents"
        />
        <StatCard
          label="Total Signals"
          value={signals.length}
          sub={`${techSignals.length} tech`}
        />
      </div>

      {/* ── Per-agent stat cards ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 xl:grid-cols-4 gap-4 items-start">
        {(['tech', 'cnn'] as const).map(key => {
          const ag    = agentStatus[key]
          const label = key === 'tech' ? 'TechAgent' : 'CNN Agent'
          const color = key === 'tech' ? 'text-purple-400' : 'text-yellow-400'
          const borderClass = key === 'tech'
            ? 'border border-purple-900/50'
            : 'border border-yellow-900/50'
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

      {/* ── Tech signal feed ── */}
      <div className="max-w-3xl">

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

      </div>

    </div>
  )
}
