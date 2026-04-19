import React, { useEffect, useRef, useState } from 'react'

// Module-level cache — survives tab switches, cleared on manual refresh
const _CACHE_TTL_MS = 2 * 60 * 1000  // 2 minutes
let _cache: { data: PerfData; trades: any[]; decisions: any[]; ts: number } | null = null

// ── Types ──────────────────────────────────────────────────────────────────────

interface MonthStats {
  month:              string
  trades:             number
  wins:               number
  losses:             number
  win_rate:           number
  total_pnl:          number
  avg_pct_pnl:        number
  avg_win:            number | null
  avg_loss:           number | null
  expectancy:         number
  monthly_return_pct: number
  open_balance:       number | null
  close_balance:      number | null
}

interface Rolling30d {
  trades:       number
  wins:         number
  win_rate_pct: number
  return_pct:   number
  total_pnl:    number
}

interface Overall {
  total_trades:    number
  total_wins:      number
  win_rate_pct:    number
  total_pnl:       number
  current_balance: number
  peak_balance:    number
}

interface Projection {
  annual_goal_usd:      number
  trailing_monthly_pct: number
  months_to_goal:       number | null
}

interface PerfData {
  months:      MonthStats[]
  rolling_30d: Rolling30d
  overall:     Overall
  projection:  Projection
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

interface Decision {
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

type AgentFilter = 'ALL' | 'TECH' | 'MOMENTUM' | 'CNN' | 'SCALP'
type TradeView   = 'ALL' | 'OPEN' | 'CLOSED'

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtHold(secs: number | null): string {
  if (!secs) return '—'
  if (secs < 3600)  return `${Math.round(secs / 60)}m`
  if (secs < 86400) return `${(secs / 3600).toFixed(1)}h`
  return `${(secs / 86400).toFixed(1)}d`
}

function fmtTime(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

const AGENT_COLORS: Record<string, string> = {
  CNN:      'text-blue-400',
  MOMENTUM: 'text-purple-400',
  SCALP:    'text-yellow-400',
  TECH:     'text-green-400',
}

const AGENT_BADGES: Record<string, string> = {
  CNN:      'bg-blue-900/40 text-blue-300',
  MOMENTUM: 'bg-purple-900/40 text-purple-300',
  SCALP:    'bg-yellow-900/40 text-yellow-300',
  TECH:     'bg-green-900/40 text-green-300',
}

// ── Stat card ─────────────────────────────────────────────────────────────────

function Stat({ label, value, sub, color }: {
  label: string; value: string; sub?: string; color?: string
}) {
  return (
    <div className="bg-gray-800 rounded-lg p-3 flex flex-col gap-0.5">
      <div className="text-xs text-gray-400">{label}</div>
      <div className={`text-xl font-bold ${color ?? 'text-white'}`}>{value}</div>
      {sub && <div className="text-xs text-gray-500">{sub}</div>}
    </div>
  )
}

// ── Filter pill ───────────────────────────────────────────────────────────────

function Pill({ active, onClick, children }: {
  active: boolean; onClick: () => void; children: React.ReactNode
}) {
  return (
    <button
      onClick={onClick}
      className={`text-xs px-2.5 py-1 rounded border transition-colors ${
        active
          ? 'bg-blue-700 border-blue-500 text-white'
          : 'bg-gray-800 border-gray-600 text-gray-400 hover:text-gray-200'
      }`}
    >
      {children}
    </button>
  )
}

// ── Section header ────────────────────────────────────────────────────────────

function SectionHeader({ title, count, children }: {
  title: string; count?: number; children?: React.ReactNode
}) {
  return (
    <div className="flex items-center justify-between mb-3">
      <div className="flex items-center gap-2">
        <span className="text-sm font-semibold text-white">{title}</span>
        {count !== undefined && (
          <span className="text-xs text-gray-500 font-mono">({count})</span>
        )}
      </div>
      {children && <div className="flex items-center gap-1.5">{children}</div>}
    </div>
  )
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function PerformanceDashboard() {
  const cached = _cache && (Date.now() - _cache.ts < _CACHE_TTL_MS) ? _cache : null
  const [data,      setData]      = useState<PerfData | null>(cached?.data ?? null)
  const [trades,    setTrades]    = useState<Trade[]>(cached?.trades ?? [])
  const [decisions, setDecisions] = useState<Decision[]>(cached?.decisions ?? [])
  const [loading,   setLoading]   = useState(!cached)
  const [error,     setError]     = useState('')

  // Filter state
  const [tradeAgent,    setTradeAgent]    = useState<AgentFilter>('ALL')
  const [tradeView,     setTradeView]     = useState<TradeView>('CLOSED')
  const [decisionAgent, setDecisionAgent] = useState<AgentFilter>('ALL')

  const load = async (force = false) => {
    if (!force && _cache && (Date.now() - _cache.ts < _CACHE_TTL_MS)) return
    try {
      setLoading(true)
      const [perfRes, tradesRes, decisionsRes] = await Promise.all([
        fetch('/api/performance'),
        fetch('/api/trades?limit=500'),
        fetch('/api/agents/decisions?signals_only=true&limit=300'),
      ])
      if (!perfRes.ok) throw new Error(`HTTP ${perfRes.status}`)
      const d = await perfRes.json()
      const t = tradesRes.ok ? await tradesRes.json() : []
      const dec = decisionsRes.ok ? await decisionsRes.json() : []
      _cache = { data: d, trades: t, decisions: dec, ts: Date.now() }
      setData(d); setTrades(t); setDecisions(dec)
      setError('')
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])  // uses cache if fresh

  if (loading) return (
    <div className="flex items-center justify-center h-48 text-gray-500 text-sm">Loading…</div>
  )
  if (error) return (
    <div className="flex items-center justify-center h-48 text-red-400 text-sm">{error}</div>
  )
  if (!data) return null

  const { months, rolling_30d, overall, projection } = data

  const breakEvenWinRate = 41.0
  const goalUsd          = projection.annual_goal_usd

  const projLabel = (() => {
    if (projection.months_to_goal === null)
      return projection.trailing_monthly_pct <= 0 ? 'Need positive return' : 'Calculating…'
    const m = Math.round(projection.months_to_goal)
    return m < 12 ? `~${m} months` : `~${(m / 12).toFixed(1)} years`
  })()

  const winRateColor = (wr: number) =>
    wr >= 50 ? 'text-green-400' : wr >= breakEvenWinRate ? 'text-yellow-400' : 'text-red-400'

  const pnlColor = (v: number) => v >= 0 ? 'text-green-400' : 'text-red-400'

  // Filtered trades
  const filteredTrades = trades.filter(t => {
    if (tradeAgent !== 'ALL' && t.agent !== tradeAgent) return false
    if (tradeView === 'OPEN'   && t.closed_at !== null)  return false
    if (tradeView === 'CLOSED' && t.closed_at === null)  return false
    return true
  })

  // Filtered decisions
  const filteredDecisions = decisions.filter(d =>
    decisionAgent === 'ALL' || d.agent === decisionAgent
  )

  const AGENTS: AgentFilter[] = ['ALL', 'CNN', 'MOMENTUM', 'SCALP', 'TECH']

  return (
    <div className="space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white">Performance Dashboard</h2>
          <p className="text-xs text-gray-400 mt-0.5">
            Break-even win rate: ~41% &nbsp;·&nbsp; Goal: ${goalUsd.toLocaleString()}/yr
          </p>
        </div>
        <button onClick={() => load(true)} className="btn-secondary text-xs py-1.5 px-3">Refresh</button>
      </div>

      {/* Top stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat
          label="All-Time Win Rate"
          value={`${overall.win_rate_pct}%`}
          sub={`${overall.total_wins}W / ${overall.total_trades - overall.total_wins}L`}
          color={winRateColor(overall.win_rate_pct)}
        />
        <Stat
          label="All-Time P&L"
          value={`${overall.total_pnl >= 0 ? '+' : ''}$${overall.total_pnl.toFixed(2)}`}
          sub={`${overall.total_trades} closed trades`}
          color={pnlColor(overall.total_pnl)}
        />
        <Stat
          label="Trailing 30d Return"
          value={`${rolling_30d.return_pct >= 0 ? '+' : ''}${rolling_30d.return_pct}%`}
          sub={`${rolling_30d.trades} trades · ${rolling_30d.win_rate_pct}% win`}
          color={pnlColor(rolling_30d.return_pct)}
        />
        <Stat
          label={`Path to $${(goalUsd / 1000).toFixed(0)}k/yr`}
          value={projLabel}
          sub={`at ${projection.trailing_monthly_pct}%/mo`}
          color={projection.months_to_goal !== null ? 'text-blue-400' : 'text-gray-400'}
        />
      </div>

      {/* Monthly table */}
      {months.length > 0 && (
        <div className="card overflow-x-auto">
          <SectionHeader title="Month-by-Month Breakdown" />
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-400 border-b border-gray-700">
                <th className="text-left py-1.5 pr-4">Month</th>
                <th className="text-right pr-4">Trades</th>
                <th className="text-right pr-4">Win Rate</th>
                <th className="text-right pr-4">Expectancy</th>
                <th className="text-right pr-4">Return %</th>
                <th className="text-right pr-4">P&L ($)</th>
                <th className="text-right">Closing Bal</th>
              </tr>
            </thead>
            <tbody>
              {[...months].reverse().map(m => (
                <tr key={m.month} className="border-b border-gray-800 hover:bg-gray-800/40">
                  <td className="py-1.5 pr-4 text-white font-mono">{m.month}</td>
                  <td className="text-right pr-4 text-gray-300">{m.trades}</td>
                  <td className={`text-right pr-4 ${winRateColor(m.win_rate)}`}>{m.win_rate}%</td>
                  <td className={`text-right pr-4 font-mono ${pnlColor(m.expectancy)}`}>
                    {m.expectancy >= 0 ? '+' : ''}${m.expectancy.toFixed(3)}
                  </td>
                  <td className={`text-right pr-4 font-mono ${pnlColor(m.monthly_return_pct)}`}>
                    {m.monthly_return_pct >= 0 ? '+' : ''}{m.monthly_return_pct}%
                  </td>
                  <td className={`text-right pr-4 font-mono ${pnlColor(m.total_pnl)}`}>
                    {m.total_pnl >= 0 ? '+' : ''}${m.total_pnl.toFixed(2)}
                  </td>
                  <td className="text-right font-mono text-gray-300">
                    {m.close_balance != null ? `$${m.close_balance.toFixed(2)}` : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Trade Ledger ──────────────────────────────────────────────────────── */}
      <div className="card overflow-x-auto">
        <SectionHeader title="Trade Ledger" count={filteredTrades.length}>
          <div className="flex gap-1">
            {(['ALL', 'OPEN', 'CLOSED'] as TradeView[]).map(v => (
              <Pill key={v} active={tradeView === v} onClick={() => setTradeView(v)}>{v}</Pill>
            ))}
          </div>
          <div className="flex gap-1 ml-2">
            {AGENTS.map(a => (
              <Pill key={a} active={tradeAgent === a} onClick={() => setTradeAgent(a)}>{a}</Pill>
            ))}
          </div>
        </SectionHeader>

        {filteredTrades.length === 0 ? (
          <div className="text-gray-500 text-xs py-6 text-center">No trades match the current filter.</div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-400 border-b border-gray-700">
                <th className="text-left py-1.5 pr-3">Agent</th>
                <th className="text-left pr-3">Product</th>
                <th className="text-right pr-3">Entry</th>
                <th className="text-right pr-3">Exit</th>
                <th className="text-right pr-3">Size $</th>
                <th className="text-right pr-3">P&L</th>
                <th className="text-right pr-3">P&L %</th>
                <th className="text-right pr-3">Hold</th>
                <th className="text-right pr-3">Trigger</th>
                <th className="text-right">Opened</th>
              </tr>
            </thead>
            <tbody>
              {filteredTrades.map(t => (
                <tr key={t.id} className="border-b border-gray-800 hover:bg-gray-800/40">
                  <td className="py-1 pr-3">
                    <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${AGENT_BADGES[t.agent] ?? 'bg-gray-700 text-gray-300'}`}>
                      {t.agent}
                    </span>
                  </td>
                  <td className="pr-3 text-white font-mono">{t.product_id.replace('-USD', '')}</td>
                  <td className="text-right pr-3 font-mono text-gray-300">${t.entry_price.toFixed(4)}</td>
                  <td className="text-right pr-3 font-mono text-gray-300">
                    {t.exit_price != null ? `$${t.exit_price.toFixed(4)}` : <span className="text-yellow-400">Open</span>}
                  </td>
                  <td className="text-right pr-3 font-mono text-gray-300">${t.usd_open.toFixed(2)}</td>
                  <td className={`text-right pr-3 font-mono ${t.pnl != null ? pnlColor(t.pnl) : 'text-gray-500'}`}>
                    {t.pnl != null ? `${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(3)}` : '—'}
                  </td>
                  <td className={`text-right pr-3 font-mono ${t.pct_pnl != null ? pnlColor(t.pct_pnl) : 'text-gray-500'}`}>
                    {t.pct_pnl != null ? `${t.pct_pnl >= 0 ? '+' : ''}${t.pct_pnl.toFixed(2)}%` : '—'}
                  </td>
                  <td className="text-right pr-3 text-gray-400">{fmtHold(t.hold_secs)}</td>
                  <td className="text-right pr-3 text-gray-400 font-mono text-xs">
                    {t.trigger_close ?? t.trigger_open}
                  </td>
                  <td className="text-right text-gray-500">{fmtTime(t.opened_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* ── Agent Decision History ─────────────────────────────────────────────── */}
      <div className="card overflow-x-auto">
        <SectionHeader title="Agent Decision History" count={filteredDecisions.length}>
          <div className="flex gap-1">
            {AGENTS.map(a => (
              <Pill key={a} active={decisionAgent === a} onClick={() => setDecisionAgent(a)}>{a}</Pill>
            ))}
          </div>
        </SectionHeader>

        {filteredDecisions.length === 0 ? (
          <div className="text-gray-500 text-xs py-6 text-center">No decisions found.</div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-400 border-b border-gray-700">
                <th className="text-left py-1.5 pr-3">Agent</th>
                <th className="text-left pr-3">Product</th>
                <th className="text-left pr-3">Side</th>
                <th className="text-right pr-3">Confidence</th>
                <th className="text-right pr-3">Price</th>
                <th className="text-left pr-3">Reasoning</th>
                <th className="text-right">Time</th>
              </tr>
            </thead>
            <tbody>
              {filteredDecisions.map(d => (
                <tr key={d.id} className="border-b border-gray-800 hover:bg-gray-800/40">
                  <td className="py-1 pr-3">
                    <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${AGENT_BADGES[d.agent] ?? 'bg-gray-700 text-gray-300'}`}>
                      {d.agent}
                    </span>
                  </td>
                  <td className="pr-3 text-white font-mono">{d.product_id.replace('-USD', '')}</td>
                  <td className="pr-3">
                    <span className={`font-semibold ${d.side === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
                      {d.side}
                    </span>
                  </td>
                  <td className="text-right pr-3 font-mono text-gray-300">
                    {Math.round(d.confidence * 100)}%
                  </td>
                  <td className="text-right pr-3 font-mono text-gray-300">
                    ${d.price < 1 ? d.price.toFixed(5) : d.price.toFixed(2)}
                  </td>
                  <td className="pr-3 text-gray-400 max-w-xs truncate" title={d.reasoning ?? ''}>
                    {d.reasoning ?? '—'}
                  </td>
                  <td className="text-right text-gray-500 whitespace-nowrap">{fmtTime(d.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Projection detail */}
      <div className="card">
        <div className="text-sm font-semibold text-white mb-3">$50k/Year Projection</div>
        <div className="text-xs text-gray-400 space-y-1">
          <p>Current balance: <span className="text-white font-mono">${overall.current_balance.toFixed(2)}</span></p>
          <p>Trailing 30-day return: <span className={`font-mono ${pnlColor(rolling_30d.return_pct)}`}>
            {rolling_30d.return_pct >= 0 ? '+' : ''}{rolling_30d.return_pct}%
          </span></p>
          <p>Extrapolated annual: <span className="text-white font-mono">{(rolling_30d.return_pct * 12).toFixed(1)}%</span></p>
          {projection.trailing_monthly_pct > 0 && overall.current_balance > 0 && (
            <p className="mt-2 text-gray-300">
              At <span className="text-blue-300">{projection.trailing_monthly_pct}%/month</span> compounded,
              your account generates $50k/year when it reaches{' '}
              <span className="text-white font-mono">
                ${(goalUsd / (projection.trailing_monthly_pct / 100 * 12)).toFixed(0)}
              </span> — estimated in <span className="text-blue-300 font-semibold">{projLabel}</span>.
            </p>
          )}
          {projection.trailing_monthly_pct <= 0 && (
            <p className="text-yellow-400 mt-2">
              Monthly return is currently negative — win rate needs to reach ~41% before profits compound.
            </p>
          )}
          <p className="text-gray-500 mt-3">
            Projection uses trailing 30-day rate. Check again after 30+ trading days for a reliable baseline.
          </p>
        </div>
      </div>

    </div>
  )
}
