import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { Signal, Order } from '../App'

// ── Types ──────────────────────────────────────────────────────────────────────

interface DrawdownStatus {
  halted:             boolean
  halt_reason:        string
  day_start_balance:  number | null
  week_start_balance: number | null
  day_elapsed_pct:    number
  week_elapsed_pct:   number
  daily_limit:        number
  weekly_limit:       number
}

interface CNNStatus {
  torch_available:  boolean
  model_loaded:     boolean
  last_scan_at:     number | null   // unix timestamp
  next_scan_at:     number | null
  scan_count:       number
  signals_total:    number
  signals_buy:      number
  signals_sell:     number
  signals_executed: number
  dry_run:          boolean
  dry_run_balance:  number | null
  is_trading:       boolean
  drawdown:         DrawdownStatus | null
}

interface CnnScan {
  id:          number
  product_id:  string
  price:       number
  cnn_prob:    number | null
  llm_prob:    number | null
  model_prob:  number
  cnn_weight:  number | null
  llm_weight:  number | null
  side:        string        // BUY | SELL | HOLD
  strength:    number
  signal_gen:  number        // 0 | 1
  regime:      string | null
  adx:         number | null
  rsi:         number | null
  macd:        number | null
  mfi:         number | null
  stoch_k:     number | null
  atr:         number | null
  vwap_dist:   number | null
  fast_rsi:    number | null   // 5-min RSI(12) / 100
  velocity:    number | null   // 5-min price velocity, normalised [-1,1]
  vol_z:       number | null   // 5-min volume z-score, normalised [-1,1]
  scanned_at:  string
}

interface ParsedSignal {
  raw:        Signal
  cnnProb:    number | null
  llmProb:    number | null
  modelProb:  number | null
  rsi:        number | null
  macdHist:   number | null
  bbPos:      number | null
  obImb:      number | null
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function parseReasoning(reasoning: string | null | undefined): Omit<ParsedSignal, 'raw'> {
  const defaults = { cnnProb: null, llmProb: null, modelProb: null,
                     rsi: null, macdHist: null, bbPos: null, obImb: null }
  if (!reasoning) return defaults

  const num = (pattern: RegExp) => {
    const m = reasoning.match(pattern)
    return m ? parseFloat(m[1]) : null
  }

  return {
    cnnProb:   num(/cnn_prob=([\d.]+)/),
    llmProb:   num(/llm_prob=([\d.]+)/),
    modelProb: num(/model_prob=([\d.]+)/),
    rsi:       num(/RSI:\s*([\d.]+)/),
    macdHist:  num(/MACD hist:\s*([+-]?[\d.]+)/),
    bbPos:     num(/Bollinger pos:\s*([\d.]+)/),
    obImb:     num(/OB imbalance:\s*([+-]?[\d.]+)/),
  }
}

function timeAgo(unix: number | null): string {
  if (!unix) return '—'
  const secs = Math.floor(Date.now() / 1000 - unix)
  if (secs < 60)   return `${secs}s ago`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`
  return `${Math.floor(secs / 3600)}h ago`
}

function timeUntil(unix: number | null): string {
  if (!unix) return '—'
  const secs = Math.floor(unix - Date.now() / 1000)
  if (secs <= 0)   return 'now'
  if (secs < 60)   return `${secs}s`
  if (secs < 3600) return `${Math.floor(secs / 60)}m`
  return `${Math.floor(secs / 3600)}h`
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

function ProbBar({ value, color }: { value: number | null; color: string }) {
  if (value === null) return <span className="text-gray-600 text-xs">—</span>
  const pct = Math.round(value * 100)
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-20 bg-gray-700 rounded-full h-1.5 flex-shrink-0">
        <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-gray-300">{pct}%</span>
    </div>
  )
}

// ── Sub-agent types ────────────────────────────────────────────────────────────

interface AgentDecision {
  id:         number
  agent:      string        // TECH | MOMENTUM
  product_id: string
  side:       string        // BUY | SELL | HOLD
  confidence: number
  price:      number
  score:      number | null
  reasoning:  string | null
  balance:    number | null
  pnl:        number | null
  created_at: string
}

// ── Main Component ─────────────────────────────────────────────────────────────

interface Props {
  signals:  Signal[]
  orders:   Order[]
  postJSON: (url: string, opts?: RequestInit) => Promise<Response>
}

type SortKey = 'model_prob' | 'cnn_prob' | 'llm_prob' | 'strength' | 'adx' | 'rsi' | 'product_id' | 'scanned_at' | 'vwap_dist' | 'velocity'

export default function CNNDashboard({ signals, orders, postJSON }: Props) {
  const [status,      setStatus]      = useState<CNNStatus | null>(null)
  const [scanning,    setScanning]    = useState(false)
  const [training,    setTraining]    = useState(false)
  const [trainResult, setTrainResult] = useState<string | null>(null)
  const [epochs,      setEpochs]      = useState(20)
  const [trainSecs,   setTrainSecs]   = useState(0)
  const [statusMsg,   setStatusMsg]   = useState('')
  const [scans,       setScans]       = useState<CnnScan[]>([])
  const [search,      setSearch]      = useState('')
  const [sideFilter,  setSideFilter]  = useState<'ALL' | 'BUY' | 'SELL' | 'HOLD'>('ALL')
  const [sigFilter,   setSigFilter]   = useState<'ALL' | 'SIG' | 'NOSIG'>('ALL')
  const [sortKey,          setSortKey]          = useState<SortKey>('model_prob')
  const [sortAsc,          setSortAsc]          = useState(false)
  const [showScans,        setShowScans]        = useState(false)
  const [agentDecisions,   setAgentDecisions]   = useState<AgentDecision[]>([])

  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch('/api/cnn/status')
      if (r.ok) setStatus(await r.json())
    } catch {}
  }, [])

  const fetchScans = useCallback(async () => {
    try {
      const r = await fetch('/api/cnn/scans?limit=1000')
      if (r.ok) setScans(await r.json())
    } catch {}
  }, [])

  useEffect(() => {
    fetchStatus()
    const id = setInterval(fetchStatus, 15_000)
    return () => clearInterval(id)
  }, [fetchStatus])

  // Always fetch scans on mount so the header count is correct immediately
  useEffect(() => {
    fetchScans()
  }, [fetchScans])

  // While panel is open, also poll every 30s for new rows
  useEffect(() => {
    if (!showScans) return
    const id = setInterval(fetchScans, 30_000)
    return () => clearInterval(id)
  }, [showScans, fetchScans])

  const fetchAgentDecisions = useCallback(async () => {
    try {
      const r = await fetch('/api/agents/decisions?signals_only=true&limit=200')
      if (r.ok) setAgentDecisions(await r.json())
    } catch {}
  }, [])

  useEffect(() => {
    fetchAgentDecisions()
    const id = setInterval(fetchAgentDecisions, 15_000)
    return () => clearInterval(id)
  }, [fetchAgentDecisions])

  // Per-product lookup: latest Tech & Momentum decision (agentDecisions is newest-first)
  const agentByProduct = useMemo(() => {
    const map = new Map<string, { tech: AgentDecision | null; mom: AgentDecision | null }>()
    for (const d of agentDecisions) {
      const cur = map.get(d.product_id) ?? { tech: null, mom: null }
      if (d.agent === 'TECH'     && cur.tech === null) cur.tech = d
      if (d.agent === 'MOMENTUM' && cur.mom  === null) cur.mom  = d
      map.set(d.product_id, cur)
      if (cur.tech && cur.mom) continue  // both found, no need to keep scanning
    }
    return map
  }, [agentDecisions])

  // Filtered + sorted scans
  const filteredScans = useMemo(() => {
    let rows = scans
    if (search.trim()) {
      const q = search.trim().toLowerCase()
      rows = rows.filter(s => s.product_id.toLowerCase().includes(q))
    }
    if (sideFilter !== 'ALL') rows = rows.filter(s => s.side === sideFilter)
    if (sigFilter === 'SIG')   rows = rows.filter(s => s.signal_gen === 1)
    if (sigFilter === 'NOSIG') rows = rows.filter(s => s.signal_gen === 0)

    rows = [...rows].sort((a, b) => {
      const av = a[sortKey] ?? (sortKey === 'product_id' ? '' : -999)
      const bv = b[sortKey] ?? (sortKey === 'product_id' ? '' : -999)
      if (av < bv) return sortAsc ? -1 : 1
      if (av > bv) return sortAsc ?  1 : -1
      return 0
    })
    return rows
  }, [scans, search, sideFilter, sigFilter, sortKey, sortAsc])

  const handleScan = async () => {
    setScanning(true)
    setStatusMsg('Scanning all pairs…')
    try {
      const r = await postJSON('/api/cnn/scan')
      const d = await r.json()
      setStatusMsg(`Scan complete — ${d.signals_generated ?? 0} signals`)
      await fetchStatus()
      await fetchScans()
    } catch {
      setStatusMsg('Scan failed')
    } finally {
      setScanning(false)
      setTimeout(() => setStatusMsg(''), 4000)
    }
  }

  const handleTrain = async () => {
    setTraining(true)
    setTrainResult(null)
    setStatusMsg('')
    setTrainSecs(0)

    // Elapsed-time ticker
    const start = Date.now()
    const ticker = setInterval(() => {
      setTrainSecs(Math.floor((Date.now() - start) / 1000))
    }, 1000)

    try {
      const r = await postJSON(`/api/cnn/train?epochs=${epochs}`)
      const d = await r.json()
      if (d.error) {
        setTrainResult(`Error: ${d.error}`)
      } else {
        const elapsed = Math.floor((Date.now() - start) / 1000)
        setTrainResult(
          `Done — ${d.samples} samples | ${d.epochs} epochs | ${elapsed}s | loss ${d.initial_loss?.toFixed(4)} → ${d.final_loss?.toFixed(4)}`
        )
      }
    } catch {
      setTrainResult('Training failed')
    } finally {
      clearInterval(ticker)
      setTraining(false)
    }
  }

  // Filter to CNN signals only
  const cnnSignals: ParsedSignal[] = signals
    .filter(s => s.signal_type?.startsWith('CNN'))
    .map(s => ({ raw: s, ...parseReasoning(s.reasoning) }))

  // Filter to CNN-strategy orders
  const cnnOrders = orders.filter(o =>
    o.strategy?.startsWith('CNN') || o.strategy?.startsWith('DRY')
  )

  const buySigs  = cnnSignals.filter(s => s.raw.side === 'BUY').length
  const sellSigs = cnnSignals.filter(s => s.raw.side === 'SELL').length

  return (
    <div className="space-y-6">

      {/* ── Stat Cards ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard
          label="Simulated Balance"
          value={status?.dry_run_balance != null
            ? `$${status.dry_run_balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
            : '—'}
          sub={status?.dry_run ? 'DRY-RUN' : 'LIVE'}
          color={status?.dry_run_balance != null && status.dry_run_balance >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <StatCard
          label="CNN Signals (session)"
          value={status?.signals_total ?? cnnSignals.length}
          sub={`${status?.signals_buy ?? buySigs} buy · ${status?.signals_sell ?? sellSigs} sell`}
        />
        <StatCard
          label="Trades Executed"
          value={status?.signals_executed ?? cnnOrders.length}
          sub={`${status?.scan_count ?? 0} scans run`}
        />
        <StatCard
          label="Model"
          value={status?.model_loaded ? 'Loaded' : 'Untrained'}
          sub={status?.torch_available ? 'PyTorch ✓' : 'Linear fallback'}
          color={status?.model_loaded ? 'text-green-400' : 'text-amber-400'}
        />
      </div>

      {/* ── Timing row ── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <StatCard label="Last Scan"     value={timeAgo(status?.last_scan_at ?? null)} sub="every 15 min" />
        <StatCard label="Next Scan"     value={timeUntil(status?.next_scan_at ?? null)} />
        <StatCard label="Last Trained"  value={timeAgo((status as any)?.last_trained_at ?? null)}
          sub={`auto every ~1 hr · ${(status as any)?.train_count ?? 0} runs`} />
        <StatCard label="Trading"       value={status?.is_trading ? 'Active' : 'Paused'}
          color={status?.is_trading ? 'text-green-400' : 'text-gray-500'} />
      </div>

      {/* ── Drawdown Circuit Breaker ── */}
      {status?.drawdown && (
        <div className={`card p-4 ${status.drawdown.halted ? 'border border-red-700 bg-red-900/10' : ''}`}>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-white">
              Drawdown Circuit Breaker
              {status.drawdown.halted
                ? <span className="ml-2 text-xs font-bold text-red-400 bg-red-900/30 px-2 py-0.5 rounded border border-red-700">HALTED</span>
                : <span className="ml-2 text-xs font-normal text-green-400 bg-green-900/20 px-2 py-0.5 rounded border border-green-800">OK</span>
              }
            </h3>
          </div>

          {status.drawdown.halted && (
            <p className="text-xs text-red-300 mb-3 bg-red-900/20 px-3 py-2 rounded border border-red-800">
              {status.drawdown.halt_reason}
            </p>
          )}

          <div className="grid grid-cols-2 gap-4">
            {/* Daily */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-400">Daily window</span>
                <span className="text-gray-500">
                  limit {(status.drawdown.daily_limit * 100).toFixed(0)}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-1.5 mb-1">
                <div
                  className="h-1.5 rounded-full bg-blue-500"
                  style={{ width: `${Math.round(status.drawdown.day_elapsed_pct * 100)}%` }}
                />
              </div>
              <div className="text-xs text-gray-500">
                {Math.round(status.drawdown.day_elapsed_pct * 24)}h elapsed · resets in {24 - Math.round(status.drawdown.day_elapsed_pct * 24)}h
              </div>
            </div>

            {/* Weekly */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-400">Weekly window</span>
                <span className="text-gray-500">
                  limit {(status.drawdown.weekly_limit * 100).toFixed(0)}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-1.5 mb-1">
                <div
                  className="h-1.5 rounded-full bg-purple-500"
                  style={{ width: `${Math.round(status.drawdown.week_elapsed_pct * 100)}%` }}
                />
              </div>
              <div className="text-xs text-gray-500">
                {Math.round(status.drawdown.week_elapsed_pct * 7 * 24)}h elapsed · resets in {Math.round((1 - status.drawdown.week_elapsed_pct) * 7 * 24)}h
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Controls ── */}
      <div className="card p-4 flex flex-wrap items-center gap-3">
        <button
          onClick={handleScan}
          disabled={scanning}
          className="btn-primary text-sm px-4 py-2 disabled:opacity-50"
        >
          {scanning ? '⏳ Scanning…' : '🔍 Scan Now'}
        </button>

        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-400">Train epochs:</label>
          <input
            type="number"
            min={1} max={200}
            value={epochs}
            disabled={training}
            onChange={e => setEpochs(Number(e.target.value))}
            className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-white text-center disabled:opacity-50"
          />
          <button
            onClick={handleTrain}
            disabled={training}
            className="btn-secondary text-sm px-4 py-2 disabled:opacity-50"
          >
            {training ? `⏳ ${trainSecs}s elapsed…` : '🧠 Train Model'}
          </button>
        </div>

        {(statusMsg || trainResult) && (
          <span className={`text-xs px-3 py-1.5 rounded border ${
            trainResult
              ? 'text-green-300 bg-green-900/30 border-green-800'
              : 'text-blue-300 bg-blue-900/30 border-blue-800 animate-pulse'
          }`}>
            {statusMsg || trainResult}
          </span>
        )}
      </div>

      {/* ── Main grid: Signals + Orders ── */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">

        {/* CNN Signals */}
        <div>
          <h2 className="text-base font-bold text-white mb-3">
            CNN Signals
            <span className="text-sm text-gray-500 font-normal ml-2">({cnnSignals.length})</span>
          </h2>
          {cnnSignals.length === 0 ? (
            <div className="card text-center py-10">
              <p className="text-gray-500 text-sm">No CNN signals yet — click Scan Now or wait for auto-scan</p>
            </div>
          ) : (
            <div className="space-y-3 max-h-[70vh] overflow-y-auto pr-1">
              {cnnSignals.map(({ raw: s, cnnProb, llmProb, modelProb, rsi, macdHist, bbPos, obImb }) => (
                <div key={s.id} className="card p-3">
                  {/* Header */}
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                        s.side === 'BUY'
                          ? 'bg-green-900/50 text-green-400 border border-green-800'
                          : 'bg-red-900/50 text-red-400 border border-red-800'
                      }`}>{s.side}</span>
                      <span className="font-bold text-white text-sm">{s.product_id}</span>
                      {s.acted === 1 && (
                        <span className="text-xs text-blue-400 bg-blue-900/30 px-1.5 py-0.5 rounded border border-blue-800">
                          Executed
                        </span>
                      )}
                    </div>
                    <div className="text-right">
                      <div className="font-mono text-white text-sm">
                        ${s.price >= 1000
                          ? s.price.toLocaleString('en-US', { maximumFractionDigits: 2 })
                          : s.price.toFixed(4)}
                      </div>
                      <div className="text-xs text-gray-500">
                        {new Date(s.created_at).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>

                  {/* Probability bars */}
                  <div className="grid grid-cols-3 gap-2 mb-2">
                    <div>
                      <div className="text-xs text-gray-500 mb-0.5">CNN</div>
                      <ProbBar value={cnnProb} color={s.side === 'BUY' ? 'bg-blue-500' : 'bg-orange-500'} />
                    </div>
                    <div>
                      <div className="text-xs text-gray-500 mb-0.5">LLM</div>
                      <ProbBar value={llmProb} color={s.side === 'BUY' ? 'bg-purple-500' : 'bg-pink-500'} />
                    </div>
                    <div>
                      <div className="text-xs text-gray-500 mb-0.5">Blended</div>
                      <ProbBar value={modelProb} color={s.side === 'BUY' ? 'bg-green-500' : 'bg-red-500'} />
                    </div>
                  </div>

                  {/* Indicator pills */}
                  <div className="flex flex-wrap gap-2 text-xs">
                    {rsi != null && (
                      <span className={`px-2 py-0.5 rounded border ${
                        rsi < 30 ? 'text-green-300 border-green-800 bg-green-900/30'
                        : rsi > 70 ? 'text-red-300 border-red-800 bg-red-900/30'
                        : 'text-gray-400 border-gray-700 bg-gray-800'
                      }`}>RSI {rsi.toFixed(1)}</span>
                    )}
                    {macdHist != null && (
                      <span className={`px-2 py-0.5 rounded border ${
                        macdHist >= 0 ? 'text-green-300 border-green-800 bg-green-900/30'
                        : 'text-red-300 border-red-800 bg-red-900/30'
                      }`}>MACD {macdHist >= 0 ? '+' : ''}{macdHist.toFixed(5)}</span>
                    )}
                    {bbPos != null && (
                      <span className={`px-2 py-0.5 rounded border ${
                        bbPos < 0.2 ? 'text-green-300 border-green-800 bg-green-900/30'
                        : bbPos > 0.8 ? 'text-red-300 border-red-800 bg-red-900/30'
                        : 'text-gray-400 border-gray-700 bg-gray-800'
                      }`}>BB {(bbPos * 100).toFixed(0)}%</span>
                    )}
                    {obImb != null && (
                      <span className={`px-2 py-0.5 rounded border ${
                        obImb > 0.1 ? 'text-green-300 border-green-800 bg-green-900/30'
                        : obImb < -0.1 ? 'text-red-300 border-red-800 bg-red-900/30'
                        : 'text-gray-400 border-gray-700 bg-gray-800'
                      }`}>OB {obImb >= 0 ? '+' : ''}{obImb.toFixed(2)}</span>
                    )}
                    <span className="px-2 py-0.5 rounded border text-gray-400 border-gray-700 bg-gray-800">
                      strength {(s.strength * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* CNN Orders/Trades */}
        <div>
          <h2 className="text-base font-bold text-white mb-3">
            Dry-Run Trades
            <span className="text-sm text-gray-500 font-normal ml-2">({cnnOrders.length})</span>
          </h2>
          {cnnOrders.length === 0 ? (
            <div className="card text-center py-10">
              <p className="text-gray-500 text-sm">No trades yet — enable trading and run a scan</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-[70vh] overflow-y-auto pr-1">
              {cnnOrders.map(o => (
                <div key={o.order_id} className="card p-3 flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className={`text-xs font-bold px-2 py-0.5 rounded flex-shrink-0 ${
                      o.side === 'BUY'
                        ? 'bg-green-900/50 text-green-400 border border-green-800'
                        : 'bg-red-900/50 text-red-400 border border-red-800'
                    }`}>{o.side}</span>
                    <span className="font-bold text-white text-sm truncate">{o.product_id}</span>
                    <span className="text-xs text-gray-500 flex-shrink-0">{o.order_type}</span>
                  </div>

                  <div className="flex items-center gap-4 flex-shrink-0 text-right">
                    <div>
                      <div className="text-xs text-gray-500">Size</div>
                      <div className="text-sm font-mono text-white">
                        ${(o.quote_size ?? 0).toFixed(2)}
                      </div>
                    </div>
                    {o.price && (
                      <div>
                        <div className="text-xs text-gray-500">Price</div>
                        <div className="text-sm font-mono text-gray-300">
                          ${o.price >= 1000
                            ? o.price.toLocaleString('en-US', { maximumFractionDigits: 2 })
                            : o.price.toFixed(4)}
                        </div>
                      </div>
                    )}
                    <div>
                      <div className="text-xs text-gray-500">Status</div>
                      <div className={`text-xs font-medium ${
                        o.status === 'dry_run' ? 'text-amber-400'
                        : o.status === 'live'  ? 'text-green-400'
                        : o.status === 'canceled' ? 'text-gray-500'
                        : 'text-gray-400'
                      }`}>{o.status}</div>
                    </div>
                    <div className="text-xs text-gray-600">
                      {new Date(o.created_at).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Confidence Table ── */}
      <div className="card overflow-hidden">
        {/* Header toggle */}
        <button
          onClick={() => { setShowScans(v => !v); }}
          className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-800/50 transition-colors"
        >
          <div className="flex items-center gap-3">
            <span className="text-sm font-semibold text-white">CNN Confidence Ratings</span>
            <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
              {scans.length} scanned
            </span>
            {scans.filter(s => s.signal_gen === 1).length > 0 && (
              <span className="text-xs text-green-400 bg-green-900/30 border border-green-800 px-2 py-0.5 rounded">
                {scans.filter(s => s.signal_gen === 1).length} signals
              </span>
            )}
          </div>
          <span className="text-gray-500 text-sm">{showScans ? '▲' : '▼'}</span>
        </button>

        {showScans && (
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

              {/* Side filter */}
              <div className="flex rounded border border-gray-700 overflow-hidden text-xs">
                {(['ALL','BUY','SELL','HOLD'] as const).map(f => (
                  <button key={f}
                    onClick={() => setSideFilter(f)}
                    className={`px-2.5 py-1.5 ${sideFilter === f ? 'bg-blue-700 text-white' : 'text-gray-400 hover:bg-gray-700'}`}
                  >{f}</button>
                ))}
              </div>

              {/* Signal filter */}
              <div className="flex rounded border border-gray-700 overflow-hidden text-xs">
                {([['ALL','All'],['SIG','Signal'],['NOSIG','No Signal']] as [string,string][]).map(([v,l]) => (
                  <button key={v}
                    onClick={() => setSigFilter(v as 'ALL'|'SIG'|'NOSIG')}
                    className={`px-2.5 py-1.5 ${sigFilter === v ? 'bg-blue-700 text-white' : 'text-gray-400 hover:bg-gray-700'}`}
                  >{l}</button>
                ))}
              </div>

              <button onClick={fetchScans} className="text-xs text-gray-400 hover:text-white px-2 py-1.5 border border-gray-700 rounded hover:border-gray-500">
                ↺ Refresh
              </button>

              <span className="text-xs text-gray-600 ml-auto">{filteredScans.length} rows</span>
            </div>

            {/* Table */}
            <div className="overflow-x-auto max-h-[60vh] overflow-y-auto">
              <table className="w-full text-xs">
                <thead className="sticky top-0 bg-gray-900 z-10">
                  <tr className="text-gray-500 border-b border-gray-800">
                    {([
                      ['product_id','Symbol'],
                      ['model_prob','Blended'],
                      ['cnn_prob',  'CNN'],
                      ['llm_prob',  'LLM'],
                      ['strength',  'Strength'],
                      ['vwap_dist', 'VWAP Δ'],
                      ['velocity',  'Vel 5m'],
                      ['adx',       'ADX'],
                      ['rsi',       'RSI'],
                      ['mfi',       'MFI'],
                      ['stoch_k',   'StochK'],
                      ['scanned_at','Time'],
                    ] as [SortKey, string][]).map(([k, label]) => (
                      <th key={k}
                        onClick={() => { if (sortKey === k) setSortAsc(v => !v); else { setSortKey(k); setSortAsc(false) }}}
                        className="px-3 py-2 text-left cursor-pointer hover:text-white select-none whitespace-nowrap"
                      >
                        {label} {sortKey === k ? (sortAsc ? '↑' : '↓') : ''}
                      </th>
                    ))}
                    <th className="px-3 py-2 text-left text-purple-400">Tech</th>
                    <th className="px-3 py-2 text-left text-blue-400">Mom</th>
                    <th className="px-3 py-2 text-left">Regime / Signal</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredScans.length === 0 ? (
                    <tr><td colSpan={15} className="text-center py-8 text-gray-600">
                      No scans yet — run a scan first
                    </td></tr>
                  ) : filteredScans.map(s => {
                    const pct   = Math.round(s.model_prob * 100)
                    const isBuy = s.side === 'BUY'
                    const isSell= s.side === 'SELL'
                    return (
                      <tr key={s.id} className={`border-b border-gray-800/50 hover:bg-gray-800/30 ${s.signal_gen ? 'bg-blue-900/10' : ''}`}>
                        {/* Symbol */}
                        <td className="px-3 py-2 font-mono font-bold text-white whitespace-nowrap">
                          {s.product_id.replace('-USD','')}
                          {s.signal_gen === 1 && <span className="ml-1 text-blue-400">★</span>}
                        </td>

                        {/* Blended — plain number */}
                        <td className={`px-3 py-2 font-mono ${isBuy ? 'text-green-400' : isSell ? 'text-red-400' : 'text-gray-400'}`}>
                          {pct}%
                        </td>

                        {/* CNN */}
                        <td className="px-3 py-2 font-mono text-gray-300">
                          {s.cnn_prob != null ? `${Math.round(s.cnn_prob * 100)}%` : '—'}
                        </td>

                        {/* LLM */}
                        <td className="px-3 py-2 font-mono text-gray-300">
                          {s.llm_prob != null ? `${Math.round(s.llm_prob * 100)}%` : '—'}
                        </td>

                        {/* Strength — bar (≥60% fires a signal) */}
                        <td className="px-3 py-2">
                          <div className="flex items-center gap-1.5">
                            <div className="w-16 bg-gray-700 rounded-full h-1.5 flex-shrink-0">
                              <div
                                className={`h-1.5 rounded-full ${
                                  s.strength >= 0.6
                                    ? (isBuy ? 'bg-green-500' : 'bg-red-500')
                                    : s.strength > 0
                                      ? 'bg-amber-500'
                                      : 'bg-gray-600'
                                }`}
                                style={{ width: `${Math.min(s.strength * 100, 100)}%` }}
                              />
                            </div>
                            <span className={`font-mono text-xs ${
                              s.strength >= 0.6
                                ? (isBuy ? 'text-green-400' : 'text-red-400')
                                : s.strength > 0
                                  ? 'text-amber-400'
                                  : 'text-gray-600'
                            }`}>{Math.round(s.strength * 100)}%</span>
                          </div>
                        </td>

                        {/* VWAP distance */}
                        <td className={`px-3 py-2 font-mono ${
                          s.vwap_dist == null ? 'text-gray-600'
                          : s.vwap_dist > 0.02  ? 'text-red-400'    // far above VWAP — overbought
                          : s.vwap_dist < -0.02 ? 'text-green-400'  // far below VWAP — oversold
                          : 'text-gray-400'
                        }`}>
                          {s.vwap_dist != null
                            ? `${s.vwap_dist >= 0 ? '+' : ''}${(s.vwap_dist * 100).toFixed(1)}%`
                            : '—'}
                        </td>

                        {/* 5-min Velocity */}
                        <td className={`px-3 py-2 font-mono ${
                          s.velocity == null     ? 'text-gray-600'
                          : s.velocity >  0.3   ? 'text-red-400'    // rapid upward move
                          : s.velocity < -0.3   ? 'text-green-400'  // rapid downward (possible reversal)
                          : 'text-gray-400'
                        }`}>
                          {s.velocity != null
                            ? `${s.velocity >= 0 ? '+' : ''}${(s.velocity * 100).toFixed(0)}%`
                            : '—'}
                        </td>

                        {/* ADX */}
                        <td className={`px-3 py-2 font-mono ${(s.adx ?? 0) >= 25 ? 'text-amber-400' : 'text-gray-500'}`}>
                          {s.adx?.toFixed(1) ?? '—'}
                        </td>

                        {/* RSI */}
                        <td className={`px-3 py-2 font-mono ${
                          (s.rsi ?? 50) < 30 ? 'text-green-400' :
                          (s.rsi ?? 50) > 70 ? 'text-red-400' : 'text-gray-400'
                        }`}>
                          {s.rsi?.toFixed(1) ?? '—'}
                        </td>

                        {/* MFI */}
                        <td className={`px-3 py-2 font-mono ${
                          (s.mfi ?? 50) < 20 ? 'text-green-400' :
                          (s.mfi ?? 50) > 80 ? 'text-red-400' : 'text-gray-400'
                        }`}>
                          {s.mfi?.toFixed(1) ?? '—'}
                        </td>

                        {/* Stoch K */}
                        <td className={`px-3 py-2 font-mono ${
                          (s.stoch_k ?? 50) < 20 ? 'text-green-400' :
                          (s.stoch_k ?? 50) > 80 ? 'text-red-400' : 'text-gray-400'
                        }`}>
                          {s.stoch_k?.toFixed(1) ?? '—'}
                        </td>

                        {/* Time */}
                        <td className="px-3 py-2 text-gray-600 whitespace-nowrap">
                          {new Date(s.scanned_at).toLocaleTimeString()}
                        </td>

                        {/* Tech agent vote */}
                        {(() => {
                          const ag = agentByProduct.get(s.product_id)?.tech ?? null
                          if (!ag) return <td className="px-3 py-2 text-gray-700 font-mono text-xs">no data</td>
                          const isBuy  = ag.side === 'BUY'
                          const isSell = ag.side === 'SELL'
                          const isHold = ag.side === 'HOLD'
                          return (
                            <td className="px-3 py-2">
                              <div className="flex flex-col gap-0.5">
                                <span className={`px-1.5 py-0.5 rounded text-xs font-bold w-fit ${
                                  isBuy  ? 'bg-green-900/50 text-green-400 border border-green-800' :
                                  isSell ? 'bg-red-900/50   text-red-400   border border-red-800'   :
                                           'bg-gray-800     text-gray-500  border border-gray-700'
                                }`}>{ag.side}</span>
                                <span className={`font-mono text-xs ${isHold ? 'text-gray-600' : 'text-purple-400'}`}>
                                  {Math.round(ag.confidence * 100)}%
                                  {ag.score != null ? ` s=${ag.score.toFixed(2)}` : ''}
                                </span>
                                {ag.pnl != null && (
                                  <span className={`font-mono text-xs ${ag.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    {ag.pnl >= 0 ? '+' : ''}${ag.pnl.toFixed(2)}
                                  </span>
                                )}
                              </div>
                            </td>
                          )
                        })()}

                        {/* Momentum agent vote */}
                        {(() => {
                          const ag = agentByProduct.get(s.product_id)?.mom ?? null
                          if (!ag) return <td className="px-3 py-2 text-gray-700 font-mono text-xs">no data</td>
                          const isBuy  = ag.side === 'BUY'
                          const isSell = ag.side === 'SELL'
                          const isHold = ag.side === 'HOLD'
                          return (
                            <td className="px-3 py-2">
                              <div className="flex flex-col gap-0.5">
                                <span className={`px-1.5 py-0.5 rounded text-xs font-bold w-fit ${
                                  isBuy  ? 'bg-green-900/50 text-green-400 border border-green-800' :
                                  isSell ? 'bg-red-900/50   text-red-400   border border-red-800'   :
                                           'bg-gray-800     text-gray-500  border border-gray-700'
                                }`}>{ag.side}</span>
                                <span className={`font-mono text-xs ${isHold ? 'text-gray-600' : 'text-blue-400'}`}>
                                  {Math.round(ag.confidence * 100)}%
                                  {ag.score != null ? ` s=${ag.score.toFixed(2)}` : ''}
                                </span>
                                {ag.pnl != null && (
                                  <span className={`font-mono text-xs ${ag.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    {ag.pnl >= 0 ? '+' : ''}${ag.pnl.toFixed(2)}
                                  </span>
                                )}
                              </div>
                            </td>
                          )
                        })()}

                        {/* Side + signal badge */}
                        <td className="px-3 py-2">
                          <span className={`px-1.5 py-0.5 rounded text-xs font-bold ${
                            isBuy  ? 'bg-green-900/50 text-green-400 border border-green-800' :
                            isSell ? 'bg-red-900/50 text-red-400 border border-red-800' :
                                     'bg-gray-800 text-gray-500 border border-gray-700'
                          }`}>{s.side}</span>
                          {s.regime && (
                            <span className={`ml-1 px-1.5 py-0.5 rounded text-xs ${
                              s.regime === 'TRENDING'
                                ? 'text-amber-400 bg-amber-900/20 border border-amber-800'
                                : 'text-blue-400 bg-blue-900/20 border border-blue-800'
                            }`}>{s.regime === 'TRENDING' ? 'T' : 'R'}</span>
                          )}
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
