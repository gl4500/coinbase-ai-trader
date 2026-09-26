import React, { useEffect, useMemo, useState } from 'react'

// ── Types (mirrors GET /api/diagnostics response) ──────────────────────────────

type Cal = { bucket: number; n: number; win_rate: number; avg_ret: number }

type ExitRow = { trigger: string; n: number; sum_pnl: number; avg_pct: number; win_rate: number }

type AssetRow = { product_id: string; n: number; sum_pnl: number; win_rate: number }

type RegimeRow = { regime: string; n: number; sum_pnl: number }

type Funnel = { scans: number; buy_signals: number; executed: number; matured: number }

interface Diag {
  window: string
  signal_edge: { n: number; precision: number; e_return: number; calibration: Cal[] }
  exit_attribution: { by_trigger: ExitRow[]; scan_sell_share: number }
  regime_and_asset: { by_asset: AssetRow[]; by_regime: RegimeRow[] }
  signal_funnel: Funnel
}

type Window = '30d' | '90d' | 'all'
const WINDOWS: Window[] = ['30d', '90d', 'all']

// ── App ──────────────────────────────────────────────────────────────────────

export default function DiagnosticsDashboard() {
  const [window_, setWindow] = useState<Window>('30d')
  const [data, setData] = useState<Diag | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    let cancelled = false
    setErr(null)
    ;(async () => {
      try {
        const r = await fetch(`/api/diagnostics?window=${window_}`)
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const j = await r.json()
        if (!cancelled) { setData(j); setErr(null) }
      } catch (e) {
        if (!cancelled) setErr((e as Error).message)
      }
    })()
    return () => { cancelled = true }
  }, [window_, refreshKey])

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-white">Diagnostics</h2>
        <div className="flex items-center gap-2">
          <div className="flex gap-1">
            {WINDOWS.map(w => (
              <button
                key={w}
                onClick={() => setWindow(w)}
                className={`text-xs px-3 py-1 rounded border transition-colors ${
                  window_ === w
                    ? 'bg-indigo-700 border-indigo-500 text-white'
                    : 'bg-gray-800 border-gray-700 text-gray-400 hover:text-gray-200'
                }`}
              >
                {w}
              </button>
            ))}
          </div>
          <button
            className="text-xs text-gray-500 hover:text-gray-300"
            onClick={() => setRefreshKey(k => k + 1)}
          >
            ↻ refresh
          </button>
        </div>
      </div>

      {err && (
        <div className="card text-center py-12">
          <p className="text-red-400 text-sm">Diagnostics error: {err}</p>
        </div>
      )}

      {!err && !data && (
        <div className="card text-center py-12">
          <p className="text-gray-500 text-sm">Loading diagnostics…</p>
        </div>
      )}

      {!err && data && (
        <div className="space-y-4">
          <CalibrationCard cal={data.signal_edge.calibration}
            n={data.signal_edge.n} precision={data.signal_edge.precision}
            eReturn={data.signal_edge.e_return} />
          <ExitAttributionCard rows={data.exit_attribution.by_trigger}
            share={data.exit_attribution.scan_sell_share} />
          <RegimeAssetCard ra={data.regime_and_asset} />
          <FunnelCard f={data.signal_funnel} />
        </div>
      )}
    </div>
  )
}

// ── Signal edge & calibration (SVG line chart) ──────────────────────────────

function CalibrationCard({ cal, n, precision, eReturn }:
  { cal: Cal[]; n: number; precision: number; eReturn: number }) {
  const W = 360, H = 200, padL = 36, padR = 12, padT = 12, padB = 24
  const innerW = W - padL - padR
  const innerH = H - padT - padB
  const x = (b: number) => padL + b * innerW
  const y = (v: number) => padT + innerH - Math.max(0, Math.min(1, v)) * innerH

  const points = cal.map(c => `${x(c.bucket).toFixed(1)},${y(c.win_rate).toFixed(1)}`).join(' ')
  const yTicks = [0, 0.25, 0.5, 0.75, 1]

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-1">
        <h3 className="card-header mb-0">Signal edge &amp; calibration</h3>
        <span className="text-xs text-gray-500">n={n}</span>
      </div>
      <div className="text-xs text-gray-400 mb-3 font-mono">
        precision <span className="text-gray-200">{(precision * 100).toFixed(1)}%</span>
        <span className="mx-2 text-gray-700">·</span>
        E[return] <span className={eReturn >= 0 ? 'text-green-400' : 'text-red-400'}>
          {eReturn >= 0 ? '+' : ''}{(eReturn * 100).toFixed(2)}%
        </span>
      </div>

      {cal.length === 0 ? (
        <p className="text-xs text-gray-500">No calibration buckets for this window.</p>
      ) : (
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-md h-auto">
          {yTicks.map(v => (
            <g key={v}>
              <line x1={padL} x2={W - padR} y1={y(v)} y2={y(v)}
                stroke="#1f2937" strokeWidth={1} strokeDasharray="2,2" />
              <text x={padL - 6} y={y(v) + 3} textAnchor="end" fontSize="9" fill="#6b7280">
                {(v * 100).toFixed(0)}%
              </text>
            </g>
          ))}
          {/* perfect-calibration reference diagonal */}
          <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="#374151" strokeWidth={1} strokeDasharray="4,3" />
          <polyline fill="none" stroke="#818cf8" strokeWidth={1.5} points={points} />
          {cal.map(c => (
            <circle key={c.bucket} cx={x(c.bucket)} cy={y(c.win_rate)} r={3} fill="#818cf8">
              <title>{`bucket ${c.bucket.toFixed(2)} · n=${c.n} · WR ${(c.win_rate * 100).toFixed(1)}% · avg ret ${(c.avg_ret * 100).toFixed(2)}%`}</title>
            </circle>
          ))}
        </svg>
      )}
    </div>
  )
}

// ── Exit attribution (SVG horizontal bar chart of sum PnL by trigger) ──────

function ExitAttributionCard({ rows, share }: { rows: ExitRow[]; share: number }) {
  const W = 480, rowH = 22, padL = 100, padR = 60, padT = 4
  const maxAbs = useMemo(() => Math.max(1, ...rows.map(r => Math.abs(r.sum_pnl))), [rows])
  const innerW = W - padL - padR
  const H = padT * 2 + rows.length * rowH
  const zeroX = padL + innerW / 2
  const barScale = (v: number) => (v / maxAbs) * (innerW / 2)

  return (
    <div className="card">
      <h3 className="card-header mb-0">
        Exit attribution
        <span className="text-gray-500 font-normal normal-case tracking-normal ml-2">
          SCAN-SELL share {(share * 100).toFixed(0)}%
        </span>
      </h3>

      {rows.length === 0 ? (
        <p className="text-xs text-gray-500">No closed trades in this window.</p>
      ) : (
        <>
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-2xl h-auto mb-2">
            <line x1={zeroX} x2={zeroX} y1={0} y2={H} stroke="#374151" strokeWidth={1} />
            {rows.map((r, i) => {
              const w = barScale(r.sum_pnl)
              const barX = w >= 0 ? zeroX : zeroX + w
              const cy = padT + i * rowH + rowH / 2
              return (
                <g key={r.trigger}>
                  <text x={padL - 8} y={cy + 3} textAnchor="end" fontSize="10" fill="#9ca3af">
                    {r.trigger}
                  </text>
                  <rect x={barX} y={cy - 6} width={Math.abs(w)} height={12} rx={2}
                    fill={r.sum_pnl >= 0 ? '#4ade80' : '#f87171'} />
                  <text x={w >= 0 ? zeroX + Math.abs(w) + 6 : zeroX - Math.abs(w) - 6} y={cy + 3}
                    textAnchor={w >= 0 ? 'start' : 'end'} fontSize="10" fill="#6b7280">
                    {r.sum_pnl >= 0 ? '+' : ''}${r.sum_pnl.toFixed(2)}
                  </text>
                </g>
              )
            })}
          </svg>
          <table className="text-xs w-full max-w-2xl font-mono">
            <thead>
              <tr className="text-gray-500 text-left">
                <th className="font-normal pb-1">trigger</th>
                <th className="font-normal pb-1">n</th>
                <th className="font-normal pb-1">avg %</th>
                <th className="font-normal pb-1">win rate</th>
              </tr>
            </thead>
            <tbody>
              {rows.map(r => (
                <tr key={r.trigger} className="text-gray-300">
                  <td className="py-0.5">{r.trigger}</td>
                  <td className="py-0.5">{r.n}</td>
                  <td className={`py-0.5 ${r.avg_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {r.avg_pct >= 0 ? '+' : ''}{(r.avg_pct * 100).toFixed(2)}%
                  </td>
                  <td className="py-0.5">{(r.win_rate * 100).toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}

// ── Regime + per-asset breakdown (SVG bar chart for regimes, ranked list for assets) ──

function RegimeAssetCard({ ra }: { ra: { by_asset: AssetRow[]; by_regime: RegimeRow[] } }) {
  const W = 260, rowH = 20, padL = 90, padR = 16, padT = 4
  const maxAbs = useMemo(
    () => Math.max(1, ...ra.by_regime.map(r => Math.abs(r.sum_pnl))),
    [ra.by_regime]
  )
  const innerW = W - padL - padR
  const H = padT * 2 + ra.by_regime.length * rowH
  const zeroX = padL + innerW / 2
  const barScale = (v: number) => (v / maxAbs) * (innerW / 2)

  const worstFirst = [...ra.by_asset].sort((a, b) => a.sum_pnl - b.sum_pnl).slice(0, 15)

  return (
    <div className="card">
      <h3 className="card-header mb-3">Regime &amp; asset breakdown</h3>
      <div className="flex gap-8 flex-wrap">
        <div>
          <div className="text-xs text-gray-500 mb-1">By regime</div>
          {ra.by_regime.length === 0 ? (
            <p className="text-xs text-gray-500">No labeled trades.</p>
          ) : (
            <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxWidth: W }}>
              <line x1={zeroX} x2={zeroX} y1={0} y2={H} stroke="#374151" strokeWidth={1} />
              {ra.by_regime.map((r, i) => {
                const w = barScale(r.sum_pnl)
                const barX = w >= 0 ? zeroX : zeroX + w
                const cy = padT + i * rowH + rowH / 2
                return (
                  <g key={r.regime}>
                    <text x={padL - 8} y={cy + 3} textAnchor="end" fontSize="9" fill="#9ca3af">
                      {r.regime}
                    </text>
                    <rect x={barX} y={cy - 5} width={Math.abs(w)} height={10} rx={2}
                      fill={r.sum_pnl >= 0 ? '#4ade80' : '#f87171'} />
                    <text x={w >= 0 ? zeroX + Math.abs(w) + 4 : zeroX - Math.abs(w) - 4} y={cy + 3}
                      textAnchor={w >= 0 ? 'start' : 'end'} fontSize="9" fill="#6b7280">
                      ${r.sum_pnl.toFixed(0)}
                    </text>
                  </g>
                )
              })}
            </svg>
          )}
        </div>

        <div className="min-w-[220px]">
          <div className="text-xs text-gray-500 mb-1">By asset (worst first)</div>
          {worstFirst.length === 0 ? (
            <p className="text-xs text-gray-500">No closed trades in this window.</p>
          ) : (
            <div className="space-y-0.5">
              {worstFirst.map(a => (
                <div key={a.product_id} className="flex items-center justify-between text-xs font-mono gap-3">
                  <span className="text-gray-300">{a.product_id}</span>
                  <span className="text-gray-600">n={a.n}</span>
                  <span className={a.sum_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                    {a.sum_pnl >= 0 ? '+' : ''}${a.sum_pnl.toFixed(2)}
                  </span>
                  <span className="text-gray-500">{(a.win_rate * 100).toFixed(0)}% WR</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Signal funnel (SVG horizontal funnel bars) ──────────────────────────────

function FunnelCard({ f }: { f: Funnel }) {
  const stages: [string, number][] = [
    ['scans', f.scans],
    ['BUY signals', f.buy_signals],
    ['executed', f.executed],
    ['matured', f.matured],
  ]
  const W = 420, rowH = 26, padL = 96, padR = 56, padT = 4
  const H = padT * 2 + stages.length * rowH
  const innerW = W - padL - padR
  const maxV = Math.max(1, ...stages.map(([, v]) => v))
  const barW = (v: number) => (v / maxV) * innerW

  return (
    <div className="card">
      <h3 className="card-header mb-2">Signal funnel</h3>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-lg h-auto">
        {stages.map(([label, v], i) => {
          const w = barW(v)
          const cy = padT + i * rowH + rowH / 2
          const pctOfScans = f.scans > 0 ? (v / f.scans) * 100 : 0
          return (
            <g key={label}>
              <text x={padL - 8} y={cy + 3} textAnchor="end" fontSize="10" fill="#9ca3af">
                {label}
              </text>
              <rect x={padL} y={cy - 8} width={w} height={16} rx={2} fill="#818cf8" opacity={0.85} />
              <text x={padL + w + 6} y={cy + 3} fontSize="10" fill="#6b7280">
                {v}{i > 0 && ` (${pctOfScans.toFixed(1)}%)`}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
