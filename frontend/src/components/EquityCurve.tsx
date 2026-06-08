import React, { useEffect, useMemo, useState } from 'react'

type Point = [string, number]   // [closed_at_iso, cumulative_pnl]

interface CurveResponse {
  v3: Point[]
  v45: Point[]
  days: number
}

export default function EquityCurve({ days = 7 }: { days?: number }) {
  const [data, setData] = useState<CurveResponse | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const r = await fetch(`/api/equity_curve?days=${days}`)
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const j = await r.json()
        if (!cancelled) { setData(j); setErr(null) }
      } catch (e) {
        if (!cancelled) setErr((e as Error).message)
      }
    })()
    return () => { cancelled = true }
  }, [days, refreshKey])

  const chart = useMemo(() => {
    if (!data) return null
    const all = [...data.v3, ...data.v45]
    if (all.length === 0) return null

    const xs = all.map(p => new Date(p[0]).getTime())
    const ys = all.map(p => p[1])
    const xMin = Math.min(...xs)
    const xMax = Math.max(...xs)
    const yMin = Math.min(0, ...ys)
    const yMax = Math.max(0, ...ys)
    const W = 720, H = 220, padL = 50, padR = 12, padT = 12, padB = 28
    const innerW = W - padL - padR
    const innerH = H - padT - padB

    const xScale = (t: number) =>
      padL + ((t - xMin) / Math.max(xMax - xMin, 1)) * innerW
    const yScale = (v: number) =>
      padT + innerH - ((v - yMin) / Math.max(yMax - yMin, 1)) * innerH

    const toPath = (points: Point[]) => {
      if (points.length === 0) return ''
      return points.map((p, i) => {
        const x = xScale(new Date(p[0]).getTime())
        const y = yScale(p[1])
        return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`
      }).join(' ')
    }

    const zeroY = yScale(0)
    const fmt$ = (n: number) => `${n >= 0 ? '+' : ''}$${n.toFixed(2)}`
    const fmtDate = (t: number) => {
      const d = new Date(t)
      return `${d.getMonth() + 1}/${d.getDate()}`
    }

    // y-axis grid lines (5 ticks)
    const yTicks: number[] = []
    for (let i = 0; i <= 4; i++) {
      yTicks.push(yMin + (i / 4) * (yMax - yMin))
    }

    // x-axis ticks (4 evenly spaced)
    const xTicks: number[] = []
    for (let i = 0; i <= 3; i++) {
      xTicks.push(xMin + (i / 3) * (xMax - xMin))
    }

    return {
      W, H, padL, padR, padT, padB,
      pathV3: toPath(data.v3),
      pathV45: toPath(data.v45),
      zeroY, yTicks, xTicks,
      yScale, xScale, fmt$, fmtDate,
      v3Last: data.v3.length ? data.v3[data.v3.length - 1][1] : null,
      v45Last: data.v45.length ? data.v45[data.v45.length - 1][1] : null,
    }
  }, [data])

  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-200">Equity curve — last {days}d</h3>
        <button
          className="text-xs text-gray-500 hover:text-gray-300"
          onClick={() => setRefreshKey(k => k + 1)}
        >
          ↻ refresh
        </button>
      </div>

      {err && <p className="text-xs text-red-400">{err}</p>}

      {!chart && !err && (
        <p className="text-xs text-gray-500">
          {data ? 'No trades closed in this window yet — waiting for first exits.' : 'Loading…'}
        </p>
      )}

      {chart && (
        <>
          <svg viewBox={`0 0 ${chart.W} ${chart.H}`} className="w-full h-auto">
            {/* y-axis grid */}
            {chart.yTicks.map((v, i) => (
              <g key={`y-${i}`}>
                <line x1={chart.padL} x2={chart.W - chart.padR}
                      y1={chart.yScale(v)} y2={chart.yScale(v)}
                      stroke="#1f2937" strokeWidth={1} strokeDasharray="2,2" />
                <text x={chart.padL - 6} y={chart.yScale(v) + 3}
                      textAnchor="end" fontSize="9" fill="#6b7280">
                  {chart.fmt$(v)}
                </text>
              </g>
            ))}
            {/* zero line emphasized */}
            <line x1={chart.padL} x2={chart.W - chart.padR}
                  y1={chart.zeroY} y2={chart.zeroY}
                  stroke="#4b5563" strokeWidth={1} />
            {/* x-axis ticks */}
            {chart.xTicks.map((t, i) => (
              <text key={`x-${i}`} x={chart.xScale(t)}
                    y={chart.H - chart.padB + 14}
                    textAnchor="middle" fontSize="9" fill="#6b7280">
                {chart.fmtDate(t)}
              </text>
            ))}
            {/* v3 line (cyan) */}
            <path d={chart.pathV3} stroke="#06b6d4" strokeWidth={1.5}
                  fill="none" />
            {/* v45 line (amber) */}
            <path d={chart.pathV45} stroke="#f59e0b" strokeWidth={1.5}
                  fill="none" />
          </svg>

          {/* legend + latest values */}
          <div className="flex items-center justify-between mt-2 text-xs font-mono">
            <div className="flex items-center gap-3">
              <span className="flex items-center gap-1">
                <span className="inline-block w-3 h-0.5 bg-cyan-500" />
                <span className="text-gray-400">8001 v3</span>
                {chart.v3Last !== null && (
                  <span className={chart.v3Last >= 0 ? 'text-green-400' : 'text-red-400'}>
                    {chart.fmt$(chart.v3Last)}
                  </span>
                )}
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block w-3 h-0.5 bg-amber-500" />
                <span className="text-gray-400">8002 v45</span>
                {chart.v45Last !== null && (
                  <span className={chart.v45Last >= 0 ? 'text-green-400' : 'text-red-400'}>
                    {chart.fmt$(chart.v45Last)}
                  </span>
                )}
              </span>
            </div>
            {chart.v3Last !== null && chart.v45Last !== null && (
              <span className="text-gray-500">
                Δ (v45 − v3): <span className={(chart.v45Last - chart.v3Last) >= 0 ? 'text-green-400' : 'text-red-400'}>
                  {chart.fmt$(chart.v45Last - chart.v3Last)}
                </span>
              </span>
            )}
          </div>
        </>
      )}
    </div>
  )
}
