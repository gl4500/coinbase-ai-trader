import React, { useEffect, useMemo, useState } from 'react'

interface Product {
  product_id: string
  base_currency: string
  quote_currency: string
}

type Interval = '15' | '60' | '240' | 'D'
const INTERVALS: { label: string, value: Interval }[] = [
  { label: '15m', value: '15' },
  { label: '1h',  value: '60' },
  { label: '4h',  value: '240' },
  { label: '1D',  value: 'D' },
]

/** TradingView free "basic studies" — no account required. Each maps to a
 *  STUDY_ID@tv-basicstudies string the widget understands via the `studies`
 *  URL param. Click to toggle on/off; selected studies auto-load into the
 *  chart. The chart's own fx toolbar still lets the user add more ad-hoc. */
type Study = {
  label: string
  id: string        // TradingView study id
  defaultOn?: boolean
}
const STUDIES: Study[] = [
  { label: 'RSI',           id: 'RSI@tv-basicstudies',           defaultOn: true },
  { label: 'MACD',          id: 'MACD@tv-basicstudies',          defaultOn: true },
  { label: 'Bollinger',     id: 'BB@tv-basicstudies',            defaultOn: true },
  { label: 'EMA',           id: 'MAExp@tv-basicstudies' },
  { label: 'SMA',           id: 'MASimple@tv-basicstudies' },
  { label: 'Stoch RSI',     id: 'StochasticRSI@tv-basicstudies' },
  { label: 'ATR',           id: 'ATR@tv-basicstudies' },
  { label: 'ADX',           id: 'ADX@tv-basicstudies' },
  { label: 'VWAP',          id: 'VWAP@tv-basicstudies' },
  { label: 'OBV',           id: 'OBV@tv-basicstudies' },
  { label: 'Ichimoku',      id: 'IchimokuCloud@tv-basicstudies' },
  { label: 'MFI',           id: 'MFI@tv-basicstudies' },
  { label: 'CCI',           id: 'CCI@tv-basicstudies' },
  { label: 'Parabolic SAR', id: 'ParabolicSAR@tv-basicstudies' },
  { label: 'Williams %R',   id: 'WilliamsR@tv-basicstudies' },
  { label: 'ROC',           id: 'ROC@tv-basicstudies' },
]

/** Convert a Coinbase product_id (BTC-USD) to a TradingView symbol.
 *  Coinbase pairs map cleanly to COINBASE:{BASE}{QUOTE}. */
function tradingViewSymbol(pid: string): string {
  const [base, quote] = pid.split('-')
  return `COINBASE:${base}${quote}`
}

export default function PriceChart() {
  const [products, setProducts] = useState<Product[]>([])
  const [pid, setPid] = useState<string>('BTC-USD')
  const [interval, setInterval] = useState<Interval>('60')
  const [err, setErr] = useState<string | null>(null)
  const [selectedStudies, setSelectedStudies] = useState<Set<string>>(
    () => new Set(STUDIES.filter(s => s.defaultOn).map(s => s.id))
  )

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const r = await fetch('/api/products')
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        const j = await r.json()
        if (cancelled) return
        const ps: Product[] = (j.products ?? j ?? [])
          .filter((p: Product) => p.quote_currency === 'USD')
          .sort((a: Product, b: Product) => a.product_id.localeCompare(b.product_id))
        setProducts(ps)
      } catch (e) {
        if (!cancelled) setErr((e as Error).message)
      }
    })()
    return () => { cancelled = true }
  }, [])

  const symbol = useMemo(() => tradingViewSymbol(pid), [pid])

  const iframeUrl = useMemo(() => {
    const studiesJson = JSON.stringify(Array.from(selectedStudies))
    const params = new URLSearchParams({
      symbol,
      interval,
      theme: 'dark',
      style: '1',                    // 1 = candles
      timezone: 'America/New_York',
      hide_top_toolbar: '0',
      hide_legend: '0',
      hide_side_toolbar: '0',
      allow_symbol_change: '1',
      save_image: '0',
      studies: studiesJson,
      locale: 'en',
    })
    return `https://www.tradingview.com/widgetembed/?${params.toString()}`
  }, [symbol, interval, selectedStudies])

  const toggleStudy = (id: string) => {
    setSelectedStudies(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return (
    <div className="space-y-4">
      <div className="card p-4">
        <div className="flex items-center justify-between flex-wrap gap-3 mb-3">
          <div>
            <h2 className="text-lg font-bold text-white">Price Chart</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              TradingView widget · <span className="font-mono text-gray-500">{symbol}</span>
              {' · '}
              <span className="text-gray-500">data: TradingView (Coinbase feed)</span>
            </p>
          </div>
          <div className="flex items-center gap-2">
            <label htmlFor="pid-select" className="text-xs text-gray-400">Coin:</label>
            <select
              id="pid-select"
              value={pid}
              onChange={(e) => setPid(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs font-mono text-gray-100 min-w-[140px]"
            >
              {err && <option>{err}</option>}
              {!err && products.length === 0 && <option>Loading…</option>}
              {products.map(p => (
                <option key={p.product_id} value={p.product_id}>{p.product_id}</option>
              ))}
            </select>
            <div className="flex items-center gap-0.5 ml-2">
              {INTERVALS.map(iv => (
                <button
                  key={iv.value}
                  onClick={() => setInterval(iv.value)}
                  className={`text-xs px-2 py-1 rounded font-mono ${
                    interval === iv.value
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {iv.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Indicator toggle row — TradingView free basic studies, no account needed */}
        <div className="mb-3">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-gray-400 mr-1">Indicators:</span>
            {STUDIES.map(s => {
              const on = selectedStudies.has(s.id)
              return (
                <button
                  key={s.id}
                  onClick={() => toggleStudy(s.id)}
                  className={`text-xs px-2 py-0.5 rounded border transition-colors ${
                    on
                      ? 'bg-blue-700 border-blue-500 text-white'
                      : 'bg-gray-800 border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-600'
                  }`}
                  title={`${on ? 'Hide' : 'Show'} ${s.label}`}
                >
                  {s.label}
                </button>
              )
            })}
            {selectedStudies.size > 0 && (
              <button
                onClick={() => setSelectedStudies(new Set())}
                className="text-xs px-2 py-0.5 rounded border bg-transparent border-gray-700 text-gray-500 hover:text-gray-300 ml-1"
                title="Clear all indicators"
              >
                clear
              </button>
            )}
          </div>
          <p className="text-[10px] text-gray-600 mt-1">
            {selectedStudies.size === 0
              ? 'No indicators loaded — click any above to add. Use the fx button inside the chart for more.'
              : `${selectedStudies.size} indicator${selectedStudies.size === 1 ? '' : 's'} loaded. The fx button in the chart toolbar can add ad-hoc more without a TradingView account.`}
          </p>
        </div>

        <div className="bg-[#131722] rounded overflow-hidden" style={{ height: '600px' }}>
          <iframe
            key={`${symbol}-${interval}-${Array.from(selectedStudies).sort().join(',')}`}
            src={iframeUrl}
            title={`TradingView chart ${symbol}`}
            width="100%"
            height="600"
            frameBorder="0"
            allowTransparency
            scrolling="no"
            allowFullScreen
          />
        </div>

        <p className="text-xs text-gray-600 mt-2">
          Free TradingView embed widget. Data is from TradingView's Coinbase feed, not our
          internal candles. For chart actions linked to your trades, see Performance tab.
        </p>
      </div>
    </div>
  )
}
