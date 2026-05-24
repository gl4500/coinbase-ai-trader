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
      studies: '[]',
      locale: 'en',
    })
    return `https://www.tradingview.com/widgetembed/?${params.toString()}`
  }, [symbol, interval])

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

        <div className="bg-[#131722] rounded overflow-hidden" style={{ height: '600px' }}>
          <iframe
            key={`${symbol}-${interval}`}
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
