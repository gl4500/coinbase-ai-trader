import React, { useState } from 'react'
import { Product } from '../App'

interface Props {
  products:        Product[]
  onSelectProduct: (p: Product) => void
}

export default function MarketBrowser({ products, onSelectProduct }: Props) {
  const [search, setSearch] = useState('')

  const filtered = products.filter(p =>
    p.product_id.toLowerCase().includes(search.toLowerCase()) ||
    p.base_currency.toLowerCase().includes(search.toLowerCase()) ||
    (p.display_name || '').toLowerCase().includes(search.toLowerCase())
  )

  const fmtPrice = (n: number) =>
    n >= 1000 ? n.toLocaleString('en-US', { maximumFractionDigits: 2 }) : n.toFixed(4)

  const fmtVol = (v: number) =>
    v >= 1e9 ? `$${(v / 1e9).toFixed(2)}B`
    : v >= 1e6 ? `$${(v / 1e6).toFixed(2)}M`
    : v >= 1e3 ? `$${(v / 1e3).toFixed(1)}K`
    : `$${v.toFixed(0)}`

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-white">
          Markets <span className="text-sm text-gray-500 font-normal">({filtered.length})</span>
        </h2>
        <input
          type="text"
          placeholder="Search BTC, ETH…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-200 w-48 focus:outline-none focus:border-blue-500"
        />
      </div>

      {filtered.length === 0 ? (
        <div className="card text-center py-12">
          <p className="text-gray-500 text-sm">
            {products.length === 0
              ? 'No products loaded — click Scan to fetch data'
              : 'No products match your search'}
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          {filtered.map(p => {
            const pct = p.price_pct_change_24h ?? 0
            return (
              <div
                key={p.product_id}
                onClick={() => onSelectProduct(p)}
                className="card hover:border-blue-500/50 cursor-pointer transition-all"
              >
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-white text-base">{p.base_currency}</span>
                      <span className="text-xs text-gray-500">/ {p.quote_currency}</span>
                    </div>
                    <div className="text-xs text-gray-500 mt-0.5">{p.display_name || p.product_id}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono font-bold text-white text-sm">
                      ${fmtPrice(p.price ?? 0)}
                    </div>
                    <div className={`text-xs font-medium ${pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {pct >= 0 ? '+' : ''}{pct.toFixed(2)}%
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div>
                    <div className="text-gray-500">24h Vol</div>
                    <div className="text-gray-300">{fmtVol(p.volume_24h ?? 0)}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">24h High</div>
                    <div className="text-green-400">${fmtPrice(p.high_24h ?? 0)}</div>
                  </div>
                  <div>
                    <div className="text-gray-500">24h Low</div>
                    <div className="text-red-400">${fmtPrice(p.low_24h ?? 0)}</div>
                  </div>
                </div>

                <div className="mt-3 flex justify-end">
                  <span className="text-xs text-blue-400">View Order Book →</span>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
