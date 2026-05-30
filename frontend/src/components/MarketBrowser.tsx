import React, { useState } from 'react'
import { Product } from '../App'

interface Props {
  products:        Product[]
  onSelectProduct: (p: Product) => void
}

type SortKey = 'product_id' | 'price' | 'price_pct_change_24h' | 'volume_24h' | 'high_24h' | 'low_24h'

const COLUMNS: { key: SortKey; label: string; align: 'left' | 'right' }[] = [
  { key: 'product_id',           label: 'Market',   align: 'left'  },
  { key: 'price',                label: 'Price',    align: 'right' },
  { key: 'price_pct_change_24h', label: '24h %',    align: 'right' },
  { key: 'volume_24h',           label: '24h Vol',  align: 'right' },
  { key: 'high_24h',             label: '24h High', align: 'right' },
  { key: 'low_24h',              label: '24h Low',  align: 'right' },
]

export default function MarketBrowser({ products, onSelectProduct }: Props) {
  const [search,  setSearch]  = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('volume_24h')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  const fmtPrice = (n: number) =>
    n >= 1000 ? n.toLocaleString('en-US', { maximumFractionDigits: 2 }) : n.toFixed(4)

  const fmtVol = (v: number) =>
    v >= 1e9 ? `$${(v / 1e9).toFixed(2)}B`
    : v >= 1e6 ? `$${(v / 1e6).toFixed(2)}M`
    : v >= 1e3 ? `$${(v / 1e3).toFixed(1)}K`
    : `$${v.toFixed(0)}`

  const filtered = products.filter(p =>
    p.product_id.toLowerCase().includes(search.toLowerCase()) ||
    p.base_currency.toLowerCase().includes(search.toLowerCase()) ||
    (p.display_name || '').toLowerCase().includes(search.toLowerCase())
  )

  const sorted = [...filtered].sort((a, b) => {
    let cmp: number
    if (sortKey === 'product_id') {
      cmp = a.product_id.localeCompare(b.product_id)
    } else {
      cmp = ((a[sortKey] as number) ?? 0) - ((b[sortKey] as number) ?? 0)
    }
    return sortDir === 'asc' ? cmp : -cmp
  })

  const toggleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(key === 'product_id' ? 'asc' : 'desc')
    }
  }

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

      {sorted.length === 0 ? (
        <div className="card text-center py-12">
          <p className="text-gray-500 text-sm">
            {products.length === 0
              ? 'No products loaded — click Scan to fetch data'
              : 'No products match your search'}
          </p>
        </div>
      ) : (
        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800">
                {COLUMNS.map(col => (
                  <th
                    key={col.key}
                    onClick={() => toggleSort(col.key)}
                    className={`px-3 py-2.5 text-xs font-semibold uppercase tracking-wide text-gray-500
                               cursor-pointer select-none hover:text-gray-300 transition-colors
                               ${col.align === 'right' ? 'text-right' : 'text-left'}`}
                  >
                    {col.label}
                    <span className="ml-1 text-blue-400">
                      {sortKey === col.key ? (sortDir === 'asc' ? '▲' : '▼') : ''}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map(p => {
                const pct = p.price_pct_change_24h ?? 0
                return (
                  <tr
                    key={p.product_id}
                    onClick={() => onSelectProduct(p)}
                    className="border-b border-gray-800/60 last:border-0 hover:bg-gray-800/50
                               cursor-pointer transition-colors"
                  >
                    <td className="px-3 py-2.5">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-white">{p.base_currency}</span>
                        <span className="text-xs text-gray-500">/ {p.quote_currency}</span>
                      </div>
                      <div className="text-xs text-gray-500">{p.display_name || p.product_id}</div>
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono text-white">
                      ${fmtPrice(p.price ?? 0)}
                    </td>
                    <td className={`px-3 py-2.5 text-right font-mono font-medium
                                   ${pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {pct >= 0 ? '+' : ''}{pct.toFixed(2)}%
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono text-gray-300">
                      {fmtVol(p.volume_24h ?? 0)}
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono text-green-400">
                      ${fmtPrice(p.high_24h ?? 0)}
                    </td>
                    <td className="px-3 py-2.5 text-right font-mono text-red-400">
                      ${fmtPrice(p.low_24h ?? 0)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
