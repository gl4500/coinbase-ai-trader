import React, { useCallback, useEffect, useState } from 'react'
import { Product } from '../App'

interface BookEntry { price: number; size: number }
interface Book { bids: BookEntry[]; asks: BookEntry[] }

interface Props { product: Product | null }

export default function OrderBook({ product }: Props) {
  const [book, setBook] = useState<Book>({ bids: [], asks: [] })

  const fetchBook = useCallback(async () => {
    if (!product) return
    try {
      const resp = await fetch(`/api/orderbook/${product.product_id}`)
      if (resp.ok) {
        const data = await resp.json()
        setBook({
          bids: (data.bids || []).slice(0, 15),
          asks: (data.asks || []).slice(0, 15),
        })
      }
    } catch {}
  }, [product])

  useEffect(() => {
    fetchBook()
    const id = setInterval(fetchBook, 3000)
    return () => clearInterval(id)
  }, [fetchBook])

  if (!product) return (
    <div className="card text-center py-12">
      <p className="text-gray-500 text-sm">Select a market to view the order book</p>
    </div>
  )

  const fmtPrice = (n: number) =>
    n >= 1000 ? n.toLocaleString('en-US', { maximumFractionDigits: 2 }) : n.toFixed(4)

  const fmtSize  = (n: number) => n >= 1 ? n.toFixed(4) : n.toFixed(6)

  const maxBidSize = Math.max(...book.bids.map(b => b.size), 1)
  const maxAskSize = Math.max(...book.asks.map(a => a.size), 1)

  const spread = book.asks.length && book.bids.length
    ? book.asks[0].price - book.bids[0].price
    : null

  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <h2 className="text-lg font-bold text-white">{product.product_id} Order Book</h2>
        <span className="font-mono text-xl font-bold text-white">
          ${fmtPrice(product.price ?? 0)}
        </span>
        <span className={`text-sm font-medium ${(product.price_pct_change_24h ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
          {(product.price_pct_change_24h ?? 0) >= 0 ? '+' : ''}
          {(product.price_pct_change_24h ?? 0).toFixed(2)}%
        </span>
        {spread != null && (
          <span className="text-xs text-gray-500">
            Spread: ${spread.toFixed(4)} ({((spread / book.asks[0].price) * 100).toFixed(3)}%)
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Bids */}
        <div className="card">
          <h3 className="text-sm font-semibold text-green-400 mb-3">Bids (Buy Orders)</h3>
          <div className="space-y-0.5">
            <div className="grid grid-cols-3 text-xs text-gray-500 mb-1 px-1">
              <span>Price (USD)</span>
              <span className="text-right">Size ({product.base_currency})</span>
              <span className="text-right">Total</span>
            </div>
            {book.bids.length === 0
              ? <p className="text-xs text-gray-500 text-center py-4">No bids</p>
              : book.bids.map((b, i) => (
                <div key={i} className="relative grid grid-cols-3 text-xs py-0.5 px-1 rounded overflow-hidden">
                  <div
                    className="absolute inset-0 bg-green-900/20 rounded"
                    style={{ width: `${(b.size / maxBidSize) * 100}%` }}
                  />
                  <span className="relative text-green-400 font-mono">${fmtPrice(b.price)}</span>
                  <span className="relative text-right text-gray-300 font-mono">{fmtSize(b.size)}</span>
                  <span className="relative text-right text-gray-500 font-mono">${(b.price * b.size).toFixed(2)}</span>
                </div>
              ))
            }
          </div>
        </div>

        {/* Asks */}
        <div className="card">
          <h3 className="text-sm font-semibold text-red-400 mb-3">Asks (Sell Orders)</h3>
          <div className="space-y-0.5">
            <div className="grid grid-cols-3 text-xs text-gray-500 mb-1 px-1">
              <span>Price (USD)</span>
              <span className="text-right">Size ({product.base_currency})</span>
              <span className="text-right">Total</span>
            </div>
            {book.asks.length === 0
              ? <p className="text-xs text-gray-500 text-center py-4">No asks</p>
              : book.asks.map((a, i) => (
                <div key={i} className="relative grid grid-cols-3 text-xs py-0.5 px-1 rounded overflow-hidden">
                  <div
                    className="absolute inset-0 bg-red-900/20 rounded"
                    style={{ width: `${(a.size / maxAskSize) * 100}%` }}
                  />
                  <span className="relative text-red-400 font-mono">${fmtPrice(a.price)}</span>
                  <span className="relative text-right text-gray-300 font-mono">{fmtSize(a.size)}</span>
                  <span className="relative text-right text-gray-500 font-mono">${(a.price * a.size).toFixed(2)}</span>
                </div>
              ))
            }
          </div>
        </div>
      </div>
    </div>
  )
}
