import React, { useState } from 'react'
import { Position, PortfolioSummary } from '../App'

interface Props {
  positions: Position[]
  portfolio: PortfolioSummary
}

export default function PositionTracker({ positions, portfolio }: Props) {
  return (
    <div>
      {/* Portfolio summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        {[
          { label: 'Open Positions', value: portfolio.open_positions.toString(), color: 'text-white' },
          { label: 'Total Value',    value: `$${portfolio.total_value.toFixed(2)}`, color: 'text-white' },
          { label: 'Total Cost',     value: `$${portfolio.total_cost.toFixed(2)}`,  color: 'text-gray-400' },
          {
            label: 'Total P&L',
            value: `${portfolio.total_pnl >= 0 ? '+' : ''}$${portfolio.total_pnl.toFixed(2)} (${portfolio.pct_pnl.toFixed(1)}%)`,
            color: portfolio.total_pnl >= 0 ? 'text-green-400' : 'text-red-400',
          },
        ].map(({ label, value, color }) => (
          <div key={label} className="card">
            <div className="text-xs text-gray-500 mb-1">{label}</div>
            <div className={`font-bold text-sm ${color}`}>{value}</div>
          </div>
        ))}
      </div>

      {/* Positions table */}
      {positions.length === 0 ? (
        <div className="card text-center py-12">
          <p className="text-gray-500 text-sm">No open positions</p>
        </div>
      ) : (
        <div className="card overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-gray-500 border-b border-gray-800">
                <th className="text-left py-2 pr-4">Pair</th>
                <th className="text-right py-2 pr-4">Side</th>
                <th className="text-right py-2 pr-4">Size</th>
                <th className="text-right py-2 pr-4">Avg Price</th>
                <th className="text-right py-2 pr-4">Current</th>
                <th className="text-right py-2 pr-4">Value</th>
                <th className="text-right py-2">P&L</th>
              </tr>
            </thead>
            <tbody>
              {positions.map(p => {
                const pnl    = p.cash_pnl ?? 0
                const pctPnl = p.pct_pnl  ?? 0
                const fmtP   = (n: number) => n >= 1000
                  ? n.toLocaleString('en-US', { maximumFractionDigits: 2 })
                  : n.toFixed(4)
                return (
                  <tr key={p.product_id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                    <td className="py-2 pr-4">
                      <span className="font-medium text-white">{p.product_id}</span>
                    </td>
                    <td className="py-2 pr-4 text-right">
                      <span className={`text-xs px-1.5 py-0.5 rounded ${p.side === 'BUY' ? 'bg-green-900/40 text-green-400' : 'bg-red-900/40 text-red-400'}`}>
                        {p.side}
                      </span>
                    </td>
                    <td className="py-2 pr-4 text-right font-mono text-gray-300">
                      {p.size.toFixed(6)} {p.base_currency}
                    </td>
                    <td className="py-2 pr-4 text-right font-mono text-gray-400">
                      ${fmtP(p.avg_price)}
                    </td>
                    <td className="py-2 pr-4 text-right font-mono text-white">
                      ${fmtP(p.current_price ?? 0)}
                    </td>
                    <td className="py-2 pr-4 text-right font-mono text-gray-300">
                      ${(p.current_value ?? 0).toFixed(2)}
                    </td>
                    <td className="py-2 text-right">
                      <div className={`font-mono text-sm ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                      </div>
                      <div className={`text-xs ${pctPnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {pctPnl >= 0 ? '+' : ''}{pctPnl.toFixed(2)}%
                      </div>
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
