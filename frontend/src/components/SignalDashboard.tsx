import React, { useState } from 'react'
import { Signal } from '../App'

interface Props { signals: Signal[] }

export default function SignalDashboard({ signals }: Props) {
  const [filter, setFilter] = useState<'ALL' | 'BUY' | 'SELL'>('ALL')

  const filtered = signals.filter(s => filter === 'ALL' || s.side === filter)

  const bar = (val: number | null | undefined, color: string) => {
    if (val == null) return <span className="text-gray-600">—</span>
    const w = Math.round(Math.abs(val) * 100)
    return (
      <div className="flex items-center gap-1.5">
        <div className="w-16 bg-gray-700 rounded-full h-1.5">
          <div className={`h-1.5 rounded-full ${color}`} style={{ width: `${w}%` }} />
        </div>
        <span className="text-xs text-gray-400">{(val * 100).toFixed(0)}%</span>
      </div>
    )
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-white">
          Signals <span className="text-sm text-gray-500 font-normal">({filtered.length})</span>
        </h2>
        <div className="flex gap-1">
          {(['ALL', 'BUY', 'SELL'] as const).map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`text-xs px-3 py-1 rounded border transition-colors ${
                filter === f
                  ? f === 'BUY'  ? 'bg-green-800 border-green-600 text-green-200'
                  : f === 'SELL' ? 'bg-red-800 border-red-600 text-red-200'
                                 : 'bg-blue-800 border-blue-600 text-blue-200'
                  : 'bg-gray-800 border-gray-700 text-gray-400 hover:text-gray-200'
              }`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="card text-center py-12">
          <p className="text-gray-500 text-sm">
            No signals yet — click Scan or wait for the next auto-scan
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {filtered.map(s => (
            <div key={s.id} className="card">
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                    s.side === 'BUY'
                      ? 'bg-green-900/50 text-green-400 border border-green-800'
                      : 'bg-red-900/50 text-red-400 border border-red-800'
                  }`}>
                    {s.side}
                  </span>
                  <span className="font-bold text-white">{s.product_id}</span>
                  <span className="text-xs text-gray-500">{s.signal_type}</span>
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

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-2">
                <div>
                  <div className="text-xs text-gray-500 mb-0.5">Strength</div>
                  {bar(s.strength, s.side === 'BUY' ? 'bg-green-500' : 'bg-red-500')}
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-0.5">RSI</div>
                  {s.rsi != null
                    ? <span className={`text-xs font-mono ${s.rsi < 30 ? 'text-green-400' : s.rsi > 70 ? 'text-red-400' : 'text-gray-300'}`}>
                        {s.rsi.toFixed(1)}
                      </span>
                    : <span className="text-gray-600 text-xs">—</span>
                  }
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-0.5">MACD Hist</div>
                  {s.macd != null
                    ? <span className={`text-xs font-mono ${s.macd >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {s.macd >= 0 ? '+' : ''}{s.macd.toFixed(6)}
                      </span>
                    : <span className="text-gray-600 text-xs">—</span>
                  }
                </div>
                <div>
                  <div className="text-xs text-gray-500 mb-0.5">BB Position</div>
                  {s.bb_position != null
                    ? <span className={`text-xs font-mono ${s.bb_position < 0.2 ? 'text-green-400' : s.bb_position > 0.8 ? 'text-red-400' : 'text-gray-300'}`}>
                        {(s.bb_position * 100).toFixed(0)}%
                      </span>
                    : <span className="text-gray-600 text-xs">—</span>
                  }
                </div>
              </div>

              {s.reasoning && (
                <p className="text-xs text-gray-500 leading-relaxed border-t border-gray-800 pt-2 mt-1 whitespace-pre-line">
                  {s.reasoning.slice(0, 300)}{s.reasoning.length > 300 ? '…' : ''}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
