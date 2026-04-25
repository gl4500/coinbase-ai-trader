import React, { useEffect, useState } from 'react'

interface Counts {
  cnn_scans:        number
  cnn_signals:      number
  cnn_buy:          number
  cnn_sell:         number
  train_count:      number
  llm_calls:        number
  llm_prompt_tok:   number
  llm_response_tok: number
  llm_total_tok:    number
  tech_scans:       number
  tech_signals:     number
}

const EMPTY: Counts = {
  cnn_scans: 0, cnn_signals: 0, cnn_buy: 0, cnn_sell: 0, train_count: 0,
  llm_calls: 0, llm_prompt_tok: 0, llm_response_tok: 0, llm_total_tok: 0,
  tech_scans: 0, tech_signals: 0,
}

function Stat({ label, value, color = 'text-white' }: { label: string; value: number | string; color?: string }) {
  return (
    <div className="flex flex-col items-center px-4 py-2 bg-gray-800 rounded-lg min-w-[72px]">
      <span className={`text-lg font-bold tabular-nums ${color}`}>{value}</span>
      <span className="text-[10px] text-gray-500 mt-0.5 text-center leading-tight">{label}</span>
    </div>
  )
}

export default function FiringCounter() {
  const [counts, setCounts] = useState<Counts>(EMPTY)
  const [lastUpdated, setLastUpdated] = useState('')

  useEffect(() => {
    async function fetch_counts() {
      try {
        const [cnnRes, agentsRes] = await Promise.all([
          fetch('/api/cnn/status'),
          fetch('/api/agents/status'),
        ])
        const cnn    = cnnRes.ok    ? await cnnRes.json()    : {}
        const agents = agentsRes.ok ? await agentsRes.json() : {}

        const tech  = agents.tech       ?? {}

        setCounts({
          cnn_scans:        cnn.scan_count                                  ?? 0,
          cnn_signals:      cnn.signals_total                               ?? 0,
          cnn_buy:          cnn.signals_buy                                 ?? 0,
          cnn_sell:         cnn.signals_sell                                ?? 0,
          train_count:      cnn.train_count                                 ?? 0,
          llm_calls:        cnn.llm_calls                                   ?? 0,
          llm_prompt_tok:   cnn.llm_prompt_tokens                           ?? 0,
          llm_response_tok: cnn.llm_response_tokens                         ?? 0,
          llm_total_tok:    cnn.llm_total_tokens                            ?? 0,
          tech_scans:       tech.scan_count                                 ?? 0,
          tech_signals:     (tech.signals_buy ?? 0) + (tech.signals_sell ?? 0),
        })
        setLastUpdated(new Date().toLocaleTimeString())
      } catch {}
    }

    fetch_counts()
    const id = setInterval(fetch_counts, 10_000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="card mb-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-300">Agent Firing Counters</h2>
        <span className="text-[10px] text-gray-600">updated {lastUpdated}</span>
      </div>

      <div className="flex flex-wrap gap-2">
        {/* CNN */}
        <div className="flex gap-2 items-center">
          <span className="text-[10px] text-gray-500 font-semibold uppercase tracking-widest w-8">CNN</span>
          <Stat label="Scans" value={counts.cnn_scans} color="text-blue-300" />
          <Stat label="Signals" value={counts.cnn_signals} />
          <Stat label="BUY" value={counts.cnn_buy} color="text-green-400" />
          <Stat label="SELL" value={counts.cnn_sell} color="text-red-400" />
          <Stat label="Trains" value={counts.train_count} color="text-purple-400" />
        </div>

        <div className="w-px bg-gray-700 self-stretch mx-1" />

        {/* LLM tokens */}
        <div className="flex gap-2 items-center">
          <span className="text-[10px] text-gray-500 font-semibold uppercase tracking-widest w-8">LLM</span>
          <Stat label="Calls" value={counts.llm_calls} color="text-yellow-300" />
          <Stat label="In tok" value={counts.llm_prompt_tok.toLocaleString()} color="text-orange-300" />
          <Stat label="Out tok" value={counts.llm_response_tok.toLocaleString()} color="text-orange-300" />
          <Stat label="Total tok" value={counts.llm_total_tok.toLocaleString()} color="text-amber-400" />
        </div>

        <div className="w-px bg-gray-700 self-stretch mx-1" />

        {/* Tech */}
        <div className="flex gap-2 items-center">
          <span className="text-[10px] text-gray-500 font-semibold uppercase tracking-widest w-8">TECH</span>
          <Stat label="Scans" value={counts.tech_scans} color="text-blue-300" />
          <Stat label="Signals" value={counts.tech_signals} />
        </div>

      </div>
    </div>
  )
}
