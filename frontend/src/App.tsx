import React, { useCallback, useEffect, useRef, useState, Component, ErrorInfo, ReactNode } from 'react'

class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null }
  static getDerivedStateFromError(error: Error) { return { error } }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('[ErrorBoundary]', error, info.componentStack) }
  render() {
    if (this.state.error) return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center p-8">
        <div className="card max-w-lg w-full">
          <p className="text-red-400 font-semibold mb-2">Render error</p>
          <pre className="text-xs text-gray-400 whitespace-pre-wrap break-all">{(this.state.error as Error).message}</pre>
          <button className="btn-secondary mt-4 text-xs" onClick={() => this.setState({ error: null })}>Retry</button>
        </div>
      </div>
    )
    return this.props.children
  }
}

import MarketBrowser    from './components/MarketBrowser'
import OrderBook        from './components/OrderBook'
import PositionTracker  from './components/PositionTracker'
import SignalDashboard  from './components/SignalDashboard'
import CNNDashboard     from './components/CNNDashboard'
import AgentsDashboard  from './components/AgentsDashboard'
import LogViewer        from './components/LogViewer'

// ── Types ──────────────────────────────────────────────────────────────────────

export interface Product {
  product_id:            string
  base_currency:         string
  quote_currency:        string
  display_name:          string
  price:                 number
  price_pct_change_24h:  number
  volume_24h:            number
  high_24h:              number
  low_24h:               number
  spread?:               number
  is_tracked:            number
  last_updated?:         string
}

export interface Position {
  product_id:     string
  base_currency:  string
  side:           string
  size:           number
  avg_price:      number
  current_price?: number
  initial_value?: number
  current_value?: number
  cash_pnl?:      number
  pct_pnl?:       number
}

export interface Signal {
  id:           number
  product_id:   string
  signal_type:  string
  side:         string
  price:        number
  strength:     number
  rsi?:         number
  macd?:        number
  bb_position?: number
  reasoning?:   string
  acted:        number
  created_at:   string
}

export interface Order {
  order_id:        string
  product_id:      string
  side:            string
  order_type:      string
  price?:          number
  base_size?:      number
  quote_size?:     number
  status:          string
  filled_size:     number
  avg_fill_price?: number
  strategy?:       string
  created_at:      string
}

export interface PortfolioSummary {
  open_positions: number
  total_value:    number
  total_cost:     number
  total_pnl:      number
  pct_pnl:        number
}

interface AppData {
  is_trading: boolean
  dry_run:    boolean
  portfolio:  PortfolioSummary
  positions:  Position[]
  signals:    Signal[]
  orders:     Order[]
  products:   Product[]
}

const TABS = ['Markets', 'Order Book', 'Positions', 'Signals', 'CNN', 'Agents', 'Logs'] as const
type Tab = typeof TABS[number]

// ── App ────────────────────────────────────────────────────────────────────────

export default function App() {
  const [activeTab, setActiveTab]         = useState<Tab>('Markets')
  const [wsConnected, setWsConnected]     = useState(false)
  const [statusMessage, setStatusMessage] = useState('')
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null)
  const apiKeyRef = useRef<string>(localStorage.getItem('cb_api_key') ?? '')

  // Authenticated fetch helper — automatically includes X-API-Key when set
  const postJSON = useCallback((url: string, opts: RequestInit = {}) =>
    fetch(url, {
      ...opts,
      method: opts.method ?? 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKeyRef.current, ...(opts.headers ?? {}) },
    }), [])

  const [appData, setAppData] = useState<AppData>({
    is_trading: false,
    dry_run:    true,
    portfolio:  { open_positions: 0, total_value: 0, total_cost: 0, total_pnl: 0, pct_pnl: 0 },
    positions:  [],
    signals:    [],
    orders:     [],
    products:   [],
  })

  const wsRef = useRef<WebSocket | null>(null)

  const connectWS = useCallback(() => {
    const ws = new WebSocket(`ws://${window.location.hostname}:8001/ws`)
    wsRef.current = ws
    ws.onopen  = () => setWsConnected(true)
    ws.onclose = () => { setWsConnected(false); setTimeout(connectWS, 3000) }
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'state') {
          setAppData(prev => ({
            ...prev,
            ...msg,
            portfolio: { ...prev.portfolio, ...(msg.portfolio ?? {}) },
            positions: Array.isArray(msg.positions) ? msg.positions : prev.positions,
            signals:   Array.isArray(msg.signals)   ? msg.signals   : prev.signals,
            orders:    Array.isArray(msg.orders)     ? msg.orders    : prev.orders,
            products:  Array.isArray(msg.products)   ? msg.products  : prev.products,
          }))
        } else if (msg.type === 'trading_status') {
          setAppData(prev => ({ ...prev, is_trading: !!msg.is_trading, dry_run: !!msg.dry_run }))
        } else if (msg.type === 'price_update' && msg.product_id) {
          setAppData(prev => ({
            ...prev,
            products: prev.products.map(p =>
              p.product_id === msg.product_id
                ? { ...p, price: msg.price, price_pct_change_24h: msg.pct_change ?? p.price_pct_change_24h }
                : p
            ),
          }))
        }
      } catch {}
    }
  }, [])

  useEffect(() => {
    connectWS()
    return () => wsRef.current?.close()
  }, [connectWS])

  // Fetch API key once on mount
  useEffect(() => {
    fetch('/api/status')
      .then(r => r.json())
      .then(d => { if (d.app_api_key) { apiKeyRef.current = d.app_api_key; localStorage.setItem('cb_api_key', d.app_api_key) } })
      .catch(() => {})
  }, [])

  const flash = (msg: string) => {
    setStatusMessage(msg)
    setTimeout(() => setStatusMessage(''), 3000)
  }

  const handleToggleTrading = async () => {
    const enabling = !appData.is_trading
    const ep       = enabling ? '/api/trading/enable' : '/api/trading/disable'
    const resp     = await postJSON(ep)
    if (resp.ok) {
      setAppData(prev => ({ ...prev, is_trading: enabling }))
      flash(enabling ? 'Trading enabled' : 'Trading paused — closing Brave…')
    } else {
      flash(`Failed (${resp.status})`)
    }
  }

  const handleScanMarkets = async () => {
    flash('Scanning…')
    const resp = await postJSON('/api/scanner/run')
    const data = await resp.json()
    flash(`Updated ${data.tracked_products || 0} products`)
  }

  const portfolio: PortfolioSummary = {
    open_positions: appData.portfolio?.open_positions ?? 0,
    total_value:    appData.portfolio?.total_value    ?? 0,
    total_cost:     appData.portfolio?.total_cost     ?? 0,
    total_pnl:      appData.portfolio?.total_pnl      ?? 0,
    pct_pnl:        appData.portfolio?.pct_pnl        ?? 0,
  }

  return (
    <ErrorBoundary>
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-gray-900/95 backdrop-blur border-b border-gray-800">
        <div className="max-w-screen-2xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="text-2xl">🪙</div>
            <div>
              <h1 className="text-lg font-bold text-white leading-none">Coinbase Trader</h1>
              <p className="text-xs text-gray-500 mt-0.5">Advanced Trade · RSI · MACD · CNN signals</p>
            </div>
          </div>

          {/* Portfolio summary */}
          <div className="hidden md:flex items-center gap-6 text-sm">
            <div className="text-center">
              <div className="text-xs text-gray-500">Positions</div>
              <div className="font-bold text-white">{portfolio.open_positions}</div>
            </div>
            <div className="text-center">
              <div className="text-xs text-gray-500">Value</div>
              <div className="font-bold text-white">${portfolio.total_value.toFixed(2)}</div>
            </div>
            <div className="text-center">
              <div className="text-xs text-gray-500">P&L</div>
              <div className={`font-bold ${portfolio.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {portfolio.total_pnl >= 0 ? '+' : ''}${portfolio.total_pnl.toFixed(2)}
                <span className="text-xs ml-1">({portfolio.pct_pnl.toFixed(1)}%)</span>
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <div className={`flex items-center gap-1.5 text-xs ${wsConnected ? 'text-green-400' : 'text-gray-500'}`}>
              <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400 animate-pulse' : 'bg-gray-600'}`} />
              {wsConnected ? 'Live' : 'Offline'}
            </div>
            {statusMessage && <span className="text-xs text-blue-400 animate-pulse">{statusMessage}</span>}
            {appData.dry_run && <span className="badge-amber text-xs">DRY-RUN</span>}
            <button onClick={handleScanMarkets} className="btn-secondary text-xs py-1.5 px-3">
              🔍 Scan
            </button>
            <button
              onClick={handleToggleTrading}
              className={`text-xs px-3 py-1.5 rounded border transition-colors ${
                appData.is_trading
                  ? 'bg-green-700 border-green-500 text-white hover:bg-green-600'
                  : 'bg-gray-800 border-gray-600 text-gray-400 hover:text-gray-200'
              }`}
            >
              {appData.is_trading ? '⏸ Pause' : '▶ Start'}
            </button>
          </div>
        </div>

        {/* Tab bar */}
        <div className="max-w-screen-2xl mx-auto px-4 pb-2 flex gap-1 overflow-x-auto">
          {TABS.map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`tab whitespace-nowrap ${activeTab === tab ? 'tab-active' : 'tab-inactive'}`}
            >
              {tab}
            </button>
          ))}
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-screen-2xl mx-auto px-4 py-6">
        {activeTab === 'Markets' && (
          <MarketBrowser
            products={appData.products}
            onSelectProduct={(p) => { setSelectedProduct(p); setActiveTab('Order Book') }}
          />
        )}
        {activeTab === 'Order Book' && (
          <OrderBook product={selectedProduct} />
        )}
        {activeTab === 'Positions' && (
          <PositionTracker positions={appData.positions} portfolio={portfolio} />
        )}
        {activeTab === 'Signals' && (
          <SignalDashboard signals={appData.signals} />
        )}
        {activeTab === 'CNN' && (
          <CNNDashboard signals={appData.signals} orders={appData.orders} postJSON={postJSON} />
        )}
        {activeTab === 'Agents' && (
          <AgentsDashboard />
        )}
        {activeTab === 'Logs' && (
          <LogViewer />
        )}
      </main>
    </div>
    </ErrorBoundary>
  )
}
