import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

// Top-level fallback — catches crashes in App itself
function FatalError({ error }: { error: Error }) {
  return (
    <div style={{ background: '#030712', minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '2rem' }}>
      <div style={{ background: '#111827', border: '1px solid #374151', borderRadius: '12px', padding: '1.5rem', maxWidth: '600px', width: '100%' }}>
        <p style={{ color: '#f87171', fontWeight: 600, marginBottom: '0.5rem' }}>Coinbase Trader — Fatal render error</p>
        <pre style={{ color: '#9ca3af', fontSize: '12px', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{error.message}</pre>
        <button style={{ marginTop: '1rem', padding: '6px 16px', background: '#1f2937', border: '1px solid #374151', borderRadius: '8px', color: '#d1d5db', cursor: 'pointer' }}
          onClick={() => window.location.reload()}>Reload</button>
      </div>
    </div>
  )
}

class RootBoundary extends React.Component<{ children: React.ReactNode }, { error: Error | null }> {
  state = { error: null }
  static getDerivedStateFromError(e: Error) { return { error: e } }
  render() {
    return this.state.error ? <FatalError error={this.state.error} /> : this.props.children
  }
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RootBoundary>
      <App />
    </RootBoundary>
  </React.StrictMode>
)
