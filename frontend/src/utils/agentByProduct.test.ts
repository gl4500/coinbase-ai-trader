import { describe, it, expect } from 'vitest'
import { buildAgentByProduct, AgentDecision } from './agentByProduct'

function makeDecision(overrides: Partial<AgentDecision>): AgentDecision {
  return {
    id: 1, agent: 'TECH', product_id: 'XRP-USD',
    side: 'BUY', confidence: 0.7, price: 1.33,
    score: 0.7, reasoning: null, balance: 900, pnl: null,
    created_at: '2026-04-12T15:00:00Z',
    ...overrides,
  }
}

describe('buildAgentByProduct', () => {
  it('returns empty map for no decisions', () => {
    const map = buildAgentByProduct([])
    expect(map.size).toBe(0)
  })

  it('maps TECH decisions correctly', () => {
    const d = makeDecision({ agent: 'TECH', product_id: 'XRP-USD' })
    const map = buildAgentByProduct([d])
    expect(map.get('XRP-USD')?.tech).toEqual(d)
  })

  it('keeps the newest decision per agent (first in list = newest)', () => {
    const newer = makeDecision({ id: 2, agent: 'TECH', product_id: 'XRP-USD', side: 'BUY',  confidence: 0.8 })
    const older  = makeDecision({ id: 1, agent: 'TECH', product_id: 'XRP-USD', side: 'SELL', confidence: 0.4 })
    const map = buildAgentByProduct([newer, older])  // newest first
    expect(map.get('XRP-USD')?.tech?.confidence).toBe(0.8)
    expect(map.get('XRP-USD')?.tech?.side).toBe('BUY')
  })

  it('handles multiple products independently', () => {
    const decisions = [
      makeDecision({ agent: 'TECH', product_id: 'XRP-USD' }),
      makeDecision({ agent: 'TECH', product_id: 'ADA-USD', side: 'SELL' }),
    ]
    const map = buildAgentByProduct(decisions)
    expect(map.get('XRP-USD')?.tech).not.toBeNull()
    expect(map.get('ADA-USD')?.tech).not.toBeNull()
    expect(map.get('ADA-USD')?.tech?.side).toBe('SELL')
  })

  it('unknown agent is ignored', () => {
    const d = makeDecision({ agent: 'UNKNOWN', product_id: 'XRP-USD' })
    const map = buildAgentByProduct([d])
    // Entry is created but tech is null
    expect(map.get('XRP-USD')?.tech).toBeNull()
  })
})
