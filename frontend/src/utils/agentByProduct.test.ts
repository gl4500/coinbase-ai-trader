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
    expect(map.get('XRP-USD')?.mom).toBeNull()
    expect(map.get('XRP-USD')?.scalp).toBeNull()
  })

  it('maps MOMENTUM decisions correctly', () => {
    const d = makeDecision({ agent: 'MOMENTUM', product_id: 'ADA-USD' })
    const map = buildAgentByProduct([d])
    expect(map.get('ADA-USD')?.mom).toEqual(d)
    expect(map.get('ADA-USD')?.tech).toBeNull()
    expect(map.get('ADA-USD')?.scalp).toBeNull()
  })

  it('maps SCALP decisions correctly', () => {
    const d = makeDecision({ agent: 'SCALP', product_id: 'XRP-USD', side: 'BUY' })
    const map = buildAgentByProduct([d])
    expect(map.get('XRP-USD')?.scalp).toEqual(d)
    expect(map.get('XRP-USD')?.tech).toBeNull()
    expect(map.get('XRP-USD')?.mom).toBeNull()
  })

  it('keeps the newest decision per agent (first in list = newest)', () => {
    const newer = makeDecision({ id: 2, agent: 'SCALP', product_id: 'XRP-USD', side: 'BUY',  confidence: 0.8 })
    const older  = makeDecision({ id: 1, agent: 'SCALP', product_id: 'XRP-USD', side: 'SELL', confidence: 0.4 })
    const map = buildAgentByProduct([newer, older])  // newest first
    expect(map.get('XRP-USD')?.scalp?.confidence).toBe(0.8)
    expect(map.get('XRP-USD')?.scalp?.side).toBe('BUY')
  })

  it('handles multiple products independently', () => {
    const decisions = [
      makeDecision({ agent: 'TECH',     product_id: 'XRP-USD' }),
      makeDecision({ agent: 'MOMENTUM', product_id: 'ADA-USD' }),
      makeDecision({ agent: 'SCALP',    product_id: 'XRP-USD', side: 'BUY' }),
    ]
    const map = buildAgentByProduct(decisions)
    expect(map.get('XRP-USD')?.tech).not.toBeNull()
    expect(map.get('XRP-USD')?.scalp).not.toBeNull()
    expect(map.get('ADA-USD')?.mom).not.toBeNull()
    expect(map.get('ADA-USD')?.scalp).toBeNull()  // SCALP didn't scan ADA
  })

  it('all three agents on same product', () => {
    const decisions = [
      makeDecision({ agent: 'TECH',     product_id: 'XRP-USD', side: 'BUY'  }),
      makeDecision({ agent: 'MOMENTUM', product_id: 'XRP-USD', side: 'HOLD' }),
      makeDecision({ agent: 'SCALP',    product_id: 'XRP-USD', side: 'BUY'  }),
    ]
    const map = buildAgentByProduct(decisions)
    const votes = map.get('XRP-USD')!
    expect(votes.tech?.side).toBe('BUY')
    expect(votes.mom?.side).toBe('HOLD')
    expect(votes.scalp?.side).toBe('BUY')
  })
})
