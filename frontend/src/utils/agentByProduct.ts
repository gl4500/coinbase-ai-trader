/**
 * Pure function extracted from CNNDashboard's agentByProduct useMemo.
 * Builds a per-product map of the latest decision from each agent.
 * Exported so it can be unit tested without mounting the component.
 */

export interface AgentDecision {
  id:         number
  agent:      string        // TECH | MOMENTUM | SCALP
  product_id: string
  side:       string        // BUY | SELL | HOLD
  confidence: number
  price:      number
  score:      number | null
  reasoning:  string | null
  balance:    number | null
  pnl:        number | null
  created_at: string
}

export interface AgentVotes {
  tech:  AgentDecision | null
  mom:   AgentDecision | null
  scalp: AgentDecision | null
}

export function buildAgentByProduct(decisions: AgentDecision[]): Map<string, AgentVotes> {
  const map = new Map<string, AgentVotes>()
  for (const d of decisions) {
    const cur = map.get(d.product_id) ?? { tech: null, mom: null, scalp: null }
    if (d.agent === 'TECH'     && cur.tech  === null) cur.tech  = d
    if (d.agent === 'MOMENTUM' && cur.mom   === null) cur.mom   = d
    if (d.agent === 'SCALP'    && cur.scalp === null) cur.scalp = d
    map.set(d.product_id, cur)
  }
  return map
}
