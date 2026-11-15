# CLAUDE.md — polymarket_app (Claude adapter)

@AGENTS.md

`AGENTS.md` (imported above) is the shared project contract for Claude Code and Codex: scope, roles,
workflow, commits and peer review, security gate, tests, port discipline, run isolation, the session
link, checklists and the mandatory reference `docs/agents/architecture-and-invariants.md`. This file
adds only what applies to Claude. Where they disagree, stop and ask the operator.

---

## Session hygiene — compact periodically

Long Claude Code sessions burn context fast — especially subagent-driven implementation runs, multi-file refactors, and brainstorm → spec → plan → execute cycles. The assistant should **suggest `/compact`** at natural breakpoints rather than let context grow unbounded.

Triggers to suggest `/compact`:
- After 5+ subagent dispatches within one feature
- When estimated conversation exceeds ~100k tokens
- Before starting a NEW feature in a session that already shipped one
- At natural milestones — after a commit lands, after a sweep module finishes, after a brainstorm/plan cycle completes
- When the user mentions cost, slowness, or long-context concerns

How to apply:
- Surface as a single-line suggestion at the breakpoint: *"Context is getting long — want to `/compact` before we continue?"*
- Don't compact silently or force it; the user decides
- `/compact` preserves project state (memory files, recent commits, working tree) — safe at any natural breakpoint
- After `/compact` fires, the same rule applies to the next chunk of work

---

## Memory

- Update relevant memory files immediately after every code change.
- Do not wait until end of session.

---

## CLAUDE.md ↔ Memory Sync Rule

**The repo contract and memory must always be updated together, and memory must be updated after every code change.**

The repo contract (`AGENTS.md` for shared rules, this `CLAUDE.md` for Claude-only rules) and the persistent memory files at
`C:\Users\gl450\.claude\projects\C--Users-gl450\memory\` are the two halves of the same contract.

### After every code change — required steps
1. Architecture change (agents, endpoints, file structure) → update `coinbase_trader_architecture.md`.
2. Bug fixed → append the bug + fix to `coinbase_trader_bugs_fixed.md`.
3. Threshold changed → update `coinbase_trader_thresholds.md`.
4. Rule added or modified → update the matching memory file AND the matching `AGENTS.md` section (or this `CLAUDE.md`, for a Claude-only rule) in the same response.
5. Never commit code without committing any corresponding `AGENTS.md` / `CLAUDE.md` update in the same or immediately following commit.

Relevant memory files for this repo (`coinbase-ai-trader` / polymarket_app):
| Memory file | Mirrors |
|---|---|
| `feedback_tdd_workflow.md` | `AGENTS.md` TDD Workflow section |
| `feedback_scope_restriction.md` | `AGENTS.md` Scope section |
| `feedback_shell_cleanup.md` | `AGENTS.md` Shell cleanup section |
| `feedback_sync_rule.md` | Memory sync rule |
| `feedback_python_clean_functions.md` | `AGENTS.md` Code Style + new-code authoring rules |
| `feedback_verify_your_own_reader.md` | `AGENTS.md` Session-link coordination section |
| `feedback_xgb_focus_not_cnn.md` | XGB-only scope; CNN frozen for new feature work |
| `coinbase_trader_architecture.md` | `docs/agents/architecture-and-invariants.md` Architecture Quick Reference + per-session change log |
| `coinbase_trader_schema.md` | DB column lists + code landmarks |

Note: `trading_app_*.md` memory files belong to a different project (not polymarket_app) and should be ignored when working in this repo.

---

## Required Skills — Invoke Before Acting

These trigger clauses are binding. Invoke the matching skill via the `Skill` tool before the action, not after.

| Trigger | Skill |
|---|---|
| Writing or modifying code | `superpowers:test-driven-development` |
| Proposing a fix for a bug or failing test | `superpowers:systematic-debugging` |
| Claiming work complete, or before committing | `superpowers:verification-before-completion` |
| Touching anything under `frontend/` | `frontend-design` |
| Receiving any code-review feedback | `superpowers:receiving-code-review` |
| Before requesting review on a completed change | `superpowers:requesting-code-review` |

A global `SessionStart` hook in `~/.claude/settings.json` also echoes this list at the start of every session so it is always in context.
