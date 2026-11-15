# CLAUDE.md — polymarket_app (Claude adapter)

@AGENTS.md

**If the `AGENTS.md` text is not in your context, read `AGENTS.md` in the repository root (the folder
that holds this `CLAUDE.md`) in full before doing anything else.** Claude Code skips the import above
when a session starts in a subdirectory (e.g. `tools/session_bridge/`): the root file then counts as an external
import that needs approval, and a non-interactive session never asks. Start sessions at the repository root.

`AGENTS.md` (imported above) is the shared project contract for Claude Code and Codex: scope, roles,
workflow, commits and peer review, security gate, tests, port discipline, run isolation, the session
link, checklists and the mandatory references in `docs/agents/`. This file
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

## Memory ↔ repo rule (pointers, not copies)

**A rule's text lives once, in this repo:** shared rules in `AGENTS.md` and its mandatory references,
Claude-only rules in this `CLAUDE.md`. The memory folder `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\`
holds what the repo does not: why a rule exists, incident history, operator quotes, rules that apply
across projects, and project state (architecture, bugs, thresholds, session log). Where memory covers a
repo rule it holds a pointer to the section, a one-line summary and the why, never a second full copy.
If memory and this repo disagree on a rule, this repo (in the task's own checkout) wins: follow it and
fix the memory pointer in the same session.

### After every code change — required steps
1. Architecture change (agents, endpoints, file structure) → update `coinbase_trader_architecture.md`.
2. Bug fixed → append the bug + fix to `coinbase_trader_bugs_fixed.md`.
3. Threshold changed → update `coinbase_trader_thresholds.md`.
4. Rule added or modified → change it in the matching `AGENTS.md` section or mandatory reference (or this `CLAUDE.md`, for a Claude-only rule). In the same response, update the memory pointer, summary line and why/history; don't copy the rule text into memory.
5. Never commit code without committing any corresponding `AGENTS.md` / `CLAUDE.md` update in the same or immediately following commit.

Memory files that point into this repo (`coinbase-ai-trader` / polymarket_app):
| Memory file | Points to (memory keeps why/history) |
|---|---|
| `feedback_no_pytest_during_trading.md` | `AGENTS.md` TDD Workflow → Test windows; Live-backend guard |
| `feedback_backend_port_isolation.md` | `AGENTS.md` Backend port discipline; `promote-model` skill |
| `feedback_no_restart_during_retrain.md` | `AGENTS.md` Backend port discipline; checklists |
| `feedback_codex_review_before_merge.md` | `AGENTS.md` Roles; Commit Standards → Peer review before merge |
| `feedback_xgb_focus_not_cnn.md` | `AGENTS.md` Code Style (XGB-only) |
| `feedback_python_clean_functions.md` | `docs/agents/coding-standards.md` (memory keeps the cross-project copy) |
| `feedback_tdd_workflow.md`, `feedback_scope_restriction.md`, `feedback_shell_cleanup.md`, `feedback_verify_your_own_reader.md` | General rules in memory; this repo's procedure in the matching `AGENTS.md` section |
| `feedback_sync_rule.md` | This section |
| `coinbase_trader_architecture.md` | Project state; normative invariants are in `docs/agents/architecture-and-invariants.md` |
| `coinbase_trader_schema.md` | DB column lists + code landmarks (project state) |

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
