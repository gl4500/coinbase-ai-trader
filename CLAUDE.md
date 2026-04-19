# Claude Code — Project Instructions

## Scope

Primary working directory: `C:\Users\gl450\polymarket_app\`
Do not modify files outside this directory.

---

## Fix Workflow (REQUIRED)

When you find bugs, inconsistencies, or issues — during an audit, code review,
or in response to a user report — follow this sequence every time:

1. **Create a task list before touching code.**
   Use `TaskCreate` for each distinct fix. Each task must identify:
   - The file
   - The problem
   - The intended fix

2. **Fix in order. Mark tasks as you go.**
   Set `in_progress` when starting a task, `completed` immediately when done.
   Do not batch completions.

3. **Run tests after each fix**, not just at the end.
   If a test fails, resolve it before moving to the next task.

4. **No fix without a task.**
   If you discover a new problem mid-fix, create a task for it first.

---

## Memory

- Update relevant memory files immediately after every code change.
- Do not wait until end of session.

---

## Tests

- Write tests before or alongside fixes where practical.
- Prefer per-module test runs over the full suite.
- Kill any background processes started during test runs when done.

---

## Code Style

- Edit existing files rather than creating new ones.
- Do not add features, refactoring, or abstractions beyond the task scope.
- No comments unless the *why* is non-obvious.
