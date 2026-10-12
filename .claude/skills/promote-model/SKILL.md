---
name: promote-model
description: Use when shipping a new XGB model or inference path in polymarket_app — the 8002 shadow validation and 8001 promotion procedure.
---

# Promote a new XGB model / inference path (8002 shadow → 8001)

Port rules live in CLAUDE.md ("Backend port discipline"): 8001 is the live backend; new or
experimental backends run on 8002+. This is the procedure for moving a new model through them.

1. Land code + migration (creates new telemetry column nullable).
2. Operator-train the new model → unsuffixed artifacts (or horizon-suffixed during sweep).
3. Operator-start the new backend on **port 8002**: `PORT=8002 python main.py` from `backend/` cwd.
4. Verify new telemetry column populates in `cnn_scans` (rows from PID 8002 backend will have non-NULL).
5. Compare new vs old metrics over a shadow week (or however long the cutover plan specifies).
6. **If new model shows promise** → copy new-model artifacts to the unsuffixed paths the 8001 backend loads, then restart the 8001 backend (which will pick up the new artifacts). The promotion is just a file-copy + restart.
7. If new model doesn't show promise → kill the 8002 backend, leave 8001 untouched, iterate on the model.

Never kill 8001 mid-session unless: (a) the operator explicitly approves, OR (b) you're swapping
artifacts after the promotion-gate criteria are met.
