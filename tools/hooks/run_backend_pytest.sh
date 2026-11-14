#!/usr/bin/env bash
# pre-commit "pytest-backend" hook: run the fast backend suite with the PROJECT interpreter.
#
# Interpreter resolution (first hit wins), never a bare `python` from PATH:
#   1. $PYTEST_PYTHON, if set (must exist)
#   2. <main checkout>/.venv  — main checkout = parent of `git rev-parse --git-common-dir`,
#      so linked worktrees (which have no .venv of their own) use the main checkout's venv
#   3. <this checkout top level>/.venv
# If none exists the commit is refused with a message: a PATH python on this machine is a
# different interpreter without the project's dependencies (TD-2).
set -uo pipefail

fail() { echo "pytest-backend: $*" >&2; exit 1; }

top="$(git rev-parse --show-toplevel)" || fail "not inside a git checkout"
common="$(git rev-parse --path-format=absolute --git-common-dir)" || fail "cannot resolve git common dir"
main_root="$(dirname "$common")"

PYTHON=""
if [[ -n "${PYTEST_PYTHON:-}" ]]; then
    [[ "$PYTEST_PYTHON" == /* || "$PYTEST_PYTHON" =~ ^[A-Za-z]:[/\\] ]] || fail "PYTEST_PYTHON must be an absolute path, got '$PYTEST_PYTHON'"
    [[ -f "$PYTEST_PYTHON" ]] || fail "PYTEST_PYTHON='$PYTEST_PYTHON' does not exist"
    PYTHON="$PYTEST_PYTHON"
else
    for root in "$main_root" "$top"; do
        for cand in "$root/.venv/Scripts/python.exe" "$root/.venv/bin/python"; do
            if [[ -f "$cand" ]]; then
                PYTHON="$cand"
                break 2
            fi
        done
    done
fi
[[ -n "$PYTHON" ]] || fail "no project .venv under '$main_root' or '$top'; refusing to fall back to PATH python (set PYTEST_PYTHON to override)"

# Stub settings so no test can reach a real account (same values as before this hook moved).
export COINBASE_API_KEY_NAME="organizations/test/apiKeys/test-key"
export COINBASE_API_PRIVATE_KEY="stub"
export APP_API_KEY="pre-commit-test-key"
export DRY_RUN="true"
export DATABASE_URL=":memory:"
export LOG_LEVEL="WARNING"

cd "$top" || fail "cannot cd to '$top'"
echo "pytest-backend: using $PYTHON"

status=0
"$PYTHON" -m pytest backend/tests -m "not slow and not integration" -x -q --tb=short --no-header || status=$?

if [[ $status -ne 0 ]]; then
    echo "pytest-backend: FAILED (exit $status) - commit blocked" >&2
fi
exit "$status"
