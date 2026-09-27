from pathlib import Path
from typing import Any
import sys
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='backslashreplace')
from session_bridge.store import Store

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DB = PROJECT_ROOT / '.coordination' / 'session-link.sqlite3'

# The single home for what may be dispatched. BOTH routes read this, because an allow-list
# copied per route is the classic one-fact-in-two-places defect: the copy nobody checks
# drifts, and an operation becomes reachable one way and refused the other.
ALLOWED_OPERATIONS = frozenset(
    {'send', 'inbox', 'wait', 'ack', 'claim', 'update_task', 'status'})

# Operations whose whole cost is the import. `import crewai.flow` measures ~4.3s against
# ~0.25s for sqlite3, to run one membership test and one getattr, so these take the direct
# route by default. Everything else keeps the Flow.
FAST_OPERATIONS = frozenset({'send', 'inbox', 'wait', 'ack'})


def _validated(operation):
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError('unknown bridge operation')
    return operation


def dispatch_direct(database, role, operation, payload):
    """Route straight to the store, without importing CrewAI.

    Same validation and same result as `dispatch`; a test probes both routes and pins them
    to identical answers. This exists for latency only -- the Flow route below remains
    reachable, since AGENTS.md is explicit that turning off CrewAI's instrumentation is the
    operator's decision rather than a performance fix.
    """
    return getattr(Store(str(database), role), _validated(operation))(**payload)


def route_for(operation, requested=None):
    """Which route to use. `requested` ('fast'/'flow') wins so the instrumented path is
    always available; otherwise message operations go fast and the rest keep the Flow."""
    if requested not in (None, 'auto', 'fast', 'flow'):
        raise ValueError("route must be auto, fast or flow")
    if requested in ('fast', 'flow'):
        return requested
    return 'fast' if operation in FAST_OPERATIONS else 'flow'


def dispatch_routed(database, role, operation, payload, requested=None):
    """Entry point for callers that do not care which route runs."""
    if route_for(_validated(operation), requested) == 'fast':
        return dispatch_direct(database, role, operation, payload)
    return dispatch(database, role, operation, payload)

_FLOW = None


def _session_link_flow():
    """Build the CrewAI Flow on first use, not at import.

    Deferring this is the whole latency fix: the import measures ~4.3s against ~0.25s for
    sqlite3. `Flow[LinkState]` is evaluated at class-definition time, so the class has to be
    created inside the function rather than merely referenced there. Cached, so repeated
    Flow-route calls in one process pay the import once.
    """
    global _FLOW
    if _FLOW is not None:
        return _FLOW

    from crewai.flow.flow import Flow, listen, start
    from pydantic import BaseModel, Field

    class LinkState(BaseModel):
        database: str = str(DEFAULT_DB)
        role: str = 'codex'
        operation: str = 'status'
        payload: dict[str, Any] = Field(default_factory=dict)

    class SessionLinkFlow(Flow[LinkState]):
        @start()
        def validate_request(self):
            # Reads the SHARED allow-list, so the two routes cannot drift apart.
            return _validated(self.state.operation)

        @listen(validate_request)
        def execute_request(self, operation):
            store = Store(self.state.database, self.state.role)
            return getattr(store, operation)(**self.state.payload)

    _FLOW = SessionLinkFlow
    return _FLOW


def dispatch(database, role, operation, payload):
    """The CrewAI-routed path, with its instrumentation intact."""
    return _session_link_flow()().kickoff(inputs={
        'database': str(database), 'role': role, 'operation': operation, 'payload': payload})

def kickoff():
    import json
    print(json.dumps(dispatch(DEFAULT_DB, 'codex', 'status', {}), indent=2))

def plot():
    _session_link_flow()().plot('session_link')
