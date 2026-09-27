from pathlib import Path
from typing import Any
import sys
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='backslashreplace')
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field
from session_bridge.store import Store

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DB = PROJECT_ROOT / '.coordination' / 'session-link.sqlite3'

class LinkState(BaseModel):
    database: str = str(DEFAULT_DB)
    role: str = 'codex'
    operation: str = 'status'
    payload: dict[str, Any] = Field(default_factory=dict)

class SessionLinkFlow(Flow[LinkState]):
    @start()
    def validate_request(self):
        if self.state.operation not in {'send', 'inbox', 'wait', 'ack', 'claim', 'update_task', 'status'}:
            raise ValueError('unknown bridge operation')
        return self.state.operation

    @listen(validate_request)
    def execute_request(self, operation):
        store = Store(self.state.database, self.state.role)
        return getattr(store, operation)(**self.state.payload)

def dispatch(database, role, operation, payload):
    return SessionLinkFlow().kickoff(inputs={
        'database': str(database), 'role': role, 'operation': operation, 'payload': payload})

def kickoff():
    import json
    print(json.dumps(dispatch(DEFAULT_DB, 'codex', 'status', {}), indent=2))

def plot():
    SessionLinkFlow().plot('session_link')
