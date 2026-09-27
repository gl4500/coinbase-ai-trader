"""Absolute-path entry point for MCP, CLI and lightweight Claude hooks."""
import argparse
import json
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE / 'src'))


def main():
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--role', choices=['claude', 'codex'], required=True)
    parser.add_argument('--db', default=str(ROOT / '.coordination' / 'session-link.sqlite3'))
    parser.add_argument('mode', choices=['mcp', 'call', 'hook'])
    parser.add_argument('operation', nargs='?', default='status')
    parser.add_argument('--payload', default='{}')
    parser.add_argument('--payload-file')
    # 'auto' sends message ops down the direct route and the rest through the
    # CrewAI Flow; 'flow' forces the instrumented path for any operation.
    parser.add_argument('--route', choices=['auto', 'fast', 'flow'], default='auto')
    args = parser.parse_args()
    if args.mode == 'hook':
        from session_bridge.hooks import handle_hook
        handle_hook(args.db, args.role, HERE / 'bridge.py')
        return
    # Keep stdout exclusively JSON-RPC/JSON, including background event output.
    protocol_stdout = sys.stdout
    protocol_stdin = sys.stdin
    sys.stdout = sys.stderr
    # Framework interactive prompts must never consume MCP protocol messages.
    sys.stdin = io.StringIO('')
    from session_bridge.main import dispatch_routed
    if args.mode == 'call':
        payload = json.loads(Path(args.payload_file).read_text(encoding='utf-8-sig')
                             if args.payload_file else args.payload)
        result = dispatch_routed(args.db, args.role, args.operation, payload,
                                 requested=args.route)
        protocol_stdout.write(json.dumps(result, indent=2) + '\n')
        protocol_stdout.flush()
    else:
        from session_bridge.server import run
        run(args.db, args.role, protocol_stdin, protocol_stdout)


if __name__ == '__main__':
    main()
