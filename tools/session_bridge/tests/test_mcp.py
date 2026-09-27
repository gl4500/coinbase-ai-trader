"""Real two-process MCP transport test; never writes the operational mailbox."""
import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

LAUNCHER = Path(__file__).resolve().parents[1] / 'bridge.py'


class MCPTests(unittest.TestCase):
    def test_round_trip_and_hook(self):
        with tempfile.TemporaryDirectory() as temp:
            db = str(Path(temp) / 'test.sqlite3')
            with open(Path(temp) / 'server.log', 'w', encoding='utf-8') as log:
                asyncio.run(asyncio.wait_for(self.exchange(db, log), 90))

    async def exchange(self, db, log):
        def params(role):
            return StdioServerParameters(command=sys.executable, args=[str(LAUNCHER), '--role', role, '--db', db, 'mcp'])
        async with stdio_client(params('codex'), errlog=log) as (cr, cw):
            async with ClientSession(cr, cw) as codex:
                await codex.initialize()
                self.assertEqual(len((await codex.list_tools()).tools), 7)  # +await_message (long-poll)
                sent = await codex.call_tool('send_message', {'recipient': 'claude', 'body': 'integration ping', 'request_key': 'test-ping'})
                self.assertFalse(sent.isError, sent)
                mid = json.loads(sent.content[0].text)['id']
                hook = subprocess.run([sys.executable, str(LAUNCHER), '--role', 'claude', '--db', db, 'hook'],
                    input=json.dumps({'hook_event_name': 'PostToolUse', 'session_id': 'test-only', 'cwd': 'temporary'}),
                    text=True, capture_output=True, timeout=10, check=True)
                context = json.loads(hook.stdout)['hookSpecificOutput']['additionalContext']
                self.assertIn('integration ping', context)
                async with stdio_client(params('claude'), errlog=log) as (rr, rw):
                    async with ClientSession(rr, rw) as claude:
                        await claude.initialize()
                        inbox = await claude.call_tool('read_inbox', {})
                        self.assertFalse(inbox.isError, inbox)
                        self.assertIn(mid, str(inbox))
                        ack = await claude.call_tool('acknowledge_message', {'message_id': mid})
                        self.assertFalse(ack.isError, ack)
                        reply = await claude.call_tool('send_message', {'recipient': 'codex', 'body': 'integration pong', 'request_key': 'test-pong'})
                        self.assertFalse(reply.isError, reply)
                        received = await codex.call_tool('read_inbox', {})
                        self.assertIn('integration pong', str(received))
                        bad = await claude.call_tool('acknowledge_message', {'message_id': json.loads(reply.content[0].text)['id']})
                        self.assertTrue(bad.isError)


if __name__ == '__main__':
    unittest.main()
