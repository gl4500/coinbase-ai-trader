"""Watch the mailbox and queue bounded check-ins to an existing Codex thread.

No automatic acknowledgements; receipts are written only by the assistant.
This process does not wake Claude. Claude needs its own in-session timer.
"""
import argparse
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / '.coordination'


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def decision(state, receipt, now):
    if not state.get('token'):
        return 'queue'
    if receipt and receipt.get('token') == state['token']:
        return 'queue' if now - receipt['at'] >= 120 else 'responded'
    return 'stale' if now - state['queued_at'] > 180 else 'waiting'


def mailbox(now):
    with sqlite3.connect(f'{(STATE / "session-link.sqlite3").as_uri()}?mode=ro', uri=True, timeout=5) as db:
        return {
            role: {
                'pending': db.execute('SELECT COUNT(*) FROM messages WHERE recipient=? AND acknowledged IS NULL', (role,)).fetchone()[0],
                'last_ack_at': db.execute('SELECT MAX(acknowledged) FROM messages WHERE recipient=?', (role,)).fetchone()[0],
                'oldest_pending_age_seconds': db.execute('SELECT ? - MIN(created) FROM messages WHERE recipient=? AND acknowledged IS NULL', (now, role)).fetchone()[0],
            } for role in ('codex', 'claude')
        }


def run(thread):
    # OS-owned loopback lock disappears on exit/crash, unlike stale PID files.
    lock = socket.socket()
    if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
        lock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    lock.bind(('127.0.0.1', 47839))
    executable = shutil.which('codex')
    if not executable:
        raise RuntimeError('codex CLI not found')
    stop = STATE / 'ping-monitor.stop'
    state_path = STATE / 'ping-cycle.json'
    receipt_path = STATE / 'ping-receipt.json'
    state = read(state_path)
    if state.get('thread') != thread:
        state = {}
    while not stop.exists():
        now = time.time()
        status = {'pid': os.getpid(), 'thread': thread, 'monitor_checked_at': now,
                  'poll_seconds': 30, 'cycle_seconds': 120, 'stale_seconds': 180,
                  'claude_wake': 'requires Claude in-session timer'}
        try:
            status['mailbox'] = mailbox(now)
            action = decision(state, read(receipt_path), now)
            status['assistant_cycle'] = action
            if action == 'queue':
                token = str(uuid.uuid4())
                state = {'thread': thread, 'token': token, 'queued_at': now}
                # Persist BEFORE queueing: an ambiguous failure must not flood the thread.
                write(state_path, state)
                prompt = (
                    f'Regular session-link cycle {token} (user requested). Read Claude inbox via '
                    f'{ROOT}/tools/session_bridge/bridge.py --role codex call inbox; process and ACK '
                    'messages, reply to requests, and continue existing repair/review work. '
                    'After checking, record actual assistant receipt by running '
                    f'{ROOT}/.coordination-runtime/Scripts/python.exe '
                    f'{ROOT}/tools/session_bridge/ping_monitor.py receipt --token {token}. '
                    'Do not create other sessions. If user has stopped coordination, run monitor stop.'
                )
                result = subprocess.run([executable, 'queue', '--thread', thread, '--message', prompt],
                                        capture_output=True, text=True, timeout=20, cwd=ROOT)
                status['queue_result'] = {'returncode': result.returncode,
                                          'output': (result.stdout + result.stderr)[-2000:]}
                if result.returncode:
                    status['assistant_cycle'] = 'queue_failed'
                else:
                    status['assistant_cycle'] = 'queued_awaiting_receipt'
            status['challenge'] = state
            status['receipt'] = read(receipt_path)
        except Exception as exc:
            status['error'] = f'{type(exc).__name__}: {exc}'
        write(STATE / 'ping-status.json', status)
        # A stop request is observed within one second.
        for _ in range(30):
            if stop.exists():
                break
            time.sleep(1)
    write(STATE / 'ping-status.json', {'pid': os.getpid(), 'stopped_at': time.time()})
    lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'receipt', 'status', 'stop'])
    parser.add_argument('--thread')
    parser.add_argument('--token')
    args = parser.parse_args()
    STATE.mkdir(exist_ok=True)
    if args.action == 'run':
        if not args.thread:
            parser.error('--thread required')
        run(args.thread)
    elif args.action == 'receipt':
        state = read(STATE / 'ping-cycle.json')
        if not args.token or state.get('token') != args.token:
            parser.error('receipt must match current challenge token')
        write(STATE / 'ping-receipt.json', {'token': args.token, 'at': time.time()})
    elif args.action == 'stop':
        (STATE / 'ping-monitor.stop').touch()
    else:
        status = read(STATE / 'ping-status.json')
        status['monitor_stale'] = time.time() - status.get('monitor_checked_at', 0) > 65
        print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
