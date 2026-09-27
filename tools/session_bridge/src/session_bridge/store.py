"""Cooperative local mailbox. Roles are routing labels, not authentication."""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

ROLES = {'codex', 'claude'}


def _checked_seconds(value, field, low, high):
    """A finite number in (low, high], without coercion.

    bool is excluded deliberately: it is an int subclass, so True would otherwise pass as
    one second and a caller's typo would silently become a policy.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{field} must be a number, got {value!r}')
    value = float(value)
    if value != value or value in (float('inf'), float('-inf')):
        raise ValueError(f'{field} must be finite, got {value!r}')
    if not low < value <= high:
        raise ValueError(f'{field} must be within ({low}, {high}], got {value!r}')
    return value


class Store:
    def __init__(self, path, role):
        if role not in ROLES:
            raise ValueError('role must be codex or claude')
        self.path, self.role = Path(path), role
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, sender TEXT NOT NULL, recipient TEXT NOT NULL,
                    body TEXT NOT NULL, created REAL NOT NULL, acknowledged REAL,
                    request_key TEXT NOT NULL, UNIQUE(sender, request_key));
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
                    scopes TEXT NOT NULL, status TEXT NOT NULL, note TEXT NOT NULL,
                    updated REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (
                    role TEXT NOT NULL, session_id TEXT NOT NULL, cwd TEXT NOT NULL,
                    last_seen REAL NOT NULL, PRIMARY KEY(role, session_id));
                CREATE TABLE IF NOT EXISTS notices (
                    message_id TEXT NOT NULL, session_id TEXT NOT NULL, shown REAL NOT NULL,
                    PRIMARY KEY(message_id, session_id));
            ''')

    @contextmanager
    def connect(self, write=False):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute('BEGIN IMMEDIATE')
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def send(self, recipient, body, request_key):
        if recipient not in ROLES or recipient == self.role:
            raise ValueError('recipient must be the other session role')
        if not isinstance(body, str) or not body.strip() or len(body) > 20000:
            raise ValueError('body must contain 1..20000 characters')
        if not isinstance(request_key, str) or not request_key or len(request_key) > 200:
            raise ValueError('request_key required, max 200 characters')
        with self.connect(True) as db:
            old = db.execute('SELECT * FROM messages WHERE sender=? AND request_key=?',
                             (self.role, request_key)).fetchone()
            if old:
                if old['recipient'] != recipient or old['body'] != body:
                    raise ValueError('request_key already used with different content')
                return dict(old)
            mid = str(uuid.uuid4())
            db.execute('INSERT INTO messages VALUES (?,?,?,?,?,NULL,?)',
                       (mid, self.role, recipient, body, time.time(), request_key))
            return dict(db.execute('SELECT * FROM messages WHERE id=?', (mid,)).fetchone())

    INBOX_WINDOW = 50
    # What a reader can actually afford is BYTES, not messages. Measured over 535 real
    # messages on 2026-09-27: median 614 bytes, p90 3,577, max 6,669 -- so fifty messages is
    # ~7.7k tokens of context typically and 57k in the worst case observed, for one read.
    # 16000 keeps a normal read near 4k tokens while staying under send()'s own 20000-char
    # cap, so a message that exceeds the whole budget remains constructible and therefore
    # has to be handled rather than assumed away. The count cap only stops pathological
    # row counts now; the budget is what usually binds on heavy traffic.
    INBOX_MAX_BYTES = 16000

    def pending_count(self):
        """How many unacknowledged messages this role actually has, window or no window.

        `inbox()` is bounded, and a bound that cannot be seen is indistinguishable from a
        quiet link. This is the number that makes truncation observable.
        """
        with self.connect() as db:
            return db.execute(
                'SELECT COUNT(*) FROM messages WHERE recipient=? AND acknowledged IS NULL',
                (self.role,)).fetchone()[0]

    def inbox(self, limit=None, max_bytes=None):
        """The NEWEST unacknowledged messages, returned oldest-first.

        Newest, not oldest, and the distinction is the whole point. This read `ORDER BY
        created LIMIT 50` until 2026-09-27, which meant that once fifty unacknowledged
        messages accumulated every later arrival fell outside the window: the reader saw a
        frozen snapshot of old mail and concluded the peer had stopped replying. It had not.
        A backlog silently disabled the link, and `wait()` -- which reads through here --
        could not wake on a message it was unable to see.

        Newest-bias trades one blind spot for a smaller one: old mail leaves the window
        instead. That direction is self-correcting, because acknowledging what you can see
        brings the rest back, whereas the old direction got worse the longer it ran.
        Bounded twice, because a count alone bounds the wrong quantity: at most `limit`
        messages AND at most `max_bytes` of body. The newest message is always delivered
        even when it exceeds the whole budget by itself -- otherwise a legitimate message
        becomes permanently unreadable and the link wedges, which is worse than the defect
        above. Trimming for the budget therefore comes off the OLD end only.

        Callers that need to know the window bit should ask `pending_count()`.
        """
        window = self.INBOX_WINDOW if limit is None else int(limit)
        budget = self.INBOX_MAX_BYTES if max_bytes is None else int(max_bytes)
        if window < 1:
            raise ValueError('limit must be at least 1')
        if budget < 1:
            raise ValueError('max_bytes must be at least 1')
        with self.connect() as db:
            newest = [dict(r) for r in db.execute(
                'SELECT * FROM messages WHERE recipient=? AND acknowledged IS NULL '
                'ORDER BY created DESC, id DESC LIMIT ?',
                (self.role, window))]
        kept = []
        billed = 0
        for row in newest:
            cost = len(row['body'].encode('utf-8'))
            if kept and billed + cost > budget:
                break
            kept.append(row)
            billed += cost
        # Selected newest-first so both bounds keep the right end; presented oldest-first
        # because reading order carries the thread of the conversation.
        return list(reversed(kept))

    MAX_WAIT_SECS = 900.0
    MAX_POLL_SECS = 30.0

    def wait(self, timeout_secs=300.0, poll_secs=1.0):
        """Block until mail arrives for this role, then return it. Long-poll, not a loop.

        `inbox()` answers "is there anything?" and costs a round trip every time it says no.
        A peer with nothing to do paid for each of those answers. This spends wall clock
        inside one call instead, so waiting is free and only arrival costs anything.

        Three properties are load-bearing:
          * it checks BEFORE sleeping, so a deliverable message is never delayed;
          * it holds no write transaction between checks, or the sender would block behind
            the very reader waiting for them;
          * a timeout is reported AS a timeout. An empty list cannot distinguish "quiet
            link" from "gave up", and the caller acts differently on each.

        Returns {status: 'messages'|'timeout', messages, waited_secs, timeout_secs}.
        Like notice(), reading is not acknowledging -- the caller still owes an ack().
        """
        timeout_secs = _checked_seconds(timeout_secs, 'timeout_secs', 0.0, self.MAX_WAIT_SECS)
        poll_secs = _checked_seconds(poll_secs, 'poll_secs', 0.0, self.MAX_POLL_SECS)
        if poll_secs > timeout_secs:
            raise ValueError('poll_secs cannot exceed timeout_secs')

        started = time.monotonic()
        while True:
            pending = self.inbox()
            if pending:
                total = self.pending_count()
                return {'status': 'messages', 'messages': pending,
                        'pending_total': total,
                        'truncated': max(0, total - len(pending)),
                        'waited_secs': round(time.monotonic() - started, 3),
                        'timeout_secs': timeout_secs}
            remaining = timeout_secs - (time.monotonic() - started)
            if remaining <= 0:
                return {'status': 'timeout', 'messages': [],
                        'pending_total': 0, 'truncated': 0,
                        'waited_secs': round(time.monotonic() - started, 3),
                        'timeout_secs': timeout_secs}
            # Never sleep past the deadline: a poll interval wider than the remaining budget
            # would overshoot and turn a short timeout into a long stall.
            time.sleep(min(poll_secs, remaining))

    def ack(self, message_id):
        with self.connect(True) as db:
            row = db.execute('SELECT * FROM messages WHERE id=? AND recipient=?',
                             (message_id, self.role)).fetchone()
            if not row:
                raise ValueError('message not found in your inbox')
            db.execute('UPDATE messages SET acknowledged=COALESCE(acknowledged,?) WHERE id=?',
                       (time.time(), message_id))
            return dict(db.execute('SELECT * FROM messages WHERE id=?', (message_id,)).fetchone())

    @staticmethod
    def normalize_scopes(scopes):
        result = []
        for value in scopes:
            value = value.replace('\\', '/').rstrip('/').lower()
            if not value or value.startswith('/') or ':' in value or any(c in value for c in '*?[]'):
                raise ValueError('scope must be an explicit repo-relative file or directory')
            if '..' in PurePosixPath(value).parts or value == '.':
                raise ValueError('scope cannot escape or claim the entire project')
            result.append(str(PurePosixPath(value)))
        return sorted(set(result))

    def claim(self, task_id, title, scopes):
        if not task_id or len(task_id) > 200 or not title or len(title) > 1000:
            raise ValueError('task_id and title required, max 200 and 1000 characters')
        scopes = self.normalize_scopes(scopes)
        with self.connect(True) as db:
            old = db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
            if old:
                if old['owner'] == self.role and old['title'] == title and json.loads(old['scopes']) == scopes:
                    return self.task_dict(old)
                raise ValueError('task already claimed; use update_task or a new task ID')
            for task in db.execute("SELECT * FROM tasks WHERE status != 'done' AND owner != ?", (self.role,)):
                for a in scopes:
                    for b in json.loads(task['scopes']):
                        if a == b or a.startswith(b + '/') or b.startswith(a + '/'):
                            raise ValueError(f"scope overlaps {task['id']} owned by {task['owner']}: {b}")
            db.execute('INSERT INTO tasks VALUES (?,?,?,?,?,?,?)',
                       (task_id, self.role, title, json.dumps(scopes), 'active', '', time.time()))
            return self.task_dict(db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone())

    @staticmethod
    def task_dict(row):
        result = dict(row)
        result['scopes'] = json.loads(result['scopes'])
        return result

    def update_task(self, task_id, status, note):
        if status not in {'active', 'blocked', 'done'} or len(note) > 20000:
            raise ValueError('status must be active, blocked, or done; note max 20000 characters')
        with self.connect(True) as db:
            old = db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
            if not old or old['owner'] != self.role:
                raise ValueError('only task owner can update it')
            if old['status'] == 'done' and status != 'done':
                raise ValueError('completed tasks cannot be reopened; claim a new task')
            db.execute('UPDATE tasks SET status=?,note=?,updated=? WHERE id=?',
                       (status, note, time.time(), task_id))
            return self.task_dict(db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone())

    def status(self):
        with self.connect() as db:
            return {'role': self.role, 'database': str(self.path),
                    'tasks': [self.task_dict(r) for r in db.execute('SELECT * FROM tasks ORDER BY updated')],
                    'sessions': [dict(r) for r in db.execute('SELECT * FROM sessions ORDER BY last_seen DESC')],
                    'pending': [dict(r) for r in db.execute(
                        'SELECT sender,recipient,COUNT(*) AS count FROM messages WHERE acknowledged IS NULL GROUP BY sender,recipient')]}

    def notice(self, session_id, cwd):
        """An actual hook heartbeat. Showing a notice does not acknowledge it."""
        now = time.time()
        with self.connect(True) as db:
            db.execute('INSERT INTO sessions VALUES (?,?,?,?) ON CONFLICT(role,session_id) DO UPDATE SET cwd=excluded.cwd,last_seen=excluded.last_seen',
                       (self.role, session_id, cwd, now))
            rows = list(db.execute('''SELECT m.* FROM messages m LEFT JOIN notices n
                ON m.id=n.message_id AND n.session_id=? WHERE m.recipient=?
                AND m.acknowledged IS NULL AND (n.shown IS NULL OR n.shown < ?)
                ORDER BY m.created LIMIT 5''', (session_id, self.role, now - 300)))
            for r in rows:
                db.execute('INSERT INTO notices VALUES (?,?,?) ON CONFLICT(message_id,session_id) DO UPDATE SET shown=excluded.shown',
                           (r['id'], session_id, now))
            return [dict(r) for r in rows]
