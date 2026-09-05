from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from .models import Message, SessionState

class SQLiteStore:
    def __init__(self, path: str = "agent.db") -> None:
        self.path = path
        self._init()

    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    messages_json TEXT NOT NULL DEFAULT '[]',
                    iteration_count INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (user_id, session_id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS analogy_boards (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    situation TEXT NOT NULL DEFAULT '',
                    frame_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (user_id, session_id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS analogy_candidates (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    case_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    mapping TEXT NOT NULL DEFAULT '',
                    analogy_break TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (user_id, session_id, case_id)
                )
            """)

    def load_session(self, user_id: str, session_id: str) -> SessionState:
        with self._connect() as c:
            row = c.execute(
                "SELECT * FROM sessions WHERE user_id=? AND session_id=?",
                (user_id, session_id),
            ).fetchone()
        if not row:
            state = SessionState.new(user_id, session_id)
            self.save_session(state)
            return state
        messages = [Message(**m) for m in json.loads(row["messages_json"])]
        return SessionState(
            user_id=row["user_id"], session_id=row["session_id"],
            summary=row["summary"], messages=messages,
            iteration_count=row["iteration_count"], updated_at=row["updated_at"]
        )

    def save_session(self, state: SessionState) -> None:
        payload = json.dumps([asdict(m) for m in state.messages], ensure_ascii=False)
        with self._connect() as c:
            c.execute("""
                INSERT INTO sessions(user_id,session_id,summary,messages_json,iteration_count,updated_at)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(user_id,session_id) DO UPDATE SET
                  summary=excluded.summary,
                  messages_json=excluded.messages_json,
                  iteration_count=excluded.iteration_count,
                  updated_at=excluded.updated_at
            """, (state.user_id, state.session_id, state.summary, payload,
                  state.iteration_count, state.updated_at))

    def set_analogy_frame(self, user_id: str, session_id: str, situation: str, frame: dict) -> None:
        import time
        with self._connect() as c:
            c.execute("""
                INSERT INTO analogy_boards(user_id,session_id,situation,frame_json,updated_at)
                VALUES(?,?,?,?,?)
                ON CONFLICT(user_id,session_id) DO UPDATE SET
                  situation=excluded.situation,
                  frame_json=excluded.frame_json,
                  updated_at=excluded.updated_at
            """, (user_id, session_id, situation, json.dumps(frame, ensure_ascii=False), time.time()))

    def record_analogy_candidate(
        self,
        user_id: str,
        session_id: str,
        *,
        case_id: str,
        status: str,
        reason: str,
        mapping: str,
        analogy_break: str,
    ) -> None:
        import time
        with self._connect() as c:
            c.execute("""
                INSERT INTO analogy_candidates(
                    user_id,session_id,case_id,status,reason,mapping,analogy_break,updated_at
                ) VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(user_id,session_id,case_id) DO UPDATE SET
                  status=excluded.status,
                  reason=excluded.reason,
                  mapping=excluded.mapping,
                  analogy_break=excluded.analogy_break,
                  updated_at=excluded.updated_at
            """, (
                user_id, session_id, case_id, status, reason, mapping, analogy_break, time.time()
            ))

    def get_analogy_board(self, user_id: str, session_id: str) -> dict:
        with self._connect() as c:
            board = c.execute(
                "SELECT situation,frame_json,updated_at FROM analogy_boards WHERE user_id=? AND session_id=?",
                (user_id, session_id),
            ).fetchone()
            rows = c.execute(
                """
                SELECT case_id,status,reason,mapping,analogy_break,updated_at
                FROM analogy_candidates
                WHERE user_id=? AND session_id=?
                ORDER BY updated_at, case_id
                """,
                (user_id, session_id),
            ).fetchall()
        return {
            "situation": board["situation"] if board else "",
            "frame": json.loads(board["frame_json"]) if board else {},
            "candidates": [
                {
                    "case_id": row["case_id"],
                    "status": row["status"],
                    "reason": row["reason"],
                    "mapping": row["mapping"],
                    "analogy_break": row["analogy_break"],
                }
                for row in rows
            ],
        }
