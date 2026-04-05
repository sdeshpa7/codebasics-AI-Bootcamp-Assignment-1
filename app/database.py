"""
app/database.py
Persistent storage for FinSolve logging and chat history.
"""
import sqlite3
import threading
from contextlib import contextmanager
from app.config import settings
import json
import os

_db_lock = threading.Lock()

def init_db():
    try:
        db_path = settings.sqlite_db_path
    except AttributeError:
        # Fallback if config is outdated
        db_path = "data/finsolve.db"

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with _db_lock:
        with sqlite3.connect(db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    llm_response TEXT NOT NULL,
                    user_roles TEXT NOT NULL,
                    employee_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    violation_type TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

@contextmanager
def get_connection():
    try:
        db_path = settings.sqlite_db_path
    except AttributeError:
        db_path = "data/finsolve.db"
        
    with _db_lock:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()

def log_chat(session_id: str, query: str, response: str, roles: list[str], employee_id: str = "Unknown"):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_history (session_id, user_query, llm_response, user_roles, employee_id) VALUES (?, ?, ?, ?, ?)",
            (session_id, query, response, json.dumps(roles), employee_id)
        )

def log_violation(violation_type: str, reason: str, session_id: str = None):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO audit_logs (violation_type, reason, session_id) VALUES (?, ?, ?)",
            (violation_type, reason, session_id)
        )
