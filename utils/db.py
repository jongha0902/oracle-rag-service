import sqlite3
from contextlib import contextmanager
from utils.config import Config
 
DB_PATH = Config.DB_PATH

class DatabaseError(Exception):
    pass

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise DatabaseError(f"[DB 오류] {e}")
    finally:
        conn.close()