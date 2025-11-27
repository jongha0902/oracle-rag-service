import oracledb
import logging
from contextlib import contextmanager
from utils.config import Config

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    pass

# ✅ Oracle Instant Client 초기화 (Thick Mode)
# 11g와 통신하기 위해 필수입니다.
try:
    oracledb.init_oracle_client(lib_dir=Config.ORA_LIB_DIR)
    logger.info(f"✅ Oracle Client 초기화 완료: {Config.ORA_LIB_DIR}")
except Exception as e:
    logger.warning(f"⚠️ Oracle Client 초기화 주의 (이미 초기화되었거나 경로 오류): {e}")

@contextmanager
def get_conn():
    conn = None
    try:
        conn = oracledb.connect(
            user=Config.ORA_USER,
            password=Config.ORA_PASSWORD,
            dsn=Config.ORA_DSN
        )
        yield conn
        conn.commit()
    except oracledb.Error as e:
        if conn: conn.rollback()
        error_obj, = e.args
        logger.error(f"Oracle Error: {error_obj.message}")
        raise DatabaseError(f"[Oracle 오류] {error_obj.message}")
    except Exception as e:
        if conn: conn.rollback()
        logger.error(f"DB Error: {e}")
        raise DatabaseError(f"[DB 오류] {e}")
    finally:
        if conn:
            conn.close()