# utils/config.py

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 1. Oracle 연결 정보
    ORA_USER: str
    ORA_PASSWORD: str
    ORA_DSN: str
    
    # 2. Oracle 11g 연결을 위한 Instant Client 경로 (필수)
    ORA_LIB_DIR: str

    # 3. 벡터 스토어 저장 경로
    SCHEMA_STORE_PATH: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

try:
    Config = Settings()
except Exception as e:
    print("❌ [설정 오류] .env 파일을 확인해주세요:", e)
    raise