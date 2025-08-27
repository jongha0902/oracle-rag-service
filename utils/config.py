import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    DB_PATH = os.getenv("DB_PATH", "C:/sqlite3/ets_api.db")
    API_SALT = os.getenv("API_SALT", "ets-ai-secret-api-salt")