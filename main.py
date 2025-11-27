import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
from utils.db import (
    DatabaseError
)
from utils.exception_handler import (
    handle_http_exception,
    handle_validation_error,
    handle_type_error,
    handle_unexpected_exception,
    handle_database_error
)

from utils.api.rag import router as rag_router
from utils.ollama_rag import initialize_vectorstore

# 🔧 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ lifespan 이벤트 설정 (앱 수명 주기 동안 초기화 및 정리 작업 수행)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ✅ 앱 실행 전에 수행할 초기화 작업
    logger.info("🚀 앱 시작: DB Vectorstore 초기화 중...")
    initialize_vectorstore()
    yield  # 👈 여기서 FastAPI 앱이 실행됩니다 (요청 수신 가능 상태로 진입)

    # 🛑 앱 종료 직전에 실행할 정리 작업 (옵션)
    logger.info("🛑 앱 종료")

# ✅ FastAPI 앱 생성 및 설정
app = FastAPI(lifespan=lifespan)

# ✅ CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 보안상 실제 운영 시 도메인 지정 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 예외 핸들러 등록
app.add_exception_handler(HTTPException, handle_http_exception)
app.add_exception_handler(RequestValidationError, handle_validation_error)
app.add_exception_handler(TypeError, handle_type_error)
app.add_exception_handler(Exception, handle_unexpected_exception)
app.add_exception_handler(DatabaseError, handle_database_error)

# ✅ 실제 API 경로 등록 (분리된 router 사용)
app.include_router(rag_router)
