from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
import logging

from utils.ollama_rag import rag_with_history

router = APIRouter()
logger = logging.getLogger(__name__)

# ----------------------------------------------------
# 👇 메인 엔드포인트
# ----------------------------------------------------
@router.post("/ask")
async def ask_question(
    query: str = Form(...),
    session_id: str = Form("default_session")
):
    try:
        answer = rag_with_history(
            question=query,
            session_id=session_id
        )
        
        return JSONResponse(status_code=200, content={"answer": answer})

    except Exception as e:
        logger.exception("API Error")
        raise HTTPException(status_code=500, detail=str(e))