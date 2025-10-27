from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from utils.ollama_rag import rag_with_ollama

router = APIRouter()

# 💬 질문 응답 (POST 방식)
@router.post("/ask")
async def ask_question(
    request: Request
):
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="요청 본문이 유효한 JSON이 아닙니다.")

    question = data.get("query", "").strip()
    query_type = data.get("type", "").strip()

    if not question:
        res = {"message": "질문이 없습니다."}
        return JSONResponse(status_code=400, content=res)

    # ✅ 실제 응답 처리 (현재는 테스트 응답)
    answer = rag_with_ollama(question, query_type)
    answer = "API Gateway Test...."

    res = {"answer": answer}
    return JSONResponse(status_code=200, content=res)
