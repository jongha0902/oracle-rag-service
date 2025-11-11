# utils/api/rag.py

from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
import io
import zipfile

# ----------------------------------------------------
# 👇 RAG 관련 함수
# ----------------------------------------------------
from utils.ollama_rag import rag_with_ollama, rag_with_context, ask_llm_only

# ----------------------------------------------------
# 👇 다양한 파일 파싱을 위한 라이브러리 임포트
# ----------------------------------------------------
try:
    import pandas as pd
except ImportError:
    pd = None
    logging.warning("pandas가 설치되지 않았습니다. 엑셀 파일 처리가 비활성화됩니다.")

try:
    import openpyxl
except ImportError:
    openpyxl = None
    logging.warning("openpyxl이 설치되지 않았습니다. .xlsx 파일 처리가 비활성화됩니다.")

try:
    import xlrd
except ImportError:
    xlrd = None
    logging.warning("xlrd가 설치되지 않았습니다. .xls 파일 처리가 비활성화됩니다.")

try:
    from PyPDF2 import PdfReader
    from PyPDF2.errors import FileNotDecryptedError
except ImportError:
    PdfReader = None
    FileNotDecryptedError = None
    logging.warning("PyPDF2가 설치되지 않았습니다. .pdf 파일 처리가 비활성화됩니다.")

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    logging.warning("BeautifulSoup4가 설치되지 않았습니다. .xml, .jsp, .html 파일 처리가 비활성화됩니다.")
# ----------------------------------------------------


router = APIRouter()
logger = logging.getLogger(__name__)
# ----------------------------------------------------


# ----------------------------------------------------
# 👇 파일 확장자별 텍스트 추출 헬퍼 함수
# ----------------------------------------------------
async def read_file_content(f: UploadFile) -> str:
    """
    업로드된 파일(UploadFile)을 받아, 확장자에 맞는 파서를 사용해 텍스트를 추출합니다.
    """
    filename = f.filename.lower()
    content_bytes = await f.read()

    try:
        # 1️⃣ XLSX (엑셀)
        if filename.endswith('.xlsx'):
            if not pd or not openpyxl:
                raise ImportError("pandas/openpyxl이 설치되지 않아 .xlsx 파일을 읽을 수 없습니다.")
            try:
                with pd.ExcelFile(io.BytesIO(content_bytes), engine='openpyxl') as xls:
                    sheets = []
                    for sheet_name in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name=sheet_name)
                        sheet_text = f"--- 시트: {sheet_name} ---\n{df.to_string(index=False)}"
                        sheets.append(sheet_text)
                    return "\n\n".join(sheets)
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail=f"'{f.filename}'은 손상되었거나 암호화된 .xlsx 파일입니다.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f".xlsx 파일 처리 중 오류: {e}")

        # 2️⃣ XLS
        elif filename.endswith('.xls'):
            if not pd or not xlrd:
                raise ImportError("pandas/xlrd가 설치되지 않아 .xls 파일을 읽을 수 없습니다.")
            try:
                with pd.ExcelFile(io.BytesIO(content_bytes), engine='xlrd') as xls:
                    sheets = []
                    for sheet_name in xls.sheet_names:
                        df = pd.read_excel(xls, sheet_name=sheet_name)
                        sheets.append(f"--- 시트: {sheet_name} ---\n{df.to_string(index=False)}")
                    return "\n\n".join(sheets)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f".xls 파일 처리 중 오류: {e}")

        # 3️⃣ HTML/XML/JSP
        elif filename.endswith(('.xml', '.jsp', '.html')):
            if not BeautifulSoup:
                raise ImportError("BeautifulSoup4가 설치되지 않아 .xml/.jsp/.html 파일을 읽을 수 없습니다.")
            try:
                try:
                    text = content_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    text = content_bytes.decode('cp949')
                soup = BeautifulSoup(text, 'lxml')
                return soup.get_text(separator="\n", strip=True)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"HTML/XML 파일 처리 중 오류: {e}")

        # 4️⃣ PDF
        elif filename.endswith('.pdf'):
            if not PdfReader:
                raise ImportError("PyPDF2가 설치되지 않아 .pdf 파일을 읽을 수 없습니다.")
            try:
                reader = PdfReader(io.BytesIO(content_bytes))
                if reader.is_encrypted:
                    raise HTTPException(status_code=400, detail=f"'{f.filename}'은 암호화된 PDF입니다.")
                pdf_texts = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(pdf_texts)
            except FileNotDecryptedError:
                raise HTTPException(status_code=400, detail=f"'{f.filename}'은 암호화된 PDF입니다.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"PDF 처리 중 오류: {e}")

        # 5️⃣ 기본 텍스트 파일
        else:
            try:
                return content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                return content_bytes.decode('cp949')

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"파일 파싱 중 오류 ({filename}): {e}")
        raise HTTPException(status_code=400, detail=f"'{f.filename}' 처리 중 오류 발생: {e}")


# ----------------------------------------------------
# 👇 메인 RAG 엔드포인트 (수정 없음)
# ----------------------------------------------------
@router.post("/ask")
async def ask_question(
    query: str = Form(...),
    type: str = Form(...),
    file: Optional[List[UploadFile]] = File(None)
):
    question = query.strip()
    query_type = type.strip()

    try:
        # 🧠 타입 1: 전력거래 RAG
        if query_type == "1":
            if not question:
                raise HTTPException(status_code=400, detail="타입 1은 질문이 필수입니다.")
            if file:
                logger.warning("타입 1은 파일을 지원하지 않습니다. 파일이 무시됩니다.")
            response = rag_with_ollama(question, query_type="1")

        # 📂 타입 2: 파일 RAG or LLM Only
        elif query_type == "2":
            if file and len(file) > 0:
                logger.info(f"📬 타입 2 (파일 RAG): {len(file)}개 파일 수신됨")
                file_contents = []
                for f in file:
                    try:
                        extracted = await read_file_content(f)
                        file_contents.append(extracted)
                        logger.info(f" - 파일 처리 완료: {f.filename} ({len(extracted)}자)")
                    except HTTPException as he:
                        raise he
                combined_context = "\n\n".join(file_contents)
                response = rag_with_context(question, combined_context)
            else:
                if not question:
                    raise HTTPException(status_code=400, detail="파일이 없을 때는 질문이 필수입니다.")
                logger.info("📬 타입 2 (LLM Only): 파일 없음 → LLM 직접 호출")
                response = ask_llm_only(question)

        else:
            raise HTTPException(status_code=400, detail=f"유효하지 않은 프롬프트 타입({query_type})입니다.")

        return JSONResponse(status_code=200, content={"answer": response})

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"서버 내부 오류: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")