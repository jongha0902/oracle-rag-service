# utils/api/rag.py

from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import JSONResponse
# 👇 3개의 함수를 모두 임포트합니다.
from utils.ollama_rag import rag_with_ollama, rag_with_context, ask_llm_only
from typing import List, Optional
import logging
import io # 👈 파일 스트림 처리를 위해 임포트
import zipfile # 👈 [신규] BadZipFile 오류를 잡기 위해 임포트

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
# 👇 [신규] 파일 확장자별 텍스트 추출 헬퍼 함수
# ----------------------------------------------------
async def read_file_content(f: UploadFile) -> str:
    """
    업로드된 파일(UploadFile)을 받아, 확장자에 맞는 파서를 사용해 텍스트를 추출합니다.
    """
    filename = f.filename.lower()
    content_bytes = await f.read()
    
    try:
        # 1. 엑셀 (.xlsx) - [수정됨]
        if filename.endswith('.xlsx'):
            if not pd or not openpyxl:
                raise ImportError("pandas/openpyxl 라이브러리가 설치되지 않아 .xlsx 파일을 읽을 수 없습니다.")
            
            try:
                # --- 1차 시도: .xlsx (openpyxl)로 열기 ---
                xls_file = io.BytesIO(content_bytes)
                xls = pd.ExcelFile(xls_file, engine='openpyxl')
                
                all_sheets_text = []
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    sheet_text = f"--- 시트: {sheet_name} ---\n{df.to_string()}"
                    all_sheets_text.append(sheet_text)
                return "\n\n".join(all_sheets_text)

            # 👇 [핵심 수정] BadZipFile 오류는 암호화 또는 손상된 파일임
            except zipfile.BadZipFile:
                logger.warning(f"파일 '{f.filename}'은(는) .xlsx(zip) 형식이 아닙니다. 암호화되었거나 손상되었을 수 있습니다.")
                # 'xlrd'로 fallback하지 않고 바로 오류 반환
                raise HTTPException(
                    status_code=400,
                    detail=f"파일 '{f.filename}'이(가) 암호화되었거나, 손상되었거나, 유효한 .xlsx 파일이 아닙니다."
                )
                
            except Exception as e_openpyxl:
                # (openpyxl의 다른 오류들 - 예: 암호화)
                if "encrypted" in str(e_openpyxl).lower():
                     raise HTTPException(
                        status_code=400, 
                        detail=f"파일 '{f.filename}'이(가) 암호화된 엑셀 파일일 수 있습니다."
                    )
                raise e_openpyxl # 그 외의 openpyxl 오류

        # 2. 엑셀 (.xls)
        elif filename.endswith('.xls'):
            if not pd or not xlrd:
                raise ImportError("pandas/xlrd 라이브러리가 설치되지 않아 .xls 파일을 읽을 수 없습니다.")
            
            try:
                xls_file = io.BytesIO(content_bytes)
                xls = pd.ExcelFile(xls_file, engine='xlrd')
                all_sheets_text = []
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    sheet_text = f"--- 시트: {sheet_name} ---\n{df.to_string()}"
                    all_sheets_text.append(sheet_text)
                return "\n\n".join(all_sheets_text)
            except Exception as e:
                # 👈 .xls 암호화는 "Can't find workbook" 오류 등을 발생시킴
                if "workbook" in str(e).lower() or "encrypted" in str(e).lower():
                    raise HTTPException(
                        status_code=400, 
                        detail=f"파일 '{f.filename}'이(가) 암호화되었거나 유효한 .xls 파일이 아닙니다."
                    )
                raise e

        # 3. XML, JSP, HTML (.xml, .jsp, .html)
        elif filename.endswith(('.xml', '.jsp', '.html')):
            if not BeautifulSoup:
                 raise ImportError("BeautifulSoup4 라이브러리가 설치되지 않아 .xml/.jsp 파일을 읽을 수 없습니다.")
            
            try:
                text = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                text = content_bytes.decode('cp949') # 한글 Windows 기본 인코딩
            
            soup = BeautifulSoup(text, 'lxml')
            return soup.get_text(separator="\n", strip=True)

        # 4. PDF (.pdf)
        elif filename.endswith('.pdf'):
            if not PdfReader:
                raise ImportError("PyPDF2가 설치되지 않아 .pdf 파일을 읽을 수 없습니다.")

            pdf_file = io.BytesIO(content_bytes)
            reader = PdfReader(pdf_file)

            if reader.is_encrypted:
                raise HTTPException(
                    status_code=400, 
                    detail=f"파일 '{f.filename}'은(는) 암호화되어 있어 읽을 수 없습니다. 암호를 해제한 후 다시 업로드해주세요."
                )

            pdf_text = []
            for page in reader.pages:
                page_content = page.extract_text()
                if page_content:
                    pdf_text.append(page_content)
            return "\n\n".join(pdf_text)
            
        # 5. 기타 (기본 텍스트 파일 .txt 등)
        else:
            try:
                return content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                return content_bytes.decode('cp949')

    except FileNotDecryptedError:
        raise HTTPException(
            status_code=400, 
            detail=f"파일 '{f.filename}'은(는) 암호화되어 있어 읽을 수 없습니다. 암호를 해제한 후 다시 업로드해주세요."
        )
    except Exception as e:
        logger.error(f"파일 파싱 중 오류 발생 ({filename}): {e}", exc_info=True)
        # 파싱 실패 시 사용자에게 오류 전달
        raise HTTPException(
            status_code=400, 
            detail=f"{e}"
        )


# ----------------------------------------------------
# 👇 [수정 없음] 챗봇 메인 엔드포인트
# ----------------------------------------------------
@router.post("/ask")
async def ask_question(
    query: str = Form(...),
    type: str = Form(...),
    file: Optional[List[UploadFile]] = File(None)
):
    
    question = query.strip()
    query_type = type.strip()
    
    rag_response = None

    try:
        if query_type == "1":
            # --- (타입 1 로직은 동일) ---
            if not question:
                raise HTTPException(status_code=400, detail="타입 1은 질문(query)이 필수입니다.")
            if file and len(file) > 0:
                logger.warning("타입 1은 파일을 지원하지 않습니다. (파일 무시됨)")
            rag_response = rag_with_ollama(question, query_type="1") 

        elif query_type == "2":
            # --- (타입 2 로직) ---
            if file and len(file) > 0:
                # --- A. 파일이 "있는" 경우 (파일 RAG) ---
                file_contents = []
                logger.info(f"📬 타입 2 (파일 RAG): {len(file)}개의 파일 수신")
                
                for f in file:
                    logger.info(f" - 파일 읽기 시작: {f.filename}")
                    
                    try:
                        extracted_text = await read_file_content(f)
                        file_contents.append(extracted_text)
                        logger.info(f" - 파일 읽기 완료: {f.filename} (추출된 텍스트 {len(extracted_text)}자)")
                    except HTTPException as he:
                        raise he 
                
                combined_context = "\n\n".join(file_contents)

                rag_response = rag_with_context(question, combined_context)
            
            else:
                # --- B. 파일이 "없는" 경우 (LLM 직접 호출) ---
                if not question: 
                     raise HTTPException(status_code=400, detail="타입 2에서 파일이 없는 경우, 질문(query)은 필수입니다.")
                
                logger.info("📬 타입 2 (LLM Only): 파일 없음. LLM으로 직접 질문합니다.")
                rag_response = ask_llm_only(question)
            
        else:
            raise HTTPException(status_code=400, detail=f"유효하지 않은 프롬프트 타입({query_type})입니다.")

        # 최종 응답 반환
        return JSONResponse(status_code=200, content={"answer": rag_response})

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"처리 중 예외 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")