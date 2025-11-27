import os
import shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.config import Config
from utils.db_full_schema import get_full_db_schema, search_db_metadata, get_all_table_names

# -----------------------------------------------------------------
# 1. 모델 및 임베딩 초기화
# -----------------------------------------------------------------
llm = ChatOllama(
    model="gpt-oss:20b-cloud",
    temperature=0.1, 
    base_url="http://localhost:11434"
)

embeddings = HuggingFaceEmbeddings(model_name="C:\\Users\\User\\.cache\\huggingface\\hub\\models--intfloat--multilingual-e5-large\\snapshots\\0dc5580a448e4284468b8909bae50fa925907bc5")

# -----------------------------------------------------------------
# 2. 대화 기록 저장소 (In-Memory for Test)
# -----------------------------------------------------------------
store = {}

def get_session_history(session_id: str):
    """테스트용 메모리 저장소 (서버 재시작 시 초기화됨)"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# -----------------------------------------------------------------
# 3. 초기화 함수 (서버 시작 시 호출)
# -----------------------------------------------------------------
# utils/ollama_rag.py

def initialize_vectorstore():
    """앱 시작 시 벡터 스토어가 존재하는지 확인하고, 없으면 생성합니다."""
    print("🚀 [초기화] DB 스키마 벡터 스토어 확인 중...")
    
    try:
        # 1. 벡터 스토어 파일(index.faiss)이 이미 있는지 확인
        index_path = os.path.join(Config.SCHEMA_STORE_PATH, "index.faiss")

        if os.path.exists(Config.SCHEMA_STORE_PATH) and os.path.exists(index_path):
            print(f"✅ [초기화 스킵] 기존 벡터 스토어가 발견되었습니다. (경로: {Config.SCHEMA_STORE_PATH})")
            print("   (※ DB 변경 사항을 반영하려면 'data/schema_store' 폴더를 삭제 후 재시작하세요.)")
            return

        # 2. 없으면 생성 로직 수행
        print("⚡ 기존 스토어가 없습니다. DB 스키마 추출 및 벡터화를 시작합니다...")

        docs = get_full_db_schema()
        if not docs:
            print("⚠️ [초기화 주의] DB에서 추출된 객체가 없습니다.")
            return

        print(f"   - 추출된 DB 객체 수: {len(docs)}개")

        lc_docs = [Document(page_content=d["content"], metadata={"name": d["name"]}) for d in docs]
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = splitter.split_documents(lc_docs)
        
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(Config.SCHEMA_STORE_PATH)
        print("✅ [초기화 완료] DB 벡터화 및 저장 성공!")
        
    except Exception as e:
        print(f"❌ [초기화 실패] 오류 발생: {e}")

# -----------------------------------------------------------------
# 4. 벡터 리트리버 관리
# -----------------------------------------------------------------
def get_db_retriever():
    if os.path.exists(os.path.join(Config.SCHEMA_STORE_PATH, "index.faiss")):
        vectorstore = FAISS.load_local(Config.SCHEMA_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        return vectorstore.as_retriever(search_kwargs={"k": 10})
    else:
        print("⚠️ 벡터 스토어가 없습니다. (초기화 실패 가능성)")
        return None

# -----------------------------------------------------------------
# 5. 키워드 추출용 (의도 파악)
# -----------------------------------------------------------------
def extract_keyword(question: str):
    prompt = f"""
    당신은 DB 검색 쿼리 추출기입니다.
    질문: '{question}'
    
    1. 사용자가 특정 테이블명, 컬럼명, 또는 비즈니스 용어(예: MEP, 인사, 급여 등)를 찾고 있다면 그 '핵심 단어' 하나만 출력하세요.
    2. '테이블', '목록', '전체', '보여줘' 같은 일반적인 요청 단어는 무시하세요.
    3. 검색할 구체적 대상이 없다면 "FALSE"라고만 출력하세요.
    
    Output example:
    - "MEP 테이블 있어?" -> MEP
    - "전체 테이블 리스트 줘" -> FALSE
    - "사용자 정보 어디 있어?" -> 사용자
    """
    return llm.invoke(prompt).content.strip()

# -----------------------------------------------------------------
# 6. 통합 RAG 실행 함수 (Hybrid Search 적용)
# -----------------------------------------------------------------
def rag_with_history(question: str, session_id: str = "default"):
    
    retrieved_context = ""
    
    # 1. [우선순위 1] 키워드 추출 (의도 파악)
    # 질문에 특정 대상(MEP, USER 등)이 있는지 먼저 확인합니다.
    keyword = extract_keyword(question)
    
    # 2. 로직 분기 처리
    if keyword != "FALSE" and len(keyword) > 1:
        # (A) 구체적인 검색어가 있는 경우 (예: "MEP 테이블 찾아줘", "MEP가 포함된 전체 테이블")
        print(f"🔎 메타데이터 조건 검색 수행: '{keyword}'")
        meta_result = search_db_metadata(keyword)
        retrieved_context += f"\n[DB 메타데이터 검색 결과 (키워드: {keyword})]\n{meta_result}\n"
        
        # 필요하다면 여기서 벡터 검색도 병행 가능 (Hybrid)
        retriever = get_db_retriever()
        if retriever:
            docs = retriever.invoke(question)
            vec_result = "\n\n".join([f"--- {d.metadata.get('name')} ---\n{d.page_content}" for d in docs])
            retrieved_context += f"\n[관련 스키마 정보 (유사도 검색)]\n{vec_result}\n"

    elif any(x in question for x in ["전체 테이블", "모든 테이블", "테이블 목록", "테이블 리스트"]):
        # (B) 검색어는 없는데 '전체'를 달라고 한 경우 (예: "그냥 전체 테이블 다 보여줘")
        print("💡 조건 없는 전체 목록 조회 요청 감지")
        retrieved_context = f"[전체 테이블 목록]\n{get_all_table_names()}"
        
    else:
        # (C) 그 외 일반적인 질문 -> 벡터 검색만 수행
        print("📚 일반 RAG 검색 수행")
        retriever = get_db_retriever()
        if retriever:
            docs = retriever.invoke(question)
            vec_result = "\n\n".join([f"--- {d.metadata.get('name')} ---\n{d.page_content}" for d in docs])
            retrieved_context += f"\n[관련 스키마 정보 (유사도 검색)]\n{vec_result}\n"

    system_prompt = """너는 Oracle Database 전문가이자 데이터 분석가다.
                        아래 제공된 [참고 정보]와 [대화 기록]을 바탕으로 사용자의 질문에 답변하라.
                        - [DB 메타데이터 검색 결과]가 있다면 그 정보를 최우선으로 사용하여 테이블 이름을 정확히 답변하라.
                        - 질문이 특정 테이블이나 컬럼을 찾고 있다면, 정확한 이름을 제시하라.
                    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("system", "[참고 정보]\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    print(f"🔍 질문(Session={session_id}): {question}")
    
    final_context = retrieved_context if retrieved_context.strip() else "제공된 문서나 DB 정보가 없습니다. 일반적인 지식으로 답변합니다."
    
    answer = chain_with_history.invoke(
        {"question": question, "context": final_context},
        config={"configurable": {"session_id": session_id}}
    )
    
    return {
        "answer": answer
    }