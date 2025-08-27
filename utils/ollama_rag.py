import os
import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 경로 설정
vectorstore_path = "/home/kecpuser/workspace/ollama_fastapi/app/data/vectorstore"
embedding_model_path = "/home/kecpuser/huggingface/hub/models--intfloat--multilingual-e5-large/snapshots/0dc5580a448e4284468b8909bae50fa925907bc5"
pdf_path = "/home/kecpuser/workspace/ollama_fastapi/app/data/전력시장운영규칙.pdf"
txt_path = "/home/kecpuser/workspace/ollama_fastapi/app/data/cleaned_text.txt"

vectorstore = None  # ✅ 전역 vectorstore 선언

# PDF → 텍스트 변환
def extract_text_from_pdf(path: str) -> str:
    text = ""
    with open(path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                cleaned = page_text.replace("-\n", "").replace("\n", " ").strip()
                text += cleaned + "\n"
    print(f"[PDF] 전체 페이지 수: {len(reader.pages)} / 추출 완료")

    return text

def save_text(text: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

# Vectorstore 구축/로딩
def create_vectorstore():
    global vectorstore
    
    if not os.path.exists(txt_path):
        print("📄 PDF에서 텍스트 추출 중...")
        extracted = extract_text_from_pdf(pdf_path)
        save_text(extracted, txt_path)
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600, chunk_overlap=300, separators=["\n\n", "\n", ".", " "]
    )

    documents = splitter.create_documents([raw_text])
    print(f"📑 문서 분할: 전체 {len(documents)}개")
    filtered_docs = [doc for doc in documents if len(doc.page_content.strip()) > 80]
    print(f"✅ 필터 후 문서: {len(filtered_docs)}개")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_path)

    if not os.path.exists(vectorstore_path):
        print("💾 FAISS 벡터스토어 생성 중...")
        vectorstore = FAISS.from_documents(filtered_docs, embeddings)
        vectorstore.save_local(vectorstore_path)
    else:
        print("📦 FAISS 벡터스토어 로드 중...")
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

    return vectorstore

def build_context_for_question(question: str, k=20, score_threshold=0.4):
    results = vectorstore.similarity_search_with_score(question, k=k)

    # score 오름차순 정렬
    results.sort(key=lambda x: x[1])

    # score 필터링
    filtered = [(doc, score) for doc, score in results if score <= score_threshold]

    if not filtered:
        return "관련 문서를 찾을 수 없습니다."

    context_parts = [
        f"[score={score:.4f}]\n{doc.page_content}"
        for doc, score in filtered
    ]

    return "\n\n".join(context_parts)

custom_prompt = PromptTemplate.from_template(
"""
너는 '전력거래시장 규칙' 전문 기반의 RAG QA 전문가다.
아래 [문서 내용]을 참고하여 [질문]에 대해 아래의 답변 형식과 규칙에 따라 답변하라.

[답변 형식 예시]
1. 정의
(질문에 해당하는 공식/수식이 있으면 반드시 그대로 복사해 답변하라. 문서 내에서 일치하는 공식/수식이 없으면 "문서에서 확인할 수 없습니다."라고 답변하라.)

2. 설명
A. (조건/예외명)
(수식 및 설명)
...

[답변 작성 규칙]
- 1번 항목(정의)에는 질문에 해당하는 공식/수식만 반드시 그대로 복사해 답변하라.
- 중간 변수(ex: MP, GP, TLF 등)는 그대로 두고, 내부 수식은 확장하지 말라.
- 공식/수식은 반드시 문서 내 등장한 그 모습 그대로 복사해라.
- 추론, 변형, 해석, 요약, 다른 용어로의 변환 모두 절대 하지 말라.
- 답변 마지막에 **[END]**만 출력하라.

[문서 내용]
{context}

[질문]
{question}
"""
)

custom_prompt2 = PromptTemplate.from_template(
"""
너는 '전력거래시장 규칙' 전문 기반의 RAG QA 전문가다.
아래 [문서 내용]을 바탕으로 [질문]에 대해 답변하라.
공식/수식은 절대 변경하지말고, 문서 내용에 나온 그대로 답변하라.

[문서 내용]
{context}

[질문]
{question}
"""
)



#def query_ollama(prompt: str, model: str = "kanana-1.5-2.1b") -> str:
#def query_ollama(prompt: str, model: str = "gemma3-merged") -> str:
def query_ollama(prompt: str, model: str = "kanana-1.5-8b-instruct") -> str:
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,           # 창의성 최소화, 결정적 답변
            #"top_k": 40,                  # 후보군 좁게
            #"top_p": 0.7,                # 확률 상위 85%만 후보
            #"repeat_penalty": 1.15,       # 반복 억제
            #"presence_penalty": 1.2,      # 중복 억제
            #"frequency_penalty": 1.1,     # 자주 등장 단어 억제
            #"penalize_newline": True,     # 줄바꿈 반복 억제
            "num_predict": 2024,           # 충분한 길이
            "num_ctx": 32768,              # 가능한 한 크게 (모델 한계까지)
            "stop": ["[END]", "<|end_of_text|>"] # 필요시 프롬프트 종료 문자 지정
        }
    }

    try:
        res = requests.post(url, headers=headers, json=data, timeout=180)
        res.raise_for_status()
        answer = res.json().get("response", "").strip()
        return answer if answer else "문서에서 관련된 정보를 찾을 수 없습니다."
    except requests.exceptions.RequestException as e:
        return f"🚫 Ollama 오류: {e}"

def clean_ollama_answer(raw_answer: str):
    stop_tokens = ["[END]", "<|end_of_text|>"]
    min_idx = len(raw_answer)

    for token in stop_tokens:
        idx = raw_answer.find(token)
        if idx != -1:
            min_idx = min(min_idx, idx + len(token))

    return raw_answer[:min_idx].strip()

def rag_with_ollama(question: str, query_type: str):

    if query_type in ("0", "1"):
        context_str = build_context_for_question(question, k=12)
    else:
        context_str = ""

    if str(query_type) == "0":
        prompt = custom_prompt.format(context=context_str, question=question)
    elif str(query_type) == "1":
        prompt = custom_prompt2.format(context=context_str, question=question)
    else:
        prompt = f"[질문]\n{question}"

    

    print(f"\n📝 최종 Prompt:\n{prompt}\n")
    answer = query_ollama(prompt)
    print(f"\n🔍 질문: {question}\n💡 답변: {answer}")

    return {
        "rag_context": context_str,
        "answer": clean_ollama_answer(answer)
    }