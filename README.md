# LLM 기반 FastAPI AI Assistant

이 프로젝트는 FastAPI를 기반으로 다양한 LLM(대형 언어 모델)을 활용한 AI 서비스를 제공합니다.
다양한 전략 패턴, RAG(Retrieval-Augmented Generation), LLMChain, 그리고 에이전트 기반 툴 사용 기능을 포함하고 있습니다.

## 🔧 구성 기능

### ✅ RESTful API
- 기본적인 CRUD 연습용 엔드포인트 `/items`

### ✅ LLMProvider 전략 패턴
- OpenAI, Groq 기반 LLM을 선택적으로 사용하도록 추상화

### ✅ RAG (Retrieval-Augmented Generation)
- 문서 임베딩: `sentence-transformers/all-MiniLM-L6-v2`
- 벡터 저장소: OpenSearch 기반 Elasticsearch 호환 벡터 DB
- 질문 응답 `/rag`

### ✅ LLMChain
- LangChain의 `LLMChain`을 이용한 프롬프트 템플릿 기반 응답 `/llmchain`

### ✅ LangChain Agent
- Tool: Calculator (PythonREPLTool 기반)
- 커스텀 OutputParser를 통해 Final Answer를 안정적으로 추출
- LLM: Groq 기반 `llama3-8b-8192`
- 시스템 프롬프트를 통한 행동 유도
- `/agent`

---

## 💡 용어 설명

- **RAG**: 외부 문서를 검색한 뒤 해당 내용을 바탕으로 답변을 생성하는 방식입니다.
- **LLMChain**: PromptTemplate과 LLM의 체인을 구성하여 정형화된 응답을 생성합니다.
- **Agent**: LLM이 Tool을 선택하고 실행하며, 최종 응답을 생성하는 지능형 컨트롤러입니다.
- **Tool**: 계산기, 검색기, 파일 조회 등 LLM이 사용할 수 있는 기능 모듈입니다.
- **OutputParser**: LLM이 생성한 응답에서 Action/Observation/Final Answer를 파싱하여 흐름을 제어합니다.

---

## 🛠️ 실행 방법

1. `.env` 파일에 API Key 설정:
```env
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
```

2. 패키지 설치:
```bash
pip install -r requirements.txt
```

3. Elasticsearch (OpenSearch) 실행:
```bash
docker run -d --name opensearch -p 9200:9200 -e "discovery.type=single-node" -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=admin" opensearchproject/opensearch:2
```

4. FastAPI 실행:
```bash
1. source ./venv/bin/activate # 가상환경 활성화
2. uvicorn main:app --reload # FastAPI 서버 실행
3. curl -X GET "http://localhost:8000/docs" # Swagger UI
4. deactivate # 가상환경 비활성화
```

---

## 📚 향후 계획

- LangGraph를 활용한 상태 기반 LLM Agent 설계
- LangFlow 연동 및 시각화
- 멀티 모달 Tool 및 LangSmith 연동
