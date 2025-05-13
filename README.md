# 🚀 FastAPI 기반 LLM 연동 API (Groq / OpenAI 전략 패턴 적용)

이 프로젝트는 FastAPI 기반의 RESTful API 서버에서 OpenAI 또는 Groq의 LLM(Language Model) API를 전략 패턴으로 동적으로 연동하는 예제입니다.

## 🧩 주요 기능

- ✅ RESTful API 설계 (CRUD 엔드포인트)
- 🤖 OpenAI GPT-3.5-turbo 기반 응답
- 🧠 OpenAI 사용 불가 시 자동으로 Groq(Llama3)로 fallback
- 🧰 전략 패턴(Strategy Pattern)을 활용한 LLM 연동 구조
- 🔐 .env를 통한 API 키 관리

---

## 📦 사용된 기술 스택

| 기술         | 설명                            |
|--------------|---------------------------------|
| **FastAPI**  | 고성능 Python 웹 프레임워크     |
| **Groq SDK** | Groq LPU 기반 LLM 클라이언트    |
| **OpenAI SDK** | ChatGPT 모델 연동용 Python SDK |
| **dotenv**   | 환경변수 관리 (.env)            |
| **Pydantic** | 데이터 유효성 검사 및 모델링    |
| **Strategy Pattern** | 유연한 LLM 선택 구조 구현 |

---

## 📁 프로젝트 구조
```
├── main.py # FastAPI 앱 엔트리포인
├── .env # API 키 저장용 환경파일
├── requirements.txt # 의존성 리스트
└── README.md # 현재 문서
```

## 📦 설치 및 실행
1. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```
2. **환경변수 설정**
   `.env` 파일에 OpenAI와 Groq API 키를 추가합니다.
   ```env
    OPENAI_API_KEY=your_openai_api_key
    GROQ_API_KEY=your_groq_api_key
    ```
3. **서버 실행**
```bash
  1. source venv/bin/activate
  2. uvicorn main:app --reload
  3. deactivate
```

5. **API 테스트**
```bash
  Swagger UI 사용:
  - 브라우저 열기: http://localhost:8000/docs
  - /chat 엔드포인트에서 질문을 입력하여 응답 확인
  - /items API를 통해 CRUD도 테스트 가능
```
