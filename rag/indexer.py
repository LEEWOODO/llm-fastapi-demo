import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from rag.vectorstore import vectorstore


def extract_text(filename: str, content: bytes) -> str:
    if filename.endswith(".txt"):
        return content.decode("utf-8")
    elif filename.endswith(".pdf"):
        with fitz.open(stream=content, filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. .txt 또는 .pdf만 허용됩니다.")


def index_file_to_opensearch(filename: str, content: bytes) -> int:
    text = extract_text(filename, content)

    # ✅ LLM tokenizer 불러오기 (t5-small 기준)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # ✅ 토큰 기준 분할기
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        length_function=lambda x: len(tokenizer.encode(x, truncation=False))
    )

    chunks = splitter.split_text(text)
    vectorstore.add_texts(chunks)
    return len(chunks)