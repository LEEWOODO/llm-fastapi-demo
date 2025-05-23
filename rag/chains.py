from langchain.chains import RetrievalQA
from llm.provider import FalconLLMProvider
from rag.vectorstore import vectorstore

rag_chain = RetrievalQA.from_chain_type(
    llm=FalconLLMProvider().get_chain(),  # ✅ 반드시 get_chain()으로 Runnable 반환
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    return_source_documents=True
)
