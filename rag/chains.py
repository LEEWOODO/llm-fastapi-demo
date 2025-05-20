from langchain.chains import RetrievalQA

from llm.provider import llm
from rag.vectorstore import vectorstore

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    return_source_documents=True
)
