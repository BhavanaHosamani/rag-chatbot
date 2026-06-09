import os
from langchain.chains import RetrievalQA
from rag.custom_llm import ChatOpenAINoStop, http_client

def create_qa_chain(vector_store):
    llm = ChatOpenAINoStop(
        model="openai/gpt-4o-mini",
        openai_api_key=os.getenv("API_KEY"),
        openai_api_base=os.getenv("API_BASE"),
        temperature=0.7,
        http_client=http_client
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever()
    )