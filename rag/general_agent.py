import os
from langchain.schema import HumanMessage, SystemMessage
from rag.custom_llm import ChatOpenAINoStop, http_client

def create_general_agent():
    llm = ChatOpenAINoStop(
        model="openai/gpt-4o-mini",
        openai_api_key=os.getenv("API_KEY"),
        openai_api_base=os.getenv("API_BASE"),
        temperature=0.7,
        http_client=http_client
    )

    def run(question: str) -> str:
        messages = [
            SystemMessage(content="You are a helpful general knowledge assistant."),
            HumanMessage(content=question)
        ]
        response = llm.invoke(messages)
        return response.content

    return run