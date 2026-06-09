import os 
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from rag.tools import pdf_tool

def create_agent():

    llm = ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )

    tools = [pdf_tool]

    system_prompt = """You are a highly knowledgeable and helpful AI assistant that answers questions thoroughly and in detail.

When answering questions from a PDF document:
- Always provide complete, detailed, and well-structured answers
- Explain concepts clearly with examples if needed
- Use bullet points or numbered lists when listing multiple items
- Cover all relevant aspects of the topic
- If the document has specific details, include them in your answer
- Never give one-line or vague answers
- Minimum 4-6 sentences for every answer
- If asked to explain something, give a full explanation

You have access to a PDF tool. Always use it when the question is about an uploaded document.
"""

    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        agent_kwargs={
            "system_message": system_prompt
        }
    )

    return agent_executor