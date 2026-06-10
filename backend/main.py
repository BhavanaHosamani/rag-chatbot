import os
import shutil
import httpx

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from rag.loader import load_documents
from rag.splitter import split_documents
from rag.embeddings import get_embeddings
from rag.vector_store import create_vector_store
from rag.tools import set_vector_store, search_pdf, get_pdf_list, delete_pdf_store
from rag.agent import create_agent

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# =========================
# OpenRouter Configuration
# =========================

os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

http_client = httpx.Client(timeout=120.0)

# =========================
# FastAPI App
# =========================

app = FastAPI(title="RAG Chatbot API", version="2.0.0")

@app.get("/")
async def home():
    return {"message": "RAG Chatbot API is running!", "status": "success"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Agent & LLM Init
# =========================

agent_executor = create_agent()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# =========================
# Request Models
# =========================

class QueryRequest(BaseModel):
    question: str
    chat_history: list = []
    pdf_name: Optional[str] = None   # which PDF to query (None = query all)

# =========================
# Upload PDF Endpoint
# =========================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a single PDF. Each PDF gets its own vector store."""

    os.makedirs("storage/documents", exist_ok=True)

    # Save file to disk
    file_path = f"storage/documents/{file.filename}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load only this specific file
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vector_store = create_vector_store(chunks, embeddings)

    # Store under the PDF's filename
    set_vector_store(file.filename, vector_store)

    return {
        "message": f"{file.filename} uploaded and processed successfully!",
        "pdf_name": file.filename,
        "chunks": len(chunks)
    }

# =========================
# List PDFs Endpoint
# =========================

@app.get("/pdfs")
async def list_pdfs():
    """Return all uploaded PDFs."""
    return {"pdfs": get_pdf_list()}

# =========================
# Delete PDF Endpoint
# =========================

@app.delete("/pdfs/{pdf_name}")
async def delete_pdf(pdf_name: str):
    """Delete a specific PDF and its vector store."""

    # Remove from memory
    deleted = delete_pdf_store(pdf_name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"{pdf_name} not found.")

    # Remove file from disk
    file_path = f"storage/documents/{pdf_name}"
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"message": f"{pdf_name} deleted successfully."}

# =========================
# Ask Question Endpoint
# =========================

@app.post("/ask")
async def ask_question(request: QueryRequest):
    question = request.question
    pdf_name = request.pdf_name  # frontend passes the selected PDF name

    context = search_pdf(question, pdf_name=pdf_name)

    if context and context not in [
        "No PDF uploaded yet. Please upload and process a PDF first.",
        "Selected PDF not found. Please re-upload.",
    ]:
        messages = [
            SystemMessage(content="""
You are a highly knowledgeable AI assistant.

You will be given context extracted from a PDF document and a question.

Answer only using the provided context.

Rules:
- Give detailed answers
- Use bullet points when appropriate
- Minimum 5-8 sentences
- Include important facts and details
- Be clear and structured
- If information is missing, say so
"""),
            HumanMessage(content=f"""
Context from PDF:

{context}

Question:

{question}

Provide a detailed answer.
""")
        ]

        response = llm.invoke(messages)

        return {
            "answer": response.content,
            "mode": "pdf",
            "steps": [],
            "sources": []
        }

    else:
        result = agent_executor.invoke({"input": question})

        steps = result.get("intermediate_steps", [])
        formatted_steps = []
        for step in steps:
            if isinstance(step, tuple):
                action, observation = step
                formatted_steps.append({
                    "agent": str(action.tool) if hasattr(action, "tool") else "",
                    "input": str(action.tool_input) if hasattr(action, "tool_input") else "",
                    "output": str(observation)
                })

        return {
            "answer": result["output"],
            "mode": "agent",
            "steps": formatted_steps,
            "sources": []
        }

# =========================
# Run Locally
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)