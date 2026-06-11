import os
import httpx
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from rag.splitter import split_documents
from rag.embeddings import get_embeddings
from rag.vector_store import create_vector_store
from rag.tools import set_vector_store, search_pdf, get_pdf_list, delete_pdf_store
from rag.agent import create_agent
from rag.auth import (
    sign_up, sign_in, verify_token,
    register_pdf_for_user, get_pdfs_for_user, delete_pdf_record
)
from rag.supabase_storage import (
    upload_pdf_to_supabase,
    download_pdf_from_supabase,
    delete_pdf_from_supabase,
    list_pdfs_from_supabase,
)

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader

# =========================
# Config
# =========================

os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
http_client = httpx.Client(timeout=120.0)

# =========================
# App
# =========================

app = FastAPI(title="RAG Chatbot API", version="4.0.0")

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
# Agent & LLM
# =========================

agent_executor = create_agent()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# =========================
# Auth Helper
# =========================

def get_current_user(authorization: str = None) -> dict:
    """Extract and verify JWT from Authorization header."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header.")
    token = authorization.replace("Bearer ", "").strip()
    try:
        return verify_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

# =========================
# Startup — Restore PDFs
# =========================

@app.on_event("startup")
async def restore_pdfs_on_startup():
    """On Render restart, rebuild all users' vector stores from Supabase."""
    print("🔄 Restoring PDFs from Supabase...")
    from rag.auth import get_service_client
    try:
        client = get_service_client()
        response = client.table("user_pdfs").select("user_id, pdf_name").execute()
        rows = response.data or []
    except Exception as e:
        print(f"❌ Could not fetch user_pdfs: {e}")
        return

    embeddings = get_embeddings()
    for row in rows:
        user_id = row["user_id"]
        pdf_name = row["pdf_name"]
        store_key = f"{user_id}/{pdf_name}"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp_path = tmp.name
            download_pdf_from_supabase(user_id, pdf_name, tmp_path)
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            chunks = split_documents(documents)
            vector_store = create_vector_store(chunks, embeddings)
            set_vector_store(store_key, vector_store)
            os.unlink(tmp_path)
            print(f"  ✅ Restored: {store_key}")
        except Exception as e:
            print(f"  ❌ Failed to restore {store_key}: {e}")

    print(f"✅ Restored {len(rows)} PDF(s).")

# =========================
# Request Models
# =========================

class SignUpRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class QueryRequest(BaseModel):
    question: str
    chat_history: list = []
    pdf_name: Optional[str] = None

# =========================
# Auth Endpoints
# =========================

@app.post("/auth/signup")
async def signup(request: SignUpRequest):
    try:
        data = sign_up(request.email, request.password)
        return {"message": "Account created successfully!", **data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/login")
async def login(request: LoginRequest):
    try:
        data = sign_in(request.email, request.password)
        return {"message": "Login successful!", **data}
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/auth/logout")
async def logout(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    token = authorization.replace("Bearer ", "").strip()
    from rag.auth import sign_out
    sign_out(token)
    return {"message": "Logged out successfully."}

# =========================
# Upload PDF
# =========================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    user = get_current_user(authorization)
    user_id = user["user_id"]

    file_bytes = await file.read()

    # Upload to Supabase under user's folder
    upload_pdf_to_supabase(user_id, file.filename, file_bytes)

    # Register in user_pdfs table
    register_pdf_for_user(user_id, file.filename)

    # Build vector store — key is "user_id/filename"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    chunks = split_documents(documents)
    embeddings = get_embeddings()
    vector_store = create_vector_store(chunks, embeddings)
    store_key = f"{user_id}/{file.filename}"
    set_vector_store(store_key, vector_store)
    os.unlink(tmp_path)

    return {
        "message": f"{file.filename} uploaded successfully!",
        "pdf_name": file.filename,
        "chunks": len(chunks)
    }

# =========================
# List PDFs
# =========================

@app.get("/pdfs")
async def list_pdfs(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    pdfs = get_pdfs_for_user(user["user_id"])
    return {"pdfs": pdfs}

# =========================
# Delete PDF
# =========================

@app.delete("/pdfs/{pdf_name}")
async def delete_pdf(
    pdf_name: str,
    authorization: Optional[str] = Header(None)
):
    user = get_current_user(authorization)
    user_id = user["user_id"]

    # Delete from Supabase Storage
    delete_pdf_from_supabase(user_id, pdf_name)

    # Delete from user_pdfs table
    delete_pdf_record(user_id, pdf_name)

    # Delete from in-memory vector store
    store_key = f"{user_id}/{pdf_name}"
    delete_pdf_store(store_key)

    return {"message": f"{pdf_name} deleted successfully."}

# =========================
# Ask Question
# =========================

@app.post("/ask")
async def ask_question(
    request: QueryRequest,
    authorization: Optional[str] = Header(None)
):
    user = get_current_user(authorization)
    user_id = user["user_id"]
    question = request.question
    pdf_name = request.pdf_name

    # Build store key scoped to this user
    store_key = f"{user_id}/{pdf_name}" if pdf_name else None
    context = search_pdf(question, pdf_name=store_key)

    no_pdf_msgs = [
        "No PDF uploaded yet. Please upload and process a PDF first.",
        "Selected PDF not found. Please re-upload.",
    ]

    if context and context not in no_pdf_msgs:
        messages = [
            SystemMessage(content="""
You are a highly knowledgeable AI assistant.
Answer only using the provided PDF context.
Rules:
- Give detailed answers
- Use bullet points when appropriate
- Minimum 5-8 sentences
- Include important facts and details
- Be clear and structured
- If information is missing, say so
"""),
            HumanMessage(content=f"Context from PDF:\n\n{context}\n\nQuestion:\n\n{question}\n\nProvide a detailed answer.")
        ]
        response = llm.invoke(messages)
        return {"answer": response.content, "mode": "pdf", "steps": [], "sources": []}

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
        return {"answer": result["output"], "mode": "agent", "steps": formatted_steps, "sources": []}


# =========================
# Run Locally
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)