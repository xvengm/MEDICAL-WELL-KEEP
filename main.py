from __future__ import annotations

from typing import Dict, Optional, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel, Field

from rag_agent import RAGAgent


# App & middleware

app = FastAPI(title="Medical RAG API", version="2.1")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates 
templates = Jinja2Templates(directory=str(Path("templates")))

# RAG agent 
agent = RAGAgent()


SESSIONS: Dict[str, str] = {}



# Pydantic API models

class AskRequest(BaseModel):
    session_id: str = Field(..., description="The session id returned from /api/upload")
    question: str = Field(..., min_length=3)


class RAGResponse(BaseModel):
    session_id: Optional[str] = None
    summary: str
    confidence: str
    main_issue: str
    key_findings: str
    treatment_plan: str
    important_notes: str
    codes: List[str] = []
    follow_up_questions: List[str] = []
    disclaimer: str
    limitations: List[str] = []



# Frontend page

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})



# Health check

@app.get("/api/health")
async def health():
    return {"status": "ok"}



# Upload endpoint

ALLOWED_EXTS = {".pdf", ".txt", ".doc", ".docx", ".png", ".jpg", ".jpeg"}

@app.post("/api/upload", response_model=RAGResponse)
async def api_upload(file: UploadFile = File(...)):
    name = (file.filename or "").lower()
    suffix = Path(name).suffix

    if suffix not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTS))}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        result = agent.process_document(data, name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {exc}")

    session_id = result.get("session_id")
    ctx = result.pop("context", "")  
    if session_id and ctx:
        SESSIONS[session_id] = ctx

    return RAGResponse(**result)



# Follow-up questions

@app.post("/api/ask", response_model=RAGResponse)
async def api_ask(payload: AskRequest = Body(...)):
    session_id = payload.session_id
    question = payload.question.strip()

    context = SESSIONS.get(session_id)
    if not context:
        raise HTTPException(status_code=400, detail="Unknown or expired session_id. Upload a document first.")

    try:
        result = agent.process_question(question, context)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process question: {exc}")

    return RAGResponse(**result)
