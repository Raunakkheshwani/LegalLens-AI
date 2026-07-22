"""
FastAPI wrapper exposing the LangGraph legal document review pipeline
as a real HTTP service.

Run: uvicorn main:app --reload
Docs auto-generated at: http://127.0.0.1:8000/docs
"""

import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File , HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

import config 
from document_store import get_or_build_document_store
from graph import build_graph

app= FastAPI(title="LegalLens API", version="0.1.0")
graph_app = build_graph()

UPLOAD_DIR = "data/uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ReviewRequest(BaseModel):
    document_id: str 
    document_path: str       # returned by /upload, passed back in by the client
    query:str

class ResumeRequest(BaseModel):
    thread_id: str
    correction: str | None= None        # None/empty = approve as-is

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not (file.filename.lower().endswith(".pdf") or file.filename.lower().endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only PDF or TXT files are supported.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # This triggers document_store.py's content-addressable caching —
    # cache hit if this exact file+config was already indexed, else builds fresh.
    _, document_id = get_or_build_document_store(save_path)

    return {"document_id": document_id, "document_path": save_path}


@app.post("/review")
def review(req: ReviewRequest):
    thread_id = f"api-{uuid.uuid4().hex[:12]}"
    thread_config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "document_path": req.document_path,
        "document_id": "",
        "query": req.query,
        "messages": [HumanMessage(content=req.query)],
        "retrieved_chunks": [],
        "draft_answer": "",
        "critique": "",
        "needs_revision": False,
        "ever_needed_revision": False,
        "loop_count": 0,
        "final_answer": "",
    }

    graph_app.invoke(initial_state, config=thread_config)
    state_snapshot = graph_app.get_state(thread_config)

    if state_snapshot.next:
        # Paused — needs expert reviewer sign-off before finalizing
        return {
            "status": "pending_review",
            "thread_id": thread_id,
            "draft_answer": state_snapshot.values["draft_answer"],
            "critique": state_snapshot.values["critique"],
        }

    # Completed on its own — no review needed
    result = state_snapshot.values
    return {
        "status": "completed",
        "thread_id": thread_id,
        "final_answer": result["final_answer"],
        "document_id": result["document_id"],
    }


@app.post("/review/resume")
def resume_review(req: ResumeRequest):
    thread_config = {"configurable": {"thread_id": req.thread_id}}
    state_snapshot = graph_app.get_state(thread_config)

    if not state_snapshot.next:
        raise HTTPException(status_code=400, detail="This session is not awaiting review.")

    if req.correction:
        graph_app.update_state(thread_config, {"draft_answer": req.correction})

    result = graph_app.invoke(None, config=thread_config)

    return {
        "status": "completed",
        "thread_id": req.thread_id,
        "final_answer": result["final_answer"],
        "document_id": result["document_id"],
    }