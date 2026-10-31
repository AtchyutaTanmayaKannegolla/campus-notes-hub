import os
import shutil
import sqlite3
import time
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from pypdf import PdfReader
from groq import Groq

app = FastAPI(title="Campus Notes Hub with Groq AI")

# Paste your Groq API Key here
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # paste your full key here
groq_client = Groq(api_key=GROQ_API_KEY)

AI_MODEL = "openai/gpt-oss-120b"

UPLOAD_DIR = "uploads"
DB_FILE = "notes.db"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------- Database Setup -----------------
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                subject TEXT NOT NULL,
                semester INTEGER NOT NULL,
                category TEXT NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
init_db()

# Request schemas for chat endpoints
class StudyChatRequest(BaseModel):
    prompt: str

class DocChatRequest(BaseModel):
    note_id: int
    question: str

# ----------------- Core Routes -----------------

@app.get("/", response_class=HTMLResponse)
def serve_home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/notes")
def fetch_notes(semester: str = "All", search: str = ""):
    with get_db() as conn:
        query = "SELECT * FROM notes WHERE 1=1"
        params = []

        if semester != "All":
            query += " AND semester = ?"
            params.append(int(semester))

        if search.strip():
            query += " AND (title LIKE ? OR subject LIKE ?)"
            search_param = f"%{search.strip()}%"
            params.extend([search_param, search_param])

        query += " ORDER BY id DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

@app.post("/api/upload")
async def upload_note(
    title: str = Form(...),
    subject: str = Form(...),
    semester: int = Form(...),
    category: str = Form(...),
    file: UploadFile = File(...)
):
    safe_filename = f"{int(time.time())}_{file.filename.replace(' ', '_')}"
    saved_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with get_db() as conn:
        conn.execute("""
            INSERT INTO notes (title, subject, semester, category, filename, filepath)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (title, subject, semester, category, file.filename, saved_path))

    return {"message": "Uploaded successfully"}

@app.get("/api/download/{note_id}")
def download_note(note_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="File not found")

        filepath = row["filepath"]
        original_name = row["filename"]

        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail="Physical file missing on server")

        return FileResponse(filepath, filename=original_name)

@app.delete("/api/notes/{note_id}")
def delete_note(note_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Not found")

        filepath = row["filepath"]
        if os.path.exists(filepath):
            os.remove(filepath)

        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
    return {"message": "Deleted successfully"}

# ----------------- Groq AI Endpoints -----------------

# Option 1: General Academic Study Assistant
@app.post("/api/ai/general-chat")
def general_study_chat(req: StudyChatRequest):
    try:
        completion = groq_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an encouraging academic tutor for college engineering students. "
                        "Explain concepts clearly with intuitive examples, concise points, and practical analogies."
                    )
                },
                {"role": "user", "content": req.prompt}
            ],
            temperature=0.6,
            max_tokens=1024
        )
        return {"answer": completion.choices[0].message.content}
    except Exception as e:
        return {"answer": f"Groq API Error: {str(e)}"}

# Option 2: Document Q&A (RAG on uploaded PDF)
@app.post("/api/ai/doc-chat")
def doc_chat(req: DocChatRequest):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM notes WHERE id = ?", (req.note_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Note not found")

    filepath = row["filepath"]
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File missing on server")

    # Extract text from the PDF
    extracted_text = ""
    try:
        reader = PdfReader(filepath)
        for page in reader.pages[:15]:
            extracted_text += (page.extract_text() or "") + "\n"
    except Exception as e:
        return {"answer": f"Could not read document contents: {str(e)}"}

    if not extracted_text.strip():
        return {"answer": "This document seems to be empty or contains scanned images without selectable text."}

    # Query Groq using the extracted PDF content as context
    try:
        completion = groq_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are an academic reviewer for the study document titled '{row['title']}' "
                        f"for the subject '{row['subject']}'. Answer student questions strictly using "
                        "the document excerpt provided."
                    )
                },
                {
                    "role": "user",
                    "content": f"Document Excerpt:\n{extracted_text[:12000]}\n\nStudent question: {req.question}"
                }
            ],
            temperature=0.4,
            max_tokens=1024
        )
        return {"answer": completion.choices[0].message.content}
    except Exception as e:
        return {"answer": f"Groq AI Error: {str(e)}"}