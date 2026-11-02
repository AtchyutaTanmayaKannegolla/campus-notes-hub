import io
import os
import time
import urllib.request
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
from supabase import Client, create_client

load_dotenv()

app = FastAPI(title="Campus Notes Hub with Groq AI & Supabase")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Credentials & Configuration -----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_UPLOAD_KEY = os.getenv("ADMIN_UPLOAD_KEY", "crrcoe2026")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
AI_MODEL = "openai/gpt-oss-120b"


# ----------------- Request Schemas -----------------
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
  query = supabase.table("notes").select("*").order("id", desc=True)

  if semester != "All":
    query = query.eq("semester", semester)

  response = query.execute()
  notes = response.data or []

  if search.strip():
    s = search.strip().lower()
    notes = [
        n
        for n in notes
        if s in n.get("title", "").lower() or s in n.get("subject", "").lower()
    ]

  return notes


@app.post("/api/upload")
async def upload_note(
    title: str = Form(...),
    subject: str = Form(...),
    semester: str = Form(...),
    category: str = Form(...),
    admin_key: str = Form(...),
    file: UploadFile = File(...),
):
  # Admin contributor validation
  if admin_key != ADMIN_UPLOAD_KEY:
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Unauthorized: Invalid contributor passkey.",
    )

  file_bytes = await file.read()
  safe_filename = (
      f"{int(time.time())}_{file.filename.replace(' ', '_')}"
  )

  # 1. Upload directly to Supabase storage bucket 'notes'
  supabase.storage.from_("notes").upload(
      path=safe_filename,
      file=file_bytes,
      file_options={"content-type": "application/pdf"},
  )

  # 2. Get permanent public file URL
  public_url = supabase.storage.from_("notes").get_public_url(safe_filename)

  # 3. Store metadata in Supabase table
  note_record = {
      "title": title,
      "subject": subject,
      "semester": str(semester),
      "category": category,
      "filename": file.filename,
      "filepath": safe_filename,  # path inside Supabase bucket
      "file_url": public_url,
  }
  supabase.table("notes").insert(note_record).execute()

  return {"message": "Uploaded successfully"}


@app.get("/api/download/{note_id}")
def download_note(note_id: int):
  res = supabase