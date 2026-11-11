import os
import io
import base64
from dotenv import load_dotenv
import httpx
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import groq
from pypdf import PdfReader
import google.generativeai as genai

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
ADMIN_UPLOAD_KEY = os.getenv("ADMIN_UPLOAD_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Reads comma-separated keys from GEMINI_API_KEYS or falls back to GEMINI_API_KEY
RAW_GEMINI_KEYS = os.getenv("GEMINI_API_KEYS", os.getenv("GEMINI_API_KEY", ""))
GEMINI_KEY_POOL = [k.strip() for k in RAW_GEMINI_KEYS.split(",") if k.strip()]

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_client = groq.Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

TEXT_MODEL = "openai/gpt-oss-120b"

app = FastAPI(title="Campus Notes Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    image_base64: str | None = None


class DocChatRequest(BaseModel):
    note_id: int
    question: str
    image_base64: str | None = None


def analyze_image_with_gemini(prompt: str, image_data_url: str) -> str:
    """Rotates across configured Gemini API keys on quota limits (429)."""
    if not GEMINI_KEY_POOL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No Gemini API keys configured on the server."
        )

    try:
        if "," in image_data_url:
            header, b64_str = image_data_url.split(",", 1)
            mime_type = header.split(";")[0].replace("data:", "").strip()
        else:
            b64_str = image_data_url
            mime_type = "image/png"

        image_bytes = base64.b64decode(b64_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {str(e)}")

    # Standard production vision model supported by Google AI Studio keys
    models_to_try = [
        "gemini-1.5-flash"
    ]
    last_err = None

    # Rotate through keys across different projects
    for key_idx, api_key in enumerate(GEMINI_KEY_POOL):
        genai.configure(api_key=api_key)

        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    [prompt, {"mime_type": mime_type, "data": image_bytes}],
                    request_options={"timeout": 25.0}
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                print(f"Key {key_idx + 1} with {model_name} failed: {e}")

                # If quota/rate-limit hit, switch immediately to the next API key
                if "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                    break
                continue

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Vision inference error: {str(last_err)}"
    )


# Supports both GET and HEAD for Render health checks
@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def serve_home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/notes")
async def list_notes():
    res = supabase.table("notes").select("*").order("created_at", desc=True).execute()
    return res.data


@app.post("/api/notes/upload")
async def upload_note(
    title: str = Form(...),
    subject: str = Form(...),
    semester: str = Form(...),
    category: str = Form(...),
    file: UploadFile = File(...),
    admin_key: str = Form(...)
):
    expected_key = os.getenv("ADMIN_UPLOAD_KEY")
    if not expected_key or admin_key.strip() != expected_key.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid passkey. Upload access denied."
        )

    file_bytes = await file.read()
    sanitized_filename = f"{os.urandom(6).hex()}_{file.filename.replace(' ', '_')}"
    filepath = f"uploads/{sanitized_filename}"

    supabase.storage.from_("notes").upload(
        path=filepath,
        file=file_bytes,
        file_options={"content-type": "application/pdf"}
    )

    file_url = supabase.storage.from_("notes").get_public_url(filepath)

    insert_res = supabase.table("notes").insert({
        "title": title.strip(),
        "subject": subject.strip(),
        "semester": semester.strip(),
        "category": category.strip(),
        "filename": file.filename,
        "filepath": filepath,
        "file_url": file_url
    }).execute()

    return {"message": "Upload successful", "data": insert_res.data}


@app.delete("/api/notes/{note_id}")
async def delete_note(note_id: int, x_admin_key: str = Header(None)):
    expected_key = os.getenv("ADMIN_UPLOAD_KEY")
    if not x_admin_key or x_admin_key.strip() != (expected_key or "").strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid passkey. Deletion access denied."
        )

    res = supabase.table("notes").select("filepath").eq("id", note_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Note not found.")

    filepath = res.data[0].get("filepath")
    if filepath:
        supabase.storage.from_("notes").remove([filepath])

    supabase.table("notes").delete().eq("id", note_id).execute()

    return {"message": "Note deleted successfully"}


@app.get("/api/notes/{note_id}/download")
async def download_note(note_id: int):
    res = supabase.table("notes").select("file_url").eq("id", note_id).execute()
    if not res.data or not res.data[0].get("file_url"):
        raise HTTPException(status_code=404, detail="File URL not found.")
    return RedirectResponse(url=res.data[0]["file_url"])


@app.post("/api/ai/chat")
async def study_bot_chat(payload: ChatRequest):
    if payload.image_base64:
        try:
            prompt = (
                "You are Campus StudyBot, an expert academic tutor. "
                "Analyze this diagram, screenshot, or handwritten note and provide a clear, structured explanation:\n\n"
                f"{payload.message or 'Explain this image in detail.'}"
            )
            result = analyze_image_with_gemini(prompt, payload.image_base64)
            return {"response": result}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image vision processing error: {str(e)}")

    if not groq_client:
        raise HTTPException(status_code=500, detail="Groq API Key is not configured on the server.")

    try:
        response = groq_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are Campus StudyBot, an expert academic tutor. Format your answers with clean Markdown, concise bullet points, and code blocks where applicable."
                },
                {"role": "user", "content": payload.message}
            ],
            temperature=0.4,
            max_tokens=1024
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"StudyBot inference failed: {str(e)}")


@app.post("/api/ai/doc-chat")
async def document_qa(payload: DocChatRequest):
    res = supabase.table("notes").select("file_url, title").eq("id", payload.note_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Note document not found.")

    file_url = res.data[0]["file_url"]
    title = res.data[0]["title"]

    try:
        # Non-blocking async download with a 12s timeout
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.get(file_url)
            pdf_bytes = resp.content

        reader = PdfReader(io.BytesIO(pdf_bytes))
        extracted_text = ""
        for i, page in enumerate(reader.pages[:6]):
            page_text = page.extract_text() or ""
            extracted_text += f"\n--- Page {i+1} ---\n" + page_text

        context = extracted_text[:4000] if extracted_text.strip() else "No text extracted from PDF."
        doc_prompt = (
            f"You are an academic document assistant analyzing notes titled '{title}'.\n\n"
            f"Document Excerpt:\n{context}\n\n"
            f"User Question/Prompt: {payload.question or 'Explain the attached screenshot in the context of this document.'}"
        )

        if payload.image_base64:
            result = analyze_image_with_gemini(doc_prompt, payload.image_base64)
            return {"response": result}

        if not groq_client:
            raise HTTPException(status_code=500, detail="Groq API Key is not configured on the server.")

        response = groq_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"You are an academic document assistant analyzing notes titled '{title}'. Answer clearly using the provided document excerpt in clean Markdown."
                },
                {"role": "user", "content": doc_prompt}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        return {"response": response.choices[0].message.content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Q&A processing error: {str(e)}")