# Campus Notes Hub with Groq AI

A full-stack academic resource hub and AI study companion built with FastAPI, SQLite, and Groq LLMs.

**Live Demo:** [https://campus-notes-hub.onrender.com](https://campus-notes-hub.onrender.com)

## Features
- **Centralized Resource Sharing:** Upload, search, filter, download, and delete semester notes, PYQs, and syllabus copies.
- **Document Q&A (RAG):** Context-aware question answering directly from uploaded PDFs powered by Groq's `openai/gpt-oss-120b` and `pypdf`.
- **Campus StudyBot:** Persistent peer-tutor chat assistant explaining engineering concepts and exam solutions with clean Markdown rendering.
- **Secure Config:** Environment-isolated credentials using `python-dotenv`.

## Tech Stack
- **Backend:** Python, FastAPI, Uvicorn, SQLite
- **AI Inference:** Groq SDK (`openai/gpt-oss-120b`), PyPDF
- **Frontend:** HTML5, CSS3, JavaScript, Marked.js

## Setup & Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt