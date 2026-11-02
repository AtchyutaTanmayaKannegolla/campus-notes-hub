# Campus Notes Hub with Groq AI & Supabase

A full-stack academic resource hub and AI study companion built with FastAPI, Supabase (PostgreSQL & Object Storage), and Groq LLMs.

**Live Demo:** [https://campus-notes-hub.onrender.com](https://campus-notes-hub.onrender.com)

---

## Features

- **Permanent Cloud Storage & Persistence:** File binaries are hosted in a public Supabase Storage bucket (`notes`), while document metadata persists across restarts in a Supabase PostgreSQL database.
- **Passkey-Protected Contributor Uploads:** Uploads are restricted by an administrative passkey (`ADMIN_UPLOAD_KEY`), keeping the repository secure against spam and unauthorized files.
- **In-Memory Document Q&A (RAG):** Context-aware question answering directly from uploaded PDFs without writing temporary files to local disk. PDFs are streamed in-memory via public URLs and analyzed using `pypdf` and Groq.
- **Campus StudyBot:** A persistent floating study assistant for computer science and engineering coursework, formulas, and conceptual explanations, rendered with full Markdown and table parsing via `Marked.js`.
- **Dynamic Resource Discovery:** Real-time client-side and server-side filtering by semester and keyword search across titles and subjects.
- **Instant Cloud Downloads:** Direct redirects to public Supabase bucket storage for note downloads.

---

## Tech Stack

- **Backend:** Python, FastAPI, Uvicorn, Pydantic
- **Database & Storage:** Supabase (PostgreSQL Database & Supabase Cloud Storage)
- **AI Inference:** Groq SDK (`openai/gpt-oss-120b`), PyPDF
- **Frontend:** Vanilla HTML5, CSS3, JavaScript (ES6+), Marked.js
- **Hosting & Deployment:** Render

---

## Environment Variables

Create a `.env` file in the root directory (or configure them in your Render dashboard under **Environment**):

```env
SUPABASE_URL=[https://your-project.supabase.co](https://your-project.supabase.co)
SUPABASE_KEY=your_supabase_anon_or_service_role_key
ADMIN_UPLOAD_KEY=your_secret_admin_passkey_here
GROQ_API_KEY=gsk_your_groq_api_key