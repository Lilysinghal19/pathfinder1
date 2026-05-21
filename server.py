"""
server.py  --  AI Skill Gap Analyzer + Career Coach + Voice Agent



Endpoints:
  GET  /                    -> index.html
  GET  /api/roles           -> list of available roles
  POST /api/analyze         -> resume upload + skill gap analysis
  POST /api/chat            -> career chatbot
  GET  /api/token           -> LiveKit JWT for voice agent
  POST /api/reset           -> clear session gap context

Also runs livekit_agent.py as a subprocess automatically.
"""

import asyncio
import datetime
import io
import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# ── Credentials ───────────────────────────────────────────────────────
GCP_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp_key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS

_hf = os.getenv("HF_TOKEN") or os.getenv("hf_token")
if _hf:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = _hf
    os.environ["HF_TOKEN"] = _hf

# ── Imports from your existing modules ───────────────────────────────
from chain import analyze_skill_gap, chat_with_knowledge_base
from retriever import get_available_roles, get_vectorstore, get_embeddings
import config
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from PIL import Image
import pdfplumber
import fitz

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ── LiveKit token ─────────────────────────────────────────────────────
from livekit.api import AccessToken, VideoGrants
from livekit.protocol.agent_dispatch import CreateAgentDispatchRequest
from livekit.protocol.room import CreateRoomRequest
from livekit.api import LiveKitAPI

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("server")

groq_client = Groq(api_key=config.GROQ_API_KEY)

# In-memory session store: session_id -> gap_context dict
_sessions: dict[str, dict] = {}

app = FastAPI(title="AI Skill Gap Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)



def extract_skills_llm(resume_text: str) -> list[str]:
    prompt = f"""You are a resume skill parser.
Your ONLY job: extract technical skills EXPLICITLY written in this resume.
Rules:
1. ONLY include skills that appear word-for-word in the resume text
2. DO NOT add, infer, or assume any skill not written
3. Normalize: lowercase, fix obvious typos
4. Split combined skills: "pandas, numpy" -> separate items
5. Keep multi-word skills intact: "machine learning", "deep learning"
6. Include: frameworks, libraries, languages, tools, platforms, methods
7. Exclude: soft skills, job titles, company names, degree names, years, percentages
8. If heading has skills extract those too
9. If skills are written together with operators extract them separately

Resume text:
{resume_text[:4000]}

Return ONLY a comma-separated list. Nothing else."""
    resp = groq_client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=600,
    )
    raw = resp.choices[0].message.content.strip()
    skills = []
    for s in raw.split(","):
        s = s.strip().lower().lstrip("0123456789.-) ")
        if not s or len(s) < 2:
            continue
        if any(bad in s for bad in [
            "year", "month", "%", "winner", "university",
            "bachelor", "master", "phd", "cgpa", "gpa",
            "internship", "training", "certification", "award"
        ]):
            continue
        skills.append(s)
    return list(dict.fromkeys(skills))


_db_skills_cache: list[str] | None = None

def get_all_skills_from_db() -> list[str]:
    global _db_skills_cache
    if _db_skills_cache is not None:
        return _db_skills_cache
    vs = get_vectorstore()
    data = vs.get(include=["metadatas"])
    skills = set()
    for m in data["metadatas"]:
        for s in m.get("skills_text", "").split(", "):
            if s.strip():
                skills.add(s.strip().lower())
    _db_skills_cache = sorted(skills)
    return _db_skills_cache


def map_skills_to_db(extracted: list[str], threshold: float = 0.75) -> list[str]:
    if not extracted:
        return []
    db_skills = get_all_skills_from_db()
    emb = get_embeddings()
    ext_vecs = emb.embed_documents(extracted)
    db_vecs  = emb.embed_documents(db_skills)
    mapped = []
    for i, skill in enumerate(extracted):
        sims = cosine_similarity([ext_vecs[i]], db_vecs)[0]
        best_idx = sims.argmax()
        if sims[best_idx] >= threshold:
            mapped.append(db_skills[best_idx])
    return list(set(mapped))


def extract_text_from_pdf(path: str) -> str:
    try:
        doc   = fitz.open(path)
        pages = [page.get_text() for page in doc]
        doc.close()
        text  = "\n".join(pages).strip()
        if len(text) >= 100:
            return text
    except Exception:
        pass
    try:
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if len(text.strip()) >= 100:
            return text.strip()
    except Exception:
        pass
    try:
        doc   = fitz.open(path)
        texts = []
        for page in doc:
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            texts.append(pytesseract.image_to_string(img, config="--psm 6"))
        doc.close()
        return "\n".join(texts).strip()
    except Exception:
        return ""


def extract_text_from_image(path: str) -> str:
    return pytesseract.image_to_string(Image.open(path))


def build_gap_context_str(gap: dict, insight: str) -> str:
    return (
        f"Target Role   : {gap['target_role']}\n"
        f"Match Score   : {gap['match_pct']}%\n"
        f"Gap Score     : {gap['gap_pct']}%\n"
        f"User Skills   : {', '.join(gap['user_skills'])}\n"
        f"Matching      : {', '.join(gap['matching']) or 'None'}\n"
        f"Missing       : {', '.join(gap['missing'][:10])}\n\n"
        f"AI Insight:\n{insight}"
    )
    
# ═════════════════════════════════════════════════════════════════════
# API ROUTES
# ═════════════════════════════════════════════════════════════════════

@app.get("/api/roles")
async def get_roles():
    return {"roles": get_available_roles()}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    target_role: str = Form(...),
    tone: str        = Form("concise"),
    session_id: str  = Form(None),
):
    if not session_id:
        session_id = str(uuid.uuid4())

    suffix = ".pdf" if "pdf" in file.content_type else ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        file_path = tmp.name

    try:
        resume_text = (
            extract_text_from_pdf(file_path)
            if suffix == ".pdf"
            else extract_text_from_image(file_path)
        )
    finally:
        os.unlink(file_path)

    if not resume_text.strip():
        raise HTTPException(400, "Could not extract text from file")

    raw_skills = extract_skills_llm(resume_text)
    if not raw_skills:
        raise HTTPException(400, "No skills detected in resume")

    skills     = map_skills_to_db(raw_skills)
    skills_str = ", ".join(skills)
    result     = analyze_skill_gap(
        user_skills_str=skills_str,
        target_role=target_role,
        tone=tone,
    )

    if result["error"]:
        raise HTTPException(400, result["error"])

    gap     = result["gap_report"]
    insight = result["explanation"]
    ctx_str = build_gap_context_str(gap, insight)

    # Store in memory + write gap_context.json for livekit_agent.py
    _sessions[session_id] = {
        "gap": gap, "insight": insight, "ctx_str": ctx_str,
        "chat_history": [],
    }

    with open("gap_context.json", "w", encoding="utf-8") as f:
        json.dump({
            "target_role": gap["target_role"],
            "match_pct":   gap["match_pct"],
            "gap_pct":     gap["gap_pct"],
            "matching":    gap["matching"],
            "missing":     gap["missing"],
            "user_skills": gap["user_skills"],
            "ai_insight":  insight,
        }, f, indent=2)

    return JSONResponse({
        "session_id":  session_id,
        "gap":        json.loads(json.dumps(gap,default=str)),
        "insight":     str(insight),
        "user_skills": raw_skills,
    })


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    sess = _sessions.get(req.session_id)
    if not sess:
        raise HTTPException(404, "Session not found. Run analysis first.")

    answer = chat_with_knowledge_base(
        question=req.message,
        insight=sess["ctx_str"],
        session_id=req.session_id,
    )
    sess["chat_history"].append({"role": "user",      "content": req.message})
    sess["chat_history"].append({"role": "assistant",  "content": answer})
    if len(sess["chat_history"]) > 20:
        sess["chat_history"] = sess["chat_history"][-20:]

    return {"answer": answer, "history": sess["chat_history"]}


@app.post("/api/reset")
async def reset(session_id: str = Form(...)):
    _sessions.pop(session_id, None)
    if Path("gap_context.json").exists():
        os.remove("gap_context.json")
    return {"ok": True}


@app.get("/api/token")
async def get_token(room: str = "career-coach", identity: str = "user"):
    lk_url = os.environ.get("LIVEKIT_URL", "")
    lk_key = os.environ.get("LIVEKIT_API_KEY", "")
    lk_sec = os.environ.get("LIVEKIT_API_SECRET", "")
    if not all([lk_url, lk_key, lk_sec]):
        raise HTTPException(500, "LiveKit env vars not set in .env")

    try:
        async with LiveKitAPI(url=lk_url, api_key=lk_key, api_secret=lk_sec) as lkapi:
            try:
                await lkapi.room.create_room(
                    CreateRoomRequest(name=room, empty_timeout=600)
                )
            except Exception:
                pass
            try:
                await lkapi.agent_dispatch.create_dispatch(
                    CreateAgentDispatchRequest(agent_name="career-coach", room=room)
                )
            except Exception:
                pass

        token = (
            AccessToken(api_key=lk_key, api_secret=lk_sec)
            .with_identity(identity)
            .with_name(f"User-{identity}")
            .with_grants(VideoGrants(
                room_join=True, room=room,
                can_publish=True, can_subscribe=True, can_publish_data=True,
            ))
            .with_ttl(datetime.timedelta(hours=1))
            .to_jwt()
        )
        return JSONResponse({"token": token, "room": room, "serverUrl": lk_url})
    except Exception as e:
        log.exception("Token generation failed")
        raise HTTPException(500, str(e))


# ── Serve frontend ────────────────────────────────────────────────────
@app.get("/")
async def root():
    idx = STATIC / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    raise HTTPException(404, "index.html not found in static/")

app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

def start_livekit_agent():
    """Start livekit_agent.py as a background subprocess."""
    agent_file = Path(__file__).parent / "livekit_agent.py"
    if not agent_file.exists():
        log.warning("livekit_agent.py not found - voice assistant will not work")
        return None
    log.info("Starting livekit_agent.py as background process...")
    proc = subprocess.Popen(
        [sys.executable, str(agent_file), "dev"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc


if __name__ == "__main__":
    # Start LiveKit agent in background
    agent_proc = start_livekit_agent()

    print("\n[INFO] Server starting at http://localhost:8000")
    print("[INFO] Open http://localhost:8000 in your browser\n")

    try:
        uvicorn.run(
            "server:app",
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info",
        )
    finally:
        if agent_proc:
            agent_proc.terminate()
            log.info("livekit_agent.py stopped")