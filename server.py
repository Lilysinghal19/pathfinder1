"""
server.py  --  AI Skill Gap Analyzer + Career Coach + Voice Agent
==================================================================
Replaces Streamlit (oo.py). Run with:
    python server.py

Opens at: http://localhost:8000

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

# ═════════════════════════════════════════════════════════════════════
# HELPERS  (ported from oo.py)
# ═════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# DYNAMIC SKILL EXTRACTION PIPELINE
# 3-stage: Extract → Cosine-similarity DB mapping → LLM-as-Judge
#
# NO hardcoded skill lists, aliases, or implication rules.
# Everything is driven by:
#   1. The resume text itself
#   2. The live skills in your ChromaDB vector store
#   3. The LLM judging against both
# ─────────────────────────────────────────────────────────────────────

# ── Stage 0: Text cleaning ────────────────────────────────────────────

def _clean_skill(s: str) -> str:
    """Lowercase, strip punctuation prefixes, normalise whitespace."""
    import re
    s = s.strip().lower()
    s = re.sub(r"^[\s\d.\-\)\(●•►→]+", "", s)  # leading noise
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_noise(s: str) -> bool:
    """
    Return True if the string is clearly NOT a transferable technical skill.
    Fully dynamic — no hardcoded skill names.
    """
    import re
    if not s or len(s) < 2:
        return True
    # Contains percentage or is purely numeric
    if "%" in s or re.fullmatch(r"[\d.]+", s):
        return True
    # Too long to be a skill name (sentence-length)
    if len(s.split()) > 6:
        return True
    # Generic bad-word categories (never skill names)
    NOISE_PATTERNS = [
        r"\b(year|month|winner|university|bachelor|master|phd|cgpa|gpa)\b",
        r"\b(college|institute|government|innovation|hackathon)\b",
        r"\b(b\.?tech|m\.?tech|certification|internship|training|award)\b",
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|"
        r"march|april|june|july|august|september|october|november|december)\b",
        r"\b(present|ongoing|current|today)\b",
    ]
    for pat in NOISE_PATTERNS:
        if re.search(pat, s, re.I):
            return True
    return False


def _parse_skill_list(raw: str) -> list[str]:
    """Parse a comma-separated LLM output into a clean deduped skill list."""
    seen, out = set(), []
    for token in raw.split(","):
        s = _clean_skill(token)
        if _is_noise(s):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


# ── Stage 1: Extractor LLM ────────────────────────────────────────────

def extract_skills_llm(resume_text: str) -> list[str]:
    """
    Pass 1 — Broad extraction.

    Strategy: extract WIDE — it is better to include a borderline skill
    than to miss a real one.  The judge (Pass 3) will clean up noise.
    We intentionally ask for skills from every section of the resume.
    """
    prompt = (
        "You are a technical skill extractor for resumes.\n\n"
        "TASK: Extract every technical skill from ALL sections of this resume.\n\n"
        "INCLUDE:\n"
        "- Programming languages (e.g. python, sql, java, c++)\n"
        "- ML/AI frameworks and libraries (e.g. pytorch, keras, scikit-learn, langchain)\n"
        "- Cloud and DevOps tools (e.g. docker, kubernetes, git, aws, gcp)\n"
        "- Data tools (e.g. pandas, numpy, matplotlib, seaborn, power bi, tableau)\n"
        "- ML concepts as skills (e.g. rag, nlp, deep learning, computer vision, "
        "generative ai, machine learning, transformer models)\n"
        "- Databases (e.g. mysql, postgresql, chromadb, mongodb)\n"
        "- Web frameworks (e.g. fastapi, flask, streamlit, django)\n"
        "- Any tool/platform/method used in projects or experience sections\n\n"
        "IMPORTANT RULES:\n"
        "- Extract from Skills section, Experience bullets, Projects, AND Certifications\n"
        "- If the experience says 'built NLP models' — include 'nlp'\n"
        "- If experience says 'Computer Vision models' — include 'computer vision'\n"
        "- If experience says 'Generative AI models' — include 'generative ai'\n"
        "- If Python is used anywhere (web scraping, scripts, etc.) — include 'python'\n"
        "- Keep abbreviations as-is: lstm, gru, rnn, rag, nlp, llm, cot\n"
        "- Normalize: lowercase, fix typos, split comma-joined items\n"
        "- Do NOT include: company names, city names, university names, "
        "degree types, CGPA, years, percentages, project names, "
        "award names, satellite/domain-specific jargon that is not a "
        "general transferable skill\n\n"
        f"Resume:\n{resume_text[:6000]}\n\n"
        "Return ONLY a comma-separated list of skills. "
        "One line. No numbering. No explanation."
    )
    resp = groq_client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=700,
    )
    raw    = resp.choices[0].message.content.strip()
    skills = _parse_skill_list(raw)
    log.info("[STAGE1-EXTRACT] %d raw skills extracted", len(skills))
    return skills


# ── Stage 2: Cosine-similarity DB mapping ────────────────────────────

_db_skills_cache: list[str] | None = None

def get_all_skills_from_db() -> list[str]:
    """Return all unique skill names from ChromaDB, cached after first load."""
    global _db_skills_cache
    if _db_skills_cache is not None:
        return _db_skills_cache
    vs   = get_vectorstore()
    data = vs.get(include=["metadatas"])
    seen = set()
    for m in data["metadatas"]:
        for s in m.get("skills_text", "").split(", "):
            s = s.strip().lower()
            if s:
                seen.add(s)
    _db_skills_cache = sorted(seen)
    log.info("[DB] Loaded %d unique skills from vector store", len(_db_skills_cache))
    return _db_skills_cache


def map_skills_to_db(extracted: list[str]) -> list[str]:
    """
    Stage 2 — Dynamic cosine-similarity mapping.

    For every extracted skill, find the closest DB skill by embedding
    cosine similarity.  Uses TWO thresholds:

      HIGH (>= 0.80): accept directly — very confident match
      MID  (>= 0.55): accept only if the DB skill is a SUBSTRING of the
                       extracted skill or vice versa, OR if the extracted
                       skill is short (≤ 4 chars, e.g. "nlp", "rag").
                       This handles abbreviation-to-full-form mismatches.
      BELOW 0.55:     reject — likely noise or hallucination

    Why two thresholds instead of one?
      - "nlp" vs "natural language processing" has cosine ~0.60 but IS the same skill
      - "rag" vs "retrieval augmented generation" is ~0.58 but IS the same skill
      - A single threshold of 0.80 misses these; 0.55 with no guard lets garbage through
    """
    if not extracted:
        return []

    db_skills = get_all_skills_from_db()
    if not db_skills:
        log.warning("[MAP] DB is empty — returning extracted skills as-is")
        return extracted

    emb      = get_embeddings()
    ext_vecs = emb.embed_documents(extracted)
    db_vecs  = emb.embed_documents(db_skills)
    sims     = cosine_similarity(ext_vecs, db_vecs)  # shape: (len_ext, len_db)

    mapped, seen = [], set()
    for i, skill in enumerate(extracted):
        row      = sims[i]
        best_idx = int(row.argmax())
        best_sim = float(row[best_idx])
        db_match = db_skills[best_idx]

        accept = False
        if best_sim >= 0.80:
            # High confidence — always accept
            accept = True
        elif best_sim >= 0.55:
            # Medium confidence — accept only with substring or short-abbrev guard
            short = len(skill.replace(" ", "")) <= 4   # "nlp", "rag", "gru" etc.
            substr = (skill in db_match) or (db_match in skill)
            accept = short or substr

        if accept:
            if db_match not in seen:
                seen.add(db_match)
                mapped.append(db_match)
            log.debug("[MAP] %-30s -> %-30s  sim=%.3f  accept=%s",
                      skill, db_match, best_sim, accept)
        else:
            log.debug("[MAP] %-30s -> NO MATCH (best=%.3f %s)",
                      skill, best_sim, db_match)

    log.info("[STAGE2-MAP] %d/%d skills mapped to DB", len(mapped), len(extracted))
    return mapped


# ── Stage 3: LLM-as-a-Judge ───────────────────────────────────────────

def judge_skills_llm(
    extracted_raw:  list[str],   # Pass 1 output
    mapped_to_db:   list[str],   # Pass 2 output
    db_skills:      list[str],   # all skills in DB
    resume_text:    str,
) -> list[str]:
    """
    Stage 3 — LLM-as-a-Judge (Verifier LLM).

    Inputs given to the judge:
      A. Skills extracted verbatim from the resume (Pass 1)
      B. Those skills mapped to DB canonical names (Pass 2)
      C. FULL list of all skills that exist in the DB
      D. Original resume text for grounding

    The judge's job:
      1. From list B, REMOVE any skill the candidate clearly does NOT have
         based on the resume text (catches false cosine matches)
      2. From list C (DB skills), ADD any skill that IS in the resume
         but was missed by Pass 1 or Pass 2
      3. Return the final, accurate, clean list

    This makes the mapping fully dynamic — the judge sees the LIVE DB
    contents and can recover any skill from it that belongs to this person.
    """
    if not db_skills:
        return mapped_to_db

    db_sample = db_skills[:300]   # cap to avoid token overflow

    prompt = (
        "You are a senior technical recruiter and skill verification expert.\n\n"
        "You have four inputs:\n\n"
        f"A) Skills extracted from resume (raw):\n{', '.join(extracted_raw)}\n\n"
        f"B) Those skills matched to our database (may have errors):\n{', '.join(mapped_to_db)}\n\n"
        f"C) All skills that exist in our database (use this to find missed skills):\n"
        f"{', '.join(db_sample)}\n\n"
        "D) Original resume text (ground truth):\n"
        f"{resume_text[:5000]}\n\n"
        "YOUR TASKS:\n\n"
        "TASK 1 — REMOVE from list B any skill that the candidate does NOT actually have.\n"
        "  - A cosine match might have mapped 'librosa' to 'audio processing' incorrectly\n"
        "  - Remove any DB skill in list B that has no connection to what is written in the resume\n\n"
        "TASK 2 — ADD from list C any skill that the candidate CLEARLY HAS based on the resume,\n"
        "  but which is missing from list B. Examples of what to look for:\n"
        "  - Resume says 'NLP models/pipelines' -> add 'natural language processing' or 'nlp' if in list C\n"
        "  - Resume says 'Computer Vision models' -> add 'computer vision' if in list C\n"
        "  - Resume says 'Generative AI models' -> add 'generative ai' if in list C\n"
        "  - Resume uses Python throughout -> add 'python' if in list C\n"
        "  - Resume mentions a skill by abbreviation (e.g. RAG) -> add full form if it is in list C\n"
        "  - Resume mentions a skill in Experience/Projects that list A/B missed\n\n"
        "TASK 3 — DEDUPLICATE: if list B has both the abbreviation and full form of the same skill,\n"
        "  keep whichever form exists in list C. Remove the other.\n\n"
        "RULES:\n"
        "  - Only add skills from list C (must exist in the database)\n"
        "  - Only add a skill if there is CLEAR EVIDENCE in the resume text\n"
        "  - Do not infer or hallucinate skills\n"
        "  - Do not add soft skills\n\n"
        "Return ONLY a comma-separated list of the final verified skills.\n"
        "One line. No explanation. No numbering."
    )

    resp = groq_client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=600,
    )
    raw      = resp.choices[0].message.content.strip()
    verified = _parse_skill_list(raw)

    # Safety: only keep skills that actually exist in the DB
    db_set   = set(db_skills)
    safe     = [s for s in verified if s in db_set]
    # If judge hallucinated all new skills, fall back to mapped_to_db
    if not safe:
        log.warning("[STAGE3-JUDGE] No DB-valid skills in judge output — using Stage 2 result")
        return mapped_to_db

    log.info("[STAGE3-JUDGE] Final: %d skills  (added %d, removed %d)",
             len(safe),
             len(set(safe) - set(mapped_to_db)),
             len(set(mapped_to_db) - set(safe)))
    return safe



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

    # ── 3-Stage Dynamic Skill Extraction Pipeline ──────────────────
    # Stage 1: Extractor LLM — broad verbatim extraction from resume
    stage1_raw = extract_skills_llm(resume_text)
    if not stage1_raw:
        raise HTTPException(400, "No skills detected in resume")
    log.info("[PIPELINE] Stage 1 done: %d raw skills", len(stage1_raw))

    # Stage 2: Cosine-similarity mapping to DB canonical skill names
    #   - uses dual threshold (0.80 high / 0.55+guard mid)
    #   - returns only skills that exist in the DB
    stage2_mapped = map_skills_to_db(stage1_raw)
    log.info("[PIPELINE] Stage 2 done: %d mapped to DB", len(stage2_mapped))

    # Stage 3: LLM-as-a-Judge — sees raw extraction, mapped result,
    #   FULL DB skill list, and original resume. Removes false matches,
    #   recovers missed skills directly from DB. 100% dynamic.
    all_db_skills = get_all_skills_from_db()
    skills = judge_skills_llm(
        extracted_raw = stage1_raw,
        mapped_to_db  = stage2_mapped,
        db_skills     = all_db_skills,
        resume_text   = resume_text,
    )
    log.info("[PIPELINE] Stage 3 done: %d final skills", len(skills))

    # Display skills: use Stage 1 raw (human-readable) for the UI
    # DB-mapped skills (Stage 3 output) go into gap analysis
    raw_skills = stage1_raw  # shown in "Your Extracted Skills" card
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
        "gap":         gap,
        "insight":     insight,
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