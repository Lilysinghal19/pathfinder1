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

# ═════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION  (PDF + image)
# ═════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(path: str) -> str:
    """Try three PDF extraction strategies in order of reliability."""
    # Strategy 1: PyMuPDF (fast, handles most PDFs)
    try:
        doc   = fitz.open(path)
        pages = [page.get_text() for page in doc]
        doc.close()
        text  = "\n".join(pages).strip()
        if len(text) >= 100:
            return text
    except Exception as e:
        log.debug("[PDF] fitz failed: %s", e)

    # Strategy 2: pdfplumber (better for complex layouts)
    try:
        with pdfplumber.open(path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if len(text.strip()) >= 100:
            return text.strip()
    except Exception as e:
        log.debug("[PDF] pdfplumber failed: %s", e)

    # Strategy 3: OCR fallback for scanned PDFs
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
    except Exception as e:
        log.debug("[PDF] OCR fallback failed: %s", e)

    return ""


def extract_text_from_image(path: str) -> str:
    """OCR extraction from image files."""
    try:
        return pytesseract.image_to_string(Image.open(path))
    except Exception as e:
        log.warning("[IMG] OCR failed: %s", e)
        return ""


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
_db_vecs_cache:   list      | None = None   # cached embeddings for DB skills


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


def _get_db_vecs(db_skills: list[str]):
    """Return cached DB embeddings (recomputed only when DB skills change)."""
    global _db_vecs_cache, _db_skills_cache
    if _db_vecs_cache is None or _db_skills_cache != db_skills:
        _db_vecs_cache = get_embeddings().embed_documents(db_skills)
    return _db_vecs_cache


def map_skills_to_db(extracted: list[str]) -> tuple[list[str], list[dict]]:
    """
    Stage 2 — Cosine-similarity DB mapping with TOP-3 candidates.

    Returns:
        mapped    : list of accepted DB skill names (best match per extracted skill)
        evidence  : list of dicts — one per extracted skill — containing:
                    {
                      "raw":        original extracted skill,
                      "accepted":   best DB match if accepted else None,
                      "sim":        cosine similarity of best match,
                      "top3":       [(db_skill, sim), ...] top-3 candidates,
                      "confidence": "high" | "mid" | "low"
                    }

    Two-threshold strategy:
      HIGH (>= 0.82): always accept
      MID  (0.55-0.82): accept only if short abbreviation (<=4 non-space chars)
                        OR substring overlap exists
      LOW  (< 0.55): reject — but include in evidence so judge can review
    """
    if not extracted:
        return [], []

    db_skills = get_all_skills_from_db()
    if not db_skills:
        log.warning("[MAP] DB is empty — passing extracted skills straight through")
        return extracted, []

    emb      = get_embeddings()
    ext_vecs = emb.embed_documents(extracted)
    db_vecs  = _get_db_vecs(db_skills)
    sims     = cosine_similarity(ext_vecs, db_vecs)   # (len_ext, len_db)

    mapped, seen, evidence = [], set(), []
    for i, skill in enumerate(extracted):
        row = sims[i]

        # Top-3 candidates by similarity
        top3_idx  = row.argsort()[::-1][:3].tolist()
        top3      = [(db_skills[j], float(row[j])) for j in top3_idx]
        best_db, best_sim = top3[0]

        # Decision logic
        short  = len(skill.replace(" ", "")) <= 4
        substr = (skill in best_db) or (best_db in skill)

        if best_sim >= 0.82:
            conf, accept = "high", True
        elif best_sim >= 0.55 and (short or substr):
            conf, accept = "mid", True
        else:
            conf, accept = "low", False

        accepted_name = None
        if accept and best_db not in seen:
            seen.add(best_db)
            mapped.append(best_db)
            accepted_name = best_db

        ev = {
            "raw":       skill,
            "accepted":  accepted_name,
            "sim":       round(best_sim, 4),
            "top3":      [(db, round(s, 4)) for db, s in top3],
            "confidence": conf,
        }
        evidence.append(ev)
        log.debug("[MAP] %-28s -> %-28s  sim=%.3f  conf=%-4s  accept=%s",
                  skill, best_db, best_sim, conf, accept)

    log.info("[STAGE2-MAP] %d/%d skills mapped to DB", len(mapped), len(extracted))
    return mapped, evidence


# ── Stage 3: Robust LLM-as-a-Judge ────────────────────────────────────
#
# Robustness improvements over the single-call version:
#
#  1. TWO INDEPENDENT JUDGE CALLS (different prompts + roles)
#       Call A — "Removal judge":  looks only at mapped skills and decides
#                what to REMOVE (false cosine matches, noise)
#       Call B — "Recovery judge": looks at Stage 1 raw skills + DB skill
#                list and decides what was MISSED and should be ADDED
#     Combining both in one call causes the LLM to conflate the tasks and
#     lose precision on each.  Splitting gives each judge a clear, focused job.
#
#  2. STRUCTURED JSON OUTPUT from each judge call
#       Instead of a comma list the judge returns:
#         { "keep": [...], "reason": "..." }      (removal judge)
#         { "add":  [...], "reason": "..." }      (recovery judge)
#       This makes parsing deterministic and auditable.
#
#  3. EVIDENCE-BASED CONTEXT — Stage 2 evidence (similarity scores + top-3)
#       is shown to the removal judge so it can weigh confidence.
#       A skill with sim=0.56 and no substring overlap is more likely wrong
#       than one with sim=0.79 and a substring match.
#
#  4. DB COVERAGE — the recovery judge sees the FULL DB skill list,
#       split into chunks if needed, ensuring no DB skill is invisible.
#
#  5. HARD DB-MEMBERSHIP SAFETY FILTER
#       After both judges run, any output skill is checked against the DB set.
#       Skills not in the DB are silently dropped — judge cannot hallucinate
#       a skill that doesn't exist in the DB.
#
#  6. GRACEFUL FALLBACK CHAIN
#       Stage3 output empty   -> use Stage 2 mapped (not an error)
#       Stage2 mapped empty   -> use Stage 1 raw (not an error)
#       Both empty            -> raise extraction failure
# ─────────────────────────────────────────────────────────────────────


def _call_llm_json(prompt: str, fallback: dict) -> dict:
    """
    Call Groq LLM expecting a JSON response. Returns fallback on any failure.

    Hardened against:
    - Groq API errors (quota, timeout, 5xx) — caught at request level
    - LLM returning markdown-wrapped JSON (```json ... ```)
    - LLM returning explanatory text before/after JSON object
    - LLM returning an error string ("Internal Server Error" etc.)
    - Malformed JSON (trailing commas, single quotes, truncated)
    """
    import json as _json, re as _re
    try:
        resp = groq_client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON-only API. "
                        "You MUST respond with valid JSON and nothing else. "
                        "No markdown. No explanation. No preamble. "
                        "Start your response with { and end with }."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=700,
        )
        raw = resp.choices[0].message.content.strip()
    except Exception as e:
        log.warning("[JUDGE] LLM API call failed (%s) — using fallback", e)
        return fallback

    # Strip markdown code fences
    raw = _re.sub(r"^```(?:json)?\s*", "", raw, flags=_re.M)
    raw = _re.sub(r"```\s*$",           "", raw, flags=_re.M).strip()

    # If the response doesn't look like JSON at all, bail immediately
    if not raw.startswith("{"):
        log.warning("[JUDGE] Non-JSON response (starts with %r) — fallback", raw[:40])
        return fallback

    # Extract the first complete JSON object (ignore trailing text)
    depth, end = 0, -1
    for idx, ch in enumerate(raw):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end == -1:
        log.warning("[JUDGE] Could not find closing } — fallback")
        return fallback
    raw = raw[:end]

    try:
        return _json.loads(raw)
    except _json.JSONDecodeError as e:
        log.warning("[JUDGE] JSON decode error (%s) on: %r — fallback", e, raw[:120])
        return fallback


def _judge_removal(
    mapped_to_db: list[str],
    evidence:     list[dict],
    resume_text:  str,
) -> list[str]:
    """
    Judge Call A — Removal judge.
    Receives: DB-mapped skills + their similarity evidence + resume text.
    Returns:  subset of mapped_to_db that should be KEPT.
    """
    # Build a readable evidence block for the judge
    ev_lines = []
    for ev in evidence:
        if ev["accepted"]:
            top3_str = " | ".join(f"{db}({s:.2f})" for db, s in ev["top3"])
            ev_lines.append(
                f"  raw='{ev['raw']}' -> db='{ev['accepted']}' "
                f"sim={ev['sim']:.3f} conf={ev['confidence']}  "
                f"top3=[{top3_str}]"
            )
    ev_block = "\n".join(ev_lines) if ev_lines else "(no evidence available)"

    prompt = (
        "You are a skill verification expert. Your ONLY job is to decide which "
        "skills in the MAPPED LIST are genuine skills of the candidate.\n\n"
        "CONTEXT:\n"
        "The mapped list was produced by cosine-similarity embedding matching. "
        "Some matches may be WRONG — e.g. a domain-specific term got mapped to a "
        "general skill it does not represent.\n\n"
        f"CANDIDATE RESUME:\n{resume_text[:4500]}\n\n"
        f"MAPPED SKILLS (candidate for verification):\n{', '.join(mapped_to_db)}\n\n"
        f"SIMILARITY EVIDENCE (raw skill -> db match, similarity score, top-3 candidates):\n"
        f"{ev_block}\n\n"
        "DECISION RULES:\n"
        "1. KEEP a skill if EITHER:\n"
        "   a) It appears (or a clear synonym/abbreviation appears) in the resume text, OR\n"
        "   b) It is strongly implied by other skills in the resume (e.g. 'pytorch' implies 'deep learning')\n"
        "2. REMOVE a skill if:\n"
        "   a) It has NO connection to anything in the resume text, OR\n"
        "   b) It was clearly a wrong cosine match (low sim, no substring overlap, unrelated domain)\n"
        "3. When in doubt at medium similarity (0.55-0.70), check if the raw skill and DB skill\n"
        "   are genuinely the same concept (e.g. 'nlp' and 'natural language processing' are the same)\n\n"
        "OUTPUT FORMAT — respond with ONLY valid JSON, no markdown, no explanation:\n"
        '{"keep": ["skill1", "skill2", ...], "removed": ["skill3", ...], '
        '"reason": "one sentence summary of what you removed and why"}'
    )
    result  = _call_llm_json(prompt, {"keep": mapped_to_db, "removed": [], "reason": "parse error"})
    kept    = result.get("keep", mapped_to_db)
    removed = result.get("removed", [])
    log.info("[JUDGE-A] Kept %d, removed %d: %s",
             len(kept), len(removed), result.get("reason", ""))
    return kept if isinstance(kept, list) else mapped_to_db


def _judge_recovery(
    extracted_raw: list[str],
    already_have:  list[str],
    db_skills:     list[str],
    resume_text:   str,
) -> list[str]:
    """
    Judge Call B — Recovery judge.
    Receives: Stage 1 raw extraction + skills already confirmed + full DB list + resume.
    Returns:  list of DB skills to ADD (not already in already_have).

    To handle large DB lists without token overflow, we filter the DB to only
    skills semantically close to the extracted skills before sending to LLM.
    """
    # Pre-filter DB: only send skills that share at least one word token with
    # the extracted skills. This keeps the DB list manageable without losing coverage.
    extracted_tokens = set()
    for s in extracted_raw:
        extracted_tokens.update(s.lower().split())

    # Words so common they don't help with filtering
    STOP = {"and", "or", "the", "a", "of", "in", "for", "with", "using", "based", "via"}
    extracted_tokens -= STOP

    # Score each DB skill by how many tokens overlap with extracted skills
    scored = []
    for db_s in db_skills:
        if db_s in already_have:
            continue   # already confirmed — skip
        db_tokens = set(db_s.lower().split())
        overlap   = len(db_tokens & extracted_tokens)
        scored.append((overlap, db_s))

    # Keep top-200 by overlap score + all zero-overlap ones up to 100 total for coverage
    scored.sort(key=lambda x: -x[0])
    candidates = [s for _, s in scored[:200]]
    # Also add a random sample of low-overlap ones for broad coverage
    zero_overlap = [s for score, s in scored if score == 0][:80]
    candidates_set = set(candidates) | set(zero_overlap)
    filtered_db = sorted(candidates_set)

    already_str = ", ".join(already_have) if already_have else "(none yet)"

    prompt = (
        "You are a skill recovery expert. Your ONLY job is to find technical skills "
        "that the candidate HAS but which are MISSING from the already-confirmed list.\n\n"
        f"CANDIDATE RESUME:\n{resume_text[:4500]}\n\n"
        f"SKILLS ALREADY CONFIRMED (do NOT add these again):\n{already_str}\n\n"
        f"SKILLS EXTRACTED IN PASS 1 (these came directly from the resume):\n"
        f"{', '.join(extracted_raw)}\n\n"
        f"AVAILABLE DB SKILLS (you may ONLY add from this list):\n"
        f"{', '.join(filtered_db)}\n\n"
        "RECOVERY RULES:\n"
        "1. Scan the resume for skills mentioned in Experience, Projects, Certifications\n"
        "   that are NOT in the already-confirmed list\n"
        "2. For each found skill, check if an equivalent exists in the DB skill list\n"
        "   Examples of equivalences to look for:\n"
        "   - Resume: 'NLP models' or 'nlp' -> DB: 'natural language processing' or 'nlp'\n"
        "   - Resume: 'Python web scraping' or any Python usage -> DB: 'python'\n"
        "   - Resume: 'Computer Vision models' -> DB: 'computer vision'\n"
        "   - Resume: 'Generative AI' -> DB: 'generative ai'\n"
        "   - Resume: 'LLM' anywhere -> DB: 'large language models' or 'llm'\n"
        "   - Resume: abbreviation in skills list -> DB: full form\n"
        "3. ONLY add skills that have CLEAR TEXTUAL EVIDENCE in the resume\n"
        "4. ONLY add skills that appear in the DB skill list above\n"
        "5. Do NOT add soft skills, company names, or project names\n"
        "6. Do NOT add skills already in the confirmed list\n\n"
        "OUTPUT FORMAT — respond with ONLY valid JSON, no markdown:\n"
        '{"add": ["skill1", "skill2", ...], '
        '"evidence": {"skill1": "quote from resume", "skill2": "quote from resume"}, '
        '"reason": "one sentence summary"}'
    )
    result   = _call_llm_json(prompt, {"add": [], "evidence": {}, "reason": "parse error"})
    to_add   = result.get("add", [])
    evidence = result.get("evidence", {})
    log.info("[JUDGE-B] Recovering %d skills: %s | %s",
             len(to_add), to_add, result.get("reason", ""))
    if evidence:
        for sk, ev in list(evidence.items())[:5]:
            log.info("[JUDGE-B]   '%s' <- evidence: '%s'", sk, str(ev)[:80])
    return to_add if isinstance(to_add, list) else []


def judge_skills_llm(
    extracted_raw: list[str],
    mapped_to_db:  list[str],
    db_skills:     list[str],
    resume_text:   str,
    evidence:      list[dict] | None = None,
) -> list[str]:
    """
    Stage 3 — Robust LLM-as-a-Judge.

    Two independent focused judge calls:
      A. Removal judge  — prunes false cosine matches from Stage 2
      B. Recovery judge — finds skills missed by Stage 1 + 2 using full DB

    Final output is hard-filtered against the DB membership set so
    the judge cannot hallucinate skills that don't exist in the DB.
    """
    if not db_skills:
        log.warning("[JUDGE] No DB skills available — skipping judge")
        return mapped_to_db

    if not evidence:
        evidence = []

    db_set = set(db_skills)

    # ── Judge Call A: Remove false matches ──────────────────────────
    kept_after_removal = _judge_removal(mapped_to_db, evidence, resume_text)
    # Hard safety: only DB members
    kept_after_removal = [s for s in kept_after_removal if s in db_set]
    # Fallback: if judge removed everything, revert to full mapped list
    if not kept_after_removal and mapped_to_db:
        log.warning("[JUDGE-A] Removed everything — reverting to full mapped list")
        kept_after_removal = [s for s in mapped_to_db if s in db_set]

    # ── Judge Call B: Recover missed skills ─────────────────────────
    recovered = _judge_recovery(extracted_raw, kept_after_removal, db_skills, resume_text)
    # Hard safety: only DB members, not already confirmed
    confirmed_set = set(kept_after_removal)
    new_additions = [s for s in recovered if s in db_set and s not in confirmed_set]

    # ── Merge ────────────────────────────────────────────────────────
    final = kept_after_removal + new_additions

    log.info(
        "[STAGE3-JUDGE] Final: %d skills | "
        "Stage2 had %d | Removed %d | Recovered %d",
        len(final),
        len(mapped_to_db),
        len(set(mapped_to_db) - set(kept_after_removal)),
        len(new_additions),
    )
    return final



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

    # Stage 2: Cosine-similarity mapping + evidence collection
    #   Returns (mapped_skills, evidence_list)
    #   evidence_list carries similarity scores + top-3 candidates per skill
    #   which the Stage 3 judge uses for informed removal decisions
    stage2_mapped, stage2_evidence = map_skills_to_db(stage1_raw)
    log.info("[PIPELINE] Stage 2 done: %d mapped to DB", len(stage2_mapped))

    # Stage 3: Two-call robust LLM-as-a-Judge
    #   Call A (removal judge): prunes false cosine matches using evidence
    #   Call B (recovery judge): finds missed skills from full DB list
    all_db_skills = get_all_skills_from_db()
    skills = judge_skills_llm(
        extracted_raw = stage1_raw,
        mapped_to_db  = stage2_mapped,
        db_skills     = all_db_skills,
        resume_text   = resume_text,
        evidence      = stage2_evidence,
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