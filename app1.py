import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
import streamlit.components.v1 as components

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gcp_key.json"



load_dotenv()

#tokeN ──────────────────────────────────────────────────
_hf = os.getenv("HF_TOKEN") or os.getenv("hf_token")
if _hf:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = _hf
    os.environ["HF_TOKEN"] = _hf

from chain import analyze_skill_gap, chat_with_knowledge_base
from retriever import get_available_roles, get_vectorstore, get_embeddings

import pytesseract
from PIL import Image
import pdfplumber
import fitz
import io
from groq import Groq
import config
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# Tesseract path for Windows — adjust if installed elsewhere
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title = "AI Skill Gap Analyzer",
    page_icon  = "🚀",
    layout     = "wide",
)

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
    font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] {
    background: #f1f5f9;
}
.stButton>button {
    background: linear-gradient(90deg, #2E5BFF, #4F46E5);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}
.card {
    backdrop-filter: blur(10px);
    background: rgba(255,255,255,0.85);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}
.tag {
    display: inline-block;
    padding: 6px 10px;
    margin: 4px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    color: white;
}
.chat-bubble-user {
    background: #ede9fe;
    border-radius: 14px 14px 2px 14px;
    padding: 10px 14px;
    margin: 6px 0 6px auto;
    color: #3730a3;
    max-width: 80%;
    text-align: right;
}
.chat-bubble-ai {
    background: #f0fdf4;
    border-left: 3px solid #16a34a;
    border-radius: 14px 14px 14px 2px;
    padding: 10px 14px;
    margin: 6px auto 6px 0;
    color: #166534;
    max-width: 80%;
}
.chat-wrap {
    max-height: 400px;
    overflow-y: auto;
    padding: 12px;
    background: #fafafa;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SESSION STATE — persists analysis result and chat history
# across Streamlit reruns
# ══════════════════════════════════════════════════════════════
if "analysis_done"   not in st.session_state:
    st.session_state.analysis_done   = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "ai_insight"      not in st.session_state:
    st.session_state.ai_insight      = ""
if "gap_context_str" not in st.session_state:
    st.session_state.gap_context_str = ""
if "chat_history"    not in st.session_state:
    st.session_state.chat_history    = []   # [{role, content}, ...]

# ══════════════════════════════════════════════════════════════
# GROQ CLIENT
# ══════════════════════════════════════════════════════════════
client = Groq(api_key=config.GROQ_API_KEY)

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def render_tags(skills: list, color: str) -> str:
    return " ".join([
        f"<span class='tag' style='background:{color}'>{s}</span>"
        for s in skills
    ])


def extract_skills_llm(resume_text: str, target_role: str = "") -> list[str]:
    """
    Two-pass extraction:
    Pass 1: Extract all explicitly mentioned technical skills
    Pass 2: Validate against role context
    No inferring. No adding. Only what is in the resume.
    """
    prompt = f"""You are a resume skill parser.

Your ONLY job: extract technical skills EXPLICITLY written in this resume.

Rules:
1. ONLY include skills that appear word-for-word in the resume text
2. DO NOT add, infer, or assume any skill not written
3. Normalize: lowercase, fix obvious typos (eg "pytoch" → "pytorch")
4. Split combined skills: "pandas, numpy" → separate items
5. Keep multi-word skills intact: "machine learning", "deep learning", "natural language processing"
6. Include: frameworks, libraries, languages, tools, platforms, methods, certifications
7. Exclude: soft skills, job titles, company names, degree names, years, percentages
8.if heading have any skills then you extract that also 
9.if skills are written together with any operators so extract them separately

Resume text:
{resume_text[:4000]}

Return ONLY a comma-separated list. Nothing else."""

    response = client.chat.completions.create(
        model       = config.GROQ_MODEL,
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0,
        max_tokens  = 600,
    )

    raw = response.choices[0].message.content.strip()

    # Clean the response
    skills = []
    for s in raw.split(","):
        s = s.strip().lower()
        # Remove numbering artifacts like "1. python" → "python"
        s = s.lstrip("0123456789.-) ")
        # Skip empty, too short, or obviously wrong
        if not s or len(s) < 2:
            continue
        if any(bad in s for bad in [
            "year", "month", "%", "winner", "university",
            "bachelor", "master", "phd", "cgpa", "gpa",
            "internship", "training", "certification", "award"
        ]):
            continue
        skills.append(s)

    return list(dict.fromkeys(skills))  # deduplicate preserving order

    


@st.cache_data
def get_all_skills_from_db() -> list[str]:
    """Fetch all unique skills from ChromaDB. Cached."""
    vs   = get_vectorstore()
    data = vs.get(include=["metadatas"])
    skills = set()
    for m in data["metadatas"]:
        # Fixed: split on ", " to keep multi-word skills intact
        for s in m.get("skills_text", "").split(", "):
            if s.strip():
                skills.add(s.strip().lower())
    return sorted(skills)


def map_skills_to_db(
    extracted_skills: list[str],
    threshold       : float = 0.75,
) -> list[str]:
    """Map extracted skills to closest canonical skills in ChromaDB."""
    if not extracted_skills:
        return []
    db_skills = get_all_skills_from_db()
    emb       = get_embeddings()
    ext_vecs  = emb.embed_documents(extracted_skills)
    db_vecs   = emb.embed_documents(db_skills)
    mapped = []
    for i, skill in enumerate(extracted_skills):
        sims       = cosine_similarity([ext_vecs[i]], db_vecs)[0]
        best_idx   = sims.argmax()
        best_score = sims[best_idx]
        if best_score >= threshold:
            mapped.append(db_skills[best_idx])
    return list(set(mapped))


def extract_text_from_pdf(file_path: str) -> str:
    """Three-layer extraction: PyMuPDF → pdfplumber → OCR."""
    # Layer 1: PyMuPDF
    try:
        doc   = fitz.open(file_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        text  = "\n".join(pages).strip()
        if len(text) >= 100:
            return text
    except Exception:
        pass

    # Layer 2: pdfplumber
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if len(text.strip()) >= 100:
            return text.strip()
    except Exception:
        pass

    # Layer 3: OCR for scanned/image PDFs
    try:
        doc   = fitz.open(file_path)
        texts = []
        for page in doc:
            mat  = fitz.Matrix(2, 2)
            pix  = page.get_pixmap(matrix=mat)
            img  = Image.open(io.BytesIO(pix.tobytes("png")))
            texts.append(pytesseract.image_to_string(img, config="--psm 6"))
        doc.close()
        return "\n".join(texts).strip()
    except Exception as e:
        st.warning(f"OCR failed: {e}")
        return ""


def extract_text_from_image(file_path: str) -> str:
    img = Image.open(file_path)
    return pytesseract.image_to_string(img)


def build_gap_context_str(gap: dict, insight: str) -> str:
    """
    Build a rich context string combining gap data + AI insight.
    This is passed to chat_with_knowledge_base as extra_context.
    """
    return f"""Target Role   : {gap['target_role']}
Match Score   : {gap['match_pct']}%
Gap Score     : {gap['gap_pct']}%
User Skills   : {', '.join(gap['user_skills'])}
Matching      : {', '.join(gap['matching']) or 'None'}
Missing       : {', '.join(gap['missing'][:10])}

AI Insight:
{insight}"""


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='font-size:38px; margin-bottom:4px;'>🚀 AI Skill Gap Analyzer</h1>
<p style='color:#64748b; font-size:15px; margin-top:0;'>
  Upload resume → Get instant AI insights → Chat with your career coach
</p>
""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 1 — UPLOAD + ANALYZE
# ══════════════════════════════════════════════════════════════
st.markdown("### 📊 Skill Gap Analysis")

col_left, col_right = st.columns([2, 1])

with col_left:
    uploaded_file = st.file_uploader(
        "📄 Upload Resume",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Supports text PDFs, scanned PDFs, and images",
    )

with col_right:
    roles       = get_available_roles()
    target_role = st.selectbox("🎯 Select Target Role", roles)
    tone        = st.selectbox(
        "Advice Tone",
        ["concise", "encouraging", "strict", "technical"],
        index=0,
    )

analyze_btn = st.button("🔍 Analyze Skill Gap", use_container_width=True)

# ── Run analysis on button click ──────────────────────────────
if analyze_btn and uploaded_file:

    suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".png"

    with st.spinner("Extracting resume text..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        resume_text = (
            extract_text_from_pdf(file_path)
            if uploaded_file.type == "application/pdf"
            else extract_text_from_image(file_path)
        )

    if not resume_text.strip():
        st.error("❌ Could not extract text from this file.")
        st.stop()

    with st.spinner("Extracting skills with AI..."):
        raw_skills = extract_skills_llm(resume_text)

    if not raw_skills:
        st.error("❌ No skills detected in resume.")
        st.stop()

    with st.spinner("Mapping skills to database..."):
        
        skills = map_skills_to_db(raw_skills)
        skills_str = ", ".join(skills)

        

    with st.spinner("Running skill gap analysis..."):
        result = analyze_skill_gap(
            user_skills_str = skills_str,
            target_role     = target_role,
            tone            = tone,
        )

    if result["error"]:
        st.error(f"Analysis error: {result['error']}")
        if result.get("suggestions"):
            st.info("Did you mean: " + ", ".join(
                s["role"] for s in result["suggestions"]
            ))
    else:
        # Store everything in session state
        st.session_state.analysis_done   = True
        st.session_state.analysis_result = result
        st.session_state.ai_insight      = result["explanation"]
        st.session_state.gap_context_str = build_gap_context_str(
            result["gap_report"], result["explanation"]
        )
        # Reset chat when new analysis runs
        st.session_state.chat_history = []

elif analyze_btn and not uploaded_file:
    st.warning("Please upload a resume file first.")
    if "voice_input" not in st.session_state:
        st.session_state.voice_input = ""

# ══════════════════════════════════════════════════════════════
# SECTION 2 — RESULTS
# Rendered from session_state so they persist after chat reruns
# ══════════════════════════════════════════════════════════════
if st.session_state.analysis_done and st.session_state.analysis_result:

    result = st.session_state.analysis_result
    gap    = result["gap_report"]

    st.markdown("---")

    # ── Extracted vs Mapped Skills ────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='card'>
        <h4>🧾 Extracted Skills</h4>
        {render_tags(gap['user_skills'], '#6366f1')}
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='card'>
        <h4>🧠 Mapped Skills</h4>
        {render_tags(gap['user_skills'], '#0ea5e9')}
        </div>
        """, unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"<div class='card'><h4>🎯 Match</h4><h2>{gap['match_pct']}%</h2></div>", unsafe_allow_html=True)
    m2.markdown(f"<div class='card'><h4>⚠️ Gap</h4><h2>{gap['gap_pct']}%</h2></div>", unsafe_allow_html=True)
    m3.markdown(f"<div class='card'><h4>✅ Matching</h4><h2>{len(gap['matching'])}</h2></div>", unsafe_allow_html=True)
    m4.markdown(f"<div class='card'><h4>❌ Missing</h4><h2>{len(gap['missing'])}</h2></div>", unsafe_allow_html=True)

    # ── Bar chart ─────────────────────────────────────────────
    fig = go.Figure()
    fig.add_bar(
        x            = ["Match %", "Gap %"],
        y            = [gap["match_pct"], gap["gap_pct"]],
        text         = [f"{gap['match_pct']}%", f"{gap['gap_pct']}%"],
        textposition = "auto",
        marker_color = ["#16a34a", "#dc2626"],
    )
    fig.update_layout(
        template   = "plotly_white",
        height     = 260,
        showlegend = False,
        margin     = dict(t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Matching vs Missing ───────────────────────────────────
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"""
        <div class='card'>
        <h4>✅ Matching Skills</h4>
        {render_tags(gap['matching'], '#16a34a')}
        </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""
        <div class='card'>
        <h4>❌ Missing Skills</h4>
        {render_tags(gap['missing'][:12], '#dc2626')}
        </div>
        """, unsafe_allow_html=True)

    # ── AI Insight ────────────────────────────────────────────
    st.subheader("🤖 AI Insight")
    st.markdown(f"""
    <div class='card'>
    <p style='white-space: pre-wrap; line-height: 1.7;'>
    {st.session_state.ai_insight}
    </p>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SECTION 3 — CAREER CHATBOT
    # Lives OUTSIDE the analyze button block so it persists
    # Context = AI insight + gap data passed as extra_context
    # Memory = last 10 messages via session_state + LangChain store
    # ══════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("💬 Career Chatbot")

    st.markdown(
        f"<p style='color:#64748b; font-size:13px;'>"
        f"Context loaded: <b>{gap['target_role']}</b> | "
        f"Match <b>{gap['match_pct']}%</b> | "
        f"Gap <b>{gap['gap_pct']}%</b> — "
        f"Ask anything about your career, skills, or learning path."
        f"</p>",
        unsafe_allow_html=True,
    )

    # Suggested quick questions
    if not st.session_state.chat_history:
        st.markdown("**Quick questions to get started:**")
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("Which skill to learn first?"):
                st.session_state._pending_q = (
                    "Which missing skill should I learn first and why?"
                )
        with qc2:
            if st.button("How long to close this gap?"):
                st.session_state._pending_q = (
                    "How long will it take to close my skill gap "
                    "if I study 2 hours a day?"
                )
        with qc3:
            if st.button("What projects should I build?"):
                st.session_state._pending_q = (
                    "What projects should I build to demonstrate "
                    "my skills for this role?"
                )

    # Render chat history
    if st.session_state.chat_history:
        chat_html = "<div class='chat-wrap'>"
        for msg in st.session_state.chat_history:
            content = msg["content"].replace("\n", "<br>")
            if msg["role"] == "user":
                chat_html += (
                    f"<div class='chat-bubble-user'>👤 {content}</div>"
                )
            else:
                chat_html += (
                    f"<div class='chat-bubble-ai'>🤖 {content}</div>"
                )
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    # Chat input — always visible when analysis is done
    user_query = st.chat_input(
        "Ask about your career, skills, or learning path..."
    )

    # Handle pending quick question from button click
    if "_pending_q" in st.session_state:
        user_query = st.session_state.pop("_pending_q")

    # Process the message
    if user_query:
        with st.spinner("Thinking..."):
            answer = chat_with_knowledge_base(
                question   = user_query,
                insight    = st.session_state.gap_context_str,
                session_id = "streamlit_session",
            )

        # Append to display history
        st.session_state.chat_history.append(
            {"role": "user",      "content": user_query}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

        # Keep display history trimmed to 20 items (10 pairs)
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = (
                st.session_state.chat_history[-20:]
            )

        st.rerun()

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()
# ══════════════════════════════════════════════════════════════
# 🎙️ VOICE ASSISTANT — WITH START/STOP RECORDING + STOP TTS
# ══════════════════════════════════════════════════════════════

import sounddevice as sd
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
import threading
import pyttsx3
import numpy as np

SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 30  # safety cap — recording stops earlier via button

# ───────── SESSION STATE FOR VOICE ─────────
if "voice_recording"   not in st.session_state:
    st.session_state.voice_recording   = False   # True while mic is open
if "voice_audio_frames" not in st.session_state:
    st.session_state.voice_audio_frames = []     # raw PCM chunks
if "tts_speaking"      not in st.session_state:
    st.session_state.tts_speaking      = False
if "tts_stop_flag"     not in st.session_state:
    st.session_state.tts_stop_flag     = False   # signal thread to stop
if "tts_engine_ref"    not in st.session_state:
    st.session_state.tts_engine_ref    = None    # pyttsx3 engine instance

# ───────── LOAD WHISPER ─────────
@st.cache_resource
def load_voice_models():
    return WhisperModel("small", compute_type="int8")

whisper_model = load_voice_models()

# ───────── RECORDING HELPERS ─────────
# KEY INSIGHT: Streamlit re-imports the module on every rerun, so
# module-level globals (Queue, threading.Event, thread refs) are RESET
# each time the user clicks a button. The solution is to store the
# thread reference and a shared list inside a st.cache_resource
# singleton — cache_resource objects persist for the entire server
# lifetime, surviving all reruns.

import time as _time

@st.cache_resource
def _get_recorder_state():
    """
    Returns a single dict that lives for the whole Streamlit server lifetime.
    All recording state lives here so it is never reset by reruns.
    """
    return {
        "thread":    None,
        "stop_evt":  threading.Event(),
        "chunks":    [],          # list[np.ndarray]
        "error":     "",          # PortAudio error string, or ""
        "running":   False,
    }


def _record_worker(state: dict):
    state["error"]   = ""
    state["chunks"]  = []
    state["running"] = True
    state["stop_evt"].clear()

    def _cb(indata, frames, time_info, sd_status):
        state["chunks"].append(indata.copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=2048,
            callback=_cb,
        ):
            state["stop_evt"].wait(timeout=MAX_RECORD_SECONDS)
    except Exception as exc:
        state["error"] = str(exc)
    finally:
        state["running"] = False


def start_recording():
    state = _get_recorder_state()
    state["chunks"]  = []
    state["error"]   = ""
    state["stop_evt"].clear()
    t = threading.Thread(target=_record_worker, args=(state,), daemon=True)
    state["thread"] = t
    t.start()
    _time.sleep(0.2)   # let PortAudio open the stream before returning


def stop_recording_and_save() -> tuple:
    """Returns (wav_path, error_msg). Exactly one is non-empty."""
    state = _get_recorder_state()
    state["stop_evt"].set()

    t = state.get("thread")
    if t and t.is_alive():
        t.join(timeout=5)

    error  = state.get("error", "")
    chunks = state.get("chunks", [])

    if error:
        return "", f"Microphone error: {error}"

    if not chunks:
        return "", (
            "No audio captured — the mic stream opened but collected no frames.\n"
            "This usually means the stream was stopped immediately.\n"
            "Please speak for at least 1 second before pressing Stop."
        )

    audio_np = np.concatenate(chunks, axis=0)

    if np.abs(audio_np).max() < 50:
        return "", (
            "Mic is accessible but the recording is completely silent.\n"
            "Check that the correct mic is set as default in Windows Sound settings."
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(tmp.name, SAMPLE_RATE, audio_np)
    return tmp.name, ""



# ───────── TRANSCRIBE ─────────
# Single-pass transcription. We pass initial_prompt to strongly bias
# Whisper toward English/Hindi without a second model call.
# beam_size=5 gives good accuracy; vad_filter removes silence-based
# hallucinations without re-running detection.

def transcribe_audio(audio_path: str) -> str:
    segments, info = whisper_model.transcribe(
        audio_path,
        beam_size=5,
        best_of=5,
        language=None,                      # let Whisper detect naturally
        initial_prompt=(
            "The following is a conversation in English or Hindi. "
            "Transcribe exactly what is said."
        ),
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        temperature=0.0,                    # greedy — most accurate for clear speech
    )
    text = " ".join([seg.text for seg in segments]).strip()

    # If Whisper detected a language that is neither English nor Hindi,
    # and the text looks non-Latin/non-Devanagari, return empty so the
    # user gets a "try again" prompt instead of garbage.
    detected = getattr(info, "language", "en") or "en"
    if detected not in {"en"} and text:
        # Re-run forced to English (fast, single attempt)
        segs2, _ = whisper_model.transcribe(
            audio_path,
            beam_size=5,
            language="en",
            initial_prompt="Transcribe the following English speech:",
            condition_on_previous_text=False,
            vad_filter=True,
            temperature=0.0,
        )
        text = " ".join([s.text for s in segs2]).strip()

    return text


# ───────── TTS WITH STOP SUPPORT ─────────
# pyttsx3 on Windows uses the SAPI5 COM driver.
# COM requires CoInitialize() on the thread that calls it — daemon threads
# don't get that automatically, so engine.say() silently does nothing.
# Fix: use a regular (non-daemon) thread and call pythoncom.CoInitialize()
# explicitly at the start of the thread.

@st.cache_resource
def _get_tts_state():
    """Persistent TTS control block — survives Streamlit reruns."""
    return {
        "stop_flag": False,
        "speaking":  False,
        "thread":    None,
    }


def speak(text: str):
    """
    Speak the full text in one shot.
    - CoInitialize() ensures SAPI5 COM works on this thread (Windows).
    - The entire text is queued with a single engine.say() so SAPI5
      never cuts off early (chunking caused the 10-word cutoff).
    - A stop_flag lets stop_tts() abort between sentences via an
      onWord callback that calls engine.stop() when flagged.
    """
    tts = _get_tts_state()
    tts["stop_flag"] = False
    tts["speaking"]  = True
    st.session_state.tts_speaking = True

    def run():
        try:
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pass
   
            engine = pyttsx3.init() 
            engine.setProperty("rate", 170)
            engine.setProperty("volume", 1.0)

            # onWord callback: fires before each word is spoken.
            # If stop_flag is set, halt the engine immediately.
            def on_word(name, location, length):
                if tts["stop_flag"]:
                    engine.stop()

            engine.connect("started-word", on_word)

            # Queue the ENTIRE response as a single utterance.
            # This is the key fix — one say() = one continuous speech act.
            engine.say(text)
            engine.runAndWait()

        except Exception:
            pass
        finally:
            try:
                engine.stop()
            except Exception:
                pass
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass
            tts["speaking"]  = False
            tts["stop_flag"] = False
            st.session_state.tts_speaking = False

    t = threading.Thread(target=run, daemon=False)
    tts["thread"] = t
    t.start()


def stop_tts():
    """Set stop flag — the onWord callback will call engine.stop() on next word."""
    tts = _get_tts_state()
    tts["stop_flag"] = True
    tts["speaking"]  = False
    st.session_state.tts_speaking = False


# ══════════════════════════════════════════════════════════════
# ALEXA-STYLE REAL-TIME VOICE ASSISTANT
# ══════════════════════════════════════════════════════════════

for _k, _v in {
    "va_active":    False,
    "va_status":    "idle",
    "va_audio_b64": "",
    "va_conv":      [],
    "va_error":     "",
    "_va_last":     "",
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

import base64, wave as _wave

def _b64_to_wav(b64: str) -> str:
    raw = base64.b64decode(b64)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    if raw[:4] == b"RIFF":
        tmp.write(raw); tmp.flush()
    else:
        with _wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(16000); wf.writeframes(raw)
    return tmp.name

def _run_pipeline(b64_audio: str) -> tuple:
    """Returns (query_text, response_text)."""
    wav_path = _b64_to_wav(b64_audio)
    query = transcribe_audio(wav_path)
    if not query:
        return "", ""
    response = chat_with_knowledge_base(
        question=query + " (Reply in 2-3 concise sentences.)",
        insight=st.session_state.gap_context_str,
        session_id="va_realtime",
    )
    st.session_state.va_conv.append({"role": "user",      "text": query})
    st.session_state.va_conv.append({"role": "assistant", "text": response})
    st.session_state.chat_history.append({"role": "user",      "content": f"🎤 {query}"})
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    return query, response

# ── UI ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🎤 Voice Assistant")

st.markdown("""
<style>
@keyframes va-pulse{
  0%,100%{transform:scale(1);box-shadow:0 0 0 0 rgba(239,68,68,.45)}
  50%    {transform:scale(1.07);box-shadow:0 0 0 16px rgba(239,68,68,0)}
}
@keyframes va-bar{0%,100%{height:5px}50%{height:22px}}

.va-orb{width:82px;height:82px;border-radius:50%;border:none;cursor:pointer;
  font-size:34px;display:flex;align-items:center;justify-content:center;
  margin:0 auto;transition:all .2s;box-shadow:0 6px 24px rgba(0,0,0,.15);}
.va-orb-idle{background:linear-gradient(135deg,#2E5BFF,#4F46E5);color:#fff;}
.va-orb-idle:hover{transform:scale(1.09);}
.va-orb-live{background:#ef4444;color:#fff;animation:va-pulse 1.4s infinite;}
.va-orb-busy{background:#f59e0b;color:#fff;}
.va-orb-end {background:#6b7280;color:#fff;}

.va-wave{display:flex;align-items:center;gap:4px;height:28px;justify-content:center;margin:4px 0;}
.va-wave span{display:inline-block;width:4px;border-radius:3px;
  background:linear-gradient(#7c3aed,#4f46e5);
  animation:va-bar .85s ease-in-out infinite;}
.va-wave span:nth-child(1){animation-delay:.00s}
.va-wave span:nth-child(2){animation-delay:.12s}
.va-wave span:nth-child(3){animation-delay:.24s}
.va-wave span:nth-child(4){animation-delay:.36s}
.va-wave span:nth-child(5){animation-delay:.48s}

.va-label{text-align:center;font-size:13px;font-weight:600;
  color:#64748b;margin-top:6px;min-height:20px;}

.va-bubble-user{background:#ede9fe;color:#3730a3;
  border-radius:18px 18px 4px 18px;
  padding:10px 16px;margin:5px 0 5px auto;
  max-width:82%;font-size:14px;}
.va-bubble-ai{background:#f0fdf4;color:#166534;
  border-left:4px solid #16a34a;
  border-radius:4px 18px 18px 18px;
  padding:10px 16px;margin:5px auto 5px 0;
  max-width:82%;font-size:14px;}
.va-conv-box{max-height:300px;overflow-y:auto;
  border:1px solid #e2e8f0;border-radius:12px;
  padding:12px;background:#fff;margin-top:14px;}
</style>
""", unsafe_allow_html=True)

if st.session_state.analysis_done:

    va_status = st.session_state.va_status

    # ── Process audio from browser ────────────────────────
    if st.session_state.va_audio_b64 and va_status == "processing":
        b64 = st.session_state.va_audio_b64
        st.session_state.va_audio_b64 = ""
        with st.spinner("🧠 Thinking…"):
            _, reply = _run_pipeline(b64)
        if reply:
            st.session_state.va_status = "speaking"
            speak(reply)
            _time.sleep(0.4)
        st.session_state.va_status = "listening" if st.session_state.va_active else "ended"
        st.rerun()

    # ── Orb button ────────────────────────────────────────
    _, orb_col, _ = st.columns([2, 1, 2])
    with orb_col:
        if not st.session_state.va_active:
            if st.button("🎙️  Start", use_container_width=True, key="va_start"):
                stop_tts()
                st.session_state.va_active  = True
                st.session_state.va_status  = "listening"
                st.session_state.va_conv    = []
                st.session_state.va_error   = ""
                st.session_state._va_last   = ""
                st.rerun()
        else:
            if st.button("⏹  End Session", use_container_width=True, key="va_end"):
                stop_tts()
                st.session_state.va_active = False
                st.session_state.va_status = "ended"
                st.rerun()

    # ── Status visual ──────────────────────────────────────
    if va_status == "speaking":
        st.markdown(
            "<div class='va-wave'><span></span><span></span>"
            "<span></span><span></span><span></span></div>",
            unsafe_allow_html=True,
        )

    labels = {
        "idle":       "Press Start to begin",
        "listening":  "🔴 Listening — speak naturally, pause to send",
        "processing": "🧠 Understanding…",
        "speaking":   "🔊 Speaking…",
        "ended":      "✅ Session ended — press Start to begin again",
    }
    st.markdown(
        f"<div class='va-label'>{labels.get(va_status,'')}</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.va_error:
        st.error(st.session_state.va_error)

    # ── VAD component: injected only while session is live ──
    # JS opens the mic, monitors volume, records while you speak,
    # and when silence > 1.2 s it encodes the clip as base64 and
    # stuffs it into the hidden text_input → triggers Streamlit rerun.
    if st.session_state.va_active:
        va_input = st.text_input(
            "va_audio", value="", key="va_audio_widget",
            label_visibility="collapsed",
        )
        if va_input and va_input != st.session_state._va_last:
            st.session_state._va_last   = va_input
            st.session_state.va_audio_b64 = va_input
            st.session_state.va_status    = "processing"
            stop_tts()
            st.rerun()

        components.html("""
<script>
(function(){
  if(window._vaInit) return;
  window._vaInit = true;

  const SILENCE_MS    = 1200;
  const MIN_SPEECH_MS = 350;
  const THRESHOLD     = 10;    // volume threshold (0-255)

  let recorder, audioCtx, analyser, srcNode, stream;
  let chunks=[], recording=false, silTimer=null, speechStart=null;

  async function init(){
    try{
      stream = await navigator.mediaDevices.getUserMedia({audio:true,video:false});
    }catch(e){ console.warn("Mic denied",e); return; }

    audioCtx = new (window.AudioContext||window.webkitAudioContext)({sampleRate:16000});
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    srcNode  = audioCtx.createMediaStreamSource(stream);
    srcNode.connect(analyser);

    const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
                 ? "audio/webm;codecs=opus" : "audio/webm";
    recorder = new MediaRecorder(stream, {mimeType: mime});
    recorder.ondataavailable = e=>{ if(e.data.size>0) chunks.push(e.data); };
    recorder.onstop = async ()=>{
      const dur = speechStart ? Date.now()-speechStart : 0;
      const saved = [...chunks]; chunks=[];
      if(dur < MIN_SPEECH_MS || saved.length===0) return;
      const blob  = new Blob(saved, {type:mime});
      const ab    = await blob.arrayBuffer();
      const bytes = new Uint8Array(ab);
      let b64="";
      for(let i=0;i<bytes.length;i+=8192)
        b64 += btoa(String.fromCharCode(...bytes.slice(i,i+8192)));
      pushToInput(b64);
    };

    const freqBuf = new Uint8Array(analyser.frequencyBinCount);
    function tick(){
      analyser.getByteFrequencyData(freqBuf);
      const vol = freqBuf.reduce((a,b)=>a+b,0)/freqBuf.length;

      if(vol > THRESHOLD){
        clearTimeout(silTimer); silTimer=null;
        if(!recording){
          recording=true; speechStart=Date.now();
          chunks=[]; recorder.start(100);
        }
      } else if(recording && !silTimer){
        silTimer = setTimeout(()=>{
          if(recording){ recording=false; recorder.stop(); }
          silTimer=null;
        }, SILENCE_MS);
      }
      requestAnimationFrame(tick);
    }
    tick();
  }

  function pushToInput(b64){
    // Find the hidden text_input Streamlit rendered and set its value
    const doc = window.parent ? window.parent.document : document;
    const inputs = doc.querySelectorAll('input[type="text"]');
    for(const inp of inputs){
      try{
        const setter = Object.getOwnPropertyDescriptor(
          window.HTMLInputElement.prototype,"value").set;
        setter.call(inp, b64);
        inp.dispatchEvent(new Event("input",{bubbles:true}));
        break;
      }catch(e){}
    }
  }

  init();
})();
</script>
""", height=0)

    # ── Conversation history ───────────────────────────────
    if st.session_state.va_conv:
        st.markdown("<div class='va-conv-box'>", unsafe_allow_html=True)
        for msg in st.session_state.va_conv:
            cls  = "va-bubble-user" if msg["role"]=="user" else "va-bubble-ai"
            icon = "🎤" if msg["role"]=="user" else "🤖"
            st.markdown(
                f"<div class='{cls}'>{icon} {msg['text']}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption("🎙️ Press **Start** → speak naturally → pause → AI replies. Press **End Session** to finish.")

else:
    st.info("🔒 Run skill gap analysis first to enable the voice assistant.")
# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🚀 Skill Gap Analyzer")
    st.markdown("AI-powered career gap analysis")
    st.divider()

    st.markdown("**How it works:**")
    st.markdown("""
    1. Upload your resume (PDF or image)
    2. Select your target role
    3. AI extracts and maps your skills
    4. See your match % and missing skills
    5. Read the AI insight
    6. Chat with your career coach
    """)

    st.divider()

    if st.session_state.analysis_done and st.session_state.analysis_result:
        gap = st.session_state.analysis_result["gap_report"]
        st.markdown("**Current Analysis:**")
        st.markdown(f"🎯 **Role:** {gap['target_role']}")
        st.markdown(f"✅ **Match:** {gap['match_pct']}%")
        st.markdown(f"⚠️ **Gap:** {gap['gap_pct']}%")
        st.markdown(f"📚 **Missing:** {len(gap['missing'])} skills")
        st.markdown(f"💬 **Chat messages:** {len(st.session_state.chat_history) // 2}")

        st.divider()
        if st.button("Reset everything"):
            for key in [
                "analysis_done", "analysis_result",
                "ai_insight", "gap_context_str", "chat_history",
            ]:
                st.session_state[key] = (
                    False if key == "analysis_done"
                    else ([] if key == "chat_history" else None)
                )
            st.rerun()