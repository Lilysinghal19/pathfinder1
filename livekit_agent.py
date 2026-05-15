"""
livekit_agent.py
================
Pure LiveKit worker process. Run as:
    python livekit_agent.py dev

Stack (ALL from Google / GCP — zero Whisper, zero pyttsx3, zero Groq):
  STT : Google Cloud Speech-to-Text  (gcp_key.json)
  LLM : Gemini 2.5 Flash via Vertex AI (gcp_key.json — Vertex AI API must be enabled)
  TTS : Google Cloud Text-to-Speech  (gcp_key.json)
  VAD : Silero (local, no key needed)

System prompt mirrors oo.py's career chatbot so both interfaces
give consistent, personalised advice from the same gap analysis.
"""

import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import google as lk_google, silero

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("career_agent")

# ── Required env vars ────────────────────────────────────────────────
LIVEKIT_URL        = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY    = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]
GCP_CREDENTIALS    = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp_key.json")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS

# ── Gap context file written by oo.py after analysis ────────────────
GAP_FILE = Path(__file__).parent / "gap_context.json"

# ── Career advisor system prompt ─────────────────────────────────────
# Mirrors the logic inside oo.py's chat_with_knowledge_base() call
# so both the text chatbot and the voice agent give consistent advice.
BASE_PROMPT = """You are an expert AI career coach embedded inside an AI Skill Gap Analyzer.

Your persona:
- Warm, direct, and encouraging
- Speak in short natural sentences — you are talking aloud, not writing
- NEVER use markdown, bullet points, asterisks, or numbered lists
- Keep answers under 3 sentences unless the user explicitly asks for more
- If you must list things say them naturally: "First... second... and finally..."
- You are the voice version of the text chatbot the user has been using

Your expertise:
- Career path planning and skill gap analysis
- Python, ML, data science, deep learning, NLP, cloud, DevOps, software engineering
- Resume advice, interview preparation, job search strategy
- Learning roadmaps and resource recommendations
- Industry trends in AI and software

Coaching style:
- Always refer to the user's actual skill gap data when available
- Be specific — name the actual missing skills, not generic advice
- Give a clear next step at the end of every answer
- If you don't know something, say so and suggest where to find the answer

You are speaking aloud on a voice call. Never produce any text the user would
need to read. No bullet points, no numbered lists, no bold markers. Speak like
a real career mentor on a phone call.""".strip()


def load_gap_context() -> str:
    """
    Read gap_context.json written by oo.py.
    Returns formatted context string, or empty string if file doesn't exist yet.
    """
    if not GAP_FILE.exists():
        return ""
    try:
        with open(GAP_FILE, encoding="utf-8") as f:
            ctx = json.load(f)

        role    = ctx.get("target_role", "Unknown")
        match_p = ctx.get("match_pct",   "?")
        gap_p   = ctx.get("gap_pct",     "?")
        missing = ctx.get("missing",     [])
        matching= ctx.get("matching",    [])
        insight = ctx.get("ai_insight",  "")

        parts = [
            "\n\n--- USER SKILL GAP ANALYSIS (personalise every answer using this) ---",
            f"Target Role : {role}",
            f"Match Score : {match_p}%",
            f"Gap Score   : {gap_p}%",
            f"Has skills  : {', '.join(matching[:8]) or 'none listed'}",
            f"Missing     : {', '.join(missing[:8])  or 'none listed'}",
        ]
        if insight:
            parts.append(f"AI Insight  : {insight[:400]}")
        parts.append(
            "\nReference this data naturally. Name the actual missing "
            "skills when giving advice."
        )
        return "\n".join(parts)

    except Exception as e:
        log.warning("[GAP] Could not read gap_context.json: %s", e)
        return ""


# ═════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ═════════════════════════════════════════════════════════════════════

class CareerVoiceAgent(Agent):

    def __init__(self):
        super().__init__(instructions=BASE_PROMPT + load_gap_context())
        log.info("[AGENT] Gap context: %s",
                 "loaded ✓" if GAP_FILE.exists() else "not yet available")

    async def on_enter(self):
        greeting = (
            "Hello! I'm your AI career coach. "
            "I can see your skill gap analysis is ready. "
            "What would you like to know about your career path?"
            if GAP_FILE.exists() else
            "Hello! I'm your AI career coach. "
            "Run the skill gap analysis in the app first and I'll have "
            "full context about your goals. What can I help you with today?"
        )
        await self.session.say(greeting, allow_interruptions=True)

    async def on_user_turn_completed(self, turn_ctx, new_message):
        """Re-read gap context on every turn — picks up new analyses instantly."""
        ctx = load_gap_context()
        if ctx:
            turn_ctx.add_message(
                role    = "system",
                content = "[LIVE CONTEXT — use this for this answer]" + ctx,
            )


# ═════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════

async def entrypoint(job_ctx: agents.JobContext):
    log.info("[AGENT] Job received — room: %s", job_ctx.room.name)

    # ─────────────────────────────────────────────────────────────────
    # CRITICAL: await ctx.connect() MUST be called before using the room.
    # Without this the agent joins the room but can't publish/subscribe
    # to tracks, causing "Listening but no response" bug.
    # ─────────────────────────────────────────────────────────────────
    await job_ctx.connect()
    log.info("[AGENT] Room connected ✓")

    session = AgentSession(

        # ── VAD: Silero (local, zero latency, no API key) ─────────────
        vad=silero.VAD.load(
            min_speech_duration     = 0.05,   # 50ms voice → turn started
            min_silence_duration    = 0.45,   # 450ms quiet → turn ended
            prefix_padding_duration = 0.2,
            activation_threshold    = 0.55,
        ),

        # ── STT: Google Cloud Speech-to-Text ──────────────────────────
        # Authenticated via gcp_key.json (GOOGLE_APPLICATION_CREDENTIALS)
        stt=lk_google.STT(
            languages          = ["en-US", "hi-IN"],
            model              = "latest_short",
            spoken_punctuation = False,
            credentials_file   = GCP_CREDENTIALS,
        ),

        # ── LLM: Gemini 2.5 Flash via Vertex AI ──────────────────────
        # Authenticated via gcp_key.json (same service account as STT/TTS)
        # Requires: Vertex AI API enabled in your GCP project
        llm=lk_google.LLM.with_vertex(
            model="google/gemini-2.5-flash",
            credentials_file=GCP_CREDENTIALS,
        ),

        # ── TTS: Google Cloud Text-to-Speech ──────────────────────────
        # Authenticated via gcp_key.json (GOOGLE_APPLICATION_CREDENTIALS)
        tts=lk_google.TTS(
            voice_name       = "en-US-Neural2-D",   # male; use -F for female
            speaking_rate    = 1.05,
            credentials_file = GCP_CREDENTIALS,
        ),

        # ── Barge-in ──────────────────────────────────────────────────
        allow_interruptions       = True,
        min_interruption_duration = 0.3,
        min_interruption_words    = 0,

        # ── Silence → respond window ──────────────────────────────────
        min_endpointing_delay = 0.5,
        max_endpointing_delay = 0.8,
    )

    await session.start(room=job_ctx.room, agent=CareerVoiceAgent())
    log.info("[AGENT] Session live ✓ — listening for speech")


# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc = entrypoint,
            worker_type    = agents.WorkerType.ROOM,
        )
    )