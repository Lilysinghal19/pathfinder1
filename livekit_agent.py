"""
livekit_agent.py  —  AI Career Voice Agent
============================================
livekit-agents == 1.5.8  |  livekit-plugins-google == 1.5.9

Fixes applied:
  BUG 1: Port conflict — agent uses LiveKit Cloud, no local port needed here
  BUG 2: Vertex AI model string corrected to "gemini-2.5-flash" (no "google/" prefix)
  BUG 3: turn_ctx.add_message() IS correct in v1.5.8 (ChatContext has it) — kept
  BUG 4: Startup crash on missing .env — all vars now use get() with clear error msgs

Auth: single gcp_key.json for STT + LLM (Vertex) + TTS
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ── MUST set these BEFORE importing any livekit/google plugins ────────
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"

GCP_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp_key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS

# ── Auto-read project_id from gcp_key.json if not set in .env ────────
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    try:
        with open(GCP_CREDENTIALS) as _f:
            _proj = json.load(_f).get("project_id", "")
        if not _proj:
            raise ValueError(f"'project_id' field missing in {GCP_CREDENTIALS}")
        os.environ["GOOGLE_CLOUD_PROJECT"] = _proj
        print(f"[BOOT] Auto-set GOOGLE_CLOUD_PROJECT={_proj} from {GCP_CREDENTIALS}")
    except FileNotFoundError:
        raise RuntimeError(
            f"\n[BOOT] ERROR: GCP key file not found: '{GCP_CREDENTIALS}'\n"
            f"  → Make sure gcp_key.json is in: {Path(GCP_CREDENTIALS).absolute()}\n"
            f"  → Or set GOOGLE_APPLICATION_CREDENTIALS=/full/path/to/key.json in .env"
        )
    except Exception as e:
        raise RuntimeError(f"[BOOT] Cannot read GCP project from key file: {e}")

if not os.environ.get("GOOGLE_CLOUD_LOCATION"):
    os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

# ── BUG 4 FIX: Validate LiveKit vars with clear messages, not KeyError ─
_missing = [v for v in ["LIVEKIT_URL","LIVEKIT_API_KEY","LIVEKIT_API_SECRET"]
            if not os.environ.get(v)]
if _missing:
    raise RuntimeError(
        f"\n[BOOT] ERROR: Missing required env vars: {_missing}\n"
        f"  → Create a .env file in your project folder with:\n"
        f"    LIVEKIT_URL=wss://your-project.livekit.cloud\n"
        f"    LIVEKIT_API_KEY=APIxxxxxxxxx\n"
        f"    LIVEKIT_API_SECRET=xxxxxxxxxx\n"
        f"  → Get these from https://cloud.livekit.io → Settings → Keys"
    )

# ── Now safe to import ────────────────────────────────────────────────
from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import google as lk_google, silero

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("career_agent")

log.info("=== STARTUP CONFIG ===")
log.info("  GCP credentials : %s → %s",
         GCP_CREDENTIALS, "EXISTS ✓" if Path(GCP_CREDENTIALS).exists() else "MISSING ✗")
log.info("  GCP project     : %s", os.environ.get("GOOGLE_CLOUD_PROJECT"))
log.info("  GCP location    : %s", os.environ.get("GOOGLE_CLOUD_LOCATION"))
log.info("  Vertex AI mode  : %s", os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"))
log.info("  LiveKit URL     : %s", os.environ.get("LIVEKIT_URL"))

# ── Gap context file written by oo.py ────────────────────────────────
GAP_FILE = Path(__file__).parent / "gap_context.json"

# ── System prompt ─────────────────────────────────────────────────────
BASE_PROMPT = """You are an expert AI career coach embedded inside an AI Skill Gap Analyzer.

Your persona:
- Warm, direct, and encouraging
- Speak in short natural sentences — you are talking aloud, not writing
- NEVER use markdown, bullet points, asterisks, or numbered lists
- Keep answers under 3 sentences unless explicitly asked for more
- If you must list things, say them naturally: "First... second... and finally..."

Your expertise:
- Career path planning and skill gap analysis
- Python, ML, data science, deep learning, NLP, cloud, DevOps, software engineering
- Resume advice, interview preparation, job search strategy
- Learning roadmaps and resource recommendations

Coaching style:
- Always refer to the user's actual skill gap data when it is available
- Be specific — name the actual missing skills, not generic advice
- Give one clear next step at the end of every answer

You are on a voice call. Never produce text the user would need to read.
Speak like a real career mentor on a phone call.""".strip()


def load_gap_context() -> str:
    """Read gap_context.json written by oo.py after skill gap analysis."""
    if not GAP_FILE.exists():
        return ""
    try:
        with open(GAP_FILE, encoding="utf-8") as f:
            ctx = json.load(f)
        parts = [
            "\n\n--- USER SKILL GAP ANALYSIS ---",
            f"Target Role : {ctx.get('target_role', 'Unknown')}",
            f"Match Score : {ctx.get('match_pct', '?')}%",
            f"Gap Score   : {ctx.get('gap_pct', '?')}%",
            f"Has skills  : {', '.join(ctx.get('matching',[])[:8]) or 'none listed'}",
            f"Missing     : {', '.join(ctx.get('missing', [])[:8]) or 'none listed'}",
        ]
        if ctx.get("ai_insight"):
            parts.append(f"AI Insight  : {ctx['ai_insight'][:400]}")
        parts.append("\nReference this data naturally when advising.")
        return "\n".join(parts)
    except Exception as e:
        log.warning("[GAP] Could not load context: %s", e)
        return ""


# ═════════════════════════════════════════════════════════════════════
# AGENT
# ═════════════════════════════════════════════════════════════════════

class CareerVoiceAgent(Agent):

    def __init__(self):
        super().__init__(instructions=BASE_PROMPT + load_gap_context())
        log.info("[AGENT] Gap context: %s", "loaded ✓" if GAP_FILE.exists() else "none yet")

    async def on_enter(self):
        greeting = (
            "Hello! I'm your AI career coach. I can see your skill gap analysis. "
            "What would you like to know about your career path?"
            if GAP_FILE.exists() else
            "Hello! I'm your AI career coach. How can I help you today?"
        )
        log.info("[AGENT] Sending greeting")
        await self.session.say(greeting, allow_interruptions=True)

    async def on_user_turn_completed(
        self,
        turn_ctx,       # llm.ChatContext  — the full chat context for this turn
        new_message,    # llm.ChatMessage  — the user's transcribed message
    ) -> None:
        """
        Called AFTER STT finishes, BEFORE LLM is invoked.
        Re-read gap_context.json every turn so new analyses are
        picked up immediately without restarting the agent.

        BUG 3 CLARIFICATION:
        turn_ctx here IS a llm.ChatContext instance.
        ChatContext.add_message(role=..., content=...) is the correct v1.5.8 API.
        Verified from source: livekit/agents/llm/chat_context.py line 417.
        """
        ctx = load_gap_context()
        if ctx:
            turn_ctx.add_message(
                role    = "system",
                content = "[LIVE CONTEXT — use for this answer only]" + ctx,
            )
            log.debug("[AGENT] Gap context injected into turn")


# ═════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════

async def entrypoint(job_ctx: agents.JobContext):
    log.info("[AGENT] Job received — room: %s", job_ctx.room.name)

    try:
        # CRITICAL: await connect() before using room tracks
        await job_ctx.connect()
        log.info("[AGENT] Room connected ✓")

        session = AgentSession(

            # ── VAD: Silero (local, zero latency) ─────────────────────
            vad=silero.VAD.load(
                min_speech_duration     = 0.05,   # 50ms voice → turn start
                min_silence_duration    = 0.45,   # 450ms quiet → turn end
                prefix_padding_duration = 0.2,
                activation_threshold    = 0.55,
            ),

            # ── STT: Google Cloud Speech-to-Text ──────────────────────
            stt=lk_google.STT(
                languages          = ["en-US", "hi-IN"],
                model              = "latest_short",  # fast conversational model
                spoken_punctuation = False,
                credentials_file   = GCP_CREDENTIALS,
            ),

            # ── LLM: Gemini 2.5 Flash via Vertex AI ───────────────────
            # BUG 2 FIX: model = "gemini-2.5-flash"  (NOT "google/gemini-2.5-flash")
            # "google/" prefix is only for AI Studio API, not Vertex AI.
            # vertexai=True reads GOOGLE_CLOUD_PROJECT + GOOGLE_APPLICATION_CREDENTIALS
            # automatically — no api_key needed.
            llm=lk_google.LLM(
                model       = "gemini-2.5-flash",   # ← correct Vertex AI model string
                vertexai    = True,                  # ← enables Vertex AI auth
                temperature = 0.7,
            ),

            # ── TTS: Google Cloud Text-to-Speech ──────────────────────
            tts=lk_google.TTS(
                voice_name       = "en-US-Neural2-D",   # male; use -F for female
                speaking_rate    = 1.05,
                credentials_file = GCP_CREDENTIALS,
            ),

            # ── Turn handling ─────────────────────────────────────────
            turn_handling={
                "endpointing": {
                    "min_delay": 0.5,   # wait 500ms after silence → call LLM
                    "max_delay": 0.8,   # never wait longer than 800ms
                },
                "interruption": {
                    "enabled":      True,
                    "min_duration": 0.3,  # 300ms of speech = interrupt
                    "min_words":    0,
                },
            },
        )

        log.info("[AGENT] Session built ✓ — starting…")
        await session.start(room=job_ctx.room, agent=CareerVoiceAgent())
        log.info("[AGENT] Live ✓ — waiting for user speech")

    except Exception as e:
        log.exception("[AGENT] FATAL crash in entrypoint — full traceback:")
        raise  # re-raise so LiveKit logs it and you see it in the terminal


# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc = entrypoint,
            worker_type    = agents.WorkerType.ROOM,
        )
    )