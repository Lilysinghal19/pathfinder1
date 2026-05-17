"""
livekit_agent.py -- AI Career Voice Agent
==========================================
livekit-agents == 1.5.8 | livekit-plugins-google == 1.5.9

Run:
    python livekit_agent.py dev

To test on LiveKit Playground:
    1. Go to https://cloud.livekit.io -> your project -> Agents
    2. Click "Test in Playground" next to your registered worker
    3. The agent auto-dispatches (agent_name="" means automatic dispatch)

Auth: single gcp_key.json for STT + LLM (Vertex AI) + TTS
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# -- Set Vertex AI env vars BEFORE any google/livekit imports -----------
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"

GCP_CREDENTIALS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp_key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS

# Auto-read project_id from gcp_key.json
if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
    try:
        with open(GCP_CREDENTIALS) as _f:
            _proj = json.load(_f).get("project_id", "")
        if not _proj:
            raise ValueError("project_id field missing in " + GCP_CREDENTIALS)
        os.environ["GOOGLE_CLOUD_PROJECT"] = _proj
        sys.stdout.write("[BOOT] GOOGLE_CLOUD_PROJECT=" + _proj + "\n")
        sys.stdout.flush()
    except FileNotFoundError:
        sys.stderr.write(
            "[BOOT] ERROR: GCP key file not found: " + GCP_CREDENTIALS + "\n"
            "  Set GOOGLE_APPLICATION_CREDENTIALS in .env to the correct path.\n"
        )
        sys.exit(1)
    except Exception as e:
        sys.stderr.write("[BOOT] ERROR reading GCP key: " + str(e) + "\n")
        sys.exit(1)

if not os.environ.get("GOOGLE_CLOUD_LOCATION"):
    os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"

# Validate LiveKit env vars before importing anything
_missing = [v for v in ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
            if not os.environ.get(v)]
if _missing:
    sys.stderr.write(
        "[BOOT] ERROR: Missing env vars: " + str(_missing) + "\n"
        "  Add them to your .env file.\n"
    )
    sys.exit(1)

# -- Safe to import now -------------------------------------------------
from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import google as lk_google, silero

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("career_agent")

log.info("=== STARTUP ===")
log.info("  GCP creds file : %s (%s)",
         GCP_CREDENTIALS,
         "found" if Path(GCP_CREDENTIALS).exists() else "NOT FOUND - will fail")
log.info("  GCP project    : %s", os.environ.get("GOOGLE_CLOUD_PROJECT"))
log.info("  GCP location   : %s", os.environ.get("GOOGLE_CLOUD_LOCATION"))
log.info("  Vertex AI mode : %s", os.environ.get("GOOGLE_GENAI_USE_VERTEXAI"))
log.info("  LiveKit URL    : %s", os.environ.get("LIVEKIT_URL"))

GAP_FILE = Path(__file__).parent / "gap_context.json"

BASE_PROMPT = (
    "You are an expert AI career coach embedded inside an AI Skill Gap Analyzer.\n\n"
    "Rules:\n"
    "- Speak in short natural sentences. You are talking aloud, not writing.\n"
    "- NEVER use markdown, bullet points, asterisks, or numbered lists.\n"
    "- Keep answers under 3 sentences unless asked for more.\n"
    "- List things naturally: First... second... and finally...\n"
    "- Be warm, direct, and encouraging.\n\n"
    "Expertise: career paths, skill gaps, learning roadmaps, Python, ML, "
    "data science, deep learning, NLP, cloud, DevOps, resume advice, "
    "interview prep, job search strategy.\n\n"
    "Always refer to the user's actual skill gap data when available. "
    "Name specific missing skills. Give one clear next step per answer.\n\n"
    "You are on a voice call. Speak like a real career mentor on a phone call."
)


def load_gap_context():
    if not GAP_FILE.exists():
        return ""
    try:
        with open(GAP_FILE, encoding="utf-8") as f:
            ctx = json.load(f)
        lines = [
            "\n\n--- USER SKILL GAP ANALYSIS ---",
            "Target Role : " + str(ctx.get("target_role", "Unknown")),
            "Match Score : " + str(ctx.get("match_pct", "?")) + "%",
            "Gap Score   : " + str(ctx.get("gap_pct", "?")) + "%",
            "Has skills  : " + (", ".join(ctx.get("matching", [])[:8]) or "none listed"),
            "Missing     : " + (", ".join(ctx.get("missing",  [])[:8]) or "none listed"),
        ]
        if ctx.get("ai_insight"):
            lines.append("AI Insight  : " + ctx["ai_insight"][:400])
        lines.append("\nReference this data. Name actual missing skills when advising.")
        return "\n".join(lines)
    except Exception as e:
        log.warning("[GAP] %s", e)
        return ""


# ======================================================================
# AGENT
# ======================================================================

class CareerVoiceAgent(Agent):

    def __init__(self):
        super().__init__(instructions=BASE_PROMPT + load_gap_context())
        log.info("[AGENT] Gap context: %s",
                 "loaded" if GAP_FILE.exists() else "not yet available")

    async def on_enter(self):
        if GAP_FILE.exists():
            greeting = (
                "Hello! I'm your AI career coach. "
                "I can see your skill gap analysis is ready. "
                "What would you like to know about your career path?"
            )
        else:
            greeting = (
                "Hello! I'm your AI career coach. "
                "How can I help you with your career today?"
            )
        log.info("[AGENT] Sending greeting...")
        await self.session.say(greeting, allow_interruptions=True)

    async def on_user_turn_completed(self, turn_ctx, new_message):
        """
        Called after STT finishes, before LLM is called.
        Injects fresh gap context on every turn.
        turn_ctx is a llm.ChatContext instance.
        ChatContext.add_message(role, content) is verified correct in v1.5.8
        from livekit/agents/llm/chat_context.py line 417.
        """
        ctx = load_gap_context()
        if ctx:
            turn_ctx.add_message(
                role="system",
                content="[LIVE CONTEXT - use for this answer only]" + ctx,
            )


# ======================================================================
# ENTRYPOINT
# ======================================================================

async def entrypoint(job_ctx: agents.JobContext):
    log.info("[AGENT] Job received - room: %s", job_ctx.room.name)

    try:
        # CRITICAL: must call connect() before accessing room media
        await job_ctx.connect()
        log.info("[AGENT] Room connected")

        session = AgentSession(

            # VAD: Silero (local, zero latency, no API key)
            vad=silero.VAD.load(
                min_speech_duration=0.05,
                min_silence_duration=0.45,
                prefix_padding_duration=0.2,
                activation_threshold=0.55,
            ),

            # STT: Google Cloud Speech-to-Text
            # credentials_file uses the service account JSON
            stt=lk_google.STT(
                languages=["en-US", "hi-IN"],
                model="latest_short",
                spoken_punctuation=False,
                credentials_file=GCP_CREDENTIALS,
            ),

            # LLM: Gemini 2.5 Flash via Vertex AI
            # vertexai=True reads GOOGLE_CLOUD_PROJECT + GOOGLE_APPLICATION_CREDENTIALS
            # model string is just "gemini-2.5-flash" -- no "google/" prefix
            llm=lk_google.LLM(
                model="gemini-2.5-flash",
                vertexai=True,
                temperature=0.7,
            ),

            # TTS: Google Cloud Text-to-Speech
            tts=lk_google.TTS(
                voice_name="en-US-Neural2-D",
                speaking_rate=1.05,
                credentials_file=GCP_CREDENTIALS,
            ),

            # Turn handling -- typed dict format, verified from turn.py source
            # EndpointingOptions keys: mode, min_delay, max_delay
            # InterruptionOptions keys: enabled, mode
            turn_handling={
                "endpointing": {
                    "mode":      "fixed",
                    "min_delay": 0.5,
                    "max_delay": 0.8,
                },
                "interruption": {
                    "enabled": True,
                    "mode":    "vad",
                },
            },
        )

        log.info("[AGENT] Session built - starting...")
        await session.start(room=job_ctx.room, agent=CareerVoiceAgent())
        log.info("[AGENT] Live - waiting for user speech")

    except Exception as e:
        log.exception("[AGENT] FATAL crash - this is why the agent disconnects:")
        raise


# ======================================================================
if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            # agent_name="" (default) = AUTOMATIC dispatch
            # The worker joins every room that gets created, including
            # LiveKit Playground test sessions.
            # Do NOT set agent_name unless you want explicit-only dispatch.
            worker_type=agents.WorkerType.ROOM,
        )
    )