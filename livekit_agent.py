"""
livekit_agent.py  —  Continuous Real-time Voice Agent
=======================================================
Compatible with: livekit-agents == 1.5.8

Pipeline:
  Browser Mic (WebRTC)  →  LiveKit Room  →  Silero VAD
  →  Google STT  →  Groq LLM  →  Google TTS  →  Browser Speaker

Run:
  python livekit_agent.py dev
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── LiveKit Agent SDK ──────────────────────────────────────────
from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import (
    google as lk_google,
    groq   as lk_groq,
    silero,
)

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("voice_agent")

# ── Credentials ────────────────────────────────────────────────
LIVEKIT_URL        = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY    = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]
GROQ_API_KEY       = os.environ["GROQ_API_KEY"]
GCP_CREDENTIALS    = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp_key.json")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS

# ── Gap context file (written by oo.py after analysis) ─────────
GAP_CONTEXT_FILE = Path(__file__).parent / "gap_context.json"

# ── Base system prompt ─────────────────────────────────────────
BASE_SYSTEM_PROMPT = """
You are an expert AI career coach and voice assistant built into an AI Skill Gap Analyzer app.

Your personality:
- Warm, encouraging, and direct
- Speak naturally in short conversational sentences — you are talking, not writing
- NEVER use markdown, bullet points, asterisks, or numbered lists in your spoken responses
- Keep responses under 3 sentences unless the user explicitly asks for more detail
- If you need to list things, say them naturally: "First... second... and finally..."

Your knowledge covers:
- Career paths, skill gaps, and personalised learning roadmaps
- Tech skills: Python, ML, data science, software engineering, cloud
- Resume advice, interview prep, and job search strategy

When you don't know something, say so honestly and suggest a next step.
""".strip()


def load_gap_context() -> str:
    """
    Reads gap_context.json written by oo.py after skill-gap analysis.
    Returns a formatted string to inject into the LLM context,
    or empty string if no analysis has been run yet.
    """
    if not GAP_CONTEXT_FILE.exists():
        return ""
    try:
        with open(GAP_CONTEXT_FILE, "r", encoding="utf-8") as f:
            ctx = json.load(f)
        missing_preview  = ", ".join(ctx.get("missing",  [])[:6])
        matching_preview = ", ".join(ctx.get("matching", [])[:6])
        return (
            f"\n\nCURRENT USER ANALYSIS (personalise every answer using this):\n"
            f"- Target Role : {ctx.get('target_role', 'Unknown')}\n"
            f"- Match Score : {ctx.get('match_pct', '?')}%\n"
            f"- Skill Gap   : {ctx.get('gap_pct', '?')}%\n"
            f"- Their skills: {matching_preview or 'not available'}\n"
            f"- Missing     : {missing_preview  or 'not available'}\n"
            f"\nReference this analysis naturally when answering career questions."
        )
    except Exception as e:
        log.warning(f"[GAP CTX] Could not read gap_context.json: {e}")
        return ""


# ══════════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════════

class CareerVoiceAgent(Agent):
    """
    Continuous voice agent.
    - No push-to-talk: Silero VAD handles turn detection automatically
    - Barge-in: user can interrupt the agent mid-sentence
    - Skill-gap context injected fresh before every LLM call
    """

    def __init__(self):
        instructions = BASE_SYSTEM_PROMPT + load_gap_context()
        super().__init__(instructions=instructions)

    async def on_enter(self):
        """Opening greeting when agent joins the room."""
        if GAP_CONTEXT_FILE.exists():
            greeting = (
                "Hello! I'm your AI career coach. "
                "I can see your skill gap analysis is ready. "
                "What would you like to know about your career path?"
            )
        else:
            greeting = (
                "Hello! I'm your AI career coach. "
                "Run the skill gap analysis in the app and I'll have full context "
                "about your goals. How can I help you today?"
            )
        await self.session.say(greeting, allow_interruptions=True)

    async def on_user_turn_completed(self, turn_ctx, new_message):
        """
        Called after VAD detects end-of-speech and STT finishes,
        before the LLM is invoked.
        Re-reads gap_context.json on every turn so any new analysis
        is picked up immediately without restarting the agent.
        """
        gap_ctx = load_gap_context()
        if gap_ctx:
            turn_ctx.add_message(
                role    = "system",
                content = f"[LIVE CONTEXT UPDATE]{gap_ctx}",
            )


# ══════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════

async def entrypoint(ctx: agents.JobContext):
    log.info(f"[AGENT] Connecting to room: {ctx.room.name}")

    # Build Google TTS with explicit Neural2 voice (overrides Gemini default)
    tts_engine = lk_google.TTS(
        voice_name       = "en-US-Neural2-D",   # natural male; change to -F for female
        speaking_rate    = 1.05,
        credentials_file = GCP_CREDENTIALS,
    )

    session = AgentSession(

        # 1. VAD — Silero (local, no API key, very fast)
        vad = silero.VAD.load(
            min_speech_duration     = 0.1,   # 100 ms of speech to start a turn
            min_silence_duration    = 0.6,   # 600 ms silence → end of turn
            prefix_padding_duration = 0.3,
            activation_threshold    = 0.5,   # lower = more sensitive
        ),

        # 2. STT — Google Cloud Speech
        stt = lk_google.STT(
            languages          = ["en-US", "hi-IN"],
            model              = "latest_long",
            spoken_punctuation = False,
            credentials_file   = GCP_CREDENTIALS,
        ),

        # 3. LLM — Groq (fastest available)
        llm = lk_groq.LLM(
            model       = "llama3-8b-8192",
            api_key     = GROQ_API_KEY,
            temperature = 0.7,
        ),

        # 4. TTS — Google Neural2
        tts = tts_engine,

        # ── Barge-in / interruption settings (v1.5.8 correct params) ──
        allow_interruptions        = True,
        min_interruption_duration  = 0.3,   # user must speak ≥ 300 ms to interrupt
        min_interruption_words     = 0,     # any spoken words count as interrupt

        # ── Endpointing (how long to wait after silence before sending to LLM) ──
        min_endpointing_delay = 0.5,
        max_endpointing_delay = 1.2,
    )

    await session.start(
        room  = ctx.room,
        agent = CareerVoiceAgent(),
    )

    log.info("[AGENT] Session live — waiting for user speech")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc = entrypoint,
            worker_type    = agents.WorkerType.ROOM,
        )
    )