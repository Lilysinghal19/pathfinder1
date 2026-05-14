"""
livekit_agent.py  —  Continuous Real-time Voice Agent
Compatible with: livekit-agents == 1.5.8

VAD tuned: responds after ~0.5s silence, barge-in supported.
Uses GCP for STT/TTS (professional quality).
Groq LLM for responses.
"""

import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from livekit import agents
from livekit.agents import AgentSession, Agent
from livekit.plugins import (
    google as lk_google,
    groq   as lk_groq,
    silero,
)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("voice_agent")

LIVEKIT_URL        = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY    = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]
GROQ_API_KEY       = os.environ["GROQ_API_KEY"]
GCP_CREDENTIALS    = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp_key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_CREDENTIALS

GAP_CONTEXT_FILE = Path(__file__).parent / "gap_context.json"

# ═══════════════════════════════════════════════════════════════════════════
# CAREER COACH SYSTEM PROMPT
# Combines the expertise from oo.py with conversational warmth
# ═══════════════════════════════════════════════════════════════════════════
BASE_SYSTEM_PROMPT = """You are an expert AI career coach specializing in skill gap analysis.

Your expertise:
- Career path guidance and transitions
- Technical skill assessment and learning roadmaps
- Resume optimization and interview preparation
- Job search strategy and negotiation
- Python, Machine Learning, Data Science, Cloud, DevOps, and software engineering

Communication style:
- Speak in SHORT, natural conversational sentences (as if talking, not writing)
- NEVER use markdown, bullet points, asterisks, or numbered lists
- Keep answers to 2-3 sentences unless the user asks for more detail
- If listing items, say them naturally: "First... second... and finally..."
- Be warm, encouraging, direct, and actionable
- Reference the user's specific skill gap when relevant
- Provide practical next steps they can take today

When discussing skills:
- Acknowledge what they already know
- Prioritize the highest-impact missing skills
- Suggest concrete learning resources and timelines
- Help them build relevant projects for portfolio""".strip()


def load_gap_context() -> str:
    """Load skill gap analysis context if available."""
    if not GAP_CONTEXT_FILE.exists():
        return ""
    try:
        with open(GAP_CONTEXT_FILE, "r", encoding="utf-8") as f:
            ctx = json.load(f)
        missing  = ", ".join(ctx.get("missing",  [])[:6])
        matching = ", ".join(ctx.get("matching", [])[:6])
        return (
            f"\n\nCARDINAL CONTEXT - User's Career Profile:\n"
            f"Target Role: {ctx.get('target_role', 'Unknown')}\n"
            f"Match Score: {ctx.get('match_pct', '?')}% (already has these skills)\n"
            f"Skill Gap: {ctx.get('gap_pct', '?')}% (needs these skills)\n"
            f"Has: {matching or 'not available'}\n"
            f"Missing: {missing or 'not available'}\n\n"
            f"Reference this context naturally when answering career questions. "
            f"Help them close their gap with a practical, personalized roadmap."
        )
    except Exception as e:
        log.warning(f"[GAP CTX] {e}")
        return ""


class CareerVoiceAgent(Agent):

    def __init__(self):
        super().__init__(instructions=BASE_SYSTEM_PROMPT + load_gap_context())

    async def on_enter(self):
        greeting = (
            "Hi! I'm your AI career coach. I can see your skill gap analysis. "
            "I can help you create a personalized learning roadmap. What would you like to know?"
            if GAP_CONTEXT_FILE.exists() else
            "Hello! I'm your AI career coach. How can I help you with your career today?"
        )
        await self.session.say(greeting, allow_interruptions=True)

    async def on_user_turn_completed(self, turn_ctx, new_message):
        print("\n========== USER MESSAGE ==========")
        print(new_message)
        print("==================================\n")


async def entrypoint(ctx: agents.JobContext):
    log.info(f"[AGENT] Room: {ctx.room.name}")
    
    session = AgentSession(
        # VAD: fires after 400ms silence → fast conversational response
        vad=silero.VAD.load(
            min_speech_duration     = 0.05,
            min_silence_duration    = 0.4,
            prefix_padding_duration = 0.2,
            activation_threshold    = 0.55,
        ),
        # STT: Google Cloud — conversational speech, 92%+ accuracy
        # "latest_short" optimized for back-and-forth dialogue
        stt=lk_google.STT(
            languages          = ["en-US", "hi-IN"],
            model              = "latest_short",
            spoken_punctuation = False,
            credentials_file   = GCP_CREDENTIALS,
        ),
        # LLM: Groq Llama 3 8B — fastest inference, conversational
        llm=lk_groq.LLM(
            model       = "llama3-8b-8192",
            api_key     = GROQ_API_KEY,
            temperature = 0.7,
        ),
        # TTS: Google Neural2-D — professional, natural-sounding voice
        # Specifically tuned for career coaching tone (calm, authoritative, warm)
        tts=lk_google.TTS(
            voice_name       = "en-US-Neural2-D",  # Professional male voice
            speaking_rate    = 1.0,                # Natural speaking pace
            credentials_file = GCP_CREDENTIALS,
        ),
        # Barge-in: user can interrupt the agent anytime
        allow_interruptions       = True,
        min_interruption_duration = 0.3,
        min_interruption_words    = 0,
        # Endpointing: after silence detected by VAD, wait 0.5–0.8s then call LLM
        # This creates a natural conversational pause feeling
        min_endpointing_delay = 0.5,
        max_endpointing_delay = 0.8,
    )
    print("GOOGLE CREDS:", GCP_CREDENTIALS)
    print("CREDS EXISTS:", Path(GCP_CREDENTIALS).exists())
    print("GROQ KEY EXISTS:", bool(GROQ_API_KEY))

    await session.start(room=ctx.room, agent=CareerVoiceAgent())
    log.info("[AGENT] Live — listening for speech")


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc = entrypoint,
            worker_type    = agents.WorkerType.ROOM,
        )
    )