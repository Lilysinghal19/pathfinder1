"""
agent.py  —  LiveKit Voice Agent for AI Career Assistant
=========================================================
Uses:
  STT  : faster-whisper (local, your existing setup)
  LLM  : your chain.py (chat_with_knowledge_base)
  TTS  : Google Cloud TTS (your GCP key)
  VAD  : Silero (built into livekit-agents)

Run:
    python agent.py dev

.env.local required:
    LIVEKIT_URL=wss://your-project.livekit.cloud
    LIVEKIT_API_KEY=APIxxxxxxx
    LIVEKIT_API_SECRET=xxxxxxxxxxxxxxxx
    GOOGLE_APPLICATION_CREDENTIALS=gcp_key.json
"""

import os, sys, asyncio, logging
from dotenv import load_dotenv

load_dotenv(".env.local")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Add project dir to path so chain.py is importable ─────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from livekit import agents
from livekit.agents import AgentSession, Agent, JobContext, WorkerOptions, cli, RoomInputOptions
from livekit.plugins import silero                   # VAD
from livekit.plugins import google as lk_google      # TTS via GCP

# ── Local Whisper STT plugin ───────────────────────────────────
from livekit.agents.stt import STT, SpeechData, SpeechEvent, SpeechEventType
from livekit.agents import stt as agents_stt
from faster_whisper import WhisperModel
import numpy as np

class WhisperSTT(STT):
    """Wraps faster-whisper as a LiveKit STT plugin."""

    def __init__(self):
        super().__init__(capabilities=agents_stt.STTCapabilities(
            streaming=False, interim_results=False
        ))
        logger.info("Loading Whisper model...")
        self._model = WhisperModel("small", compute_type="int8")
        logger.info("Whisper ready.")

    async def recognize(self, buffer: agents_stt.AudioBuffer, *, language=None):
        import tempfile, scipy.io.wavfile as wav_io
        audio = agents.utils.merge_frames(buffer)
        pcm   = np.frombuffer(audio.data, dtype=np.int16)
        sr    = audio.sample_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_io.write(tmp.name, sr, pcm)
            path = tmp.name

        segs, info = self._model.transcribe(
            path,
            beam_size=5,
            language=None,
            initial_prompt="English or Hindi career conversation.",
            condition_on_previous_text=False,
            vad_filter=True,
            temperature=0.0,
        )
        text = " ".join(s.text for s in segs).strip()
        logger.info(f"Whisper transcript: {repr(text)}")
        os.unlink(path)

        return agents_stt.SpeechEvent(
            type=agents_stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[agents_stt.SpeechData(
                language=info.language or "en",
                text=text,
                confidence=1.0,
            )],
        )

# ── Load chain.py LLM ──────────────────────────────────────────
try:
    from chain import chat_with_knowledge_base
    logger.info("chain.py loaded OK")
except Exception as e:
    logger.warning(f"chain.py failed: {e}. Using stub.")
    def chat_with_knowledge_base(question, insight="", session_id=""):
        return f"chain.py not loaded. You asked: {question}"

# ── Custom LLM wrapper ─────────────────────────────────────────
from livekit.agents.llm import LLM, ChatContext, ChatChunk, Choice, ChoiceDelta
from livekit.agents.llm import LLMCapabilities
import asyncio

class ChainLLM(LLM):
    """Wraps chain.py as a LiveKit LLM plugin."""

    def __init__(self):
        super().__init__(capabilities=LLMCapabilities(supports_streaming=False))

    def chat(self, *, chat_ctx: ChatContext, **kwargs):
        return ChainLLMStream(chat_ctx, self)


class ChainLLMStream(agents.llm.LLMStream):
    def __init__(self, chat_ctx, llm):
        super().__init__(llm, chat_ctx=chat_ctx, tools=[])

    async def _run(self):
        # Get last user message
        user_msg = ""
        for m in reversed(self._chat_ctx.messages):
            if m.role == "user" and isinstance(m.content, str):
                user_msg = m.content
                break

        if not user_msg:
            return

        logger.info(f"LLM query: {repr(user_msg[:80])}")

        loop   = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            lambda: chat_with_knowledge_base(
                question=user_msg + " Answer in 2-3 concise sentences.",
                insight="",
                session_id="livekit_va",
            )
        )

        logger.info(f"LLM answer: {repr(answer[:80])}")

        self._event_ch.send_nowait(
            ChatChunk(
                id="chunk-0",
                choices=[Choice(
                    delta=ChoiceDelta(role="assistant", content=answer),
                    index=0,
                )],
            )
        )

# ── Agent entrypoint ───────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    logger.info(f"Agent joining room: {ctx.room.name}")
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=WhisperSTT(),
        llm=ChainLLM(),
        tts=lk_google.TTS(
            voice_name="en-US-Neural2-F",
            credentials_file=os.environ.get(
                "GOOGLE_APPLICATION_CREDENTIALS", "gcp_key.json"
            ),
        ),
    )

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=(
                "You are an expert AI career coach. Help users identify skill gaps, "
                "recommend learning paths, and give concise, actionable career advice. "
                "Keep answers to 2-3 sentences unless asked for more detail. "
                "Support both English and Hindi."
            ),
        ),
        room_input_options=RoomInputOptions(),
    )

    await session.generate_reply(
        instructions="Greet the user warmly and ask what career topic they want help with today."
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))