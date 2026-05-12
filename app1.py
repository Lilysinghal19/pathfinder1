"""
voice_server.py — Run this alongside your Streamlit app:
    python voice_server.py
Runs on port 8765. Streamlit embeds it via iframe on port 8766.
"""

import asyncio, json, tempfile, threading, queue, base64
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel
import pyttsx3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# ── Config ──────────────────────────────────────────────────────
SAMPLE_RATE    = 16000
CHUNK          = 1024
RMS_THRESHOLD  = 180     # energy above this = speech (tune to your mic)
SILENCE_SEC    = 1.3     # seconds of silence → utterance done
MIN_SPEECH_SEC = 0.35    # ignore clips shorter than this

# ── Load Whisper once ────────────────────────────────────────────
print("Loading Whisper model…")
whisper = WhisperModel("small", compute_type="int8")
print("Whisper ready.")

app = FastAPI()

# ── HTML UI served by FastAPI ────────────────────────────────────
# Streamlit will embed this via iframe (or you can open directly)
UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Voice Assistant</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(135deg,#eef2ff,#f8fafc);
    display: flex; flex-direction: column;
    align-items: center; justify-content: flex-start;
    min-height: 100vh; padding: 28px 16px;
  }
  h2 { color: #1e1b4b; font-size: 20px; margin-bottom: 6px; }
  .subtitle { color: #64748b; font-size: 13px; margin-bottom: 24px; }

  /* Orb */
  @keyframes pulse {
    0%,100% { transform:scale(1); box-shadow:0 0 0 0 rgba(239,68,68,.45); }
    55%      { transform:scale(1.08); box-shadow:0 0 0 18px rgba(239,68,68,0); }
  }
  @keyframes bar { 0%,100%{height:5px} 50%{height:26px} }
  .orb {
    width:90px; height:90px; border-radius:50%;
    border:none; cursor:pointer; font-size:38px;
    display:flex; align-items:center; justify-content:center;
    box-shadow:0 6px 28px rgba(0,0,0,.18);
    transition:all .2s; margin-bottom:18px;
  }
  .orb-idle    { background:linear-gradient(135deg,#2E5BFF,#4F46E5); color:#fff; }
  .orb-idle:hover { transform:scale(1.1); }
  .orb-live    { background:#ef4444; color:#fff; animation:pulse 1.5s infinite; }
  .orb-busy    { background:#f59e0b; color:#fff; cursor:default; }
  .orb-ended   { background:#6b7280; color:#fff; }

  /* Status */
  .status { font-size:14px; font-weight:600; color:#475569; min-height:22px; margin-bottom:16px; }

  /* Wave */
  .wave { display:flex; align-items:center; gap:4px; height:32px; margin-bottom:14px; }
  .wave span {
    display:inline-block; width:4px; border-radius:3px;
    background:linear-gradient(#7c3aed,#4f46e5);
    animation:bar .85s ease-in-out infinite;
  }
  .wave span:nth-child(1){animation-delay:.00s}
  .wave span:nth-child(2){animation-delay:.12s}
  .wave span:nth-child(3){animation-delay:.24s}
  .wave span:nth-child(4){animation-delay:.36s}
  .wave span:nth-child(5){animation-delay:.48s}
  .wave { visibility:hidden; }
  .wave.show { visibility:visible; }

  /* End button */
  .end-btn {
    display:none; padding:9px 30px;
    border-radius:22px; border:2px solid #6b7280;
    background:#f9fafb; color:#374151;
    font-weight:600; font-size:13px;
    cursor:pointer; margin-bottom:20px;
    transition:background .2s;
  }
  .end-btn:hover { background:#e5e7eb; }
  .end-btn.show  { display:block; }

  /* Conversation */
  .conv {
    width:100%; max-width:560px; max-height:340px;
    overflow-y:auto; border:1px solid #e2e8f0;
    border-radius:14px; padding:14px;
    background:#fff; display:flex; flex-direction:column; gap:8px;
  }
  .bubble {
    padding:10px 14px; border-radius:18px;
    font-size:14px; max-width:86%; line-height:1.5;
  }
  .u { background:#ede9fe; color:#3730a3;
       border-radius:18px 18px 4px 18px; align-self:flex-end; }
  .a { background:#f0fdf4; color:#166534;
       border-left:4px solid #16a34a;
       border-radius:4px 18px 18px 18px; align-self:flex-start; }
  .caption { font-size:12px; color:#94a3b8; margin-top:14px; text-align:center; }
</style>
</head>
<body>
<h2>🎤 AI Career Voice Assistant</h2>
<p class="subtitle">Speak naturally — pause to send — AI replies aloud</p>

<button class="orb orb-idle" id="orb" onclick="toggleSession()">🎙️</button>
<button class="end-btn" id="endBtn" onclick="endSession()">⏹ End Session</button>

<div class="wave" id="wave">
  <span></span><span></span><span></span><span></span><span></span>
</div>
<div class="status" id="status">Press the mic to begin</div>

<!-- Volume meter — shows live mic level so user can verify mic is working -->
<div id="volmeter" style="width:220px;height:8px;background:#e2e8f0;border-radius:4px;
     margin:0 auto 14px;overflow:hidden;display:none;">
  <div id="volbar" style="height:100%;width:0%;
       background:linear-gradient(90deg,#22c55e,#f59e0b,#ef4444);
       transition:width 0.05s;border-radius:4px;"></div>
</div>

<div class="conv" id="conv"></div>
<p class="caption">Speak → pause 1 s → AI answers. Press End Session to stop.</p>

<script>
const WS_URL = "ws://localhost:8765/ws";
let ws, active = false;

let mediaRec, stream, chunks = [];
let silTimer = null, recording = false, speechStart = null;
let audioCtx, analyser;

// ── Tuned constants ─────────────────────────────────────────────
// getByteFrequencyData avg across ALL bins is tiny — typical room
// noise = 0-1, whisper = 1-3, normal speech = 3-12, loud = 12-30.
// We use 2 as the trigger so even quiet voices are caught.
const SILENCE_MS    = 1400;   // ms of silence → send clip
const MIN_SPEECH_MS = 300;    // ignore clips under this duration
const THRESHOLD     = 18;     // max bin value to detect speech (0-255); raise if false triggers

let debugVol = 0;  // shown in the vol meter

function setStatus(text) {
  document.getElementById("status").textContent = text;
}
function setOrb(cls, icon) {
  const o = document.getElementById("orb");
  o.className = "orb " + cls;
  o.textContent = icon;
}
function showWave(yes) {
  document.getElementById("wave").classList.toggle("show", yes);
}
function showEnd(yes) {
  document.getElementById("endBtn").classList.toggle("show", yes);
}
function addBubble(role, text) {
  const d = document.createElement("div");
  d.className = "bubble " + (role === "user" ? "u" : "a");
  d.textContent = (role === "user" ? "🎤 " : "🤖 ") + text;
  const conv = document.getElementById("conv");
  conv.appendChild(d);
  conv.scrollTop = conv.scrollHeight;
}

async function toggleSession() {
  if (!active) await startSession();
}

async function startSession() {
  active = true;
  setOrb("orb-live", "🔴");
  showEnd(true);
  setStatus("🔴 Listening — speak, then pause…");
  document.getElementById("volmeter").style.display = "block";

  ws = new WebSocket(WS_URL);

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);

    if (msg.type === "transcript") {
      addBubble("user", msg.text);
      setOrb("orb-busy", "🧠");
      setStatus("🧠 Thinking…");
      showWave(false);
    }
    else if (msg.type === "response") {
      addBubble("assistant", msg.text);
      setOrb("orb-live", "🔊");
      setStatus("🔊 Speaking…");
      showWave(true);
    }
    else if (msg.type === "done_speaking") {
      setOrb("orb-live", "🔴");
      setStatus("🔴 Listening — speak, then pause…");
      showWave(false);
    }
    else if (msg.type === "error") {
      setStatus("⚠️ " + msg.text);
    }
  };

  ws.onopen = () => {
    startMic();
  };

  ws.onclose = () => {
    if (active) endSession();
  };

  ws.onerror = (e) => {
    setStatus("⚠️ Cannot connect to voice server. Is voice_server.py running?");
    setOrb("orb-ended", "⚠️");
    showEnd(false);
    active = false;
  };
}

// Safe base64 encoder — avoids stack overflow on large buffers
function toBase64(bytes) {
  let b64 = "", chunk = 8192;
  for (let i = 0; i < bytes.length; i += chunk) {
    b64 += btoa(String.fromCharCode(...bytes.subarray(i, i + chunk)));
  }
  return b64;
}

async function startMic() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio:true, video:false});
  } catch(e) {
    setStatus("⚠️ Mic access denied: " + e.message);
    return;
  }

  // Use browser default sample rate — resampling done server-side by ffmpeg
  audioCtx  = new AudioContext();
  analyser  = audioCtx.createAnalyser();
  analyser.fftSize = 1024;  // more bins = more accurate energy reading
  audioCtx.createMediaStreamSource(stream).connect(analyser);

  const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
               ? "audio/webm;codecs=opus" : "audio/webm";

  mediaRec = new MediaRecorder(stream, {mimeType: mime});
  mediaRec.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
  mediaRec.onstop = async () => {
    const dur = speechStart ? Date.now() - speechStart : 0;
    const saved = [...chunks]; chunks = [];
    if (dur < MIN_SPEECH_MS || saved.length === 0) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    setStatus("📤 Sending…");
    const blob = new Blob(saved, {type: mime});
    const ab   = await blob.arrayBuffer();
    const b64  = toBase64(new Uint8Array(ab));  // safe chunked encoder
    ws.send(JSON.stringify({type: "audio", data: b64, mime: mime}));
  };

  const freq  = new Uint8Array(analyser.frequencyBinCount);
  const volEl = document.getElementById("volbar");

  function tick() {
    if (!active) return;
    analyser.getByteFrequencyData(freq);

    // Use max of all bins — far more reliable than average for speech detection
    let maxVal = 0;
    for (let i = 0; i < freq.length; i++) if (freq[i] > maxVal) maxVal = freq[i];
    const vol = maxVal;  // 0-255; speech typically 20-200

    // Update visual vol meter
    if (volEl) volEl.style.width = Math.min(100, vol / 255 * 100) + "%";

    if (vol > THRESHOLD) {
      clearTimeout(silTimer); silTimer = null;
      if (!recording) {
        recording = true; speechStart = Date.now();
        chunks = []; mediaRec.start(100);
        setStatus("🔴 Recording…");
      }
    } else if (recording && !silTimer) {
      silTimer = setTimeout(() => {
        if (recording) { recording = false; mediaRec.stop(); }
        silTimer = null;
      }, SILENCE_MS);
    }
    requestAnimationFrame(tick);
  }
  tick();
}

function endSession() {
  active = false;
  recording = false;
  if (mediaRec && mediaRec.state !== "inactive") mediaRec.stop();
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (audioCtx) audioCtx.close();
  if (ws) ws.close();
  clearTimeout(silTimer);
  setOrb("orb-ended", "⏹");
  showEnd(false);
  showWave(false);
  setStatus("✅ Session ended — refresh to start again");
}
</script>
</body>
</html>
"""

# ── WebSocket endpoint ────────────────────────────────────────────
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
    return HTMLResponse(UI_HTML)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected")

    # Import chain — must be in same folder as voice_server.py
    try:
        from chain import chat_with_knowledge_base
    except Exception as e:
        print(f"WARNING: could not import chain.py ({e}). Using stub.")
        def chat_with_knowledge_base(question, insight="", session_id=""):
            return f"I received your question. Make sure chain.py is in the same folder as voice_server.py."
    insight = ""   # insight passed per-query from session context if needed

    tts_q: queue.Queue = queue.Queue()

    def tts_worker():
        """Dedicated thread for pyttsx3 — one engine, COM initialized once."""
        try:
            import pythoncom; pythoncom.CoInitialize()
        except ImportError:
            pass
        engine = pyttsx3.init()
        engine.setProperty("rate", 168)
        engine.setProperty("volume", 1.0)
        while True:
            text = tts_q.get()
            if text is None:
                break
            engine.say(text)
            engine.runAndWait()
            # Signal done so WS can tell browser to go back to listening
            asyncio.run_coroutine_threadsafe(
                ws.send_text(json.dumps({"type": "done_speaking"})),
                loop,
            )
        try:
            engine.stop()
            import pythoncom; pythoncom.CoUninitialize()
        except Exception:
            pass

    loop = asyncio.get_running_loop()
    tts_thread = threading.Thread(target=tts_worker, daemon=False)
    tts_thread.start()

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") != "audio":
                continue

            # Decode base64 audio
            audio_bytes = base64.b64decode(msg["data"])
            mime        = msg.get("mime", "audio/webm")

            # Save to temp file — convert webm → wav via soundfile/av
            tmp_in  = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
            tmp_in.write(audio_bytes); tmp_in.flush()

            tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_wav.close()

            # Convert webm/opus → wav using ffmpeg (must be installed)
            import subprocess, shutil
            if shutil.which("ffmpeg"):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_in.name,
                     "-ar", "16000", "-ac", "1", tmp_wav.name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                wav_path = tmp_wav.name
            else:
                # Fallback: treat as raw PCM (less likely to work)
                wav_path = tmp_in.name

            # Transcribe
            segs, info = whisper.transcribe(
                wav_path, beam_size=5, language=None,
                initial_prompt="Conversation in English or Hindi. Transcribe exactly.",
                condition_on_previous_text=False,
                vad_filter=True, temperature=0.0,
            )
            query = " ".join(s.text for s in segs).strip()
            if not query:
                continue

            await ws.send_text(json.dumps({"type": "transcript", "text": query}))

            # LLM
            response = chat_with_knowledge_base(
                question=query + " (Answer in 2-3 concise sentences.)",
                insight=insight,
                session_id="va_ws",
            )
            await ws.send_text(json.dumps({"type": "response", "text": response}))

            # TTS (non-blocking — queued to dedicated thread)
            tts_q.put(response)

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        tts_q.put(None)  # shut down TTS thread


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8765, log_level="info")