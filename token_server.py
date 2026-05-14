"""
token_server.py  —  LiveKit Token Server + Career Coach Voice UI
==================================================================
Mints signed JWT tokens for LiveKit room access.
Serves a professional voice interface for the AI career coach.

Requirements:
  - Run BEFORE opening in browser: python livekit_agent.py dev
  - Then run: python token_server.py
  - Open: http://localhost:7880

The LiveKit agent connects to the same room and responds to your voice.
GCP-powered STT/TTS + Groq LLM = professional voice experience.
"""

import os
import logging
import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

load_dotenv()

LIVEKIT_URL        = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY    = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

DEFAULT_ROOM = "career-coach"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("token_server")

app = FastAPI(title="LiveKit Career Coach Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ══════════════════════════════════════════════════════════════════════════
# PROFESSIONAL CAREER COACH UI
# Built with LiveKit SDK for real-time voice interaction
# ══════════════════════════════════════════════════════════════════════════
COACH_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Career Coach - Voice Interview</title>
    <script src="https://cdn.jsdelivr.net/npm/livekit-client@0.15.8/dist/livekit-client.umd.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 500px;
            width: 100%;
            padding: 40px;
            text-align: center;
        }
        
        h1 {
            color: #333;
            font-size: 28px;
            margin-bottom: 8px;
        }
        
        .subtitle {
            color: #666;
            font-size: 14px;
            margin-bottom: 30px;
        }
        
        /* Microphone Button */
        .mic-button {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            font-size: 48px;
            margin: 30px auto;
            transition: all 0.3s ease;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            display: block;
        }
        
        .mic-button.idle {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        
        .mic-button.idle:hover {
            transform: scale(1.1);
            box-shadow: 0 15px 40px rgba(102,126,234,0.4);
        }
        
        .mic-button.active {
            background: #ef4444;
            color: white;
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.7); }
            50% { box-shadow: 0 0 0 20px rgba(239,68,68,0); }
        }
        
        .mic-button.speaking {
            background: #f59e0b;
            animation: none;
        }
        
        /* Status Display */
        .status {
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            color: #666;
            margin: 20px 0;
            font-weight: 500;
        }
        
        /* Waveform */
        .waveform {
            display: flex;
            align-items: flex-end;
            justify-content: center;
            gap: 4px;
            height: 60px;
            margin: 20px 0;
        }
        
        .waveform.hidden {
            display: none;
        }
        
        .wave-bar {
            width: 4px;
            background: linear-gradient(180deg, #667eea, #764ba2);
            border-radius: 2px;
            animation: wave 0.6s ease-in-out infinite;
        }
        
        @keyframes wave {
            0%, 100% { height: 10px; }
            50% { height: 40px; }
        }
        
        .wave-bar:nth-child(1) { animation-delay: 0s; }
        .wave-bar:nth-child(2) { animation-delay: 0.1s; }
        .wave-bar:nth-child(3) { animation-delay: 0.2s; }
        .wave-bar:nth-child(4) { animation-delay: 0.3s; }
        .wave-bar:nth-child(5) { animation-delay: 0.4s; }
        
        /* Transcript Display */
        .transcript {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 15px;
            margin: 20px 0;
            text-align: left;
            min-height: 60px;
            max-height: 150px;
            overflow-y: auto;
            font-size: 13px;
        }
        
        .transcript-empty {
            color: #999;
            font-style: italic;
        }
        
        .transcript-user {
            color: #667eea;
            font-weight: 600;
            margin-bottom: 8px;
        }
        
        .transcript-agent {
            color: #764ba2;
            font-style: italic;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 30px;
            justify-content: center;
        }
        
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 13px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        .btn-secondary {
            background: #e9ecef;
            color: #333;
        }
        
        .btn-secondary:hover {
            background: #dee2e6;
        }
        
        .error {
            background: #fee;
            color: #c33;
            padding: 12px;
            border-radius: 8px;
            margin: 15px 0;
            font-size: 13px;
        }

        .log {
            background: #f0f0f0;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
            font-size: 11px;
            color: #666;
            max-height: 100px;
            overflow-y: auto;
            font-family: monospace;
            text-align: left;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Career Coach</h1>
        <p class="subtitle">Professional voice guidance powered by GCP & Groq</p>
        
        <button class="mic-button idle" id="micButton" onclick="toggleConnection()">🎤</button>
        
        <div class="waveform hidden" id="waveform">
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
            <div class="wave-bar"></div>
        </div>
        
        <div class="status" id="status">Click to connect</div>
        
        <div class="transcript" id="transcript">
            <div class="transcript-empty">Your conversation will appear here</div>
        </div>
        
        <div id="error"></div>
        <div class="log" id="log" style="display:none;"></div>
        
        <div class="controls">
            <button class="btn btn-secondary" onclick="clearTranscript()">Clear</button>
            <button class="btn btn-secondary" onclick="toggleLog()">Debug</button>
        </div>
    </div>

    <script>
        // LiveKit SDK - access the exported classes correctly
        const LivekitClient = window;
        
        let room = null;
        let connected = false;
        let localParticipant = null;
        
        const transcriptDiv = document.getElementById("transcript");
        const statusDiv = document.getElementById("status");
        const micButton = document.getElementById("micButton");
        const waveform = document.getElementById("waveform");
        const errorDiv = document.getElementById("error");
        const logDiv = document.getElementById("log");
        
        function log(msg) {
            console.log("[Career Coach]", msg);
            const line = document.createElement("div");
            line.textContent = msg;
            logDiv.appendChild(line);
            logDiv.scrollTop = logDiv.scrollHeight;
        }

        function toggleLog() {
            logDiv.style.display = logDiv.style.display === "none" ? "block" : "none";
        }
        
        async function getToken() {
            try {
                log("Fetching token...");
                const res = await fetch("/token?room=career-coach&identity=user_" + Date.now());
                if (!res.ok) {
                    throw new Error("Token failed: " + res.status + " " + res.statusText);
                }
                const data = await res.json();
                log("✓ Token received");
                log("  Server: " + data.serverUrl);
                log("  Room: " + data.room);
                return data;
            } catch (e) {
                showError("Token error: " + e.message);
                log("❌ ERROR: " + e.message);
                throw e;
            }
        }
        
        async function toggleConnection() {
            if (connected) {
                await disconnect();
            } else {
                await connect();
            }
        }
        
        async function connect() {
            try {
                log("═══ CONNECTING ═══");
                setStatus("Connecting...", "idle");
                
                const tokenData = await getToken();
                const url = tokenData.serverUrl;
                const token = tokenData.token;
                
                log("Creating Room with LiveKit SDK...");
                
                // Correct way to create a Room with LiveKit SDK
                room = new LivekitClient.Room({
                    audio: true,
                    video: false,
                });
                
                log("✓ Room object created");
                log("Setting up event listeners...");
                
                room.on("participantConnected", (participant) => {
                    log("✓ Participant connected: " + participant.name);
                    addTranscript("system", "🤖 Coach joined the room");
                    setStatus("🎙️ Coach connected - speak now!", "active");
                });
                
                room.on("participantDisconnected", (participant) => {
                    log("! Participant disconnected: " + participant.name);
                    addTranscript("system", "Coach left");
                });
                
                room.on("trackSubscribed", (track, publication, participant) => {
                    log("✓ Track subscribed: " + track.kind + " from " + participant.name);
                    if (track.kind === "audio") {
                        // Auto-play remote audio
                        const audio = document.createElement("audio");
                        audio.srcObject = new MediaStream([track]);
                        audio.autoplay = true;
                        audio.play().catch(e => log("Audio play failed: " + e.message));
                    }
                });
                
                room.on("roomFinished", () => {
                    log("! Room finished (all participants left)");
                });
                
                room.on("disconnected", () => {
                    log("! Room disconnected");
                    connected = false;
                    micButton.classList.add("idle");
                    micButton.classList.remove("active");
                    waveform.classList.add("hidden");
                    setStatus("Connection lost", "idle");
                });
                
                log("Connecting to room: " + url);
                await room.connect(url, token);
                log("✓ Connected to LiveKit room!");
                
                log("Getting local participant...");
                localParticipant = room.localParticipant;
                log("✓ Local participant: " + localParticipant.name);
                
                // Get user media
                log("Requesting microphone access...");
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                    },
                    video: false 
                });
                log("✓ Microphone access granted");
                
                log("Publishing local audio track...");
                const audioTrack = stream.getAudioTracks()[0];
                await localParticipant.publishTrack(audioTrack);
                log("✓ Audio track published!");
                
                connected = true;
                micButton.classList.remove("idle");
                micButton.classList.add("active");
                setStatus("🔴 Recording... speak now!", "active");
                waveform.classList.remove("hidden");
                log("═══ ✅ READY ═══");
                addTranscript("system", "🎤 Ready - speak naturally!");
                
            } catch (e) {
                showError("Connection failed: " + e.message);
                log("❌ ERROR: " + e.message);
                log("Stack: " + e.stack);
                connected = false;
                micButton.classList.add("idle");
                micButton.classList.remove("active");
                waveform.classList.add("hidden");
            }
        }
        
        async function disconnect() {
            try {
                log("═══ DISCONNECTING ═══");
                if (room) {
                    await room.disconnect();
                    room = null;
                    log("✓ Disconnected from room");
                }
                connected = false;
                micButton.classList.add("idle");
                micButton.classList.remove("active", "speaking");
                waveform.classList.add("hidden");
                setStatus("Disconnected - click to reconnect", "idle");
            } catch (e) {
                showError("Disconnect error: " + e.message);
                log("❌ ERROR: " + e.message);
            }
        }
        
        function setStatus(text, state = "idle") {
            statusDiv.textContent = text;
            if (state === "speaking") {
                micButton.classList.remove("active");
                micButton.classList.add("speaking");
            } else if (state === "active") {
                micButton.classList.add("active");
                micButton.classList.remove("speaking");
            }
        }
        
        function addTranscript(role, text) {
            if (transcriptDiv.querySelector(".transcript-empty")) {
                transcriptDiv.innerHTML = "";
            }
            const div = document.createElement("div");
            if (role === "user") {
                div.className = "transcript-user";
                div.textContent = "👤 You: " + text;
            } else if (role === "agent") {
                div.className = "transcript-agent";
                div.textContent = "🤖 Coach: " + text;
            } else {
                div.textContent = "ℹ️ " + text;
            }
            transcriptDiv.appendChild(div);
            transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
        }
        
        function clearTranscript() {
            transcriptDiv.innerHTML = '<div class="transcript-empty">Conversation cleared</div>';
        }
        
        function showError(msg) {
            errorDiv.innerHTML = '<div class="error">⚠️ ' + msg + '</div>';
            setTimeout(() => { errorDiv.innerHTML = ""; }, 7000);
        }
        
        // Check browser support on page load
        window.addEventListener("load", () => {
            log("═══ INITIALIZING ═══");
            log("Page loaded");
            
            if (!window.navigator.mediaDevices) {
                showError("Browser does not support audio input");
                log("❌ ERROR: No mediaDevices support");
                return;
            }
            log("✓ mediaDevices available");
            
            if (!window.LivekitClient || !window.LivekitClient.Room) {
                showError("LiveKit SDK failed to load. Refresh page.");
                log("❌ ERROR: LiveKit SDK window.Room not found");
                log("Available: " + Object.keys(window.LivekitClient || {}).join(", "));
                return;
            }
            log("✓ LiveKit SDK loaded (Room class available)");
            log("═══ READY TO CONNECT ═══");
        });
    </script>
</body>
</html>
"""


@app.get("/token")
async def get_token(room: str = DEFAULT_ROOM, identity: str = "user"):
    try:
        token = (
            AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
            .with_identity(identity)
            .with_name(f"User-{identity}")
            .with_grants(VideoGrants(
                room_join        = True,
                room             = room,
                can_publish      = True,
                can_subscribe    = True,
                can_publish_data = True,
            ))
            .with_ttl(datetime.timedelta(hours=1))
            .to_jwt()
        )
        return JSONResponse({
            "token":     token,
            "room":      room,
            "identity":  identity,
            "serverUrl": LIVEKIT_URL,
        })
    except Exception as e:
        log.exception("Token generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def serve_frontend():
    """Serve the professional career coach UI."""
    return HTMLResponse(COACH_UI_HTML)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("\n" + "="*60)
    print("✅  Career Coach Server Ready")
    print("="*60)
    print("🌐  Open in browser: http://localhost:7880")
    print("📢  Ensure livekit_agent.py is running first")
    print("    python livekit_agent.py dev")
    print("="*60 + "\n")
    uvicorn.run(
        "token_server:app",
        host    = "127.0.0.1",
        port    = 7880,
        reload  = True,
        log_level = "info",
    )