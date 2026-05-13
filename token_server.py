"""
token_server.py  —  LiveKit Token + Frontend Server
====================================================
Mints signed JWT tokens so the browser can join a LiveKit room.
Serves the frontend at http://localhost:7880

Run AFTER starting livekit_agent.py:
    python token_server.py

Then open in browser: http://localhost:7880
(Do NOT use http://0.0.0.0:7880 — that won't work in Chrome/Edge)
"""

import os
import logging
import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

load_dotenv()

LIVEKIT_URL        = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY    = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

DEFAULT_ROOM = "career-coach"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("token_server")

app = FastAPI(title="LiveKit Token Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

STATIC_DIR = Path(__file__).parent / "static_lk"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return HTMLResponse("<h1>Token server running — place index.html in ./static_lk/</h1>")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("\n✅  Token server ready.")
    print("🌐  Open in browser: http://localhost:7880\n")
    uvicorn.run(
        "token_server:app",
        host    = "127.0.0.1",   # localhost only — works in all Windows browsers
        port    = 7880,
        reload  = True,
        log_level = "info",
    )