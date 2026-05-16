"""
token_server.py
===============
FastAPI server that:
  1. Mints signed LiveKit JWT tokens for the browser
  2. Serves the frontend (static_lk/index.html)

Run alongside livekit_agent.py:
    python token_server.py

Open in browser: http://localhost:8080
"""

import datetime
import logging
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from livekit.api import AccessToken, VideoGrants

load_dotenv()

_missing_vars = [v for v in ["LIVEKIT_URL","LIVEKIT_API_KEY","LIVEKIT_API_SECRET"]
                 if not os.environ.get(v)]
if _missing_vars:
    raise RuntimeError(
        f"Missing .env vars: {_missing_vars}\n"
        f"Create a .env file with LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET"
    )
LIVEKIT_URL        = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY    = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]
DEFAULT_ROOM       = "career-coach"

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
    """Mint a LiveKit JWT. The browser uses this to join the room."""
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
            # FIX: must pass timedelta, NOT an int
            .with_ttl(datetime.timedelta(hours=1))
            .to_jwt()
        )
        log.info("Token minted for room=%s identity=%s", room, identity)
        return JSONResponse({
            "token"    : token,
            "room"     : room,
            "identity" : identity,
            "serverUrl": LIVEKIT_URL,
        })
    except Exception as e:
        log.exception("Token generation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return HTMLResponse(
        "<h2>Token server running</h2>"
        "<p>Copy <code>static_lk/index.html</code> from the output folder "
        "into your project's <code>static_lk/</code> folder, then refresh.</p>"
    )


if __name__ == "__main__":
    print("\n✅  Token server ready.")
    print("🌐  Open: http://localhost:8080\n")
    uvicorn.run(
        "token_server:app",
        host      = "127.0.0.1",
        port      = 8080,
        reload    = True,
        log_level = "info",
    )