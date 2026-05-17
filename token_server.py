"""
token_server.py
===============
FastAPI server that:
  1. Creates the LiveKit room (if it doesn't exist)
  2. Dispatches the career agent into the room
  3. Mints a signed JWT for the browser
  4. Serves the frontend from static_lk/index.html

Run: python token_server.py
Open: http://localhost:8080

Fixes:
  - No emoji in print() - Windows cp1252 console safe
  - Serves static_lk/index.html correctly
  - Creates room + dispatches agent before giving token to browser
"""

import datetime
import logging
import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from livekit.api import (
    AccessToken,
    LiveKitAPI,
    VideoGrants,
)
from livekit.protocol.agent_dispatch import CreateAgentDispatchRequest
from livekit.protocol.room import CreateRoomRequest

load_dotenv()

# -- Validate env vars up front ----------------------------------------
_missing = [v for v in ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
            if not os.environ.get(v)]
if _missing:
    print(f"[ERROR] Missing env vars: {_missing}")
    print("  Create a .env file with LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET")
    sys.exit(1)

LIVEKIT_URL        = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY    = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

# Room name every browser session joins
DEFAULT_ROOM = "career-coach"

# Agent name - must match what livekit_agent.py registers as.
# When agent_name="", LiveKit dispatches ANY available registered worker.
# Set to a specific name only if you use WorkerOptions(agent_name="...").
AGENT_NAME = ""

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("token_server")

# -- FastAPI app -------------------------------------------------------
app = FastAPI(title="LiveKit Token Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Serve frontend from static_lk/
STATIC_DIR = Path(__file__).parent / "static_lk"
STATIC_DIR.mkdir(exist_ok=True)


# -- /token endpoint ---------------------------------------------------
@app.get("/token")
async def get_token(room: str = DEFAULT_ROOM, identity: str = "user"):
    """
    1. Creates the LiveKit room (idempotent - safe to call if it exists)
    2. Dispatches the agent into the room so it's ready when browser connects
    3. Returns a signed JWT the browser uses to join
    """
    try:
        async with LiveKitAPI(
            url        = LIVEKIT_URL,
            api_key    = LIVEKIT_API_KEY,
            api_secret = LIVEKIT_API_SECRET,
        ) as lkapi:

            # Step 1: Create room (no-op if already exists)
            try:
                await lkapi.room.create_room(
                    CreateRoomRequest(
                        name          = room,
                        empty_timeout = 600,   # keep open 10 min after everyone leaves
                        max_participants = 10,
                    )
                )
                log.info("Room created/verified: %s", room)
            except Exception as e:
                # Room may already exist - that's fine
                log.debug("Room create skipped (may already exist): %s", e)

            # Step 2: Dispatch agent into the room
            # This tells LiveKit: "send a worker running livekit_agent.py into this room"
            # Without this step the agent only joins when using LiveKit Playground,
            # not when a real browser connects.
            try:
                dispatch = await lkapi.agent_dispatch.create_dispatch(
                    CreateAgentDispatchRequest(
                        agent_name = AGENT_NAME,   # "" = any available worker
                        room       = room,
                    )
                )
                log.info("Agent dispatched: dispatch_id=%s", dispatch.id)
            except Exception as e:
                # An agent may already be in the room - that's fine too
                log.warning("Agent dispatch skipped (agent may already be present): %s", e)

        # Step 3: Mint JWT token for the browser
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

        log.info("Token minted - room=%s identity=%s", room, identity)
        return JSONResponse({
            "token"    : token,
            "room"     : room,
            "identity" : identity,
            "serverUrl": LIVEKIT_URL,
        })

    except Exception as e:
        log.exception("Token endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "room": DEFAULT_ROOM}


@app.get("/")
async def root():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return HTMLResponse(
        "<h2>Token server is running</h2>"
        "<p>The file <code>static_lk/index.html</code> was not found.</p>"
        "<p>Make sure <code>static_lk/index.html</code> exists next to "
        "<code>token_server.py</code>.</p>"
    )


# Serve everything else in static_lk/ (CSS, JS, images etc.)
# Mount AFTER the routes above so /token and /health take priority
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# -- Entry point -------------------------------------------------------
if __name__ == "__main__":
    # Windows-safe print - no emoji
    print("")
    print("[INFO] Token server starting...")
    print("[INFO] Open in browser: http://localhost:8080")
    print("[INFO] Make sure livekit_agent.py dev is running in another terminal")
    print("")
    uvicorn.run(
        "token_server:app",
        host      = "127.0.0.1",
        port      = 8080,
        reload    = True,
        log_level = "info",
    )