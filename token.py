"""
token.py — Generate a LiveKit access token for the frontend.
Run: python token.py
Prints a token you paste into voice_ui.html.
"""
import os
from dotenv import load_dotenv
load_dotenv(".env.local")

from livekit.api import AccessToken, VideoGrants

api_key    = os.environ["LIVEKIT_API_KEY"]
api_secret = os.environ["LIVEKIT_API_SECRET"]

token = (
    AccessToken(api_key, api_secret)
    .with_identity("user-1")
    .with_name("Career User")
    .with_grants(VideoGrants(room_join=True, room="career-room"))
    .to_jwt()
)

print("\n=== Your LiveKit Access Token ===")
print(token)
print("\nPaste this into the token field in voice_ui.html")
print("Token is valid for 6 hours.")