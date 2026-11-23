"""Entry point Vercel: semua endpoint tersedia di bawah /api (mis. /api/health, /api/predict)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from backend.app import app as model_app

app = FastAPI()
app.mount("/api", model_app)
