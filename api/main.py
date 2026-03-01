# Created by Codex - Section 1

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.dependencies import get_config, get_session_store
from api.routers import chat, config, eval as eval_router, papers, sessions
from src.logging_utils import setup_logging

load_dotenv(override=False)


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv(override=False)
    config_obj = get_config()
    setup_logging(config_obj.log_level)
    get_session_store()
    yield


app = FastAPI(title="medical-llm-assistant", lifespan=lifespan)

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:4200").strip() or "http://localhost:4200"
allowed_origins = [frontend_origin, "http://localhost:4200"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(set(allowed_origins)),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(config.router)
app.include_router(papers.router)
app.include_router(sessions.router)

if get_config().eval_mode:
    app.include_router(eval_router.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


_frontend_dir = Path("frontend/dist/medical-llm-assistant/browser").expanduser()
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
