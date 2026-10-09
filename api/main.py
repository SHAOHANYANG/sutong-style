"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.deps import services
from api.routes import router
from api.settings import ServiceSettings


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Build the pipeline before the first request instead of during it.
    await asyncio.to_thread(services)
    yield


app = FastAPI(title="sutong-style", version="0.1.0", lifespan=lifespan)
# The demo page is served from another origin and reads the SSE stream with fetch.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ServiceSettings.from_env().allowed_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.include_router(router)
