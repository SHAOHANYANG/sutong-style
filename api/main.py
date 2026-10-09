"""FastAPI application entry point."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.deps import services
from api.routes import router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    # Build the pipeline before the first request instead of during it.
    await asyncio.to_thread(services)
    yield


app = FastAPI(title="sutong-style", version="0.1.0", lifespan=lifespan)
app.include_router(router)
