"""
SSR AI Service — FastAPI Application
Stateless internal microservice. Called only by Supabase Edge Functions.
Never writes to DB. Returns evidence; Edge Functions own all state transitions.
"""

import logging
import time
from contextlib import asynccontextmanager

_ssr_log = logging.getLogger("ssr")
_ssr_log.setLevel(logging.INFO)
if not _ssr_log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    _ssr_log.addHandler(_h)
    _ssr_log.propagate = False

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings


# ── App State ─────────────────────────────────────────────────
_start_time: float = 0.0


# ── Lifespan: pre-load model on startup ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load AI model into memory once at startup."""
    global _start_time
    _start_time = time.time()

    # Lazy import to avoid loading heavy deps at module level
    from services.yolo_service import detection_service
    detection_service.load_model()

    print(f"SSR AI Service v{settings.SERVICE_VERSION} started")
    print(f"   Model source: {settings.MODEL_SOURCE}")
    print(f"   Model loaded: {detection_service.model_loaded}")
    print(f"   AI source:    {detection_service.ai_source}")

    yield

    print("SSR AI Service shutting down")


# ── FastAPI App ───────────────────────────────────────────────
app = FastAPI(
    title="SSR AI Service",
    description="Solapur Smart Roads — AI Detection, Severity Scoring, Repair Verification",
    version=settings.SERVICE_VERSION,
    docs_url="/docs",
    lifespan=lifespan,
)

# ── CORS (permissive for hackathon — tighten for production) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth Dependency ───────────────────────────────────────────
async def verify_secret(request: Request):
    """
    Validate X-SSR-Secret header on all POST endpoints.
    Rejects browser/mobile direct calls. Only Edge Functions pass.
    """
    secret = request.headers.get("X-SSR-Secret")
    if not secret or secret != settings.AI_SERVICE_SECRET:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-SSR-Secret header. This service is internal-only."
        )


# ── Health Check ──────────────────────────────────────────────
@app.get("/health")
async def health():
    """Service readiness probe. No auth required."""
    from services.yolo_service import detection_service
    return {
        "status": "ok",
        "model_loaded": detection_service.model_loaded,
        "model_source": settings.MODEL_SOURCE,
        "ai_source": detection_service.ai_source,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "version": settings.SERVICE_VERSION,
    }


# ── Version Info ──────────────────────────────────────────────
@app.get("/version")
async def version():
    """Semantic version, model identifier, ruleset version."""
    from services.yolo_service import detection_service
    return {
        "version": settings.SERVICE_VERSION,
        "model_source": settings.MODEL_SOURCE,
        "model_identifier": detection_service.model_identifier,
        "ruleset_version": settings.RULESET_VERSION,
    }


# ── Mount Routers ─────────────────────────────────────────────
from routers import detect, severity, verify

app.include_router(detect.router, dependencies=[Depends(verify_secret)])
app.include_router(severity.router, dependencies=[Depends(verify_secret)])
app.include_router(verify.router, dependencies=[Depends(verify_secret)])


# ── Global Exception Handler ─────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all: never expose raw stack traces to callers."""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "errors": [f"Internal error: {type(exc).__name__}: {str(exc)}"],
            "processing_ms": 0,
        }
    )


# ── Entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
