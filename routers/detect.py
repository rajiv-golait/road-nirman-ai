"""
SSR AI Service — POST /detect-road-damage
YOLOv8/v12 inference only. Returns boxes + SAI.
Does NOT compute EPDO — that's a separate call to /score-severity.
"""

import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.yolo_service import detection_service
from services.image_utils import fetch_image_bytes, ImageFetchError


router = APIRouter(tags=["Detection"])


# ── Request / Response Models ─────────────────────────────────
class DetectRequest(BaseModel):
    image_url: str = Field(..., description="Supabase Storage public/signed URL")
    ticket_id: Optional[str] = Field(None, description="Ticket UUID for tracing")
    captured_at: Optional[str] = Field(None, description="ISO timestamp of capture")
    source_channel: Optional[str] = Field("app", description="app, whatsapp, portal")


class DetectResponse(BaseModel):
    success: bool
    detected: bool = False
    damage_type: str = "unknown"
    ai_confidence: float = 0.0
    total_potholes: int = 0
    bounding_boxes: list = []
    ai_severity_index: float = 0.0
    ai_source: str = "OFFLINE_ESTIMATE"
    model_version: str = "none"
    processing_ms: int = 0
    errors: list[str] = []


# ── Endpoint ──────────────────────────────────────────────────
@router.post("/detect-road-damage", response_model=DetectResponse)
async def detect_road_damage(req: DetectRequest):
    """
    Detect road damage in a photo. Returns bounding boxes, confidence,
    damage classification, and Severity AI Index (SAI).

    The Edge Function should call /score-severity separately with the
    SAI output to get the full EPDO score.
    """
    start = time.time()
    errors = []

    try:
        # 1. Fetch image from Supabase Storage URL
        image_bytes = await fetch_image_bytes(req.image_url)

        # 2. Run detection (Roboflow / Local / Heuristic)
        result = await detection_service.detect(image_bytes)

        elapsed = int((time.time() - start) * 1000)

        return DetectResponse(
            success=True,
            detected=result.detected,
            damage_type=result.damage_type,
            ai_confidence=result.ai_confidence,
            total_potholes=result.total_potholes,
            bounding_boxes=result.bounding_boxes,
            ai_severity_index=result.ai_severity_index,
            ai_source=result.ai_source,
            model_version=result.model_version,
            processing_ms=elapsed,
        )

    except ImageFetchError as e:
        elapsed = int((time.time() - start) * 1000)
        return DetectResponse(
            success=False,
            errors=[str(e)],
            ai_source=detection_service.ai_source,
            model_version=detection_service.model_identifier,
            processing_ms=elapsed,
        )

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return DetectResponse(
            success=False,
            errors=[f"Detection failed: {type(e).__name__}: {str(e)}"],
            ai_source=detection_service.ai_source,
            model_version=detection_service.model_identifier,
            processing_ms=elapsed,
        )
