"""
SSR AI Service — POST /detect-road-damage
Full detection pipeline: YOLO inference → EPDO scoring → single response.
Returns boxes, confidence, SAI, epdo_score, severity_tier, and sla_hours.
"""

import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.yolo_service import detection_service
from services.epdo_service import calculate_epdo
from services.weather_service import get_rainfall_risk
from services.image_utils import fetch_image_bytes, ImageFetchError


router = APIRouter(tags=["Detection"])


# ── Request / Response Models ─────────────────────────────────
class DetectRequest(BaseModel):
    image_url: str = Field(..., description="Supabase Storage public/signed URL")
    ticket_id: Optional[str] = Field(None, description="Ticket UUID for tracing")
    captured_at: Optional[str] = Field(None, description="ISO timestamp of capture")
    source_channel: Optional[str] = Field("app", description="app, whatsapp, portal")

    # Optional context for inline EPDO scoring (defaults produce a valid score)
    road_class: str = Field("local", description="IRC road class: arterial, collector, local")
    proximity_score: float = Field(0.5, description="Proximity to critical infra 0–1")
    lat: Optional[float] = Field(None, description="Latitude for weather lookup")
    lng: Optional[float] = Field(None, description="Longitude for weather lookup")


class DetectResponse(BaseModel):
    success: bool
    detected: bool = False
    damage_type: str = "unknown"
    ai_confidence: float = 0.0
    total_potholes: int = 0
    bounding_boxes: list = []
    ai_severity_index: float = 0.0
    # EPDO fields — now included in detection response
    epdo_score: float = 0.0
    severity_tier: str = "LOW"
    sla_hours: int = 168
    ai_source: str = "OFFLINE_ESTIMATE"
    model_version: str = "none"
    rejection_reason: str = ""
    processing_ms: int = 0
    errors: list[str] = []
    # Which backend actually produced this result (visible in JSON; use when server logs are hard to read)
    inference_path: str = ""
    inference_detail: str = ""


# ── Endpoint ──────────────────────────────────────────────────
@router.post("/detect-road-damage", response_model=DetectResponse)
async def detect_road_damage(req: DetectRequest):
    """
    Detect road damage in a photo and compute severity in a single call.

    Returns bounding boxes, confidence, damage classification,
    Severity AI Index (SAI), EPDO score, severity tier, and SLA hours.

    The /score-severity endpoint is still available for re-scoring
    with updated context (e.g. after JE verification adds dimensions).
    """
    start = time.time()

    try:
        # 1. Fetch image from Supabase Storage URL
        image_bytes = await fetch_image_bytes(req.image_url)

        # 2. Run detection (Roboflow / Local / Heuristic)
        result = await detection_service.detect(image_bytes)

        # 3. Compute EPDO inline if damage was detected
        epdo_score = 0.0
        severity_tier = "LOW"
        sla_hours = 168

        if result.detected:
            # Resolve rainfall risk from GPS if available
            rainfall_risk = None
            rainfall_mm = None
            if req.lat is not None and req.lng is not None:
                rainfall_risk, rainfall_mm = await get_rainfall_risk(req.lat, req.lng)

            epdo_result = calculate_epdo(
                sai=result.ai_severity_index,
                road_class=req.road_class,
                rainfall_risk=rainfall_risk,
                rainfall_mm=rainfall_mm,
                proximity_score=req.proximity_score,
            )
            epdo_score = epdo_result["epdo_score"]
            severity_tier = epdo_result["severity_tier"]
            sla_hours = epdo_result["sla_hours"]

        elapsed = int((time.time() - start) * 1000)

        return DetectResponse(
            success=True,
            detected=result.detected,
            damage_type=result.damage_type,
            ai_confidence=result.ai_confidence,
            total_potholes=result.total_potholes,
            bounding_boxes=result.bounding_boxes,
            ai_severity_index=result.ai_severity_index,
            epdo_score=epdo_score,
            severity_tier=severity_tier,
            sla_hours=sla_hours,
            ai_source=result.ai_source,
            model_version=result.model_version,
            rejection_reason=result.rejection_reason,
            processing_ms=elapsed,
            inference_path=result.inference_path,
            inference_detail=result.inference_detail,
        )

    except ImageFetchError as e:
        elapsed = int((time.time() - start) * 1000)
        return DetectResponse(
            success=False,
            errors=[str(e)],
            ai_source=detection_service.ai_source,
            model_version=detection_service.model_identifier,
            processing_ms=elapsed,
            inference_path="image_fetch",
            inference_detail="failed_to_load_image_bytes",
        )

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return DetectResponse(
            success=False,
            errors=[f"Detection failed: {type(e).__name__}: {str(e)}"],
            ai_source=detection_service.ai_source,
            model_version=detection_service.model_identifier,
            processing_ms=elapsed,
            inference_path="error",
            inference_detail=type(e).__name__,
        )
