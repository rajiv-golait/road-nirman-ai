"""
SSR AI Service — POST /score-severity
Deterministic EPDO formula. Pure rules — no model inference.
Edge Function calls this separately after /detect-road-damage.
"""

import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from config import settings
from services.epdo_service import calculate_epdo
from services.weather_service import get_rainfall_risk


router = APIRouter(tags=["Severity"])


# ── Request / Response Models ─────────────────────────────────
class SeverityRequest(BaseModel):
    # From /detect-road-damage output
    damage_type: str = Field("pothole", description="Damage classification")
    ai_confidence: float = Field(0.5, description="Detection confidence 0–1")
    total_potholes: int = Field(1, description="Number of detections")
    ai_severity_index: float = Field(0.5, description="SAI from detection 0–1")

    # Context — supplied by Edge Function from ticket/GPS metadata
    road_class: str = Field("local", description="IRC road class: arterial, collector, local, etc.")
    proximity_score: float = Field(0.5, description="Proximity to critical infra 0–1")

    # Rainfall — can be supplied or auto-fetched from weather API
    rainfall_risk: Optional[float] = Field(
        None,
        description="Pre-computed rainfall risk 0–1. If null, fetched from OpenWeatherMap."
    )

    # GPS — needed only if rainfall_risk is null (for weather API lookup)
    lat: Optional[float] = Field(None, description="Latitude for weather lookup")
    lng: Optional[float] = Field(None, description="Longitude for weather lookup")

    # Optional dimensions from JE verification
    dimensions: Optional[dict] = Field(
        None,
        description="JE-verified dimensions: {length_m, width_m, depth_m, area_sqm}"
    )


class SeverityResponse(BaseModel):
    success: bool
    epdo_score: float = 0.0
    severity_tier: str = "LOW"
    sla_hours: int = 168
    decision_trace: dict = {}
    ruleset_version: str = ""
    processing_ms: int = 0
    errors: list[str] = []


# ── Endpoint ──────────────────────────────────────────────────
@router.post("/score-severity", response_model=SeverityResponse)
async def score_severity(req: SeverityRequest):
    """
    Calculate EPDO severity score using the IRC-adapted formula.
    Pure deterministic rules — no model inference.

    If rainfall_risk is not supplied, the service fetches live rainfall
    from OpenWeatherMap using lat/lng and caches it for 1 hour.

    Returns full decision_trace for explainability.
    """
    start = time.time()

    try:
        # Resolve rainfall risk
        rainfall_risk = req.rainfall_risk
        rainfall_mm = None

        if rainfall_risk is None and req.lat is not None and req.lng is not None:
            # Fetch live rainfall from weather API
            rainfall_risk, rainfall_mm = await get_rainfall_risk(req.lat, req.lng)

        # Calculate EPDO
        result = calculate_epdo(
            sai=req.ai_severity_index,
            road_class=req.road_class,
            rainfall_risk=rainfall_risk,
            rainfall_mm=rainfall_mm,
            proximity_score=req.proximity_score,
        )

        elapsed = int((time.time() - start) * 1000)

        return SeverityResponse(
            success=True,
            epdo_score=result["epdo_score"],
            severity_tier=result["severity_tier"],
            sla_hours=result["sla_hours"],
            decision_trace=result["decision_trace"],
            ruleset_version=settings.RULESET_VERSION,
            processing_ms=elapsed,
        )

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return SeverityResponse(
            success=False,
            errors=[f"Severity scoring failed: {type(e).__name__}: {str(e)}"],
            ruleset_version=settings.RULESET_VERSION,
            processing_ms=elapsed,
        )
