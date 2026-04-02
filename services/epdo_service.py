"""
SSR AI Service — EPDO Scoring Engine
IRC-adapted Equivalent Property Damage Only formula.
Pure deterministic functions — no model inference, fully explainable.
"""

from typing import Optional


# ── Traffic weight (TOSM) from IRC road classification ───────
TOSM_WEIGHTS = {
    "national_highway": 1.0,
    "state_highway": 0.85,
    "arterial": 0.75,       # Akkalkot Road, MIDC Road
    "collector": 0.60,       # Zone connecting roads
    "local": 0.40,           # Residential lanes
}

# ── Severity Tier → SLA hours mapping ────────────────────────
SEVERITY_SLA = {
    "CRITICAL": 24,     # EPDO >= 8.0
    "HIGH": 48,         # EPDO >= 5.0
    "MEDIUM": 72,       # EPDO >= 3.0
    "LOW": 168,         # EPDO < 3.0 (7 days)
}


def classify_severity(epdo_score: float) -> str:
    """Map EPDO score to severity tier."""
    if epdo_score >= 8.0:
        return "CRITICAL"
    elif epdo_score >= 5.0:
        return "HIGH"
    elif epdo_score >= 3.0:
        return "MEDIUM"
    else:
        return "LOW"


def bucket_rainfall(rainfall_mm: Optional[float]) -> float:
    """
    Convert monthly cumulative rainfall to risk factor.
    Based on IMD classification for Maharashtra.
    """
    if rainfall_mm is None:
        return 0.3  # Default — dry season assumption
    if rainfall_mm > 100:
        return 1.0  # Heavy monsoon
    elif rainfall_mm > 30:
        return 0.6  # Moderate rain
    else:
        return 0.2  # Dry / light rain


def calculate_epdo(
    sai: float,
    road_class: str = "local",
    rainfall_risk: Optional[float] = None,
    rainfall_mm: Optional[float] = None,
    proximity_score: float = 0.5,
) -> dict:
    """
    Calculate EPDO severity score with full decision trace.

    Formula: (SAI × 0.40) + (TOSM × 0.30) + (R × 0.15) + (C × 0.15) × 10

    Args:
        sai: AI Severity Index from YOLO detection (0.0–1.0)
        road_class: IRC road classification
        rainfall_risk: Pre-computed rainfall risk (0–1). If None, computed from rainfall_mm.
        rainfall_mm: Monthly cumulative rainfall in mm. Used if rainfall_risk not provided.
        proximity_score: Distance-based score to critical infrastructure (0–1).

    Returns:
        dict with epdo_score, severity_tier, sla_hours, and full decision_trace.
    """
    # Clamp SAI to valid range
    sai = max(0.0, min(1.0, float(sai)))

    # TOSM from road class
    tosm = TOSM_WEIGHTS.get(road_class, 0.50)

    # Rainfall risk
    if rainfall_risk is not None:
        r = max(0.0, min(1.0, float(rainfall_risk)))
        r_source = "caller_supplied"
    else:
        r = bucket_rainfall(rainfall_mm)
        r_source = f"computed_from_{rainfall_mm}mm" if rainfall_mm is not None else "default_0.3"

    # Proximity score (clamp)
    c = max(0.0, min(1.0, float(proximity_score)))

    # ── IRC-adapted EPDO Formula ──────────────────────────────
    sai_weighted = sai * 0.40
    tosm_weighted = tosm * 0.30
    rainfall_weighted = r * 0.15
    proximity_weighted = c * 0.15

    raw = sai_weighted + tosm_weighted + rainfall_weighted + proximity_weighted
    epdo_score = round(raw * 10, 2)  # Scale to 0–10

    severity_tier = classify_severity(epdo_score)
    sla_hours = SEVERITY_SLA[severity_tier]

    return {
        "epdo_score": epdo_score,
        "severity_tier": severity_tier,
        "sla_hours": sla_hours,
        "decision_trace": {
            "sai_raw": round(sai, 4),
            "sai_weighted": round(sai_weighted, 4),
            "tosm_raw": tosm,
            "tosm_weighted": round(tosm_weighted, 4),
            "road_class": road_class,
            "rainfall_risk": round(r, 4),
            "rainfall_source": r_source,
            "rainfall_weighted": round(rainfall_weighted, 4),
            "proximity_raw": round(c, 4),
            "proximity_weighted": round(proximity_weighted, 4),
            "raw_score": round(raw, 6),
            "formula": "(SAI×0.40)+(TOSM×0.30)+(R×0.15)+(C×0.15) × 10",
        },
    }
