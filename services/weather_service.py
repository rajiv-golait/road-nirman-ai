"""
SSR AI Service — Weather Service
OpenWeatherMap integration for live rainfall data.
In-memory cache with 1-hour TTL. Falls back to seasonal defaults.
"""

import time
from typing import Optional

import httpx

from config import settings


# ── In-memory cache ───────────────────────────────────────────
# Key: "lat_lng" (rounded to 2 decimals) → (timestamp, rainfall_mm)
_cache: dict[str, tuple[float, float]] = {}


def _cache_key(lat: float, lng: float) -> str:
    """Grid-cell key for cache. ~1.1km resolution."""
    return f"{round(lat, 2)}_{round(lng, 2)}"


def _is_fresh(timestamp: float) -> bool:
    """Check if cached value is within TTL."""
    return (time.time() - timestamp) < settings.WEATHER_CACHE_TTL_S


async def get_rainfall_mm(lat: float, lng: float) -> Optional[float]:
    """
    Fetch current rainfall accumulation from OpenWeatherMap.
    Returns monthly-equivalent rainfall in mm, or None if unavailable.

    Caches results for 1 hour per ~1km grid cell.
    Falls back to None if API key is missing or request fails.
    """
    # Check cache first
    key = _cache_key(lat, lng)
    if key in _cache:
        ts, value = _cache[key]
        if _is_fresh(ts):
            return value

    # No API key configured → return None (caller will use default)
    if not settings.OPENWEATHERMAP_API_KEY:
        return None

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "lat": lat,
                    "lon": lng,
                    "appid": settings.OPENWEATHERMAP_API_KEY,
                    "units": "metric",
                },
            )

        if resp.status_code != 200:
            return None

        data = resp.json()

        # Extract rainfall from response
        # OpenWeatherMap returns rain.1h (mm in last 1 hour) or rain.3h
        rain_data = data.get("rain", {})
        rain_1h = rain_data.get("1h", 0.0)
        rain_3h = rain_data.get("3h", 0.0)

        # Estimate monthly rainfall from current rate
        # Conservative: use the higher of 1h or 3h/3, extrapolate to 30 days
        hourly_rate = max(rain_1h, rain_3h / 3.0 if rain_3h else 0.0)
        monthly_estimate = hourly_rate * 24 * 30  # Rough monthly projection

        # Also check general weather condition for context
        weather_main = data.get("weather", [{}])[0].get("main", "")
        humidity = data.get("main", {}).get("humidity", 50)

        # If it's actively raining with high humidity, boost the estimate
        if weather_main in ("Rain", "Thunderstorm") and humidity > 80:
            monthly_estimate = max(monthly_estimate, 60.0)

        # Cache the result
        _cache[key] = (time.time(), monthly_estimate)

        return round(monthly_estimate, 1)

    except Exception:
        # Network error, timeout, etc — fail gracefully
        return None


async def get_rainfall_risk(lat: float, lng: float) -> tuple[float, Optional[float]]:
    """
    Get rainfall risk factor (0–1) and raw mm for a location.

    Returns:
        (rainfall_risk, rainfall_mm)
        rainfall_risk: 0.2 (dry), 0.6 (moderate), 1.0 (heavy)
        rainfall_mm: raw value or None if unavailable
    """
    from services.epdo_service import bucket_rainfall

    rainfall_mm = await get_rainfall_mm(lat, lng)
    rainfall_risk = bucket_rainfall(rainfall_mm)

    return rainfall_risk, rainfall_mm
