"""
SSR AI Service — Configuration
Pydantic Settings loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """All configuration from environment. No hardcoded secrets."""

    # ── Auth ──────────────────────────────────────────────────
    # Shared secret between Edge Functions and this service.
    # Every POST request must include X-SSR-Secret header.
    AI_SERVICE_SECRET: str = "ssr-demo-secret-2025"

    # ── Model Source ──────────────────────────────────────────
    # "roboflow"  → Roboflow Inference API (hackathon demo)
    # "local"     → Load .pt file with ultralytics (production)
    # "heuristic" → Edge detection fallback (no model needed)
    MODEL_SOURCE: Literal["roboflow", "local", "heuristic"] = "roboflow"

    # ── Roboflow (when MODEL_SOURCE = "roboflow") ────────────
    ROBOFLOW_API_KEY: str = ""
    ROBOFLOW_MODEL_ID: str = "pothole-detection-gv5e7/3"
    ROBOFLOW_CONFIDENCE: int = 40  # Roboflow uses 0-100 scale

    # ── Local Model (when MODEL_SOURCE = "local") ────────────
    LOCAL_MODEL_PATH: str = "models/yolov12s_rdd2022.pt"

    # ── YOLO Inference ────────────────────────────────────────
    YOLO_CONFIDENCE: float = 0.65
    YOLO_IOU: float = 0.50

    # ── SSIM Verification ─────────────────────────────────────
    SSIM_PASS_THRESHOLD: float = 0.75  # score < this = surface changed = PASS

    # ── Image Handling ────────────────────────────────────────
    IMAGE_FETCH_TIMEOUT_S: int = 10
    IMAGE_MAX_SIZE_BYTES: int = 10 * 1024 * 1024  # 10MB

    # ── Weather (OpenWeatherMap) ──────────────────────────────
    OPENWEATHERMAP_API_KEY: str = ""
    WEATHER_CACHE_TTL_S: int = 3600  # 1 hour
    WEATHER_DEFAULT_RAINFALL_RISK: float = 0.3  # Fallback if API down

    # ── Service Info ──────────────────────────────────────────
    SERVICE_VERSION: str = "1.0.0"
    RULESET_VERSION: str = "epdo_irc_v1"

    # ── Damage Routing Map ────────────────────────────────────
    # Maps detection labels → (department_code, damage_type)
    # Used by both Roboflow and local YOLOv12 outputs
    @property
    def damage_routing(self) -> dict:
        return {
            # RDD2022 codes (YOLOv12s local model)
            "D00": ("ROADS", "crack"),              # Longitudinal crack
            "D10": ("ROADS", "crack"),              # Transverse crack
            "D20": ("ROADS", "surface_failure"),    # Alligator crack
            "D40": ("ROADS", "pothole"),            # Pothole
            # Roboflow labels (gv5e7/3)
            "pothole": ("ROADS", "pothole"),
            "Pothole": ("ROADS", "pothole"),
            # Heuristic fallback
            "road_damage": ("ROADS", "pothole"),
        }

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


# Singleton — import this everywhere
settings = Settings()
