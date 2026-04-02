"""
SSR AI Service — Detection Service (Hybrid)
Three backends behind one interface. Swap with a single env var.

  MODEL_SOURCE = "roboflow"  → Roboflow Inference API (hackathon demo)
  MODEL_SOURCE = "local"     → Load .pt file with ultralytics (production)
  MODEL_SOURCE = "heuristic" → Edge detection fallback (no model needed)
"""

import base64
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import httpx

from config import settings
from services.image_utils import decode_image, encode_jpeg


# ── Detection Result ──────────────────────────────────────────
@dataclass
class DetectionResult:
    """Structured detection output. Maps directly to tickets table columns."""
    detected: bool = False
    damage_type: str = "unknown"
    ai_confidence: float = 0.0
    total_potholes: int = 0
    bounding_boxes: list = field(default_factory=list)
    ai_severity_index: float = 0.0  # SAI: 0–1
    ai_source: str = "OFFLINE_ESTIMATE"
    model_version: str = "none"
    raw_classes: list = field(default_factory=list)  # Original model labels


# ── Damage Type Mapping ───────────────────────────────────────
def map_damage_type(label: str) -> str:
    """Map model-specific labels to SSR standard damage_type values."""
    routing = settings.damage_routing
    if label in routing:
        return routing[label][1]  # (dept_code, damage_type) → damage_type
    return "pothole"  # Safe default


# ── SAI Calculation ───────────────────────────────────────────
def calculate_sai(
    bounding_boxes: list,
    image_width: int,
    image_height: int
) -> float:
    """
    Severity AI Index: ratio of detected damage area to total image area.
    SAI = min(sum(box_areas) / image_area * 3.5, 1.0)

    Higher SAI = more severe damage coverage.
    """
    if not bounding_boxes or image_width == 0 or image_height == 0:
        return 0.0

    image_area = image_width * image_height
    total_box_area = 0

    for box in bounding_boxes:
        if len(box) >= 4:
            x1, y1, x2, y2 = box[:4]
            box_area = abs(x2 - x1) * abs(y2 - y1)
            total_box_area += box_area

    ratio = total_box_area / image_area
    sai = min(ratio * 3.5, 1.0)
    return round(sai, 4)


# ── Detection Service ────────────────────────────────────────
class DetectionService:
    """
    Hybrid detection service. One interface, three backends.
    Initialized once at app startup via lifespan.
    """

    def __init__(self):
        self._model = None
        self._model_loaded = False
        self._ai_source = "OFFLINE_ESTIMATE"
        self._model_identifier = "none"

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def ai_source(self) -> str:
        return self._ai_source

    @property
    def model_identifier(self) -> str:
        return self._model_identifier

    def load_model(self):
        """Load model based on MODEL_SOURCE config. Called once at startup."""

        if settings.MODEL_SOURCE == "roboflow":
            # Roboflow: no local model to load. Just verify API key exists.
            if settings.ROBOFLOW_API_KEY:
                self._model_loaded = True
                self._ai_source = "ROBOFLOW_API"
                self._model_identifier = f"roboflow/{settings.ROBOFLOW_MODEL_ID}"
                print(f"   Roboflow model: {settings.ROBOFLOW_MODEL_ID}")
            else:
                print("   ⚠️ ROBOFLOW_API_KEY not set — falling back to heuristic")
                self._ai_source = "OFFLINE_ESTIMATE"
                self._model_identifier = "heuristic_edge_v1"

        elif settings.MODEL_SOURCE == "local":
            try:
                from ultralytics import YOLO
                self._model = YOLO(settings.LOCAL_MODEL_PATH)
                self._model_loaded = True
                self._ai_source = "YOLO_LOCAL"
                self._model_identifier = f"local/{settings.LOCAL_MODEL_PATH}"
                print(f"   Local model loaded: {settings.LOCAL_MODEL_PATH}")
            except Exception as e:
                print(f"   ⚠️ Failed to load local model: {e}")
                print(f"   Falling back to heuristic mode")
                self._ai_source = "OFFLINE_ESTIMATE"
                self._model_identifier = "heuristic_edge_v1"

        else:  # heuristic
            self._model_loaded = True  # Always "ready" in heuristic mode
            self._ai_source = "OFFLINE_ESTIMATE"
            self._model_identifier = "heuristic_edge_v1"
            print("   Running in heuristic mode (no AI model)")

    async def detect(self, image_bytes: bytes) -> DetectionResult:
        """
        Run detection using the configured backend.
        Returns DetectionResult regardless of which backend is active.
        """
        if settings.MODEL_SOURCE == "roboflow" and settings.ROBOFLOW_API_KEY:
            return await self._roboflow_detect(image_bytes)
        elif settings.MODEL_SOURCE == "local" and self._model is not None:
            return self._local_detect(image_bytes)
        else:
            return self._heuristic_detect(image_bytes)

    # ── Backend 1: Roboflow Inference API ─────────────────────
    async def _roboflow_detect(self, image_bytes: bytes) -> DetectionResult:
        """Call Roboflow hosted inference API."""
        try:
            # Encode image to base64 for Roboflow
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")

            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"https://detect.roboflow.com/{settings.ROBOFLOW_MODEL_ID}",
                    params={
                        "api_key": settings.ROBOFLOW_API_KEY,
                        "confidence": settings.ROBOFLOW_CONFIDENCE,
                    },
                    data=img_b64,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

            if resp.status_code != 200:
                # API error — fall back to heuristic
                print(f"   Roboflow API error: {resp.status_code}")
                return self._heuristic_detect(image_bytes)

            data = resp.json()
            predictions = data.get("predictions", [])
            img_w = data.get("image", {}).get("width", 640)
            img_h = data.get("image", {}).get("height", 640)

            if not predictions:
                return DetectionResult(
                    detected=False,
                    ai_source="ROBOFLOW_API",
                    model_version=self._model_identifier,
                )

            # Parse Roboflow predictions
            bounding_boxes = []
            confidences = []
            raw_classes = []

            for pred in predictions:
                x = pred.get("x", 0)
                y = pred.get("y", 0)
                w = pred.get("width", 0)
                h = pred.get("height", 0)
                conf = pred.get("confidence", 0)
                cls = pred.get("class", "pothole")

                # Roboflow returns center x,y + width,height
                x1 = x - w / 2
                y1 = y - h / 2
                x2 = x + w / 2
                y2 = y + h / 2

                bounding_boxes.append([
                    round(x1, 1), round(y1, 1),
                    round(x2, 1), round(y2, 1)
                ])
                confidences.append(conf)
                raw_classes.append(cls)

            # Calculate metrics
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            sai = calculate_sai(bounding_boxes, img_w, img_h)
            primary_class = max(set(raw_classes), key=raw_classes.count)

            return DetectionResult(
                detected=True,
                damage_type=map_damage_type(primary_class),
                ai_confidence=round(avg_confidence, 4),
                total_potholes=len(predictions),
                bounding_boxes=bounding_boxes,
                ai_severity_index=sai,
                ai_source="ROBOFLOW_API",
                model_version=self._model_identifier,
                raw_classes=raw_classes,
            )

        except Exception as e:
            print(f"   Roboflow detection error: {e}")
            return self._heuristic_detect(image_bytes)

    # ── Backend 2: Local YOLOv12 .pt File ─────────────────────
    def _local_detect(self, image_bytes: bytes) -> DetectionResult:
        """Run inference with locally loaded YOLO model."""
        try:
            img = decode_image(image_bytes)
            h, w = img.shape[:2]

            results = self._model.predict(
                img,
                conf=settings.YOLO_CONFIDENCE,
                iou=settings.YOLO_IOU,
                verbose=False,
            )

            if not results or len(results[0].boxes) == 0:
                return DetectionResult(
                    detected=False,
                    ai_source="YOLO_LOCAL",
                    model_version=self._model_identifier,
                )

            result = results[0]
            bounding_boxes = []
            confidences = []
            raw_classes = []

            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, f"class_{cls_id}")

                bounding_boxes.append([round(v, 1) for v in xyxy])
                confidences.append(conf)
                raw_classes.append(cls_name)

            avg_confidence = sum(confidences) / len(confidences)
            sai = calculate_sai(bounding_boxes, w, h)
            primary_class = max(set(raw_classes), key=raw_classes.count)

            return DetectionResult(
                detected=True,
                damage_type=map_damage_type(primary_class),
                ai_confidence=round(avg_confidence, 4),
                total_potholes=len(bounding_boxes),
                bounding_boxes=bounding_boxes,
                ai_severity_index=sai,
                ai_source="YOLO_LOCAL",
                model_version=self._model_identifier,
                raw_classes=raw_classes,
            )

        except Exception as e:
            print(f"   Local YOLO error: {e}")
            return self._heuristic_detect(image_bytes)

    # ── Backend 3: Heuristic Fallback ─────────────────────────
    def _heuristic_detect(self, image_bytes: bytes) -> DetectionResult:
        """
        Edge detection-based rough damage estimate.
        No AI model needed. Marks output as OFFLINE_ESTIMATE.
        """
        try:
            img = decode_image(image_bytes)
            h, w = img.shape[:2]

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)
            # Calculate edge density as rough damage proxy
            edge_density = np.count_nonzero(edges) / (w * h)

            # Rough classification based on edge density
            if edge_density > 0.15:
                detected = True
                damage_type = "pothole"
                confidence = min(edge_density * 2.5, 0.85)
                sai = min(edge_density * 2.0, 0.8)
                count = max(1, int(edge_density * 20))
            elif edge_density > 0.08:
                detected = True
                damage_type = "crack"
                confidence = min(edge_density * 2.0, 0.60)
                sai = min(edge_density * 1.5, 0.5)
                count = 1
            else:
                detected = False
                damage_type = "unknown"
                confidence = edge_density
                sai = 0.0
                count = 0

            return DetectionResult(
                detected=detected,
                damage_type=damage_type,
                ai_confidence=round(confidence, 4),
                total_potholes=count,
                bounding_boxes=[],  # Heuristic can't produce boxes
                ai_severity_index=round(sai, 4),
                ai_source="OFFLINE_ESTIMATE",
                model_version="heuristic_edge_v1",
                raw_classes=[damage_type] if detected else [],
            )

        except Exception:
            return DetectionResult(
                ai_source="OFFLINE_ESTIMATE",
                model_version="heuristic_edge_v1",
            )


# ── Singleton ─────────────────────────────────────────────────
detection_service = DetectionService()
