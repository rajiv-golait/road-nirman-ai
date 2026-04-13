"""
SSR AI Service — Detection Service (Hybrid)
Three backends behind one interface. Swap with a single env var.

  MODEL_SOURCE = "roboflow"  → Roboflow Inference API (hackathon demo)
  MODEL_SOURCE = "local"     → Load .pt file with ultralytics (production)
  MODEL_SOURCE = "heuristic" → Edge detection fallback (no model needed)
"""

import base64
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import httpx

from config import settings
from services.image_utils import (
    decode_image, encode_jpeg, preprocess_for_detection, nms_boxes,
    is_road_scene,
)

logger = logging.getLogger("ssr.yolo")


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
    rejection_reason: str = ""  # Non-empty if image was rejected pre-detection
    # Always set so clients can see which backend ran (prints often invisible under uvicorn --reload)
    inference_path: str = ""
    inference_detail: str = ""


# ── Damage Type Mapping ───────────────────────────────────────
def map_damage_type(label: str) -> str:
    """Map model-specific labels to SSR standard damage_type values."""
    routing = settings.damage_routing
    if label in routing:
        return routing[label][1]  # (dept_code, damage_type) → damage_type
    return "pothole"  # Safe default


# ── SAI Calculation (density-weighted) ────────────────────────
def calculate_sai(
    bounding_boxes: list,
    confidences: list,
    image_width: int,
    image_height: int,
) -> float:
    """
    Severity AI Index — density-weighted damage coverage score (0–1).

    Improvements over flat ratio:
      - Confidence-weighted area: high-confidence boxes count more
      - Count bonus: multiple detections increase severity (log scale)
      - Spatial spread penalty: clustered damage = single site,
        spread-out damage = systemic problem
    """
    if not bounding_boxes or image_width == 0 or image_height == 0:
        return 0.0

    image_area = image_width * image_height
    n = len(bounding_boxes)

    # confidence-weighted area
    weighted_area = 0.0
    centroids = []
    for i, box in enumerate(bounding_boxes):
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        box_area = abs(x2 - x1) * abs(y2 - y1)
        conf = confidences[i] if i < len(confidences) else 0.5
        weighted_area += box_area * conf
        centroids.append(((x1 + x2) / 2, (y1 + y2) / 2))

    area_ratio = weighted_area / image_area

    # count bonus (log curve: 1→0, 2→0.10, 5→0.23, 10→0.33)
    count_bonus = math.log10(max(n, 1)) / 3.0

    # spatial spread factor: avg pairwise distance / image diagonal
    spread_factor = 0.0
    if len(centroids) >= 2:
        diag = math.sqrt(image_width ** 2 + image_height ** 2)
        total_dist = 0.0
        pairs = 0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dx = centroids[i][0] - centroids[j][0]
                dy = centroids[i][1] - centroids[j][1]
                total_dist += math.sqrt(dx * dx + dy * dy)
                pairs += 1
        spread_factor = (total_dist / pairs / diag) * 0.15 if pairs else 0.0

    sai = min((area_ratio * 3.5) + count_bonus + spread_factor, 1.0)
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
                logger.info("Roboflow model: %s", settings.ROBOFLOW_MODEL_ID)
            else:
                logger.warning("ROBOFLOW_API_KEY not set; requests will use heuristic until configured")
                self._ai_source = "OFFLINE_ESTIMATE"
                self._model_identifier = "heuristic_edge_v1"

        elif settings.MODEL_SOURCE == "local":
            try:
                from ultralytics import YOLO
                self._model = YOLO(settings.LOCAL_MODEL_PATH)
                self._model_loaded = True
                self._ai_source = "YOLO_LOCAL"
                self._model_identifier = f"local/{settings.LOCAL_MODEL_PATH}"
                logger.info("Local model loaded: %s", settings.LOCAL_MODEL_PATH)
            except Exception as e:
                logger.warning("Failed to load local model: %s; heuristic fallback", e)
                self._ai_source = "OFFLINE_ESTIMATE"
                self._model_identifier = "heuristic_edge_v1"

        else:  # heuristic
            self._model_loaded = True  # Always "ready" in heuristic mode
            self._ai_source = "OFFLINE_ESTIMATE"
            self._model_identifier = "heuristic_edge_v1"
            logger.info("Running in heuristic mode (no AI model)")

    async def detect(self, image_bytes: bytes) -> DetectionResult:
        """
        Run detection using the configured backend.

        Gate: checks whether the image looks like a road scene first.
        If not, returns detected=False immediately — prevents false
        positives on selfies, food photos, documents, etc.
        """
        # ── Road-scene validation gate ────────────────────────
        try:
            img = decode_image(image_bytes)
            is_road, scene_score, scene_detail = is_road_scene(img)
            if not is_road:
                logger.info("Scene rejected: %s", scene_detail)
                return DetectionResult(
                    detected=False,
                    ai_source=self._ai_source,
                    model_version=self._model_identifier,
                    rejection_reason=(
                        f"Image does not appear to be a road surface "
                        f"(scene_score={scene_score}). "
                        f"Please upload a clear photo of the damaged road."
                    ),
                    inference_path="scene_gate",
                    inference_detail=scene_detail[:200],
                )
        except Exception as e:
            logger.warning("Scene check error (proceeding): %s", e)

        # ── Detection ─────────────────────────────────────────
        if settings.MODEL_SOURCE == "roboflow" and settings.ROBOFLOW_API_KEY:
            return await self._roboflow_detect(image_bytes)
        elif settings.MODEL_SOURCE == "local" and self._model is not None:
            return self._local_detect(image_bytes)
        else:
            if settings.MODEL_SOURCE == "roboflow":
                td = "roboflow_missing_api_key"
            elif settings.MODEL_SOURCE == "local":
                td = "local_model_not_loaded"
            else:
                td = "model_source_heuristic"
            return self._heuristic_detect(image_bytes, trace_detail=td)

    # ── Backend 1: Roboflow Inference API ─────────────────────
    async def _roboflow_detect(self, image_bytes: bytes) -> DetectionResult:
        """Call Roboflow hosted inference API with preprocessing + NMS."""
        try:
            # Preprocess: denoise + CLAHE + sharpen before sending to model
            img = decode_image(image_bytes)
            enhanced = preprocess_for_detection(img)
            enhanced_bytes = encode_jpeg(enhanced, quality=90)

            img_b64 = base64.b64encode(enhanced_bytes).decode("utf-8")

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
                snippet = (resp.text or "")[:400].replace("\r", " ").replace("\n", " ")
                logger.warning(
                    "Roboflow API error: status=%s body_prefix=%s",
                    resp.status_code,
                    snippet,
                )
                return self._heuristic_detect(
                    image_bytes,
                    trace_detail=f"after_roboflow_http_{resp.status_code}",
                )

            data = resp.json()
            predictions = data.get("predictions", [])
            img_w = data.get("image", {}).get("width", 640)
            img_h = data.get("image", {}).get("height", 640)

            if not predictions:
                return DetectionResult(
                    detected=False,
                    ai_source="ROBOFLOW_API",
                    model_version=self._model_identifier,
                    inference_path="roboflow",
                    inference_detail="no_predictions_above_threshold",
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

            # NMS: remove overlapping duplicate detections
            keep_indices = nms_boxes(bounding_boxes, confidences,
                                     iou_threshold=settings.YOLO_IOU)
            bounding_boxes = [bounding_boxes[i] for i in keep_indices]
            confidences = [confidences[i] for i in keep_indices]
            raw_classes = [raw_classes[i] for i in keep_indices]

            if not bounding_boxes:
                return DetectionResult(
                    detected=False,
                    ai_source="ROBOFLOW_API",
                    model_version=self._model_identifier,
                    inference_path="roboflow",
                    inference_detail="nms_removed_all_boxes",
                )

            avg_confidence = sum(confidences) / len(confidences)
            sai = calculate_sai(bounding_boxes, confidences, img_w, img_h)
            primary_class = max(set(raw_classes), key=raw_classes.count)

            return DetectionResult(
                detected=True,
                damage_type=map_damage_type(primary_class),
                ai_confidence=round(avg_confidence, 4),
                total_potholes=len(bounding_boxes),
                bounding_boxes=bounding_boxes,
                ai_severity_index=sai,
                ai_source="ROBOFLOW_API",
                model_version=self._model_identifier,
                raw_classes=raw_classes,
                inference_path="roboflow",
                inference_detail="ok",
            )

        except Exception as e:
            logger.exception("Roboflow detection failed: %s", e)
            return self._heuristic_detect(
                image_bytes,
                trace_detail=f"after_roboflow_{type(e).__name__}",
            )

    # ── Backend 2: Local YOLOv12 .pt File ─────────────────────
    def _local_detect(self, image_bytes: bytes) -> DetectionResult:
        """Run inference with locally loaded YOLO model + preprocessing."""
        try:
            img = decode_image(image_bytes)
            enhanced = preprocess_for_detection(img)
            h, w = enhanced.shape[:2]

            results = self._model.predict(
                enhanced,
                conf=settings.YOLO_CONFIDENCE,
                iou=settings.YOLO_IOU,
                verbose=False,
            )

            if not results or len(results[0].boxes) == 0:
                return DetectionResult(
                    detected=False,
                    ai_source="YOLO_LOCAL",
                    model_version=self._model_identifier,
                    inference_path="local_yolo",
                    inference_detail="no_boxes",
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
            sai = calculate_sai(bounding_boxes, confidences, w, h)
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
                inference_path="local_yolo",
                inference_detail="ok",
            )

        except Exception as e:
            logger.warning("Local YOLO error: %s", e, exc_info=True)
            return self._heuristic_detect(
                image_bytes,
                trace_detail=f"after_local_yolo_{type(e).__name__}",
            )

    # ── Backend 3: Heuristic Fallback ─────────────────────────
    def _heuristic_detect(self, image_bytes: bytes, *, trace_detail: str = "heuristic_only") -> DetectionResult:
        """
        Morphology-based damage estimate when no AI model is available.

        Strategy (much better than raw edge density):
          1. Convert to grayscale, apply CLAHE for contrast
          2. Adaptive threshold to isolate dark regions (pothole candidates)
          3. Morphological close to merge nearby fragments
          4. Find contours; filter by area + circularity
             - Potholes: compact, roughly circular dark patches
             - Cracks: elongated, high aspect-ratio contours
          5. Generate bounding boxes from qualifying contours
        """
        try:
            img = decode_image(image_bytes)
            enhanced = preprocess_for_detection(img)
            h, w = enhanced.shape[:2]
            image_area = w * h

            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

            # Adaptive threshold: dark regions relative to local neighbourhood
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, blockSize=51, C=15,
            )

            # Morphological close: merge small fragments into damage blobs
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            # Find contours
            contours, _ = cv2.findContours(
                closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )

            # Filter contours by size and shape
            min_area = image_area * 0.005   # at least 0.5% of image
            max_area = image_area * 0.40    # at most 40% (not the whole road)

            bounding_boxes = []
            confidences = []
            raw_classes = []

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-6)

                # Classify by shape
                if circularity > 0.3 and aspect < 3.0:
                    # Compact blob → pothole candidate
                    label = "pothole"
                    conf = min(0.35 + circularity * 0.3 + (area / image_area) * 2.0, 0.70)
                elif aspect >= 3.0:
                    # Elongated → crack candidate
                    label = "crack"
                    conf = min(0.25 + (area / image_area) * 1.5, 0.55)
                else:
                    continue  # too irregular, skip

                bounding_boxes.append([
                    round(float(x), 1), round(float(y), 1),
                    round(float(x + bw), 1), round(float(y + bh), 1),
                ])
                confidences.append(round(conf, 4))
                raw_classes.append(label)

            if not bounding_boxes:
                return DetectionResult(
                    detected=False,
                    ai_source="OFFLINE_ESTIMATE",
                    model_version="heuristic_morph_v2",
                    inference_path="heuristic",
                    inference_detail=trace_detail,
                )

            # NMS to remove overlapping detections
            keep = nms_boxes(bounding_boxes, confidences, iou_threshold=0.45)
            bounding_boxes = [bounding_boxes[i] for i in keep]
            confidences = [confidences[i] for i in keep]
            raw_classes = [raw_classes[i] for i in keep]

            avg_confidence = sum(confidences) / len(confidences)
            sai = calculate_sai(bounding_boxes, confidences, w, h)
            primary_class = max(set(raw_classes), key=raw_classes.count)

            return DetectionResult(
                detected=True,
                damage_type=primary_class,
                ai_confidence=round(avg_confidence, 4),
                total_potholes=len(bounding_boxes),
                bounding_boxes=bounding_boxes,
                ai_severity_index=sai,
                ai_source="OFFLINE_ESTIMATE",
                model_version="heuristic_morph_v2",
                raw_classes=raw_classes,
                inference_path="heuristic",
                inference_detail=trace_detail,
            )

        except Exception as e:
            logger.exception("Heuristic detection failed: %s", e)
            return DetectionResult(
                ai_source="OFFLINE_ESTIMATE",
                model_version="heuristic_morph_v2",
                inference_path="heuristic",
                inference_detail=f"{trace_detail}_error_{type(e).__name__}",
            )


# ── Singleton ─────────────────────────────────────────────────
detection_service = DetectionService()
