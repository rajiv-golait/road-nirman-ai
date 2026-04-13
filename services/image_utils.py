"""
SSR AI Service — Image Utilities
Shared image fetch + decode + preprocessing used across all endpoints.
"""

import cv2
import numpy as np
import httpx
from typing import Optional

from config import settings


class ImageFetchError(Exception):
    """Raised when image URL cannot be fetched or decoded."""
    pass


async def fetch_image_bytes(url: str) -> bytes:
    """
    Download image from a Supabase Storage URL.
    Returns raw bytes. Raises ImageFetchError on any failure.
    """
    if not url:
        raise ImageFetchError(f"Invalid image URL: {url}")
        
    if not url.startswith("http"):
        # Local file testing fallback
        import os
        path = url.replace("file:///", "").replace("file://", "")
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        raise ImageFetchError(f"Invalid image URL or file not found: {url}")

    try:
        async with httpx.AsyncClient(
            timeout=settings.IMAGE_FETCH_TIMEOUT_S,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url)

        if resp.status_code != 200:
            raise ImageFetchError(
                f"Image fetch failed: {url} returned HTTP {resp.status_code}"
            )

        raw = resp.content
        if len(raw) == 0:
            raise ImageFetchError(f"Image fetch returned empty body: {url}")

        if len(raw) > settings.IMAGE_MAX_SIZE_BYTES:
            raise ImageFetchError(
                f"Image too large: {len(raw)} bytes (max {settings.IMAGE_MAX_SIZE_BYTES})"
            )

        return raw

    except httpx.TimeoutException:
        raise ImageFetchError(
            f"Image fetch timed out after {settings.IMAGE_FETCH_TIMEOUT_S}s: {url}"
        )
    except httpx.RequestError as e:
        raise ImageFetchError(f"Image fetch network error: {e}")


def decode_image(raw_bytes: bytes) -> np.ndarray:
    """
    Decode raw bytes into an OpenCV BGR numpy array.
    Raises ImageFetchError if bytes are not a valid image.
    """
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageFetchError("Failed to decode image — corrupt or unsupported format")
    return img


def preprocess_for_detection(img: np.ndarray) -> np.ndarray:
    """
    Enhance image quality before sending to detection model.

    Pipeline:
      1. Denoise — reduce phone-camera sensor noise (fast non-local means)
      2. CLAHE  — adaptive contrast on lightness channel so dark potholes
                   and shadow-filled cracks become visible to the model
      3. Sharpen — recover edge detail lost by denoising

    Operates on a copy; never mutates the input.
    """
    out = img.copy()

    # 1. Denoise (fast mode: h=6 keeps detail, templateWindowSize=7)
    # OpenCV 4.10+ Python uses hColor; older tutorials used hForColorComponents (rejected in 4.11).
    out = cv2.fastNlMeansDenoisingColored(out, None, 6, 6, 7, 21)

    # 2. CLAHE on L-channel of LAB colour space
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    out = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # 3. Mild unsharp-mask sharpen (sigma=1, amount=0.6)
    gaussian = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
    out = cv2.addWeighted(out, 1.6, gaussian, -0.6, 0)

    return out


def to_grayscale_512(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR image to 512×512 grayscale.
    Used for SSIM comparison — both images must be same dimensions.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    return resized


def to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV default) to RGB (for PIL/model input)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    """Encode OpenCV image to JPEG bytes. Used for Roboflow API calls."""
    success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ImageFetchError("Failed to encode image to JPEG")
    return buffer.tobytes()


def is_road_scene(img: np.ndarray) -> tuple[bool, float, str]:
    """
    Determine whether the image plausibly shows a road/pavement surface.

    Uses three lightweight signals (no ML model needed):

      1. Gray-dominance: roads are predominantly gray (low saturation).
         Measure % of pixels with saturation < 60 in the lower half
         (upper half may be sky/buildings — lower half is where road is).

      2. Texture uniformity: roads have large low-texture regions.
         Measure Laplacian variance on the lower half — roads sit in
         a middle band (not too smooth like a blank wall, not too busy
         like text or foliage).

      3. Dark-region presence: road surfaces are mid-to-dark luminance.
         Measure % of pixels with value 30–160 in grayscale (asphalt range).

    Returns (is_road, confidence, reason).
    """
    h, w = img.shape[:2]

    # Focus on the lower 60% of the image (where road typically is)
    lower = img[int(h * 0.4):, :]
    lh, lw = lower.shape[:2]
    total_px = lh * lw

    # 1. Gray-dominance: low-saturation pixels in lower half
    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    gray_ratio = float(np.count_nonzero(sat < 60)) / total_px

    # 2. Texture check: Laplacian variance on lower-half grayscale
    gray_lower = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray_lower, cv2.CV_64F).var())

    # 3. Asphalt-range luminance (V channel 30–160)
    val = hsv[:, :, 2]
    asphalt_ratio = float(np.count_nonzero((val >= 30) & (val <= 160))) / total_px

    # Scoring
    score = 0.0

    # Gray dominance: >40% low-sat pixels → road-like
    if gray_ratio > 0.55:
        score += 0.40
    elif gray_ratio > 0.35:
        score += 0.25
    elif gray_ratio > 0.20:
        score += 0.10

    # Texture: roads have moderate texture (lap_var 50–800)
    if 50 < lap_var < 800:
        score += 0.30
    elif 20 < lap_var < 1500:
        score += 0.15

    # Asphalt luminance: >35% pixels in dark-mid range
    if asphalt_ratio > 0.45:
        score += 0.30
    elif asphalt_ratio > 0.25:
        score += 0.15

    # Decision: score >= 0.50 → plausible road scene
    is_road = score >= 0.50
    reason = (
        f"gray_ratio={gray_ratio:.2f}, lap_var={lap_var:.0f}, "
        f"asphalt_ratio={asphalt_ratio:.2f}, scene_score={score:.2f}"
    )

    return is_road, round(score, 2), reason


def nms_boxes(boxes: list, confidences: list, iou_threshold: float = 0.45) -> list[int]:
    """
    Non-Maximum Suppression to deduplicate overlapping bounding boxes.
    Returns indices of boxes to keep.
    """
    if not boxes:
        return []

    b = np.array(boxes, dtype=np.float32)
    c = np.array(confidences, dtype=np.float32)

    indices = cv2.dnn.NMSBoxes(
        bboxes=[(int(x1), int(y1), int(x2 - x1), int(y2 - y1)) for x1, y1, x2, y2 in b],
        scores=c.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )
    # cv2.dnn.NMSBoxes returns ndarray or tuple depending on version
    if indices is None or len(indices) == 0:
        return []
    return [int(i) for i in indices.flatten()]
