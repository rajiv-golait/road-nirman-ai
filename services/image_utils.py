"""
SSR AI Service — Image Utilities
Shared image fetch + decode used by /detect-road-damage and /verify-repair.
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
    if not url or not url.startswith("http"):
        raise ImageFetchError(f"Invalid image URL: {url}")

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
