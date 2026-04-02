"""
SSR AI Service — SSIM Repair Verification
Before/after image comparison using Structural Similarity Index.
Inverse logic: low similarity = surface changed = repair verified.
"""

import hashlib

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from config import settings
from services.image_utils import decode_image, to_grayscale_512


def compare_images(before_bytes: bytes, after_bytes: bytes) -> dict:
    """
    Compare before/after road images using SSIM.

    INVERSE LOGIC:
      - SSIM score < 0.75 → surface CHANGED → REPAIR_VERIFIED ✓
      - SSIM score ≥ 0.75 → surface UNCHANGED → REPAIR_REJECTED ✗

    If verification passes, generates SHA-256 hash of the after-photo
    as a tamper-proof receipt for the Digital Measurement Book.

    Args:
        before_bytes: Raw bytes of the before-repair photo.
        after_bytes: Raw bytes of the after-repair photo.

    Returns:
        dict with ssim_score, ssim_pass, verdict, verification_hash, audit_reason.
    """
    # Decode images
    before_img = decode_image(before_bytes)
    after_img = decode_image(after_bytes)

    # Normalize to 512×512 grayscale for fair comparison
    before_gray = to_grayscale_512(before_img)
    after_gray = to_grayscale_512(after_img)

    # Compute SSIM
    score = ssim(before_gray, after_gray)
    score = round(float(score), 4)

    # Inverse logic decision
    threshold = settings.SSIM_PASS_THRESHOLD
    passed = score < threshold

    # Generate SHA-256 hash of after-photo bytes ONLY on pass
    verification_hash = None
    if passed:
        verification_hash = hashlib.sha256(after_bytes).hexdigest()

    # Build human-readable audit reason
    if passed:
        audit_reason = (
            f"Surface change detected (SSIM {score} < {threshold} threshold). "
            f"Repair verified. SHA-256 hash generated for Digital MB."
        )
        verdict = "REPAIR_VERIFIED"
    else:
        audit_reason = (
            f"Surface unchanged (SSIM {score} >= {threshold} threshold). "
            f"Road surface appears identical to before-photo. Repair rejected."
        )
        verdict = "REPAIR_REJECTED"

    return {
        "ssim_score": score,
        "ssim_pass": passed,
        "verdict": verdict,
        "verification_hash": verification_hash,
        "audit_reason": audit_reason,
    }
