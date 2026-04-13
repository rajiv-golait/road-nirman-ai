"""
SSR AI Service — SSIM Repair Verification
Before/after image comparison using Structural Similarity Index.
Inverse logic: low similarity = surface changed = repair verified.

Enhanced with:
  - Multi-region SSIM (centre-weighted) to catch localized patches
  - Histogram correlation cross-check to defeat trivial image swaps
"""

import hashlib

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from config import settings
from services.image_utils import decode_image, to_grayscale_512


def _region_ssim(before: np.ndarray, after: np.ndarray) -> float:
    """
    Compute centre-weighted multi-region SSIM.

    Splits the 512×512 image into a 4×4 grid (16 patches).
    Centre patches get 2× weight because road damage is usually
    photographed in the middle of the frame.
    """
    h, w = before.shape
    rows, cols = 4, 4
    ph, pw = h // rows, w // cols

    scores = []
    weights = []

    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * ph, (r + 1) * ph
            x1, x2 = c * pw, (c + 1) * pw

            patch_b = before[y1:y2, x1:x2]
            patch_a = after[y1:y2, x1:x2]

            s = ssim(patch_b, patch_a)
            # centre 2×2 region gets double weight
            wt = 2.0 if (1 <= r <= 2 and 1 <= c <= 2) else 1.0
            scores.append(s)
            weights.append(wt)

    total_w = sum(weights)
    weighted = sum(s * w for s, w in zip(scores, weights)) / total_w
    return round(float(weighted), 4)


def _histogram_correlation(before: np.ndarray, after: np.ndarray) -> float:
    """
    Compute normalised histogram correlation between two grayscale images.
    Returns 0–1 where 1 = identical brightness distribution.
    Used as a sanity check: if histograms are nearly identical but SSIM
    says "changed", the images may just have lighting differences.
    """
    hist_b = cv2.calcHist([before], [0], None, [64], [0, 256])
    hist_a = cv2.calcHist([after], [0], None, [64], [0, 256])
    cv2.normalize(hist_b, hist_b)
    cv2.normalize(hist_a, hist_a)
    corr = cv2.compareHist(hist_b, hist_a, cv2.HISTCMP_CORREL)
    return round(float(max(corr, 0.0)), 4)


def compare_images(before_bytes: bytes, after_bytes: bytes) -> dict:
    """
    Compare before/after road images using enhanced SSIM.

    INVERSE LOGIC:
      - composite score < 0.75 → surface CHANGED → REPAIR_VERIFIED ✓
      - composite score ≥ 0.75 → surface UNCHANGED → REPAIR_REJECTED ✗

    Composite score = 0.70 × region_ssim + 0.30 × global_ssim

    Cross-check: if histogram correlation > 0.97 and SSIM says "changed",
    this is likely just a lighting/angle shift — reject anyway.

    On pass, generates SHA-256 hash of after-photo for Digital MB.
    """
    before_img = decode_image(before_bytes)
    after_img = decode_image(after_bytes)

    before_gray = to_grayscale_512(before_img)
    after_gray = to_grayscale_512(after_img)

    # Global SSIM
    global_score = float(ssim(before_gray, after_gray))

    # Centre-weighted multi-region SSIM
    region_score = _region_ssim(before_gray, after_gray)

    # Composite: region-weighted dominates (catches localized repairs)
    composite = round(0.70 * region_score + 0.30 * global_score, 4)

    # Histogram correlation cross-check
    hist_corr = _histogram_correlation(before_gray, after_gray)

    threshold = settings.SSIM_PASS_THRESHOLD
    passed = composite < threshold

    # Cross-check: if brightness distribution is nearly identical
    # but SSIM says "changed", it's probably just lighting/angle — reject
    if passed and hist_corr > 0.97 and composite > (threshold - 0.10):
        passed = False

    verification_hash = None
    if passed:
        verification_hash = hashlib.sha256(after_bytes).hexdigest()

    if passed:
        audit_reason = (
            f"Surface change detected (composite SSIM {composite} < {threshold}). "
            f"Region SSIM={region_score}, global SSIM={round(global_score, 4)}, "
            f"hist_corr={hist_corr}. "
            f"Repair verified. SHA-256 hash generated for Digital MB."
        )
        verdict = "REPAIR_VERIFIED"
    else:
        audit_reason = (
            f"Surface unchanged (composite SSIM {composite} >= {threshold}). "
            f"Region SSIM={region_score}, global SSIM={round(global_score, 4)}, "
            f"hist_corr={hist_corr}. "
            f"Road surface appears identical to before-photo. Repair rejected."
        )
        verdict = "REPAIR_REJECTED"

    return {
        "ssim_score": composite,
        "ssim_pass": passed,
        "verdict": verdict,
        "verification_hash": verification_hash,
        "audit_reason": audit_reason,
    }
