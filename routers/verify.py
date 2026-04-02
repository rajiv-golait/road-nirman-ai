"""
SSR AI Service — POST /verify-repair
SSIM before/after comparison. Inverse logic. Fail closed.
Fetches both images by URL. Never passes on fetch error.
"""

import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.ssim_service import compare_images
from services.image_utils import fetch_image_bytes, ImageFetchError


router = APIRouter(tags=["Verification"])


# ── Request / Response Models ─────────────────────────────────
class VerifyRequest(BaseModel):
    before_image_url: str = Field(..., description="Supabase Storage URL of before-photo")
    after_image_url: str = Field(..., description="Supabase Storage URL of after-photo")
    ticket_id: Optional[str] = Field(None, description="Ticket UUID for tracing")


class VerifyResponse(BaseModel):
    success: bool
    ssim_score: Optional[float] = None
    ssim_pass: bool = False
    verdict: str = "VERIFICATION_FAILED"
    verification_hash: Optional[str] = None
    audit_reason: str = ""
    processing_ms: int = 0
    errors: list[str] = []


# ── Endpoint ──────────────────────────────────────────────────
@router.post("/verify-repair", response_model=VerifyResponse)
async def verify_repair(req: VerifyRequest):
    """
    Compare before/after road images using SSIM.

    INVERSE LOGIC:
      score < 0.75 → surface CHANGED → REPAIR_VERIFIED ✓
      score ≥ 0.75 → surface UNCHANGED → REPAIR_REJECTED ✗

    FAIL CLOSED:
      If either image cannot be fetched or decoded, returns
      success=false and ssim_pass=false. Never a false positive.

    On verification pass, generates SHA-256 hash of after-photo
    as a tamper-proof receipt for the Digital Measurement Book.
    """
    start = time.time()

    # ── Fetch before-image ────────────────────────────────────
    try:
        before_bytes = await fetch_image_bytes(req.before_image_url)
    except ImageFetchError as e:
        elapsed = int((time.time() - start) * 1000)
        return VerifyResponse(
            success=False,
            ssim_pass=False,
            verdict="VERIFICATION_FAILED",
            audit_reason=f"Failed to fetch before_image_url: {e}",
            errors=[f"Image fetch failed: before_image_url — {e}"],
            processing_ms=elapsed,
        )

    # ── Fetch after-image ─────────────────────────────────────
    try:
        after_bytes = await fetch_image_bytes(req.after_image_url)
    except ImageFetchError as e:
        elapsed = int((time.time() - start) * 1000)
        return VerifyResponse(
            success=False,
            ssim_pass=False,
            verdict="VERIFICATION_FAILED",
            audit_reason=f"Failed to fetch after_image_url: {e}",
            errors=[f"Image fetch failed: after_image_url — {e}"],
            processing_ms=elapsed,
        )

    # ── Run SSIM comparison ───────────────────────────────────
    try:
        result = compare_images(before_bytes, after_bytes)
        elapsed = int((time.time() - start) * 1000)

        return VerifyResponse(
            success=True,
            ssim_score=result["ssim_score"],
            ssim_pass=result["ssim_pass"],
            verdict=result["verdict"],
            verification_hash=result["verification_hash"],
            audit_reason=result["audit_reason"],
            processing_ms=elapsed,
        )

    except ImageFetchError as e:
        # Image decode failure (corrupt image bytes)
        elapsed = int((time.time() - start) * 1000)
        return VerifyResponse(
            success=False,
            ssim_pass=False,
            verdict="VERIFICATION_FAILED",
            audit_reason=f"Image decode failed: {e}",
            errors=[str(e)],
            processing_ms=elapsed,
        )

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return VerifyResponse(
            success=False,
            ssim_pass=False,
            verdict="VERIFICATION_FAILED",
            audit_reason=f"SSIM comparison failed: {type(e).__name__}: {e}",
            errors=[f"Verification error: {type(e).__name__}: {str(e)}"],
            processing_ms=elapsed,
        )
