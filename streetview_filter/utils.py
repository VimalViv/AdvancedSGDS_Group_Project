"""Shared utility functions for the street-view filtering pipeline.

This module consolidates helpers that were previously duplicated across
notebook cells and scripts:

* ``clamp`` / ``normalize`` / ``inverse_normalize`` -- numeric helpers
* ``weighted_sum`` -- weighted-component scoring (was inlined in every
  detector method)
* ``load_grayscale_images`` -- unified image-path loading (was duplicated
  between ``_load_placeholder_refs`` and ``_load_logo_templates``)
* ``logo_absence_score`` -- (was duplicated in ``_score_non_google`` and
  ``_score_interior``)
* ``morphology_open_bool`` -- boolean-mask morphology (was repeated three
  times in ``_compute_masks``)
* ``sample_from_dataframe`` -- stratified review sampling (was repeated
  three times in ``generate_review_samples``)
* ``utc_now_iso`` -- timestamp generation (was duplicated in
  ``_row_from_scores`` and ``run``)
* ``ensure_dir`` / ``discover_images`` / image-hash helpers -- unchanged
  but centralised here for single-source imports
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

from streetview_filter.config import RuntimeConfig


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def normalize(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low))


def inverse_normalize(value: float, low: float, high: float) -> float:
    return 1.0 - normalize(value, low, high)


def weighted_sum(components: Sequence[Tuple[float, float]]) -> float:
    """Return ``clamp(sum(weight * value for weight, value in components))``."""
    return clamp(sum(w * v for w, v in components))


# ---------------------------------------------------------------------------
# Logo / template helpers  (was duplicated in _score_non_google & _score_interior)
# ---------------------------------------------------------------------------

def logo_absence_score(logo_score: float, logo_available: bool) -> float:
    """Compute the "logo absence" signal used by multiple detectors."""
    return (1.0 - logo_score) if logo_available else 0.0


# ---------------------------------------------------------------------------
# Timestamp helper  (was duplicated in _row_from_scores & run)
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_images(input_dir: Path, cfg: RuntimeConfig) -> List[Path]:
    suffixes = {s.lower() for s in cfg.supported_extensions}
    if cfg.recursive:
        candidates = [p for p in input_dir.rglob("*") if p.is_file()]
    else:
        candidates = [p for p in input_dir.iterdir() if p.is_file()]
    return sorted([p for p in candidates if p.suffix.lower() in suffixes])


# ---------------------------------------------------------------------------
# Image-loading helpers  (was duplicated in _load_placeholder_refs & _load_logo_templates)
# ---------------------------------------------------------------------------

def load_grayscale_images(paths: List[str]) -> List[Tuple[str, np.ndarray]]:
    """Load images as grayscale from a list of file-system paths.

    Returns a list of ``(path_str, gray_array)`` for every path that exists
    and can be decoded.  Previously this logic was copy-pasted in both
    ``_load_placeholder_refs`` and ``_load_logo_templates``.
    """
    results: List[Tuple[str, np.ndarray]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            results.append((str(path), gray))
    return results


# ---------------------------------------------------------------------------
# Morphology helper  (was repeated 3x in _compute_masks)
# ---------------------------------------------------------------------------

def morphology_open_bool(
    mask: np.ndarray,
    kernel: np.ndarray,
    iterations: int = 1,
) -> np.ndarray:
    """Apply morphological opening on a boolean mask and return a boolean mask."""
    return cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=iterations,
    ).astype(bool)


# ---------------------------------------------------------------------------
# Image resizing
# ---------------------------------------------------------------------------

def resize_keep_aspect(image: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    current_max = max(h, w)
    if current_max <= max_side:
        return image, 1.0
    scale = max_side / float(current_max)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


# ---------------------------------------------------------------------------
# Perceptual hashing & similarity
# ---------------------------------------------------------------------------

def compute_dhash(gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    tiny = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = tiny[:, 1:] > tiny[:, :-1]
    return diff.astype(np.uint8).reshape(-1)


def compute_phash(gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:]) if hash_size > 1 else np.median(low)
    bits = low > med
    return bits.astype(np.uint8).reshape(-1)


def hash_similarity(bits_a: np.ndarray, bits_b: np.ndarray) -> float:
    if bits_a.shape != bits_b.shape or bits_a.size == 0:
        return 0.0
    dist = np.count_nonzero(bits_a != bits_b)
    return float(1.0 - dist / bits_a.size)


def corr_similarity(gray_a: np.ndarray, gray_b: np.ndarray, size: int = 128) -> float:
    a = cv2.resize(gray_a, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(gray_b, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)

    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0))

    if denom <= 1e-6:
        mad = float(np.mean(np.abs(a - b)))
        return inverse_normalize(mad, 0.0, 20.0)

    corr = float(np.sum(a0 * b0) / denom)
    return clamp((corr + 1.0) / 2.0)


def safe_template_match(crop: np.ndarray, template: np.ndarray) -> float:
    if crop is None or template is None:
        return 0.0
    ch, cw = crop.shape[:2]
    th, tw = template.shape[:2]
    if ch < 6 or cw < 6 or th < 6 or tw < 6:
        return 0.0

    templ = template
    if th > ch or tw > cw:
        scale = min(ch / float(th), cw / float(tw))
        if scale <= 0.2:
            return 0.0
        new_w = max(6, int(round(tw * scale)))
        new_h = max(6, int(round(th * scale)))
        templ = cv2.resize(template, (new_w, new_h), interpolation=cv2.INTER_AREA)

    th2, tw2 = templ.shape[:2]
    if th2 > ch or tw2 > cw:
        return 0.0

    res = cv2.matchTemplate(crop, templ, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val)


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------

def draw_debug_overlay(
    image_bgr: np.ndarray,
    decision: Dict[str, Any],
    scores: Dict[str, float],
) -> np.ndarray:
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    label = "INVALID" if decision["is_invalid"] else "VALID"
    color = (0, 0, 255) if decision["is_invalid"] else (40, 170, 40)

    lines = [
        f"{label} | primary={decision['primary_reason']} | conf={decision['confidence']:.2f}",
        f"invalid_score={decision['invalid_score']:.3f} hard={decision['hard_reject']} borderline={decision['borderline']}",
        f"building={scores['building_score']:.2f} open={scores['open_scene_score']:.2f} corridor={scores['road_corridor_score']:.2f}",
        f"blur={scores['blur_severity']:.2f} glare={scores['glare_score']:.2f} dark={scores['dark_score']:.2f}",
        f"interior={scores['interior_score']:.2f} occlusion={scores['occlusion_score']:.2f} placeholder={scores['placeholder_similarity']:.2f}",
        f"reasons={','.join(decision['reasons'])}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 22
    for line in lines:
        cv2.putText(canvas, line[:110], (10, y), font, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, line[:110], (10, y), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        y += 20
        if y > h - 10:
            break

    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), color, 2)
    return canvas


# ---------------------------------------------------------------------------
# DataFrame sampling  (was repeated 3x in generate_review_samples)
# ---------------------------------------------------------------------------

def sample_from_dataframe(
    df: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    bucket_name: str,
) -> Optional[pd.DataFrame]:
    """Return up to *n* randomly-sampled rows with a ``review_bucket`` column.

    Returns ``None`` when *df* is empty.  Previously this three-line pattern
    was duplicated for invalid, borderline, and per-reason sampling.
    """
    if df.empty:
        return None
    n = min(n, len(df))
    idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[idx].assign(review_bucket=bucket_name)
