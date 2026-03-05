"""
scoring.py — AHP nursery placement scoring logic.
Direct Python port of the original scoring.js used in coral-gis-api.

Two weight modes:
  Standard  — default weights, no dimension filter.
  Dimension — when nursery dimensions are supplied; boosts space weight to 40%.
"""

import math
from typing import Any, Dict, List, Optional

# ── AHP weights (each set sums to 1.0) ────────────────────────────────────────

W_STANDARD: Dict[str, float] = {
    "current": 0.35,   # light score  (depth-band derived)
    "depth":   0.25,   # temp score   (depth-band derived)
    "space":   0.20,   # space_area_m2 — higher is better
    "spacing": 0.12,   # dist_nursery_m — farther from existing nurseries is better
    "access":  0.08,   # dist_shore_m   — closer to shore is better
}

W_DIMENSION: Dict[str, float] = {
    "current": 0.25,
    "depth":   0.20,
    "space":   0.40,   # primary factor when nursery dimensions are given
    "spacing": 0.10,
    "access":  0.05,
}

# Depth-band scores derived from real sensor data
DEPTH_BAND_SCORES: Dict[str, Dict[str, float]] = {
    "0-3m":  {"light": 1.0,   "temp": 1.0  },
    "3-7m":  {"light": 0.316, "temp": 0.448},
    "7-10m": {"light": 0.0,   "temp": 0.0  },
}

MIN_SPACING_M = 2  # nurseries must be at least 2 m apart


# ── Normalisation helpers ──────────────────────────────────────────────────────

def norm_higher(values: List[float]) -> List[float]:
    """Normalise so the highest value → 1.0, lowest → 0.0."""
    mn, mx = min(values), max(values)
    if mx == mn:
        return [1.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def norm_lower(values: List[float]) -> List[float]:
    """Normalise so the lowest value → 1.0, highest → 0.0."""
    mn, mx = min(values), max(values)
    if mx == mn:
        return [1.0] * len(values)
    return [(mx - v) / (mx - mn) for v in values]


# ── Dimension helpers ──────────────────────────────────────────────────────────

def calculate_required_area(
    nursery_type: str,
    width_m:  Optional[float] = None,
    length_m: Optional[float] = None,
    radius_m: Optional[float] = None,
) -> float:
    """
    Return the required floor-plan area (m²) for the given nursery type/size.
      table  → width × length
      others → π × radius²
    """
    if nursery_type == "table":
        if width_m is None or length_m is None:
            raise ValueError("Table nursery requires 'width_m' and 'length_m'.")
        if width_m <= 0 or length_m <= 0:
            raise ValueError("'width_m' and 'length_m' must be positive.")
        return width_m * length_m
    else:
        if radius_m is None:
            raise ValueError(f"'{nursery_type}' nursery requires 'radius_m'.")
        if radius_m <= 0:
            raise ValueError("'radius_m' must be positive.")
        return math.pi * radius_m ** 2


# ── Core scoring ───────────────────────────────────────────────────────────────

def score_points(
    points: List[Dict[str, Any]],
    required_area: Optional[float] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Score and rank candidate points using AHP weights.

    Args:
        points:        List of candidate point dicts (from candidate_points table).
        required_area: When set — hard-filters points where space_area_m2 < required_area
                       and switches to dimension-aware weights (space = 40%).
        limit:         Number of top results to return.

    Returns:
        Top `limit` points sorted by suitability_score descending,
        each with a 'suitability_score' field added.
    """
    # 1. Keep only available points
    pts = [p for p in points if p.get("is_available", True)]

    # 2. Hard space filter when nursery dimensions are supplied
    if required_area is not None:
        pts = [p for p in pts if float(p.get("space_area_m2") or 0) >= required_area]

    if not pts:
        return []

    w = W_DIMENSION if required_area is not None else W_STANDARD

    # 3. Extract raw values for normalisation
    nursery_vals = [float(p.get("dist_nursery_m") or 0) for p in pts]
    space_vals   = [float(p.get("space_area_m2")  or 0) for p in pts]
    shore_vals   = [float(p.get("dist_shore_m")   or 0) for p in pts]

    score_spacing = norm_higher(nursery_vals)
    score_space   = norm_higher(space_vals)
    score_access  = norm_lower(shore_vals)

    # 4. Compute AHP suitability score for each point
    scored = []
    for i, p in enumerate(pts):
        band = p.get("depth_band") or "0-3m"
        bs   = DEPTH_BAND_SCORES.get(band, DEPTH_BAND_SCORES["0-3m"])

        suit = (
            w["current"] * bs["light"]        +
            w["depth"]   * bs["temp"]         +
            w["space"]   * score_space[i]     +
            w["spacing"] * score_spacing[i]   +
            w["access"]  * score_access[i]
        )

        p_copy = dict(p)
        p_copy["suitability_score"] = round(suit, 4)
        scored.append(p_copy)

    # 5. Sort descending, return top N
    scored.sort(key=lambda x: x["suitability_score"], reverse=True)
    return scored[:limit]


def recalculate_all_scores(
    points: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Recalculate suitability_score for all candidate points using standard weights.
    Called after a new nursery is added.

    Args:
        points: All candidate_points rows with dist_nursery_m, space_area_m2,
                dist_shore_m, depth_band, fid fields.

    Returns:
        List of { fid, suitability_score } dicts for batch DB update.
    """
    valid = [
        p for p in points
        if p.get("dist_nursery_m") is not None
        and p.get("space_area_m2")  is not None
        and p.get("dist_shore_m")   is not None
    ]

    if not valid:
        return []

    w = W_STANDARD

    nursery_vals = [float(p["dist_nursery_m"]) for p in valid]
    space_vals   = [float(p["space_area_m2"])  for p in valid]
    shore_vals   = [float(p["dist_shore_m"])   for p in valid]

    score_spacing = norm_higher(nursery_vals)
    score_space   = norm_higher(space_vals)
    score_access  = norm_lower(shore_vals)

    results = []
    for i, p in enumerate(valid):
        band = p.get("depth_band") or "0-3m"
        bs   = DEPTH_BAND_SCORES.get(band, DEPTH_BAND_SCORES["0-3m"])

        suit = (
            w["current"] * bs["light"]        +
            w["depth"]   * bs["temp"]         +
            w["space"]   * score_space[i]     +
            w["spacing"] * score_spacing[i]   +
            w["access"]  * score_access[i]
        )

        results.append({
            "fid":               p["fid"],
            "suitability_score": round(suit, 4),
        })

    return results
