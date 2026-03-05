"""
app.py — ReefSense AHP Scoring API
FastAPI service deployed on Hugging Face Spaces.

Endpoints:
  GET  /health                — service health check
  POST /calculate-area        — calculate nursery floor area from dimensions
  POST /score                 — score and rank candidate points (AHP)
  POST /recalculate-all       — recalculate scores for all points after a change
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import scoring

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ReefSense AHP Scoring API",
    description="AHP-based nursery placement scoring for the ReefSense DSS.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class CalculateAreaRequest(BaseModel):
    nursery_type: str
    width_m:  Optional[float] = None
    length_m: Optional[float] = None
    radius_m: Optional[float] = None


class ScoreRequest(BaseModel):
    points:        List[Dict[str, Any]]
    required_area: Optional[float] = None
    limit:         int = 10


class RecalculateRequest(BaseModel):
    points: List[Dict[str, Any]]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "service":   "reefsense-scoring",
        "weights": {
            "standard":  scoring.W_STANDARD,
            "dimension": scoring.W_DIMENSION,
        },
    }


@app.post("/calculate-area")
def calculate_area(req: CalculateAreaRequest):
    """
    Calculate the required floor-plan area (m²) for a nursery.
    - table   → width_m × length_m
    - others  → π × radius_m²
    """
    try:
        area = scoring.calculate_required_area(
            nursery_type=req.nursery_type,
            width_m=req.width_m,
            length_m=req.length_m,
            radius_m=req.radius_m,
        )
        return {"required_area_m2": round(area, 4)}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/score")
def score(req: ScoreRequest):
    """
    Score and rank candidate points using AHP weights.
    Switches to dimension-aware weights (space = 40%) when required_area is set.
    """
    limit  = max(1, min(300, req.limit))
    result = scoring.score_points(req.points, req.required_area, limit)
    return {
        "count":  len(result),
        "limit":  limit,
        "points": result,
    }


@app.post("/recalculate-all")
def recalculate_all(req: RecalculateRequest):
    """
    Recalculate suitability scores for all candidate points using standard weights.
    Called by the Node.js backend after a new nursery is placed.
    Returns a list of { fid, suitability_score } for batch DB update.
    """
    results = scoring.recalculate_all_scores(req.points)
    return {
        "count":  len(results),
        "scores": results,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
