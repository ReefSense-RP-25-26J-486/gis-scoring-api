"""
app.py — ReefSense AHP Scoring API
FastAPI service deployed on Hugging Face Spaces.

Endpoints:
  GET  /health                      — service health check
  POST /calculate-area              — calculate nursery floor area from dimensions
  POST /score                       — score and rank candidate points (AHP)
  POST /recalculate-all             — recalculate scores for all points after a change

  GET  /temperature/records         — all merged depth-band temperature records
  GET  /temperature/records/{id}    — single temperature record by ID
  GET  /temperature/stats           — date range, row count, available depth bands

Sensor CSV data is fetched at startup from the HuggingFace Dataset:
  https://huggingface.co/datasets/senithudara/reefsense-coral-sensor-data
"""

import csv
import io
import os
import urllib.request
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import scoring

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ReefSense AHP Scoring API",
    description="AHP-based nursery placement scoring + sensor temperature data for the ReefSense DSS.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Load CSV temperature data from HF Dataset on startup ──────────────────────

HF_DATASET_BASE = (
    "https://huggingface.co/datasets/senithudara/reefsense-coral-sensor-data"
    "/resolve/main"
)

CSV_URLS = {
    "0_3m":  f"{HF_DATASET_BASE}/coral_0_3m.csv",
    "3_7m":  f"{HF_DATASET_BASE}/coral_3_7m.csv",
    "7_10m": f"{HF_DATASET_BASE}/coral_7_10m.csv",
}


def _fetch_csv(url: str) -> List[Dict[str, str]]:
    """Download a CSV from a URL and return list of row dicts."""
    with urllib.request.urlopen(url, timeout=30) as resp:
        content = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    return list(reader)


def _f(val: str) -> Optional[float]:
    """Parse float or return None."""
    try:
        return float(val) if val and val.strip() else None
    except ValueError:
        return None


def _build_temperature_records() -> List[Dict[str, Any]]:
    """
    Fetch the three depth-band CSV files from the HF Dataset and merge by Date+Time.
    Returns a list of records each containing readings from all three bands.
    Shared environmental columns (air_temp, wind_speed, salinity, light_lux)
    are taken from the 0-3m surface file.
    """
    print("[Startup] Downloading sensor CSV files from HF Dataset...")
    rows_0_3  = _fetch_csv(CSV_URLS["0_3m"])
    rows_3_7  = _fetch_csv(CSV_URLS["3_7m"])
    rows_7_10 = _fetch_csv(CSV_URLS["7_10m"])
    print(f"[Startup] Downloaded: {len(rows_0_3)} / {len(rows_3_7)} / {len(rows_7_10)} rows")

    # Build lookup maps for 3-7m and 7-10m by Date|Time key
    map_3_7  = {f"{r['Date']}|{r['Time']}": r for r in rows_3_7}
    map_7_10 = {f"{r['Date']}|{r['Time']}": r for r in rows_7_10}

    records = []
    for idx, r0 in enumerate(rows_0_3):
        key  = f"{r0['Date']}|{r0['Time']}"
        r3   = map_3_7.get(key, {})
        r10  = map_7_10.get(key, {})

        records.append({
            "id":         idx + 1,
            "date":       r0.get("Date", ""),
            "time":       r0.get("Time", ""),
            "latitude":   _f(r0.get("latitude", "")),
            "longitude":  _f(r0.get("longitude", "")),
            # Water temperatures per depth band
            "temp3m":     _f(r0.get("water_temp", "")),
            "temp7m":     _f(r3.get("water_temp", "")),
            "temp10m":    _f(r10.get("water_temp", "")),
            # Surface environmental readings (from 0-3m file)
            "light_lux":  _f(r0.get("light_lux", "")),
            "air_temp":   _f(r0.get("air_temp", "")),
            "wind_speed": _f(r0.get("wind_speed", "")),
            "salinity":   _f(r0.get("salinity", "")),
        })

    return records


# Load once at startup — stays in memory for the lifetime of the process
try:
    TEMPERATURE_RECORDS: List[Dict[str, Any]] = _build_temperature_records()
    print(f"[Startup] Loaded {len(TEMPERATURE_RECORDS)} merged temperature records.")
except Exception as _e:
    TEMPERATURE_RECORDS = []
    print(f"[Startup] WARNING — could not load temperature CSVs: {_e}")


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


# ── AHP Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "service":   "reefsense-scoring",
        "temperature_records_loaded": len(TEMPERATURE_RECORDS),
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


# ── Temperature Data Routes ───────────────────────────────────────────────────

@app.get("/temperature/records")
def get_temperature_records(
    date:  Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    page:  int           = Query(1,    ge=1,   description="Page number (1-based)"),
    limit: int           = Query(100,  ge=1, le=1500, description="Records per page"),
):
    """
    Return merged depth-band temperature records from the three CSV files.
    All 1463 readings are available (6-hourly, Jan–Dec 2024).

    Each record includes:
      temp3m, temp7m, temp10m — water temperatures at each depth band
      light_lux, air_temp, wind_speed, salinity — surface environmental readings

    Supports optional date filtering and pagination.
    """
    data = TEMPERATURE_RECORDS

    # Optional date filter
    if date:
        data = [r for r in data if r["date"] == date]

    total = len(data)

    # Pagination
    start = (page - 1) * limit
    paged = data[start: start + limit]

    return {
        "total":   total,
        "page":    page,
        "limit":   limit,
        "count":   len(paged),
        "records": paged,
    }


@app.get("/temperature/records/{record_id}")
def get_temperature_record(record_id: int):
    """Return a single temperature record by its 1-based ID."""
    if record_id < 1 or record_id > len(TEMPERATURE_RECORDS):
        raise HTTPException(status_code=404, detail="Record not found")
    return TEMPERATURE_RECORDS[record_id - 1]


@app.get("/temperature/stats")
def get_temperature_stats():
    """Return summary statistics for the loaded temperature dataset."""
    if not TEMPERATURE_RECORDS:
        return {"total": 0, "date_range": None}

    dates = [r["date"] for r in TEMPERATURE_RECORDS if r["date"]]
    return {
        "total":       len(TEMPERATURE_RECORDS),
        "date_from":   min(dates) if dates else None,
        "date_to":     max(dates) if dates else None,
        "depth_bands": ["0-3m", "3-7m", "7-10m"],
        "columns":     ["temp3m", "temp7m", "temp10m",
                        "light_lux", "air_temp", "wind_speed", "salinity"],
        "dataset_url": "https://huggingface.co/datasets/senithudara/reefsense-coral-sensor-data",
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
