"""
tests/test_web_api.py — FastAPI contract tests for /api/evaluate.

Tests cover the request-to-job-dict contract: input validation, job
creation, and persistence of override flags.  They do NOT wait for
background thread completion (which requires live data and takes 30+
seconds per ticker).
"""
import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

import web_api
from web_api import app, _jobs

client = TestClient(app)


def _post_evaluate(payload: dict):
    return client.post("/api/evaluate", json=payload)


# ── 1. override=true without justification → 400 ─────────────────────────────

def test_evaluate_missing_justification_returns_400():
    """force_validation_override=true with no justification must be rejected
    synchronously — before a background job is spawned."""
    jobs_before = set(_jobs.keys())

    resp = _post_evaluate({"ticker": "NFLX", "force_validation_override": True})

    assert resp.status_code == 400
    assert "force_justification" in resp.json()["detail"]

    # No job must have been created
    assert set(_jobs.keys()) == jobs_before


# ── 2. normal request creates a pending/running job ───────────────────────────

def test_evaluate_without_override_creates_job():
    """A standard POST creates a job and returns its id."""
    resp = _post_evaluate({"ticker": "NFLX"})

    assert resp.status_code == 200
    job_id = resp.json().get("job_id")
    assert job_id, "Response must include a non-empty job_id"

    assert job_id in _jobs
    assert _jobs[job_id]["status"] in ("pending", "running")


# ── 3. override flags are persisted into _jobs ────────────────────────────────

def test_evaluate_with_override_persists_flags_in_job():
    """Override flags must be stored on the job dict so the background
    thread can read them without re-parsing the request."""
    resp = _post_evaluate({
        "ticker": "NFLX",
        "force_validation_override": True,
        "force_justification": "test",
    })

    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    assert _jobs[job_id]["force_validation_override"] is True
    assert _jobs[job_id]["force_justification"] == "test"


# ── 4. force=false with justification present — flags stored, force is False ──

def test_evaluate_override_without_force_ignores_justification():
    """force_validation_override=false is valid even when justification is
    present.  The flag must be stored as False; the justification is stored
    but inert."""
    resp = _post_evaluate({
        "ticker": "NFLX",
        "force_validation_override": False,
        "force_justification": "whatever",
    })

    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    assert _jobs[job_id]["force_validation_override"] is False


# ── /api/analyze/{ticker} contract tests ─────────────────────────────────────

_SYNTHETIC_SNAPSHOT = {
    "ticker": "AXON",
    "as_of_date": "2026-04-29T15:00:00Z",
    "kind": "equity",
    "price": {
        "value":   395.95,
        "source":  "FMP",
        "vintage": "2026-04-29T14:55:00Z",
    },
    "distribution": {
        "mode": "monte_carlo",
        "percentiles": {
            "p5":  304.96,
            "p25": 350.66,
            "p50": 403.84,
            "p75": 482.99,
            "p95": 545.34,
        },
    },
    "execution": {
        "target_size_pct": 1.5,
        "conviction": "Low",
    },
}


def _make_complete_job(ticker: str, snapshot: dict) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "complete",
        "progress": 100,
        "step": "Report ready",
        "result": {"snapshot": snapshot},
        "error": None,
        "force_validation_override": False,
        "force_justification": "",
    }
    return job_id


def test_analyze_endpoint_400_on_invalid_ticker():
    resp = client.get("/api/analyze/123/?job_id=anything")
    assert resp.status_code in (400, 404)


def test_analyze_endpoint_404_on_missing_job():
    resp = client.get("/api/analyze/PYPL?job_id=nonexistent-job-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_analyze_endpoint_409_on_incomplete_job():
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending", "progress": 0, "step": "Queued",
        "result": None, "error": None,
        "force_validation_override": False, "force_justification": "",
    }
    resp = client.get(f"/api/analyze/PYPL?job_id={job_id}")
    assert resp.status_code == 409


def test_analyze_endpoint_returns_valid_analysis_for_complete_job():
    job_id = _make_complete_job("AXON", _SYNTHETIC_SNAPSHOT)
    resp = client.get(f"/api/analyze/AXON?job_id={job_id}")
    assert resp.status_code == 200, resp.text
    payload = resp.json()

    assert payload["ticker"] == "AXON"
    assert payload["job_id"] == job_id
    assert "analysis" in payload
    assert payload["analysis"]["ticker"] == "AXON"

    vs = payload["validation_summary"]
    assert vs["schema_valid"] is True, f"Schema errors: {vs['schema_errors']}"
    assert vs["n1"]["passed"]
    assert vs["f1"]["passed"]
    assert vs["s1"]["passed"]
    assert vs["s1"]["unsourced_count"] == 0
