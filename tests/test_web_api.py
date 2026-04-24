"""
tests/test_web_api.py — FastAPI contract tests for /api/evaluate.

Tests cover the request-to-job-dict contract: input validation, job
creation, and persistence of override flags.  They do NOT wait for
background thread completion (which requires live data and takes 30+
seconds per ticker).
"""
import sys
import os

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
