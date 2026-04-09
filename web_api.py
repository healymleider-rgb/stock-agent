"""
web_api.py — FastAPI server for the StockEval platform.

Endpoints:
  POST /api/evaluate          → {job_id}   (starts background evaluation)
  GET  /api/jobs/{job_id}     → JobStatus  (poll for result)
  GET  /api/history/{ticker}  → list of past evaluations

Run with:
  uvicorn web_api:app --reload --port 8000
"""
from __future__ import annotations

import dataclasses
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Local imports ──────────────────────────────────────────────────────────────
from agents.orchestrator_agent import OrchestratorAgent
from agents.reporting_agent import ReportingAgent
from analysis.metrics import NormalizedMetrics, compute_core_metrics
from analysis.peer_comparison import build_peer_comparison
from memory.evaluation_memory import EvaluationMemory
from models.scorecard import Scorecard

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(title="StockEval API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory job store ────────────────────────────────────────────────────────
# {job_id: {"status": str, "progress": int, "step": str, "result": dict|None, "error": str|None}}
_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=4)
_memory = EvaluationMemory()


# ── Pydantic models ────────────────────────────────────────────────────────────
class EvaluateRequest(BaseModel):
    ticker: str


class EvaluateResponse(BaseModel):
    job_id: str


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses / enums / nested objects to plain dicts."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(i) for i in obj]
    return obj


def _serialize_scorecard(sc: Scorecard) -> dict:
    """Serialise a Scorecard dataclass to a plain dict."""
    categories = {}
    for cat_name in ("valuation", "growth", "profitability", "financial_health", "momentum", "risk"):
        cat = getattr(sc, cat_name, None)
        if cat is not None:
            categories[cat_name] = _to_dict(cat)
        else:
            categories[cat_name] = None

    return {
        "ticker": sc.ticker,
        "overall_score": round(sc.overall_score, 1),
        "stance": sc.stance.value if hasattr(sc.stance, "value") else str(sc.stance),
        "confidence": round(sc.confidence, 3),
        "categories": categories,
        "bullish_factors": list(sc.bullish_factors),
        "bearish_factors": list(sc.bearish_factors),
        "key_drivers": list(sc.key_drivers),
        "what_would_change_view": list(sc.what_would_change_view),
        "risk_flags": list(sc.risk_flags),
    }


def _extract_stock_info(state: Any, metrics: NormalizedMetrics) -> dict:
    """
    Build the stock_info dict for the API response.

    All valuation metrics (PE, PS, EV/EBITDA, market cap, price) come from
    the pre-computed NormalizedMetrics so they are guaranteed to match the
    peer comparison table and scenario analysis.
    """
    sd   = state.stock_data
    info: dict[str, Any] = {}

    # ── Profile fields ─────────────────────────────────────────────────────────
    try:
        p = sd.profile
        info["company_name"] = getattr(p, "company_name", None) or state.ticker
        _s = getattr(p, "sector",   "") or ""
        _i = getattr(p, "industry", "") or ""
        info["sector"]       = "" if _s.lower() in ("unknown", "n/a") else _s
        info["industry"]     = "" if _i.lower() in ("unknown", "n/a") else _i
        info["description"]  = getattr(p, "description", "") or ""
        info["beta"]         = getattr(p, "beta", None)
    except Exception:
        info.setdefault("company_name", state.ticker)
        info.setdefault("sector",       "")
        info.setdefault("industry",     "")
        info.setdefault("description",  "")
        info.setdefault("beta",         None)

    # ── Price and market cap — from NormalizedMetrics ─────────────────────────
    info["current_price"] = metrics.price
    info["market_cap"]    = metrics.market_cap

    print(
        f"  [STOCK_INFO] price={metrics.price}({metrics.price_source})"
        f" mktcap={metrics.market_cap}({metrics.market_cap_source})"
    )

    # ── Valuation ratios — from NormalizedMetrics ─────────────────────────────
    info["pe_ratio"]   = metrics.pe_ratio
    info["ps_ratio"]   = metrics.ps_ratio
    info["ev_ebitda"]  = metrics.ev_ebitda

    print(
        f"  [STOCK_INFO] pe={metrics.pe_ratio}({metrics.pe_source})"
        f" ps={metrics.ps_ratio}({metrics.ps_source})"
        f" ev_ebitda={metrics.ev_ebitda}({metrics.ev_ebitda_source})"
    )

    # ── Growth & PEG — from NormalizedMetrics ────────────────────────────────
    # These must be in stock_info so the frontend can display them in the
    # metrics header and the peer table target row uses the same values.
    info["eps_growth_pct"] = metrics.eps_growth_pct   # annualized %, e.g. 12.5
    info["peg_ratio"]      = metrics.peg

    # ── Other ratios — from NormalizedMetrics (provider-supplied + derived) ──
    info["pb_ratio"]      = metrics.pb_ratio
    info["dividend_yield"]= metrics.dividend_yield
    info["roe"]           = metrics.roe
    info["roa"]           = metrics.roa
    info["gross_margin"]  = metrics.gross_margin
    info["net_margin"]    = metrics.net_margin
    info["operating_margin"] = metrics.operating_margin
    info["debt_to_equity"]= metrics.debt_to_equity
    info["current_ratio"] = metrics.current_ratio

    # ── Metric provenance (helps frontend show source badges) ────────────────
    info["_sources"] = {
        "price":      metrics.price_source,
        "market_cap": metrics.market_cap_source,
        "pe_ratio":   metrics.pe_source,
        "ps_ratio":   metrics.ps_source,
        "ev_ebitda":  metrics.ev_ebitda_source,
    }

    print(
        f"  [STOCK_INFO] FINAL: pe={info['pe_ratio']}({metrics.pe_source})"
        f" ps={info['ps_ratio']}({metrics.ps_source})"
        f" ev_ebitda={info['ev_ebitda']}({metrics.ev_ebitda_source})"
        f" mktcap={info['market_cap']}({metrics.market_cap_source})"
        f" eps_growth={info['eps_growth_pct']}%"
        f" peg={info['peg_ratio']}"
    )
    return info


def _extract_valuation_range(state: Any) -> dict:
    """Extract valuation range data from agent findings."""
    try:
        vr = state.agent_findings.get("fundamental", {}).get("valuation_range")
        if vr is None:
            return {"available": False}
        d = _to_dict(vr)
        d["available"] = True
        return d
    except Exception:
        return {"available": False}


def _extract_peer_comparison(state: Any, metrics: NormalizedMetrics) -> dict:
    """
    Build peer comparison.

    Target metrics come entirely from NormalizedMetrics — the same validated
    values shown in the report header. This guarantees the peer table's
    target row is internally consistent with all other report sections.
    """
    try:
        sd       = state.stock_data
        _profile = getattr(sd, "profile", None)
        sector   = getattr(_profile, "sector",       "") or ""
        industry = getattr(_profile, "industry",     "") or ""
        company  = getattr(_profile, "company_name", None)

        target_pe        = metrics.pe_ratio
        target_ps        = metrics.ps_ratio
        target_ev_ebitda = metrics.ev_ebitda
        target_growth    = metrics.eps_growth_pct
        target_peg       = metrics.peg
        target_mkt_cap   = metrics.market_cap

        print(
            f"  [PEER API] target {state.ticker}:"
            f" sector={sector!r} industry={industry!r}"
            f" PE={target_pe}({metrics.pe_source})"
            f" PS={target_ps}({metrics.ps_source})"
            f" EV/EBITDA={target_ev_ebitda}({metrics.ev_ebitda_source})"
            f" growth={target_growth}%"
            f" PEG={target_peg}"
            f" mktcap={target_mkt_cap}({metrics.market_cap_source})"
        )

        pc = build_peer_comparison(
            target_ticker       = state.ticker,
            target_pe           = target_pe,
            target_ps           = target_ps,
            target_growth       = target_growth,
            target_peg          = target_peg,
            target_ev_ebitda    = target_ev_ebitda,
            sector              = sector,
            industry            = industry,
            target_mkt_cap      = target_mkt_cap,
            target_company_name = company,
        )

        # Serialize rows:
        # - growth_pct (percentage, e.g. 12.5) → eps_growth (decimal, e.g. 0.125)
        #   because the frontend multiplies by 100 for display
        # - company_name, ev_ebitda are now native PeerRow fields
        # Internal scoring fields (prefixed with _) must not be sent to the frontend
        _PEER_INTERNAL_FIELDS = {"_industry", "_sector", "_mkt_cap", "_source", "_total_score"}

        rows: list[dict] = []
        for r in pc.rows:
            d = _to_dict(r)
            # Strip internal scoring metadata
            for k in list(d.keys()):
                if k in _PEER_INTERNAL_FIELDS:
                    del d[k]
            gp = d.pop("growth_pct", None)
            d["eps_growth"] = (gp / 100.0) if gp is not None else None
            rows.append(d)

        print(
            f"  [PEER API] serialized {len(rows)} rows:"
            + "".join(
                f"\n    {r['ticker']}: ev_ebitda={r.get('ev_ebitda')} eps_growth={r.get('eps_growth')}"
                for r in rows
            )
        )

        return {
            "has_peers": pc.has_peers,
            "rows":      rows,
            "insights":  list(pc.insights),
        }
    except Exception as exc:
        import traceback as _tb
        print(f"  [PEER API ERROR] {exc}\n{_tb.format_exc()}")
        return {"has_peers": False, "rows": [], "insights": []}


def _extract_macro(state: Any) -> dict:
    """Extract macro data from agent findings."""
    try:
        macro = state.agent_findings.get("macro") or {}
        if not macro:
            return {"available": False}
        return {
            "available": True,
            "macro_regime": macro.get("macro_regime", "Unknown"),
            "macro_score": macro.get("macro_score", 50),
            "recession_risk_level": macro.get("recession_risk_level", "Unknown"),
            "sector_tilt": macro.get("sector_tilt", ""),
            "bullish_macro_factors": list(macro.get("bullish_macro_factors", [])),
            "bearish_macro_factors": list(macro.get("bearish_macro_factors", [])),
        }
    except Exception:
        return {"available": False}


def _extract_reporting_extras(sc: Scorecard) -> dict:
    """Extract stock type label and key tension from ReportingAgent static methods."""
    extras: dict[str, Any] = {}
    try:
        st = ReportingAgent._classify_stock_type(sc)
        if st:
            extras["stock_type_label"] = st[0]
            extras["stock_type_desc"] = st[1]
        else:
            extras["stock_type_label"] = None
            extras["stock_type_desc"] = None
    except Exception:
        extras["stock_type_label"] = None
        extras["stock_type_desc"] = None

    try:
        kt = ReportingAgent._write_key_tension(sc)
        extras["key_tension"] = kt
    except Exception:
        extras["key_tension"] = None

    return extras


# ── Background evaluation task ─────────────────────────────────────────────────

def _run_evaluation(job_id: str, ticker: str) -> None:
    """Run evaluation in a background thread, updating _jobs[job_id] as we go."""
    def _update(status: str, progress: int, step: str,
                 result: Optional[dict] = None, error: Optional[str] = None) -> None:
        _jobs[job_id].update({
            "status": status,
            "progress": progress,
            "step": step,
            "result": result,
            "error": error,
        })

    try:
        _update("running", 5, "Initialising evaluation engine…")

        orchestrator = OrchestratorAgent()

        _update("running", 15, "Fetching company profile and financial data…")

        # The orchestrator logs progress internally; we do a single evaluate call.
        # We update pseudo-progress before and after the long call.
        _update("running", 25, "Running fundamental, technical, and macro analysis…")

        scorecard, memo, state = orchestrator.evaluate(ticker)

        _update("running", 80, "Compiling investment report…")

        # Compute validated metrics once — reuse in all report sections so
        # price, market cap, PE, PS, EV/EBITDA are identical everywhere.
        # Try to reuse the NormalizedMetrics computed inside FundamentalAnalysisAgent
        # (which had quarterly data available); if not present, compute fresh.
        _fund_findings = state.agent_findings.get("fundamental", {})
        norm_metrics: NormalizedMetrics = _fund_findings.get("normalized_metrics") or \
            compute_core_metrics(state.stock_data)

        # Build result payload
        scorecard_dict  = _serialize_scorecard(scorecard)
        stock_info      = _extract_stock_info(state, norm_metrics)
        valuation_range = _extract_valuation_range(state)
        peer_comparison = _extract_peer_comparison(state, norm_metrics)
        macro = _extract_macro(state)
        reporting_extras = _extract_reporting_extras(scorecard)

        result = {
            "ticker": ticker.upper(),
            "scorecard": scorecard_dict,
            "stock_info": stock_info,
            "valuation_range": valuation_range,
            "peer_comparison": peer_comparison,
            "macro": macro,
            "memo": memo,
            "reasoning_log": list(state.reasoning_log),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            **reporting_extras,
        }

        # Persist to memory
        try:
            _memory.record_evaluation(ticker, {
                "overall_score": scorecard.overall_score,
                "stance": scorecard.stance.value if hasattr(scorecard.stance, "value") else str(scorecard.stance),
                "confidence": scorecard.confidence,
            })
        except Exception:
            pass

        _update("complete", 100, "Report ready.", result=result)

    except Exception as exc:
        tb = traceback.format_exc()
        _update("error", 0, "Evaluation failed.", error=f"{exc}\n\n{tb}")


# ── API routes ─────────────────────────────────────────────────────────────────

@app.post("/api/evaluate", response_model=EvaluateResponse)
def start_evaluation(body: EvaluateRequest) -> EvaluateResponse:
    """Start a background evaluation. Returns a job_id immediately."""
    ticker = body.ticker.strip().upper()
    if not ticker or not ticker.isalpha() or len(ticker) > 10:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol.")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "step": "Queued…",
        "result": None,
        "error": None,
    }
    _executor.submit(_run_evaluation, job_id, ticker)
    return EvaluateResponse(job_id=job_id)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Poll the status of a background evaluation job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _jobs[job_id]


@app.get("/api/history/{ticker}")
def get_history(ticker: str) -> list:
    """Return past evaluations for a ticker from EvaluationMemory."""
    ticker = ticker.strip().upper()
    try:
        return _memory.get_history(ticker, limit=20)
    except Exception:
        return []


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
