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
import logging
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Local imports ──────────────────────────────────────────────────────────────
from agents.orchestrator_agent import OrchestratorAgent
from agents.reporting_agent import ReportingAgent
from analysis.metrics import NormalizedMetrics, compute_core_metrics
from analysis.peer_comparison import build_peer_comparison
from analysis.validation_gate import ValidationGate
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
#
# Terminal statuses:
#   "complete"  — validation clear; full report in result
#   "qualified" — validation passed via manual override; full report in result,
#                 validation_log.overrides records which blocks were overridden
#   "blocked"   — validation failed, no override provided; result contains only
#                 validation_log so the caller can see what failed
#   "error"     — unhandled exception during evaluation; error field has traceback
# Non-terminal:
#   "pending"   — job queued, not yet started
#   "running"   — evaluation in progress
_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=4)
_memory = EvaluationMemory()


# ── Pydantic models ────────────────────────────────────────────────────────────
class EvaluateRequest(BaseModel):
    ticker: str
    force_validation_override: bool = False
    force_justification: Optional[str] = None


class EvaluateResponse(BaseModel):
    job_id: str


# ── Validation log serialiser ─────────────────────────────────────────────────

def _vlog_to_dict(vlog: Any) -> dict:
    """Return a JSON-serialisable dict from a ValidationLog instance."""
    return {
        "ticker":    vlog.ticker,
        "report_date": vlog.report_date,
        "status":    vlog.status,
        "is_clear":  vlog.is_clear,
        "qualified": vlog.qualified,
        "blocks": [
            {
                "block_id":    b.block_id,
                "name":        b.name,
                "passed":      b.passed,
                "failures":    b.failures,
                "corrections": [
                    {
                        "field":   c.field,
                        "old":     c.old_value,
                        "new":     c.new_value,
                        "reason":  c.reason,
                        "rule_id": c.rule_id,
                    }
                    for c in b.corrections
                ],
                "metadata": dict(b.metadata),
            }
            for b in vlog.blocks
        ],
        "overrides": list(vlog.overrides),
        "formatted": vlog.format(),
    }


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
        "position_sizing": dict(sc.position_sizing) if sc.position_sizing else {},
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

    logger.debug(
        f"[STOCK_INFO] price={metrics.price}({metrics.price_source})"
        f" mktcap={metrics.market_cap}({metrics.market_cap_source})"
    )

    # ── Valuation ratios — from NormalizedMetrics ─────────────────────────────
    info["pe_ratio"]   = metrics.pe_ratio
    info["ps_ratio"]   = metrics.ps_ratio
    info["ev_ebitda"]  = metrics.ev_ebitda

    logger.debug(
        f"[STOCK_INFO] pe={metrics.pe_ratio}({metrics.pe_source})"
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

    # ── TTM fundamentals — single stored values that feed all ratio calculations ─
    # Exposing these lets the frontend verify: P/E = price / ttm_eps (to the cent).
    info["ttm_eps"]        = metrics.ttm_eps
    info["ttm_eps_source"] = metrics.ttm_eps_source
    info["ttm_fcf"]        = metrics.ttm_fcf

    # ── Data dates — when the underlying data was sampled ────────────────────
    # price_date: date the /quote price was observed (from FMP timestamp field)
    # fundamentals_date: most recent filing date across income/balance/cashflow
    info["price_date"] = sd.quote_date  # ISO YYYY-MM-DD, or None

    _stmt_dates = [
        d for d in [
            (sd.income_statements[0].date  if sd.income_statements  else None),
            (sd.balance_sheets[0].date     if sd.balance_sheets      else None),
            (sd.cash_flows[0].date         if sd.cash_flows          else None),
        ]
        if d
    ]
    info["fundamentals_date"] = max(_stmt_dates) if _stmt_dates else None

    # ── Shares provenance — SEC EDGAR source for share count ─────────────────
    info["shares_outstanding"]       = sd.shares_outstanding
    info["shares_source"]            = sd.shares_source or None
    info["shares_filing_period_end"] = sd.shares_filing_period_end
    # Real SEC filing date comes from the income statement, not from /shares-float
    # (which only carries FMP's data-refresh timestamp).
    _stmt0 = sd.income_statements[0] if sd.income_statements else None
    info["shares_filing_date"]       = getattr(_stmt0, "filing_date", None)
    info["shares_filing_url"]        = sd.shares_filing_url
    info["shares_data_refreshed_at"] = sd.shares_data_refreshed_at

    # ── Metric provenance (helps frontend show source badges) ────────────────
    info["_sources"] = {
        "price":      metrics.price_source,
        "market_cap": metrics.market_cap_source,
        "pe_ratio":   metrics.pe_source,
        "ps_ratio":   metrics.ps_source,
        "ev_ebitda":  metrics.ev_ebitda_source,
        "ttm_eps":    metrics.ttm_eps_source,
        "ttm_fcf":    metrics.ttm_fcf_source,
    }

    logger.debug(
        f"[STOCK_INFO] FINAL: pe={info['pe_ratio']}({metrics.pe_source})"
        f" ps={info['ps_ratio']}({metrics.ps_source})"
        f" ev_ebitda={info['ev_ebitda']}({metrics.ev_ebitda_source})"
        f" mktcap={info['market_cap']}({metrics.market_cap_source})"
        f" eps_growth={info['eps_growth_pct']}%"
        f" peg={info['peg_ratio']}"
    )
    return info


def _extract_valuation_range(state: Any) -> dict:
    """Extract valuation range data from agent findings.

    The ValuationRange dataclass stores scenario assumptions as scenario_bear_pe,
    scenario_base_pe, etc.  The frontend scenario table expects pe_bear_mult,
    pe_base_mult, etc.  These aliases bridge that mismatch so the multiple rows
    in the scenario table render real computed values instead of "—".
    """
    try:
        vr = state.agent_findings.get("fundamental", {}).get("valuation_range")
        if vr is None:
            return {"available": False}
        d = _to_dict(vr)
        d["available"] = True
        # Bridge scenario_* field names → frontend-expected _mult names
        d["pe_bear_mult"] = d.get("scenario_bear_pe")
        d["pe_base_mult"] = d.get("scenario_base_pe")
        d["pe_bull_mult"] = d.get("scenario_bull_pe")
        d["ev_bear_mult"] = d.get("scenario_bear_ev")
        d["ev_base_mult"] = d.get("scenario_base_ev")
        d["ev_bull_mult"] = d.get("scenario_bull_ev")
        d["ps_bear_mult"] = d.get("scenario_bear_ps")
        d["ps_base_mult"] = d.get("scenario_base_ps")
        d["ps_bull_mult"] = d.get("scenario_bull_ps")
        return d
    except Exception:
        return {"available": False}


def _extract_peer_comparison(state: Any, metrics: NormalizedMetrics) -> dict:
    """
    Build peer comparison.

    Target metrics come entirely from NormalizedMetrics (and StockData for
    extended fields) — guaranteeing the peer table's target row is internally
    consistent with all other report sections.
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

        # ── Extended target metrics from NormalizedMetrics + StockData ────────
        # Revenue growth: derive from income statements
        target_revenue_growth: Optional[float] = None
        incs = getattr(sd, "income_statements", [])
        if len(incs) >= 2 and incs[0].revenue and incs[1].revenue and incs[1].revenue > 0:
            target_revenue_growth = round((incs[0].revenue / incs[1].revenue - 1) * 100, 1)

        # Profitability from NormalizedMetrics (populated by compute_core_metrics)
        target_gross_margin      = metrics.gross_margin
        target_operating_margin  = metrics.operating_margin
        target_net_margin        = metrics.net_margin
        target_roe               = metrics.roe

        # ROIC from ratios (not in NormalizedMetrics)
        target_roic: Optional[float] = None
        rats = getattr(sd, "ratios", [])
        if rats and rats[0].roic is not None:
            target_roic = rats[0].roic

        # Financial health from NormalizedMetrics
        target_debt_equity   = metrics.debt_to_equity
        target_current_ratio = metrics.current_ratio

        # Interest coverage from income statement
        # Guard: 0.0 is not a valid interest coverage — it means data is absent,
        # not that the company earns zero times its interest expense.
        target_interest_coverage: Optional[float] = None
        if incs:
            inc0 = incs[0]
            if inc0.operating_income and inc0.interest_expense and inc0.interest_expense > 0:
                _ic = inc0.operating_income / inc0.interest_expense
                target_interest_coverage = round(_ic, 2) if _ic > 0 else None
            elif rats and rats[0].interest_coverage is not None and rats[0].interest_coverage > 0:
                target_interest_coverage = rats[0].interest_coverage

        # Beta from profile
        target_beta = getattr(_profile, "beta", None)

        logger.debug(
            f"[PEER API] target {state.ticker}:"
            f" sector={sector!r} industry={industry!r}"
            f" PE={target_pe}({metrics.pe_source})"
            f" PS={target_ps}({metrics.ps_source})"
            f" EV/EBITDA={target_ev_ebitda}({metrics.ev_ebitda_source})"
            f" growth={target_growth}% rev_g={target_revenue_growth}%"
            f" gm={target_gross_margin} om={target_operating_margin}"
            f" de={target_debt_equity} cr={target_current_ratio}"
            f" beta={target_beta}"
            f" mktcap={target_mkt_cap}({metrics.market_cap_source})"
        )

        pc = build_peer_comparison(
            target_ticker             = state.ticker,
            target_pe                 = target_pe,
            target_ps                 = target_ps,
            target_growth             = target_growth,
            target_peg                = target_peg,
            target_ev_ebitda          = target_ev_ebitda,
            sector                    = sector,
            industry                  = industry,
            target_stock_data         = sd,
            target_mkt_cap            = target_mkt_cap,
            target_company_name       = company,
            target_revenue_growth     = target_revenue_growth,
            target_gross_margin       = target_gross_margin,
            target_operating_margin   = target_operating_margin,
            target_net_margin         = target_net_margin,
            target_roe                = target_roe,
            target_roic               = target_roic,
            target_debt_equity        = target_debt_equity,
            target_current_ratio      = target_current_ratio,
            target_interest_coverage  = target_interest_coverage,
            target_beta               = target_beta,
        )

        # Serialize rows.
        # growth_pct (percentage, e.g. 12.5) → eps_growth (decimal, e.g. 0.125)
        # revenue_growth and ebitda_growth stay as percentages (consistent with growth_pct→eps_growth convention would be confusing; keep as pct for display)
        # Internal scoring fields (prefixed with _) must not be sent to the frontend.
        _PEER_INTERNAL_FIELDS = {"_industry", "_sector", "_mkt_cap", "_source", "_total_score"}

        rows: list[dict] = []
        for r in pc.rows:
            d = _to_dict(r)
            for k in list(d.keys()):
                if k in _PEER_INTERNAL_FIELDS:
                    del d[k]
            gp = d.pop("growth_pct", None)
            d["eps_growth"] = (gp / 100.0) if gp is not None else None
            rows.append(d)

        logger.debug(
            f"[PEER API] serialized {len(rows)} rows:"
            + "".join(
                f"\n    {r['ticker']}: ev_ebitda={r.get('ev_ebitda')} eps_growth={r.get('eps_growth')}"
                f" rev_g={r.get('revenue_growth')} gm={r.get('gross_margin')}"
                f" de={r.get('debt_equity')} beta={r.get('beta')}"
                for r in rows
            )
        )

        peer_trend_insights = _compute_peer_trend_insights(state.ticker, rows)
        return {
            "has_peers":           pc.has_peers,
            "rows":                rows,
            "insights":            list(pc.insights),
            "peer_level":          pc.peer_level,
            "section_label":       pc.section_label,
            "proxy_note":          pc.proxy_note,
            "peer_trend_insights": peer_trend_insights,
        }
    except Exception as exc:
        logger.error("[PEER API ERROR] %s", exc, exc_info=True)
        return {
            "has_peers":     False,
            "rows":          [],
            "insights":      [],
            "peer_level":    1,
            "section_label": "Peer Comparison",
            "proxy_note":    "",
        }


def _extract_macro(state: Any) -> dict:
    """Extract macro data from agent findings.

    Passes the full MacroLEIAgent payload through, including Phase 1 LEI fields
    (cycle_phase, lei_trend, yield_spread_trend) so the frontend and any downstream
    consumer sees the same data the ReportingAgent uses to build the memo.
    """
    try:
        macro = state.agent_findings.get("macro") or {}
        if not macro:
            return {"available": False}

        result = {
            "available": True,
            # Core fields (always present)
            "macro_regime":           macro.get("macro_regime", "Unknown"),
            "macro_score":            macro.get("macro_score", 50),
            "recession_risk_level":   macro.get("recession_risk_level", "Unknown"),
            "sector_tilt":            macro.get("sector_tilt", ""),
            "bullish_macro_factors":  list(macro.get("bullish_macro_factors", [])),
            "bearish_macro_factors":  list(macro.get("bearish_macro_factors", [])),
            "data_coverage":          macro.get("data_coverage"),
            # Phase 1 LEI additions — None when FRED unavailable or window too short
            "cycle_phase":            macro.get("cycle_phase"),
            "lei_trend":              macro.get("lei_trend"),
            "yield_spread_trend":     macro.get("yield_spread_trend"),
            # Pre-computed reasoning — rendered as narrative in macro section
            "reasoning_summary":      macro.get("reasoning_summary", ""),
            # Observation dates — lets the frontend flag stale data
            "observation_dates":      macro.get("observation_dates", {}),
        }

        # Assertion: if cycle_phase is absent from the full payload, log it clearly
        # so silent data-flow breaks surface immediately rather than silently rendering.
        if "cycle_phase" not in macro:
            import logging
            logging.getLogger("stock_eval").warning(
                "_extract_macro: cycle_phase missing from macro payload — "
                "MacroLEIAgent payload may predate Phase 1 LEI additions"
            )

        return result
    except Exception:
        return {"available": False}


def _compute_peer_trend_insights(target_ticker: str, rows: list[dict]) -> list[str]:
    """
    Generate peer-relative trend insight bullets from already-serialized historical data.
    Uses only existing data — no new API calls.
    """
    insights: list[str] = []
    try:
        target = next((r for r in rows if r.get("is_target")), None)
        peers  = [r for r in rows if not r.get("is_target")]
        if not target or not peers:
            return insights

        t_hist = target.get("historical") or []
        if not t_hist:
            return insights

        t_curr = t_hist[0]
        tkr = target.get("ticker", target_ticker)

        def _avg(lst: list) -> Optional[float]:
            return sum(lst) / len(lst) if lst else None

        def _peer_curr(key: str) -> Optional[float]:
            vals = []
            for p in peers:
                ph = p.get("historical") or []
                if ph and isinstance(ph[0].get(key), (int, float)):
                    vals.append(ph[0][key])
            return _avg(vals)

        def _trend_delta(hist: list, key: str) -> Optional[float]:
            """Recent value minus oldest value for a metric."""
            vals = [h[key] for h in hist if isinstance(h.get(key), (int, float))]
            return (vals[0] - vals[-1]) if len(vals) >= 3 else None

        def _std(series: list) -> Optional[float]:
            if len(series) < 3:
                return None
            mean = sum(series) / len(series)
            return (sum((v - mean) ** 2 for v in series) / len(series)) ** 0.5

        # 1. Revenue growth — current vs peer average
        t_rev_g = t_curr.get("revenue_growth")
        p_rev_g = _peer_curr("revenue_growth")
        if t_rev_g is not None and p_rev_g is not None:
            diff = t_rev_g - p_rev_g
            if abs(diff) >= 3.0:
                direction = "ahead of" if diff > 0 else "behind"
                insights.append(
                    f"Revenue growing {abs(diff):.0f}pp {direction} peers "
                    f"({t_rev_g:+.1f}% vs {p_rev_g:+.1f}% peer avg)"
                )

        # 2. Operating margin — trend direction vs peers
        t_om_delta = _trend_delta(t_hist, "op_margin")
        p_om_deltas = [d for p in peers if (d := _trend_delta(p.get("historical") or [], "op_margin")) is not None]
        p_om_delta_avg = _avg(p_om_deltas)
        if t_om_delta is not None and p_om_delta_avg is not None:
            gap = t_om_delta - p_om_delta_avg
            if abs(gap) >= 0.02:
                speed = "faster" if gap > 0 else "slower"
                direction = "expanding" if t_om_delta > 0 else "contracting"
                insights.append(
                    f"Margins {direction} {speed} than peers "
                    f"({t_om_delta * 100:+.1f}pp trend vs {p_om_delta_avg * 100:+.1f}pp peer avg)"
                )

        # 3. Operating margin level — structural advantage/disadvantage
        t_om = t_curr.get("op_margin")
        p_om = _peer_curr("op_margin")
        if t_om is not None and p_om is not None:
            diff = t_om - p_om
            if abs(diff) >= 0.03:
                direction = "above" if diff > 0 else "below"
                insights.append(
                    f"Profitability structurally {direction} peers "
                    f"({t_om:.1%} vs {p_om:.1%} avg op margin)"
                )

        # 4. ROE — return quality vs peers
        t_roe = t_curr.get("roe")
        p_roe = _peer_curr("roe")
        if t_roe is not None and p_roe is not None:
            diff = t_roe - p_roe
            if abs(diff) >= 0.05:
                direction = "above" if diff > 0 else "below"
                insights.append(
                    f"Return on equity {abs(diff):.0%} {direction} peers "
                    f"({t_roe:.1%} vs {p_roe:.1%})"
                )

        # 5. Revenue consistency — stability vs peers
        t_rev_series = [h["revenue_growth"] for h in t_hist if isinstance(h.get("revenue_growth"), (int, float))]
        t_std = _std(t_rev_series)
        p_stds = []
        for p in peers:
            ph = p.get("historical") or []
            series = [h["revenue_growth"] for h in ph if isinstance(h.get("revenue_growth"), (int, float))]
            s = _std(series)
            if s is not None:
                p_stds.append(s)
        p_std_avg = _avg(p_stds)
        if t_std is not None and p_std_avg is not None and p_std_avg > 0:
            ratio = t_std / p_std_avg
            if ratio < 0.65:
                insights.append("Revenue growth more consistent than peers — lower earnings volatility risk")
            elif ratio > 1.5:
                insights.append("Revenue growth more volatile than peers — higher execution risk")

    except Exception as _e:
        logger.error("[PEER_TREND_INSIGHTS] %s", _e, exc_info=True)
    return insights[:5]


def _extract_trends(state: Any) -> Optional[dict]:
    """Extract TrendResult from fundamental agent findings for the frontend."""
    try:
        trends = state.agent_findings.get("fundamental", {}).get("trends")
        if trends is None:
            return None
        return _to_dict(trends)
    except Exception:
        return None


def _extract_quant_engine(state: Any) -> Optional[dict]:
    """
    Extract the 8-step QuantEngine result stored by ReportingAgent._build_memo().

    The dict is already plain-serialisable (built by QuantEngine.build() which
    returns primitive types and nested dicts/lists only — no dataclasses).
    Returns None when the quant engine did not run (e.g. val_range absent).
    """
    try:
        qe = state.agent_findings.get("quant_engine")
        if not qe or not qe.get("available"):
            return None
        return qe
    except Exception:
        return None


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

        # ── Pre-report validation gate ─────────────────────────────────────
        # Validation gate runs here. By default, BLOCKED status halts report
        # generation and returns status="blocked" with the validation log.
        # Callers who want to proceed despite failures must pass
        # force_validation_override=true AND force_justification to the
        # /api/evaluate endpoint — this produces status="qualified" and
        # records the override in the validation log.
        _vr_obj = _fund_findings.get("valuation_range") or \
            (state.agent_findings.get("fundamental") or {}).get("valuation_range")
        _macro_findings = state.agent_findings.get("macro") or {}
        _as_of = datetime.now(timezone.utc).isoformat()

        _force_override    = _jobs[job_id].get("force_validation_override", False)
        _force_just        = _jobs[job_id].get("force_justification") or ""

        _gate = ValidationGate()
        _snapshot = _gate.build_snapshot(
            ticker          = ticker.upper(),
            as_of_date      = _as_of,
            norm_metrics    = norm_metrics,
            stock_data      = state.stock_data,
            valuation_range = _vr_obj,
            macro_findings  = _macro_findings,
            scorecard       = scorecard,
            memo_text       = memo or "",
        )

        if not _force_override:
            # Default path — block if validation fails.
            _vlog = _gate.run(_snapshot)
            print(_vlog.format())   # audit trail
            if not _vlog.is_clear:
                _update(
                    "blocked", 100,
                    f"Report blocked: {_vlog.status}",
                    result={
                        "ticker":         ticker.upper(),
                        "as_of":          _as_of,
                        "validation_log": _vlog_to_dict(_vlog),
                    },
                )
                return
        else:
            # Override path — probe first, then re-run with overrides applied.
            _vlog_probe = _gate.run(_snapshot)
            _failed_ids = [b.block_id for b in _vlog_probe.blocks if not b.passed]

            if not _failed_ids:
                print("force_validation_override set but no failures to override — "
                      "proceeding as normal complete.")
                _vlog = _vlog_probe
            else:
                _overrides = {
                    bid: {
                        "reason_code":   "MANUAL_OVERRIDE",
                        "justification": _force_just,
                    }
                    for bid in _failed_ids
                }
                _gate_ov = ValidationGate(overrides=_overrides)
                _vlog = _gate_ov.run(_snapshot)
            print(_vlog.format())   # audit trail

        result = {
            "ticker":         ticker.upper(),
            "scorecard":      scorecard_dict,
            "stock_info":     stock_info,
            "valuation_range": valuation_range,
            "peer_comparison": peer_comparison,
            "macro":          macro,
            "memo":           memo,
            "reasoning_log":  list(state.reasoning_log),
            "trends":         _extract_trends(state),
            "quant_engine":   _extract_quant_engine(state),
            "evaluated_at":   _as_of,
            "validation_log": _vlog_to_dict(_vlog),
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

        if _force_override and _failed_ids:
            _update(
                "qualified", 100,
                f"Report ready (qualified: {len(_failed_ids)} block{'s' if len(_failed_ids) != 1 else ''} overridden)",
                result=result,
            )
        else:
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

    if body.force_validation_override and not (body.force_justification or "").strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "force_validation_override=true requires force_justification "
                "(free-text reason explaining why validation failures are acceptable)"
            ),
        )

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "step": "Queued…",
        "result": None,
        "error": None,
        "force_validation_override": body.force_validation_override,
        "force_justification":       (body.force_justification or "").strip(),
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
