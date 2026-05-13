"""
ReportingAgent

Compiles all agent findings into the final investment memo and score summary.
This is the terminal agent — it produces the human-readable output.
"""
from __future__ import annotations

import re

from analysis.memo_engine import MemoEngine, MemoInput
from analysis.metrics import compute_signal_confidence
from analysis.peer_comparison import PeerComparison, build_peer_comparison
from analysis.scorer import build_scorecard
from analysis.valuation_range import ValuationRange
from agents.base_agent import BaseAgent
from config import Config
from models.message import AgentMessage, MessageType
from models.scorecard import CategoryScore, Scorecard, Stance
from models.stock_data import StockData
from utils.helpers import format_large_number, format_pct

_memo_engine = MemoEngine()


class ReportingAgent(BaseAgent):
    name = "reporting_agent"

    def process_message(self, message: AgentMessage) -> AgentMessage:
        if message.message_type != MessageType.FINAL_SUMMARY_REQUEST:
            return self._error_response(message, "Expected FINAL_SUMMARY_REQUEST")

        payload = message.payload
        stock_data: StockData = payload.get("stock_data")
        agent_findings: dict = payload.get("agent_findings", {})
        overall_confidence: float = payload.get("confidence", 0.0)

        # ── Gather category scores from agent findings ─────────────────────────
        fund = agent_findings.get("fundamental", {})
        tech = agent_findings.get("technical", {})
        risk_findings = agent_findings.get("risk", {})

        valuation: CategoryScore = fund.get("valuation")
        growth: CategoryScore = fund.get("growth")
        profitability: CategoryScore = fund.get("profitability")
        financial_health: CategoryScore = fund.get("financial_health")
        momentum: CategoryScore = tech.get("momentum")
        risk: CategoryScore = risk_findings.get("risk")
        risk_flags: list[str] = risk_findings.get("risk_flags", [])

        # Guard: use 50/100 defaults for any missing scores
        def _default_score(name: str, weight: float) -> CategoryScore:
            return CategoryScore(
                name=name, score=50.0, weight=weight,
                reasoning="Insufficient data for this category", data_quality="missing"
            )

        w = Config.SCORE_WEIGHTS
        valuation = valuation or _default_score("valuation", w["valuation"])
        growth = growth or _default_score("growth", w["growth"])
        profitability = profitability or _default_score("profitability", w["profitability"])
        financial_health = financial_health or _default_score("financial_health", w["financial_health"])
        momentum = momentum or _default_score("momentum", w["momentum"])
        risk = risk or _default_score("risk", w["risk"])

        # ── Build final scorecard ──────────────────────────────────────────────
        macro_findings = agent_findings.get("macro")
        scorecard = build_scorecard(
            ticker=message.ticker,
            valuation=valuation,
            growth=growth,
            profitability=profitability,
            financial_health=financial_health,
            momentum=momentum,
            risk=risk,
            risk_flags=risk_flags,
            confidence=overall_confidence,
            macro_findings=macro_findings,
        )

        # ── Blend signal-agreement confidence with data-availability confidence ──
        # overall_confidence is from compute_confidence(norm_metrics) — measures
        # data completeness.  compute_signal_confidence measures whether the
        # category scores AGREE directionally.  Both must be high for true conviction.
        _all_cats = {
            "valuation":        valuation,
            "growth":           growth,
            "profitability":    profitability,
            "financial_health": financial_health,
            "momentum":         momentum,
            "risk":             risk,
        }
        signal_conf, signal_explanation = compute_signal_confidence(_all_cats, macro=macro_findings)
        # Final confidence = average of data completeness and signal agreement.
        # This ensures 99% confidence is only possible when BOTH data is complete
        # AND signals align — not just when the pipeline ran successfully.
        blended_conf = round((overall_confidence + signal_conf) / 2.0, 3)
        scorecard.confidence = blended_conf
        scorecard.confidence_explanation = signal_explanation
        print(
            f"  [CONF BLEND] data={overall_confidence:.3f}"
            f" signal={signal_conf:.3f} → blended={blended_conf:.3f}"
            f" | {signal_explanation}"
        )

        # ── Apply macro influence on confidence and stance (post-scoring) ────────
        self._apply_macro_influence(scorecard, macro_findings or {})

        # ── Single integrated confidence explanation ───────────────────────────
        # Replaces the two-part approach (signal-agreement text + macro addendum)
        # with one cohesive sentence that names both fundamental and macro drivers.
        scorecard.confidence_explanation = self._compose_confidence_explanation(
            scorecard, _all_cats, macro_findings or {}
        )

        # ── Build investment memo text ─────────────────────────────────────────
        memo = self._build_memo(scorecard, stock_data, agent_findings)

        return self._reply(
            message,
            MessageType.FINAL_SUMMARY_RESPONSE,
            payload={
                "scorecard": scorecard,
                "memo": memo,
                "risk_flags": risk_flags,
            },
            confidence=overall_confidence,
            reasoning_summary="Final investment memo compiled from all agent outputs.",
        )

    # ── Memo builder ──────────────────────────────────────────────────────────

    def _build_memo(
        self,
        sc: Scorecard,
        stock_data: StockData,
        findings: dict,
    ) -> str:
        p = stock_data.profile
        ticker = sc.ticker
        company = p.company_name if p else ticker
        sector = p.sector if p else "Unknown"
        industry = p.industry if p else "Unknown"

        # ── Valuation metrics — always from NormalizedMetrics ─────────────────
        # NormalizedMetrics is the single source of truth for all display-facing
        # numbers.  Using it here guarantees the memo header is consistent with
        # the scorecard, peer table, and scenario analysis — they all consumed
        # the same object.  Raw re-derivation from ratios/income is eliminated.
        fund          = findings.get("fundamental", {})
        norm_metrics  = fund.get("normalized_metrics")

        if norm_metrics is not None:
            price  = norm_metrics.price or stock_data.current_price
            mktcap = format_large_number(norm_metrics.market_cap or stock_data.market_cap)
            pe_val = norm_metrics.pe_ratio
            ps_val = norm_metrics.ps_ratio
            ev_val = norm_metrics.ev_ebitda
            print(
                f"  [MEMO] metrics from NormalizedMetrics:"
                f" price={price}  mktcap={norm_metrics.market_cap}({norm_metrics.market_cap_source})"
                f" pe={pe_val}({norm_metrics.pe_source})"
                f" ps={ps_val}({norm_metrics.ps_source})"
                f" ev={ev_val}({norm_metrics.ev_ebitda_source})"
            )
        else:
            # Fallback: raw derivation (no NormalizedMetrics available, e.g. test mode)
            price  = stock_data.current_price
            mktcap = format_large_number(stock_data.market_cap)
            ratios = stock_data.latest_ratios
            pe_val = ratios.pe_ratio     if ratios else None
            ps_val = ratios.ps_ratio     if ratios else None
            ev_val = ratios.ev_to_ebitda if ratios else None
            print(
                f"  [MEMO] NormalizedMetrics not in findings — falling back to raw ratios"
            )

        _fv = lambda x: f"{x:.1f}" if x is not None else "N/A"
        pe_str = _fv(pe_val)
        ps_str = _fv(ps_val)
        ev_str = _fv(ev_val)

        macro_findings = findings.get("macro", {})
        val_range: ValuationRange | None = findings.get("fundamental", {}).get("valuation_range")

        # ── Alpha engine: unified pipeline ───────────────────────────────────────
        # AlphaPipeline enforces strict layer ordering (factor → regression →
        # HRL → scenario tree → enriched MC), wraps each layer in an independent
        # try/except, runs divergence detection, and coherence checks.
        # val_range.mc is mutated in-place by the pipeline when MC re-run succeeds.
        peer_cmp              = None   # built later — pre-initialise so scoping is clean
        _regime_for_mc        = (macro_findings or {}).get("macro_regime") or "Unknown"
        _factor_profile       = None
        _regression_calib     = None
        _hrl_result           = None
        _scenario_tree        = None
        _alpha_divergence_label = "n/a"
        _alpha_divergence_flags: list = []
        _alpha_coherence_issues: list = []
        _alpha                        = None   # pre-declare for quant_engine gate below

        if val_range is not None:
            try:
                from analysis.alpha_pipeline import AlphaPipeline as _APipeline
                _alpha = _APipeline().run(
                    stock_data   = stock_data,
                    val_range    = val_range,
                    macro_regime = _regime_for_mc,
                    peer_rows    = [],          # peer_cmp not yet built at this scope
                    pe_val       = pe_val,
                    price        = price,
                )
                _factor_profile         = _alpha.factor_profile
                _regression_calib       = _alpha.regression_calib
                _hrl_result             = _alpha.hrl_result
                _scenario_tree          = _alpha.scenario_tree
                _alpha_divergence_label = _alpha.divergence_label
                _alpha_divergence_flags = _alpha.divergence_flags or []
                _alpha_coherence_issues = _alpha.coherence_issues or []
            except Exception as _alpha_err:
                print(f"  [ALPHA_PIPELINE] failed: {_alpha_err}")

        # ── Quant engine: 8-step unified result ──────────────────────────────
        # Enriches alpha outputs with explicit step-by-step audit trail and
        # stores the result in findings so web_api.py can expose it via the
        # API without re-running any computation.
        # findings IS state.agent_findings (passed by reference from orchestrator).
        if val_range is not None and _alpha is not None:
            try:
                from analysis.quant_engine import QuantEngine as _QE
                _qe_result = _QE().build(
                    ticker        = ticker,
                    alpha_outputs = _alpha,
                    val_range     = val_range,
                )
                findings["quant_engine"] = _qe_result
            except Exception as _qe_err:
                print(f"  [QUANT_ENGINE] failed: {_qe_err}")
                findings["quant_engine"] = {"available": False, "ticker": ticker}

        market_findings = findings.get("market", {}).get("findings", {})
        analyst_note = market_findings.get("analyst", {}).get("note", "")
        pt_note = market_findings.get("price_target", {}).get("note", "")
        sector_note = market_findings.get("sector", {}).get("note", "")
        _analyst_consensus_target: "float | None" = (
            market_findings.get("price_target", {}).get("consensus_target")
        )

        sentiment_findings = findings.get("sentiment", {}).get("findings", {})
        sentiment_note = sentiment_findings.get("note", "")

        # ── Run audit log ─────────────────────────────────────────────────────
        # Printed unconditionally so every run leaves a clear trail of raw
        # inputs → displayed values.  Discrepancies are flagged with ***.
        self._print_run_audit(ticker, stock_data, findings, pe_val, ps_val, ev_val)

        from datetime import datetime as _dt
        _now = _dt.now()
        _hour = _now.strftime("%I").lstrip("0") or "12"
        _ts = f"{_now.strftime('%B')} {_now.day}, {_now.year} — {_hour}:{_now.strftime('%M %p')}"

        lines: list[str] = []

        # ── Methodology preamble ──────────────────────────────────────────────
        lines += [
            "  STOCKEVAL METHODOLOGY",
            "  ─────────────────────",
            "  This report uses driver-based scenario modeling: forward revenue ×",
            "  operating margin × FCF conversion × exit multiple, calibrated against",
            "  historical peer multiples.  This differs from traditional methods:",
            "",
            "    • DCF (Discounted Cash Flow) — discounts projected FCF at WACC.",
            "      StockEval uses explicit operating drivers but doesn't apply WACC",
            "      discounting; results may diverge for high-growth tickers.",
            "    • Intrinsic value — multiplies forward EPS by historical P/E.",
            "      StockEval shows P/E × forward EPS as a 'supporting method' but",
            "      doesn't use it as primary, due to EPS volatility on many tickers.",
            "    • Comparable multiples — applies peer P/E or EV/EBITDA directly.",
            "      StockEval embeds peer multiples as the 'exit multiple' input.",
            "",
            "  When driver model diverges materially from supporting methods, the",
            "  Methodology Note in the valuation section explains why.",
            "",
        ]

        # Header
        lines += [
            "=" * 68,
            f"  INVESTMENT SNAPSHOT — {ticker}",
            f"  As of {_ts}",
            "=" * 68,
            "",
        ]

        _conf_label = self._confidence_label(sc.confidence)

        # Company info + key metrics — provenance sourced from NormalizedMetrics
        def _src(label: str) -> str:
            return f"  [{label}]" if label else ""

        _nm = norm_metrics  # alias for brevity
        _mktcap_src  = _src(_nm.market_cap_source)  if _nm else ""
        _price_src   = _src("FMP")                   # price always from FMP in live runs
        _pe_src      = _src(_nm.pe_source)           if _nm else ""
        _ps_src      = _src(_nm.ps_source)           if _nm else ""
        _ev_src      = _src(_nm.ev_ebitda_source)    if _nm else ""

        _shares_line = ""
        if _nm and _nm.shares is not None:
            _sh_fmt = f"{_nm.shares / 1e6:.1f}M"
            _sh_src = f"  [{_nm.shares_source}]" if _nm.shares_source else ""
            _sh_url = f"  {_nm.shares_filing_url}" if getattr(_nm, "shares_filing_url", None) else ""
            _shares_line = f"  Shares Out.   : {_sh_fmt}{_sh_src}{_sh_url}"

        _price_line = (
            f"  Current Price : ${price:.2f}{_price_src}" if price
            else "  Current Price : N/A"
        )

        _header_metrics = [
            f"  Company       : {company}",
            f"  Sector        : {sector}  |  Industry: {industry}",
            f"  Market Cap    : {mktcap}{_mktcap_src}",
            _price_line,
        ]
        if _shares_line:
            _header_metrics.append(_shares_line)
        _header_metrics += [
            f"  P/E Ratio     : {pe_str}{_pe_src}",
            f"  P/S Ratio     : {ps_str}{_ps_src}",
            f"  EV/EBITDA     : {ev_str}{_ev_src}",
            "",
        ]

        lines += _header_metrics

        # Sizing must be computed before ACTION derivation so that a 0% final
        # size can veto a BUY/STAGED BUY zone signal and return WAIT instead.
        _beta = getattr(stock_data.profile, "beta", None) if stock_data and stock_data.profile else None
        # Re-derive reliability from live price_history (NormalizedMetrics was computed
        # before price_history was fetched, so its beta_months may be 0).
        _ph_closes    = getattr(getattr(stock_data, "price_history", None), "closes", []) or []
        _ph_months    = int(len(_ph_closes) / 21)
        _beta_reliable = (
            _ph_months >= 24
            and (_beta is None or abs(_beta) <= 5.0)
        ) if _ph_closes else getattr(norm_metrics, "beta_reliable", True)
        _validation_for_sizing = findings.get("fundamental", {}).get("validation")
        self._build_position_sizing_section._scenario_tree_ref    = _scenario_tree
        self._build_position_sizing_section._divergence_label_ref = _alpha_divergence_label
        self._build_position_sizing_section._validation_ref       = _validation_for_sizing
        _ps_lines, _ps_data = self._build_position_sizing_section(
            sc, macro_findings, beta=_beta, beta_reliable=_beta_reliable,
            pe_val=pe_val, val_range=val_range, price=price,
            factor_profile=_factor_profile,
        )
        self._build_position_sizing_section._scenario_tree_ref    = None
        self._build_position_sizing_section._divergence_label_ref = None
        self._build_position_sizing_section._validation_ref       = None
        sc.position_sizing = _ps_data
        _final_size = float(_ps_data.get("position_size", -1))

        _outlook, _action, _action_why = self._derive_outlook_action(sc, price, val_range, final_size=_final_size)
        _action_detail = ""
        if _action == "BUY":
            _action_detail = " (price below P25 fair value — strong entry zone)"
        elif _action == "STAGED BUY":
            _action_detail = " (price between P25–P50 fair value — build gradually)"
        elif _action == "HOLD":
            _action_detail = " (maintain existing position; price near fair value)"
        elif _action == "SELL":
            _action_detail = " (thesis broken or price far above fair value)"
        elif _action == "WAIT":
            if _final_size == 0.0:
                _action_detail = " (risk/return metrics do not support entry at current price)"
            elif "Bullish" in _outlook:
                _action_detail = " (bullish thesis; price at or above fair value — await pullback)"
            else:
                _action_detail = " (price at or above fair value — no favorable entry signal)"

        _action_block: list[str] = [
            "─" * 68,
            f"  Overall Score : {sc.overall_score:.0f} / 100",
            "═" * 68,
            f"  OUTLOOK  : {_outlook}",
            f"  ACTION   : {_action}{_action_detail}",
        ]
        if _action_why:
            # Wrap long why lines at 60 chars
            _why_prefix = "  WHY     : "
            _why_wrap   = "             "
            _why_words  = _action_why.split()
            _why_line   = _why_prefix
            for _w in _why_words:
                if len(_why_line) + len(_w) + 1 > 67:
                    _action_block.append(_why_line)
                    _why_line = _why_wrap + _w
                else:
                    _why_line += ("" if _why_line.endswith(": ") else " ") + _w
            if _why_line.strip():
                _action_block.append(_why_line)
        _action_block += [
            "═" * 68,
            *(
                [f"  Stock Type    : {_st[0]} — {_st[1]}"]
                if (_st := self._classify_stock_type(sc)) else []
            ),
            f"  Confidence    : {sc.confidence:.0%} — {_conf_label}",
            *(
                [f"  Why           : {sc.confidence_explanation}"]
                if sc.confidence_explanation else []
            ),
            "─" * 68,
            "",
        ]
        lines += _action_block
        lines.append("  Category Subscores (score × weight = contribution):")

        # Use CategoryScore objects directly so data_quality == "missing" shows as N/A,
        # not as a silent 50/100 default that looks like real data.
        _cat_attrs = ["valuation", "growth", "profitability", "financial_health", "momentum", "risk"]
        total_contribution = 0.0
        for attr in _cat_attrs:
            cat_obj = getattr(sc, attr, None)
            label = attr.replace("_", " ").title().ljust(18)
            if cat_obj is None or cat_obj.data_quality == "missing":
                score_str = "   N/A"
                bar = self._score_bar(None)
                contrib_str = ""
            else:
                score_str = f"{cat_obj.score:.0f}/100"
                bar = self._score_bar(cat_obj.score)
                contrib = cat_obj.score * cat_obj.weight
                total_contribution += contrib
                contrib_str = f"  → {contrib:.1f} pts ({cat_obj.weight*100:.0f}%)"
            lines.append(f"    {label} {score_str:>7}  {bar}{contrib_str}")
        lines.append("")

        # Pre-compute strong/weak category lists — used in both Bullish/Bearish
        # filtering below and the Strengths/Risks blocks in the Investment Memo.
        strong_cats = [
            attr for attr in _cat_attrs
            if getattr(sc, attr, None)
            and getattr(sc, attr).data_quality != "missing"
            and getattr(sc, attr).score >= 70
        ]
        weak_cats = [
            attr for attr in _cat_attrs
            if getattr(sc, attr, None)
            and getattr(sc, attr).data_quality != "missing"
            and getattr(sc, attr).score < 45
        ]

        # Build term set from all strong-category reasoning + factors so that
        # Bullish Factors can be checked for semantic overlap with Strengths.
        _strength_terms: set[str] = set()
        for _cat in strong_cats:
            _cat_obj = getattr(sc, _cat)
            _strength_terms |= self._key_terms(_cat_obj.reasoning)
            for _f in getattr(_cat_obj, "factors", [])[:3]:
                _strength_terms |= self._key_terms(_f)

        # Bullish / Bearish Factors
        # Suppress:
        #   • "Macro:"      — covered in Macro Regime section
        #   • "Valuation:"  — covered in Valuation View + Range
        #   • Any category prefix whose detailed reasoning already appears in
        #     Strengths (score ≥ 70) — prefix match (fast path)
        #   • Any factor whose content overlaps heavily with Strengths reasoning
        #     — semantic dedup (slower path, catches cross-category rewording)
        _suppress = ("Macro:", "Valuation:") + tuple(attr.title() + ":" for attr in strong_cats)
        _OVERLAP_THRESH = 0.45
        bullish_filtered = [
            f for f in sc.bullish_factors
            if not any(f.startswith(p) for p in _suppress)
            and self._overlap_ratio(self._key_terms(f), _strength_terms) < _OVERLAP_THRESH
        ]
        bearish_filtered = [
            f for f in sc.bearish_factors
            if not f.startswith("Macro:")
        ]

        if bullish_filtered:
            lines.append("  Supporting Factors:")
            for f in bullish_filtered[:4]:
                lines.append(f"    + {f}")
            lines.append("")

        # Build term set from shown bullish factors so Key Drivers can skip
        # anything already stated there or in Strengths reasoning.
        _shown_terms: set[str] = _strength_terms.copy()
        for _f in bullish_filtered[:4]:
            _shown_terms |= self._key_terms(_f)

        if bearish_filtered:
            lines.append("  Risk Factors:")
            for f in bearish_filtered[:4]:
                lines.append(f"    - {f}")
            lines.append("")

        # Risk flags
        if sc.risk_flags:
            lines.append("  Risk Flags:")
            for flag in sc.risk_flags[:5]:
                lines.append(f"    ⚠  {flag}")
            lines.append("")

        # ── DATA QUALITY FLAGS — from DataIntegrityEngine ─────────────────────
        _validation = findings.get("fundamental", {}).get("validation")
        _dq_header_written = False

        def _ensure_dq_header() -> None:
            nonlocal _dq_header_written
            if not _dq_header_written:
                lines.append("  Data Quality Flags:")
                _dq_header_written = True

        if _validation is not None and _validation.flags:
            _dq_lines = _validation.report_lines()
            if _dq_lines:
                _ensure_dq_header()
                lines += _dq_lines

        # Alpha engine divergence flags (scenario tree vs factor model disagreement)
        for _dflag in _alpha_divergence_flags:
            _ensure_dq_header()
            lines.append(f"    ⚠  {_dflag}")

        # Cross-layer coherence issues from _coherence_check()
        for _cflag in _alpha_coherence_issues:
            _ensure_dq_header()
            lines.append(f"    ℹ  {_cflag}")

        if _dq_header_written:
            lines.append("")

        # ── SCENARIO TREE — narrative summary ─────────────────────────────────
        if _scenario_tree is not None:
            _st = _scenario_tree
            _st_lines: list[str] = []

            # Best / worst case
            if _st.best_case is not None and _st.worst_case is not None:
                _bc = _st.best_case
                _wc = _st.worst_case
                _st_lines.append(
                    f"    Bull Case  ({_bc.probability:.0%}):  "
                    f"{_bc.label}  →  "
                    f"${_bc.target_price:.0f}  ({_bc.expected_return:+.0%})"
                )
                _st_lines.append(
                    f"    Bear Case  ({_wc.probability:.0%}):  "
                    f"{_wc.label}  →  "
                    f"${_wc.target_price:.0f}  ({_wc.expected_return:+.0%})"
                )

            # Weighted expected return + dispersion
            _st_lines.append(
                f"    Weighted E[R]: {_st.weighted_return:+.1%}"
                f"  |  Scenario Std: {_st.scenario_std:.0%}"
                f"  |  VaR (P5): {_st.var_95:.0%}"
            )

            # Skew characterisation
            _up = _st.upside_mass
            _dn = _st.downside_mass
            if _up > 0.40 and _dn < 0.15:
                _skew_note = f"Asymmetric upside: {_up:.0%} of probability mass >+20% return."
            elif _dn > 0.35:
                _skew_note = f"Downside skewed: {_dn:.0%} of probability mass <−20% return."
            elif _up > 0.25 and _dn < 0.20:
                _skew_note = f"Modest positive skew ({_up:.0%} upside / {_dn:.0%} downside mass)."
            else:
                _skew_note = f"Balanced distribution ({_up:.0%} upside / {_dn:.0%} downside mass)."
            _st_lines.append(f"    {_skew_note}")

            # Concentration note (binary risk flag)
            if _st.concentration_3 > 0.70:
                _st_lines.append(
                    f"    ⚠  High scenario concentration: top-3 paths = "
                    f"{_st.concentration_3:.0%} of probability.  Binary risk present."
                )

            if _st_lines:
                lines.append("  1-Year Operating Scenarios  (supplementary; scenario tree)")
                lines.append("  ─────────────────────────────────────────────────────────")
                lines += _st_lines
                lines.append("")

        # Key drivers — filter out any that restate Strengths or Bullish Factors
        # NOTE: "What Would Change Our View" is owned by MemoEngine (_change_view_bullets +
        # _reframe_trigger) and rendered in the INVESTMENT MEMO block below.
        # Do NOT emit it here — it would produce a duplicate, unreframed section.
        _drivers_deduped = [
            d for d in sc.key_drivers
            if self._overlap_ratio(self._key_terms(d), _shown_terms) < _OVERLAP_THRESH
        ]
        lines.append("  Key Drivers:")
        for d in _drivers_deduped[:3]:
            lines.append(f"    • {d}")
        lines.append("")

        # Macro regime context — debug log so LEI field presence is auditable
        _lei_debug = {
            k: macro_findings.get(k)
            for k in ("macro_regime", "macro_score", "recession_risk_level",
                      "cycle_phase", "lei_trend", "yield_spread_trend")
        }
        print(f"  [MACRO_SECTION] payload for _build_macro_section: {_lei_debug}")
        lines += self._build_macro_section(macro_findings)

        # ── Valuation View ─────────────────────────────────────────────────────
        lines.append("  Valuation View")
        lines.append("  --------------")
        _peg_method = getattr(val_range, "peg_method", "") if val_range else ""
        _peg_note   = getattr(val_range, "peg_note",   "") if val_range else ""
        if val_range and val_range.peg_ratio is not None:
            if _peg_method == "revenue_cagr":
                _peg_display = f"{val_range.peg_ratio:.2f}x (rev-CAGR; EPS too volatile)"
            else:
                _peg_display = f"{val_range.peg_ratio:.2f}x"
        elif _peg_method == "not_meaningful":
            _peg_display = "not meaningful"
        else:
            _peg_display = "N/A"
        _vv_pe  = f"{pe_str}{_pe_src}"
        _vv_ps  = f"{ps_str}{_ps_src}"
        _vv_ev  = f"{ev_str}{_ev_src}"
        lines.append(f"  P/E: {_vv_pe}  |  P/S: {_vv_ps}  |  EV/EBITDA: {_vv_ev}  |  PEG: {_peg_display}")
        if val_range and val_range.peg_ratio is not None and val_range.peg_interpretation:
            lines.append(f"  {val_range.peg_interpretation}")
        if _peg_note:
            lines.append(f"  ⚠ PEG note: {_peg_note}")
        if sc.valuation and sc.valuation.data_quality != "missing":
            lines.append(f"  {sc.valuation.reasoning}")
            tension_factors = [f for f in sc.valuation.factors if "PEG" in f and "tension" not in f.lower() and len(f) > 40]
            plain_factors   = [f for f in sc.valuation.factors if f not in tension_factors]
            for f in plain_factors[:3]:
                lines.append(f"  • {f}")
        else:
            lines.append("  Valuation data unavailable.")
        _gq = self._compute_growth_quality(stock_data)
        if _gq:
            _gq_label, _gq_desc = _gq
            lines.append(f"  Growth Quality: {_gq_label} — {_gq_desc}")
        lines.append("")

        # ── Trend Summary ──────────────────────────────────────────────────────
        _trends = fund.get("trends")
        if _trends is not None:
            lines.append("  Trend Summary")
            lines.append("  -------------")
            lines.append(
                f"  Revenue Growth   {_trends.revenue_growth_sig}  {_trends.revenue_growth}"
            )
            lines.append(
                f"  Operating Margin {_trends.op_margin_sig}  {_trends.op_margin}"
            )
            lines.append(
                f"  Net Margin       {_trends.net_margin_sig}  {_trends.net_margin}"
            )
            lines.append(
                f"  ROE              {_trends.roe_sig}  {_trends.roe}"
            )
            lines.append(
                f"  ROIC             {_trends.roic_sig}  {_trends.roic}"
            )
            lines.append("")

        _growth_score = (
            sc.growth.score
            if sc.growth and sc.growth.data_quality != "missing"
            else None
        )
        lines += self._build_valuation_range_section(
            val_range, price,
            growth_score    = _growth_score,
            scenario_tree   = _scenario_tree,
            analyst_target  = _analyst_consensus_target,
        )

        # ── Peer comparison ────────────────────────────────────────────────────
        peer_cmp = None   # initialise so validation layer always has a reference
        try:
            _peg    = val_range.peg_ratio       if val_range else None
            _growth = val_range.eps_growth_rate if val_range else None
            peer_cmp = build_peer_comparison(
                target_ticker=ticker,
                target_pe=pe_val,
                target_ps=ps_val,
                target_ev_ebitda=ev_val,
                target_growth=_growth,
                target_peg=_peg,
                sector=sector,
                industry=industry,
                target_company_name=company,
                target_mkt_cap=stock_data.market_cap if stock_data else None,
                # Margin/financial profile — feeds PeerSelectionEngine.classify()
                # revenue-model inference and apply_structural_filters margin filter.
                # Without these, classification falls back to Archetype.OTHER.
                target_gross_margin    = norm_metrics.gross_margin     if norm_metrics else None,
                target_operating_margin= norm_metrics.operating_margin if norm_metrics else None,
                target_net_margin      = norm_metrics.net_margin       if norm_metrics else None,
                target_debt_equity     = norm_metrics.debt_to_equity   if norm_metrics else None,
                target_roe             = norm_metrics.roe              if norm_metrics else None,
                target_current_ratio   = norm_metrics.current_ratio    if norm_metrics else None,
                target_stock_data      = stock_data,
            )
            lines += self._build_peer_comparison_section(peer_cmp)
            lines += self._build_peer_history_section(peer_cmp)
        except Exception as _exc:
            print(f"  [REPORT] peer comparison failed — {_exc}")
            lines += [
                "  Peer Comparison",
                "  ───────────────",
                "    No valid peers — peer engine error. Do NOT fall back to sector.",
                "",
            ]

        # Analyst context
        if analyst_note or pt_note or sector_note:
            lines.append("  Market Context:")
            for note in [sector_note, analyst_note, pt_note]:
                if note:
                    lines.append(f"    {note}")
            lines.append("")

        # ── Investment Memo — MemoEngine synthesis ───────────────────────────
        lines += [
            "=" * 68,
            "  INVESTMENT MEMO",
            "=" * 68,
            "",
        ]

        _memo_input = MemoInput(
            scorecard = sc,
            company   = company,
            sector    = sector,
            industry  = industry,
            macro     = macro_findings or {},
            pe        = pe_val,
            ps        = ps_val,
            ev_ebitda = ev_val,
            price     = price,
            action    = _action,
        )
        _memo_result = _memo_engine.build(_memo_input)

        # ── Pre-render enforcement ────────────────────────────────────────────
        # _enforce_report_sections is the gate between module output and render.
        # It mutates peers and triggers in-place (correctable violations) and
        # raises for memo structure violations (code bugs).
        # peer_cmp may be None when the peer engine returned no result — section
        # A (peer filtering) is skipped in that case; B and C always run.
        self._enforce_report_sections(_memo_result, peer_cmp)

        # Cache finalized peer_cmp in findings so web_api._extract_peer_comparison
        # can reuse it instead of making a second build_peer_comparison FMP call.
        # findings IS state.agent_findings (by reference from orchestrator), so
        # web_api.py sees this value without any extra plumbing.
        if peer_cmp is not None:
            findings["peer_comparison"] = peer_cmp

        # Lock the memo — no further field mutations are permitted after this point.
        _memo_result.lock()
        print("  [ENFORCE] pre-render checks complete — memo locked")

        # ── Memo render — locked block ────────────────────────────────────────
        # MemoEngine output is written ONCE here. Nothing is appended inside
        # the memo block after this point. Position sizing and data sources are
        # separate named sections below, not part of the Investment Memo.
        _memo_rendered = _memo_result.render()
        lines += _memo_rendered.split("\n")
        lines.append("")  # single blank line between memo and position sizing

        print(f"  [MEMO] word_count={_memo_result.word_count} | locked — no post-render appending")

        lines += _ps_lines

        # ── Data sources summary ───────────────────────────────────────────────
        sources = stock_data.data_sources
        if sources:
            lines += ["─" * 68, "  Data Sources", "─" * 68, ""]

            dataset_sources = {k: v for k, v in sources.items() if "." not in k}
            field_sources   = {k: v for k, v in sources.items() if "." in k}

            _non_data = ("FMP", "unavailable", "access_restricted")
            fallback_datasets = {
                k: v for k, v in dataset_sources.items()
                if v not in _non_data
            }
            access_restricted = sorted(
                k for k, v in dataset_sources.items() if v == "access_restricted"
            )
            unavailable = sorted(
                k for k, v in dataset_sources.items() if v == "unavailable"
            )
            # Group field-level fills by provider
            fallback_fields_by_provider: dict[str, list[str]] = {}
            for fk, prov in field_sources.items():
                if prov not in _non_data:
                    fallback_fields_by_provider.setdefault(prov, []).append(
                        fk.split(".")[-1]
                    )

            lines.append("  Primary provider : FMP (Financial Modeling Prep)")

            if fallback_datasets:
                for ds, prov in sorted(fallback_datasets.items()):
                    lines.append(f"  Fallback dataset : {ds} sourced from {prov}")

            if fallback_fields_by_provider:
                for prov, fields in sorted(fallback_fields_by_provider.items()):
                    lines.append(
                        f"  Fallback fields  : {', '.join(sorted(fields))}"
                        f" sourced from {prov}"
                    )

            if access_restricted:
                lines.append(
                    f"  Access restricted: {', '.join(access_restricted)}"
                    " (HTTP 402 — endpoint requires a higher FMP plan)"
                )

            if unavailable:
                lines.append(
                    f"  Unavailable      : {', '.join(unavailable)}"
                    " (no data returned — ticker may not have this data)"
                )

            if access_restricted or unavailable:
                completeness = "partial"
            else:
                completeness = "complete"
            lines.append(f"  Result based on  : {completeness} data")
            lines.append("")

        # ── Excel reconciliation (opt-in; only when file exists) ─────────────
        _excel_data = self._load_excel_summary(ticker)
        if _excel_data is not None:
            lines += self._render_excel_reconciliation(_excel_data, val_range, price)

        lines.append("=" * 68)

        return "\n".join(lines)

    def _apply_macro_influence(self, sc: Scorecard, macro: dict) -> None:
        """
        Light-touch post-scoring adjustments driven by macro regime findings.
        Operates on the scorecard object in-place after build_scorecard() returns.

        Confidence:
          Expansion + Low risk   → +0.02
          Contraction            → -0.04
          High recession risk    → -0.04 (cumulative if also Contraction)
          Elevated recession risk→ -0.02
          All others             → 0

        Stance guardrail (belt-and-suspenders; scorer may have already applied
        this when the full overlay ran, but this covers the no-FRED path):
          recession_risk == "High" + Bullish  → Neutral
          recession_risk == "High" + Neutral  → Bearish
        """
        regime      = macro.get("macro_regime") or macro.get("regime", "")
        risk        = macro.get("recession_risk_level") or macro.get("recession_risk", "")
        cycle_phase = macro.get("cycle_phase", "")

        if not regime or regime == "Unknown":
            return

        # ── Confidence delta ───────────────────────────────────────────────────
        delta = 0.0
        if regime == "Expansion" and risk == "Low":
            delta += 0.02
        elif regime == "Contraction":
            delta -= 0.04
        if risk == "High":
            delta -= 0.04
        elif risk == "Elevated":
            delta -= 0.02

        # Cycle-phase refinement — small additional nudge (+/− 0.01)
        # Supportive phases get a slight boost; late-cycle a slight drag.
        # Contraction is already penalised above; do not double-penalise.
        if cycle_phase in ("early", "mid"):
            delta += 0.01
        elif cycle_phase == "late":
            delta -= 0.01

        sc.confidence = max(0.0, min(1.0, sc.confidence + delta))

        # ── Stance guardrail ───────────────────────────────────────────────────
        if risk == "High":
            if sc.stance == Stance.BULLISH:
                sc.stance = Stance.NEUTRAL
            elif sc.stance == Stance.NEUTRAL:
                sc.stance = Stance.BEARISH

        print(
            f"  [Macro Influence] confidence adjusted to {sc.confidence:.2%}"
            f", stance={sc.stance.value}"
            f"  (regime={regime!r}, cycle_phase={cycle_phase!r}, recession_risk={risk!r}, delta={delta:+.2f})"
        )

    @staticmethod
    def _compose_confidence_explanation(
        sc: "Scorecard",
        cats: "dict[str, CategoryScore]",
        macro: dict,
    ) -> str:
        """
        Single integrated confidence explanation.

        Format: "[Label] confidence driven by [A and B], though [C and D] limit conviction."
        Always references the four signal groups: fundamentals, valuation, momentum, macro.

        Strong = score ≥ 65 | Weak = score < 45
        """
        _LABELS = {
            "valuation":        "valuation",
            "growth":           "growth",
            "profitability":    "profitability",
            "financial_health": "balance sheet strength",
            "momentum":         "momentum",
            "risk":             "structural risk profile",
        }

        def _join(names: list[str]) -> str:
            if not names:
                return ""
            if len(names) == 1:
                return names[0]
            return ", ".join(names[:-1]) + f" and {names[-1]}"

        # Confidence tier label (prose form)
        conf = sc.confidence
        if conf >= 0.80:
            conf_label = "High"
        elif conf >= 0.65:
            conf_label = "Moderate-to-high"
        elif conf >= 0.50:
            conf_label = "Moderate"
        else:
            conf_label = "Low"

        # Category classification — ordered for display priority
        _DRIVER_ORDER   = ["profitability", "financial_health", "growth", "valuation", "momentum", "risk"]
        _LIMITER_ORDER  = ["growth", "momentum", "valuation", "profitability", "financial_health", "risk"]

        drivers: list[str] = []
        limiters: list[str] = []
        for k in _DRIVER_ORDER:
            cat = cats.get(k)
            if cat and cat.data_quality != "missing":
                if cat.score >= 65:
                    drivers.append(_LABELS[k])
        for k in _LIMITER_ORDER:
            cat = cats.get(k)
            if cat and cat.data_quality != "missing":
                if cat.score < 45:
                    limiters.append(_LABELS[k])

        # ── Macro context ──────────────────────────────────────────────────────
        regime      = macro.get("macro_regime") or macro.get("regime", "")
        risk        = macro.get("recession_risk_level") or macro.get("recession_risk", "")
        cycle_phase = macro.get("cycle_phase", "")
        macro_available = bool(regime and regime != "Unknown")

        macro_supports = (
            macro_available
            and regime in ("Expansion", "Recovery")
            and risk not in ("High", "Elevated")
        )
        macro_limits = (
            macro_available
            and (risk in ("High", "Elevated") or regime in ("Contraction", "Slowdown"))
        )

        if macro_limits:
            if risk in ("High", "Elevated"):
                if cycle_phase and cycle_phase not in ("unknown", "contraction", "") and regime:
                    limiters.append(
                        f"{risk.lower()} recession risk"
                        f" in a {cycle_phase}-cycle {regime.lower()}"
                    )
                else:
                    limiters.append(f"{risk.lower()} recession risk")
            elif regime == "Contraction":
                limiters.append("macro contraction headwinds")
            else:
                limiters.append("a softening macro backdrop")

        # ── Assemble ──────────────────────────────────────────────────────────
        if drivers and limiters:
            return (
                f"{conf_label} confidence driven by {_join(drivers[:3])},"
                f" though {_join(limiters[:3])} limit conviction."
            )
        elif drivers:
            macro_note = (
                ", with a constructive macro backdrop adding support"
                if macro_supports else ""
            )
            return f"{conf_label} confidence driven by {_join(drivers[:3])}{macro_note}."
        elif limiters:
            return (
                f"{conf_label} confidence; {_join(limiters[:3])} weigh on"
                " overall conviction."
            )
        else:
            if macro_supports:
                return (
                    f"{conf_label} confidence — no single dominant driver,"
                    " though a supportive macro backdrop provides a constructive floor."
                )
            return f"{conf_label} confidence — signals are broadly balanced across categories."

    @staticmethod
    def _confidence_label(conf: float) -> str:
        """Human-readable confidence tier for display."""
        if conf >= 0.80:
            return "High"
        elif conf >= 0.65:
            return "Moderate-High"
        elif conf >= 0.50:
            return "Moderate"
        else:
            return "Low"

    def _build_verdict_block(self, sc: "Scorecard", macro: dict) -> list[str]:
        """
        Structured final verdict:
          Line 1 : Verdict | Score | Confidence (at-a-glance)
          Blank
          Para   : main strength driver → key concern → macro context → action
        Replaces the single-line `_write_verdict()` call in the final verdict block.
        """
        conf_label = self._confidence_label(sc.confidence)
        stance_display = (
            "Strong Buy" if sc.stance == Stance.BULLISH and sc.overall_score >= 80
            else "Buy" if sc.stance == Stance.BULLISH
            else "Hold" if sc.stance == Stance.NEUTRAL
            else "Sell"
        )

        def _s(attr: str) -> "float | None":
            cat = getattr(sc, attr, None)
            return cat.score if cat and cat.data_quality != "missing" else None

        prof_s   = _s("profitability")
        fh_s     = _s("financial_health")
        growth_s = _s("growth")
        val_s    = _s("valuation")
        mom_s    = _s("momentum")
        risk_s   = _s("risk")
        stance   = sc.stance.value

        # ── S1: Primary driver of the rating ──────────────────────────────────
        strong = [
            lbl
            for s, lbl in [
                (prof_s,   "strong margins"),
                (fh_s,     "balance sheet resilience"),
                (growth_s, "above-average growth"),
                (mom_s,    "positive price momentum"),
                (risk_s,   "a favourable risk profile"),
            ]
            if s is not None and s >= 65
        ]
        if strong:
            if len(strong) == 1:
                strength_str = strong[0]
            elif len(strong) == 2:
                strength_str = f"{strong[0]} and {strong[1]}"
            else:
                strength_str = f"{', '.join(strong[:-1])}, and {strong[-1]}"
            if stance == "Bullish":
                s1 = f"The buy thesis is underpinned by {strength_str}."
            elif stance == "Bearish":
                s1 = f"Despite {strength_str}, the sell thesis prevails on balance."
            else:
                s1 = (
                    f"{strength_str.capitalize()} provides a quality floor,"
                    " but is insufficient alone to shift the rating decisively."
                )
        else:
            if stance == "Bullish":
                s1 = "No single category is exceptional, but the composite picture is modestly constructive."
            elif stance == "Bearish":
                s1 = "The absence of clear strengths across categories underpins the negative view."
            else:
                s1 = "No dominant driver emerges — the balanced profile is consistent with a hold rating."

        # ── S2: Key concern ────────────────────────────────────────────────────
        weak = [
            lbl
            for s, lbl in [
                (val_s,    "stretched valuation"),
                (growth_s, "weak near-term growth"),
                (mom_s,    "deteriorating price momentum"),
                (prof_s,   "margin pressure"),
            ]
            if s is not None and s < 45
        ]
        if weak:
            concern_str = f"{weak[0]} and {weak[1]}" if len(weak) > 1 else weak[0]
            s2 = f"The primary risk to the thesis is {concern_str}, which warrants monitoring."
        elif val_s is not None and 45 <= val_s < 55 and growth_s is not None and growth_s >= 60:
            s2 = (
                "Valuation does not offer a wide margin of safety —"
                " execution on growth is critical to sustaining the buy case."
            )
        else:
            s2 = ""

        # ── S3: Macro context ──────────────────────────────────────────────────
        # _macro_context_line() returns a self-contained sentence; use it directly.
        _mc_line = self._macro_context_line(macro)
        s3 = _mc_line if _mc_line else ""

        # ── S4: Action ─────────────────────────────────────────────────────────
        if stance == "Bullish":
            if mom_s is not None and mom_s < 45:
                s4 = (
                    "Given weak near-term momentum, a staged entry is preferable —"
                    " initiate a starter position and add on technical confirmation."
                )
            elif sc.confidence >= 0.70:
                s4 = (
                    "Overall conviction supports establishing a meaningful position;"
                    " size per the position guidance below."
                )
            else:
                s4 = (
                    "Moderate conviction supports a starter position;"
                    " hold in reserve to add as the thesis develops."
                )
        elif stance == "Bearish":
            s4 = (
                "Avoid initiating or adding exposure at current prices —"
                " revisit if valuation corrects or fundamentals improve materially."
            )
        else:
            s4 = (
                "Hold existing positions but do not add at current levels;"
                " a clearer catalyst in either direction is needed to shift the rating."
            )

        paragraph = " ".join(s for s in [s1, s2, s3, s4] if s)
        return [
            f"  {paragraph}",
            "",
        ]

    def _build_macro_section(self, macro: dict) -> list[str]:
        """
        Return a compact, investment-oriented macro section as a list of lines.
        Caller appends these directly into the memo line list.

        Structure
        ─────────
        1. Hard data line  — Regime | Score | Recession Risk (always present)
        2. LEI narrative   — prose interpreting phase + trend + implication
                             Falls back to generic verdict when phase is unknown
        3. Sector tilt     — prescriptive sector preference
        4. Key factors     — up to 2 bullish, up to 2 bearish signals
        """
        regime   = macro.get("macro_regime") or macro.get("regime", "")
        score    = macro.get("macro_score")
        risk     = macro.get("recession_risk_level") or macro.get("recession_risk", "")
        tilt     = macro.get("sector_tilt", "")
        bullish  = macro.get("bullish_macro_factors", [])
        bearish  = macro.get("bearish_macro_factors", [])

        lines: list[str] = ["  Macro Regime:", "  ────────────"]

        if not regime or regime == "Unknown":
            lines.append(
                "    Macro signals are inconclusive — LEI overlay was not available for this evaluation."
            )
            lines.append("")
            return lines

        # 1. Labeled header block
        score_str = f"{score:.0f}/100" if score is not None else "N/A"
        cycle_phase  = macro.get("cycle_phase")
        lei_trend    = macro.get("lei_trend")
        spread_trend = macro.get("yield_spread_trend")

        lines.append(f"    Macro Regime   : {regime}")
        if cycle_phase and cycle_phase not in ("unknown", None):
            phase_label = cycle_phase.replace("_", " ").title()
            lines.append(f"    Cycle Phase    : {phase_label} Cycle")
        lines.append(f"    Score          : {score_str}")
        if risk:
            lines.append(f"    Recession Risk : {risk}")

        # 2. LEI narrative — one institutional paragraph
        if cycle_phase and cycle_phase not in ("unknown", None):
            narrative = self._lei_narrative(regime, cycle_phase, lei_trend, spread_trend, risk, sector_tilt=tilt)
        else:
            # No phase data — fall back to the generic one-liner
            narrative = self._macro_verdict(regime, risk)
        lines.append(f"    {narrative}")

        # 2c. OECD CLI staleness note — flag when latest obs is > 6 months old
        snapshot = macro.get("snapshot", {})
        obs_dates = macro.get("observation_dates", {})
        cli_obs_date = obs_dates.get("oecd_cli")
        if cli_obs_date:
            try:
                from datetime import date
                obs = date.fromisoformat(cli_obs_date)
                months_old = (date.today() - obs).days // 30
                if months_old > 6:
                    lines.append(
                        f"    Note: OECD CLI data is {months_old}m old"
                        f" (obs: {cli_obs_date}) — trend signal is lagged."
                    )
            except (ValueError, TypeError):
                pass

        # 3. Sector tilt
        if tilt:
            lines.append(f"    Sector Tilt: {tilt}")

        # 4. Key factors (up to 2 each), trailing clauses stripped for brevity
        for f in bullish[:2]:
            lines.append(f"    + {f.split(' — ')[0]}")
        for f in bearish[:2]:
            lines.append(f"    ✕ {f.split(' — ')[0]}")

        lines.append("")
        return lines

    @staticmethod
    def _lei_narrative(
        regime: str,
        cycle_phase: str,
        lei_trend: "str | None",
        spread_trend: "str | None",
        recession_risk: str,
        sector_tilt: str = "",
    ) -> str:
        """
        Three-sentence institutional paragraph:
          S1 — interpretive framing (what the phase means for investors, not a restatement
               of labeled fields; "appears to be" for uncertain phases, "is" for confirmed)
          S2 — signal reading using "While X, Y, suggesting Z" or "X and Y — [implication]";
               never just a list of fragments
          S3 — "This keeps recession risk [X] and [positioning implication]" with
               sector_tilt woven in when available

        Never opens with "The macro backdrop is in a [phase] phase" — that restates labels.
        Instead leads with an investor-relevant interpretation of what the phase implies.
        """
        # ── Signal atoms ──────────────────────────────────────────────────────
        lei_pos = {
            "rising":     "leading indicators are improving",
            "falling":    "leading indicators are fading",
            "inflecting": "leading indicators are at a potential inflection",
        }.get(lei_trend or "")

        curve_pos = {
            "rising":  "the yield curve is steepening",
            "falling": "the yield curve is compressing",
        }.get(spread_trend or "")

        risk_elevated = recession_risk in ("Elevated", "High")
        risk_str = recession_risk.lower() if recession_risk else "moderate"

        def _s3_low(action: str) -> str:
            """S3 for non-elevated recession risk: 'This keeps risk [X] and [action].'"""
            return f"This keeps recession risk {risk_str} and {action}."

        def _s3_high(action: str) -> str:
            """S3 for elevated recession risk: 'Recession risk is [X]: [action].'"""
            return f"Recession risk is {risk_str}: {action}."

        def _s3(low_action: str, high_action: str = "") -> str:
            """Dispatch to elevated or normal S3 path."""
            if risk_elevated:
                return _s3_high(high_action or low_action)
            return _s3_low(low_action)

        # sector_tilt is used only in the fallback path (_macro_verdict) where the
        # action strings do not already specify sector preferences.  In phase-specific
        # paths the sector guidance is embedded in the action strings directly.

        # ── Phase narratives ───────────────────────────────────────────────────
        if cycle_phase == "early":
            if regime == "Recovery":
                s1 = "Early-cycle recovery dynamics appear to be taking hold, with the trough likely behind us and re-acceleration still developing."
                if lei_pos and curve_pos:
                    if spread_trend == "falling":
                        s2 = (
                            f"While {lei_pos}, the yield curve remains compressed —"
                            " suggesting growth momentum is re-accelerating but not yet"
                            " confirmed across all markets."
                        )
                    else:
                        s2 = (
                            f"{lei_pos.capitalize()} and {curve_pos} —"
                            " both consistent with a recovery that is gaining momentum,"
                            " though broad confirmation is still developing."
                        )
                elif lei_pos:
                    s2 = (
                        f"While {lei_pos}, the pace of re-acceleration remains gradual —"
                        " the trough appears to have passed, but momentum is still building."
                    )
                elif curve_pos:
                    s2 = (
                        f"{curve_pos.capitalize()}, which is supportive of the early-cycle"
                        " view, though broader confirmation remains pending."
                    )
                else:
                    s2 = (
                        "Early indicators are stabilising after the contraction,"
                        " though the recovery is not yet confirmed across all markets."
                    )
                s3 = _s3(
                    "favors selective cyclical and small-cap exposure"
                    " as credit conditions ease and earnings expectations reset",
                    "selective cyclical and small-cap exposure is favored,"
                    " though position sizing should reflect the elevated risk backdrop",
                )

            else:  # Slowdown approaching a turning point
                s1 = "A cyclical turning point appears to be approaching, with softening conditions beginning to stabilise."
                if lei_pos:
                    s2 = (
                        f"While conditions have been softening, {lei_pos} —"
                        " suggesting the cycle may be closer to a trough than a"
                        " continuation of the downturn."
                    )
                else:
                    s2 = (
                        "Conditions appear to be stabilising after a period of softening,"
                        " though confirmation over the next two to three data prints is needed."
                    )
                s3 = _s3(
                    "supports rotating selectively into cyclicals",
                    "rotating selectively into cyclicals is appropriate"
                    " as the re-acceleration confirms across multiple prints",
                )

        elif cycle_phase == "mid":
            s1 = (
                "Growth is broadly tracking trend and the cycle retains forward"
                " momentum, consistent with a healthy mid-cycle expansion."
            )
            if lei_pos and curve_pos:
                s2 = (
                    f"{lei_pos.capitalize()} and {curve_pos} —"
                    " no material deterioration visible, and no signs the cycle has peaked."
                )
            elif lei_pos:
                s2 = (
                    f"{lei_pos.capitalize()}, with no indication the expansion is"
                    " approaching its peak rate of growth."
                )
            elif curve_pos:
                s2 = (
                    f"{curve_pos.capitalize()},"
                    " consistent with healthy credit conditions and the mid-cycle dynamic."
                )
            else:
                s2 = (
                    "No material deterioration is visible in the leading indicator picture,"
                    " suggesting the expansion continues on a stable footing."
                )
            s3 = _s3(
                "supports broad risk appetite; cyclicals and quality growth are the"
                " preferred expression, and multiple expansion remains possible"
                " while the cycle holds",
            )  # mid-cycle is rarely elevated; low_action suffices

        elif cycle_phase == "late":
            if regime == "Expansion":
                s1 = "The cycle is maturing, with peak-rate-of-growth behind us and late-cycle dynamics increasingly in place."
                if lei_pos and curve_pos:
                    s2 = (
                        f"While the backdrop remains constructive, {lei_pos} and"
                        f" {curve_pos} — classic late-cycle signals that typically"
                        " precede a growth deceleration."
                    )
                elif lei_pos:
                    s2 = (
                        f"While overall conditions remain supportive, {lei_pos} —"
                        " suggesting the cycle is past its peak rate of acceleration."
                    )
                elif curve_pos:
                    s2 = (
                        f"While growth remains constructive, {curve_pos} —"
                        " a late-cycle dynamic that warrants a more selective approach."
                    )
                else:
                    s2 = (
                        "Late-cycle dynamics are in place, with limited scope"
                        " for further multiple expansion from current levels."
                    )
                s3 = _s3(
                    "limits scope for further multiple expansion; favor quality,"
                    " balance sheet resilience, and pricing power over growth and leverage",
                    "quality, balance sheet resilience, and pricing power become"
                    " the key differentiators — the risk/reward of adding cyclical"
                    " exposure narrows materially from here",
                )

            else:  # Slowdown / late
                s1 = (
                    "Conditions are past their cyclical peak, with a late-cycle slowdown"
                    " compressing the runway for further risk-asset outperformance."
                )
                if lei_pos and curve_pos:
                    s2 = (
                        f"Both {lei_pos} and {curve_pos} —"
                        " suggesting conditions are past their best and further weakness"
                        " lies ahead before any stabilisation."
                    )
                elif lei_pos:
                    s2 = (
                        f"While no inversion is yet confirmed, {lei_pos} —"
                        " consistent with conditions past their cyclical peak."
                    )
                else:
                    s2 = (
                        "Leading indicators are deteriorating, consistent with"
                        " conditions past their cyclical peak and no near-term recovery visible."
                    )
                s3 = _s3(
                    "narrows risk appetite significantly; defensives and quality"
                    " outperform high-multiple growth names on a relative basis",
                    "risk appetite narrows significantly; defensives and quality"
                    " outperform high-multiple growth names on a relative basis",
                )

        elif cycle_phase == "contraction":
            s1 = (
                "An active contraction is underway, presenting a clear headwind"
                " to earnings and risk-asset valuations."
            )
            if lei_trend == "inflecting":
                s2 = (
                    "While leading indicators are beginning to show early signs of"
                    " stabilisation, the contraction has not yet run its course —"
                    " any inflection must be confirmed across multiple prints before"
                    " it is treated as a genuine trough signal."
                )
            elif lei_pos:
                s2 = (
                    f"{lei_pos.capitalize()} and no recovery signals have yet emerged,"
                    " suggesting the trough is not yet visible on the current data."
                )
            else:
                s2 = (
                    "Leading indicators are declining with no recovery signals yet"
                    " visible, suggesting the trough is not yet in sight."
                )
            s3 = _s3(
                "warrants defensive positioning: reduce cyclical exposure,"
                " emphasise balance sheet resilience, and treat any recovery signals"
                " as requiring confirmation before re-risking",
                "defensive positioning is warranted — reduce cyclical exposure,"
                " emphasise balance sheet resilience, and treat any recovery signals"
                " as requiring confirmation before re-risking",
            )

        else:
            # Fallback — should not reach in practice
            s1 = f"{regime} macro regime."
            s2 = ""
            s3 = ""

        return " ".join(p for p in [s1, s2, s3] if p)

    @staticmethod
    def _macro_verdict(regime: str, recession_risk: str) -> str:
        """
        Generic one-line verdict — used when cycle_phase is unknown
        (e.g. FRED not configured, trend data unavailable).
        """
        if regime in ("Expansion", "Recovery"):
            base = "The macro backdrop is supportive — conditions are broadly constructive"
        elif regime == "Slowdown":
            base = "The macro backdrop is softening — proceed with selectivity"
        else:  # Contraction
            base = "The macro backdrop is a headwind — defensively position the portfolio"

        if recession_risk in ("Elevated", "High"):
            return f"{base}; recession risk is {recession_risk.lower()}."
        return f"{base}."

    @staticmethod
    def _macro_context_line(macro: dict) -> "str | None":
        """
        One-sentence LEI-aware macro framing for the Final Verdict block.
        Format: "A [phase description] with [key signals][, though/qualifier][; risk note]."
        Returns None if no meaningful macro data is present.
        """
        regime = macro.get("macro_regime") or macro.get("regime", "")
        if not regime or regime == "Unknown":
            return None

        cycle_phase  = macro.get("cycle_phase")
        risk         = macro.get("recession_risk_level") or macro.get("recession_risk", "")
        lei_trend    = macro.get("lei_trend")
        spread_trend = macro.get("yield_spread_trend")

        risk_elevated = risk in ("Elevated", "High")
        risk_str = risk.lower() if risk else ""

        # ── Phase phrase — sentence-start form ────────────────────────────────
        if cycle_phase and cycle_phase not in ("unknown", None):
            if cycle_phase == "contraction":
                phase_str = "An active contraction"
            else:
                _art = "An" if cycle_phase[0].lower() in "aeiou" else "A"
                phase_str = f"{_art} {cycle_phase}-cycle {regime.lower()}"
        else:
            _art = "An" if regime and regime[0].lower() in "aeiou" else "A"
            phase_str = f"{_art} {regime.lower()} macro backdrop"

        # ── Signal + qualifier ────────────────────────────────────────────────
        if lei_trend == "rising" and spread_trend == "rising":
            signal    = "improving leading indicators and a steepening yield curve"
            qualifier = ""
        elif lei_trend == "rising" and spread_trend == "falling":
            signal    = "improving leading indicators"
            qualifier = ", though the yield curve is compressing"
        elif lei_trend == "rising":
            signal    = "improving leading indicators"
            qualifier = (
                ", though growth momentum remains below trend"
                if regime == "Recovery" else ""
            )
        elif lei_trend == "falling" and spread_trend == "falling":
            signal    = "fading leading indicators and a compressing yield curve"
            qualifier = ""
        elif lei_trend == "falling":
            signal    = "fading leading indicators"
            qualifier = ""
        elif lei_trend == "inflecting":
            signal    = "leading indicators at a potential inflection point"
            qualifier = ""
        elif spread_trend == "falling":
            signal    = "a compressing yield curve"
            qualifier = ""
        elif spread_trend == "rising":
            signal    = "a steepening yield curve"
            qualifier = ""
        else:
            signal    = None
            qualifier = ""

        # ── Risk tail ──────────────────────────────────────────────────────────
        risk_tail = f"; recession risk is {risk_str}" if risk_elevated else ""

        # ── Compose ───────────────────────────────────────────────────────────
        if signal:
            return f"{phase_str} with {signal}{qualifier}{risk_tail}."
        # No trend signals — fall back to regime-level implication
        if cycle_phase == "contraction" or regime == "Contraction":
            implied = "is a headwind; defensive positioning is warranted"
        elif cycle_phase == "late":
            implied = "is maturing; favour quality over growth"
        elif regime in ("Expansion", "Recovery"):
            implied = "is broadly constructive"
        else:
            implied = "is softening; proceed with selectivity"
        return f"{phase_str} {implied}{risk_tail}."

    @staticmethod
    def _build_valuation_range_section(
        vr: "ValuationRange | None",
        current_price: "float | None",
        growth_score: "float | None" = None,
        scenario_tree: "object | None" = None,
        analyst_target: "float | None" = None,
    ) -> list[str]:
        """
        Render a bear/base/bull valuation range table with PEG validation.

        Layout:
          Valuation Range
          ───────────────
            Current Price  : $XX.XX
            Multiple range : ±20% (Standard)
            EPS CAGR used  : 12.5% (bear=flat, base=+CAGR, bull=+130% CAGR)

            Primary Driver : P/E
                                    Bear        Base        Bull
                                    ─────────   ─────────   ─────────
            P/E multiple            20.0x       25.0x       30.0x
            EPS (1yr fwd)           $5.80       $6.53       $6.74
            Implied price           $116.00     $163.25     $202.20
            vs Current              -23%        +9%         +35%
            Supporting methods (base case only): EV/EBITDA 18.0x → $148.20

            ──────────────────  ─────────   ─────────   ─────────
            Aggregate           $120.00     $157.00     $195.00
            vs Current          -20%        +5%         +30%
            → Base case implies +5% vs current price — roughly fairly valued.

            PEG Ratio : 2.00x (25.0x ÷ 12.5%)
            → PEG 2.00 — slightly expensive relative to growth
        """
        lines: list[str] = [
            "  Fair Value Range  (primary; driver model)",
            "  ──────────────────────────────────────────",
        ]

        lines += [
            "  Note: bear/base/bull describe possible operating outcomes for the",
            "  business — NOT buy/hold/sell signals.  The ACTION at the top of",
            "  this report combines these with current price to give a decision.",
            "",
        ]

        if vr is None or vr.data_quality == "missing":
            lines.append("    Valuation range not computable — insufficient data.")
            lines.append("")
            return lines

        def _fp(v: "float | None") -> str:
            return f"${v:>7.2f}" if v is not None else "    N/A"

        def _upside(v: "float | None", cp: "float | None") -> str:
            if v is None or cp is None or cp == 0:
                return "   N/A"
            pct = (v - cp) / cp * 100
            sign = "+" if pct >= 0 else ""
            return f"{sign}{pct:.0f}%"

        price_str = f"${current_price:.2f}" if current_price else "N/A"
        lines.append(f"    Current Price : {price_str}")

        # ── Quality tier context ───────────────────────────────────────────────
        if vr.scenario_bear_mult is not None and vr.scenario_bull_mult is not None:
            _spread = round((1.0 - vr.scenario_bear_mult) * 100)
            _tier = "High quality (±15%)" if _spread == 15 else \
                    "Low quality (±25%)"  if _spread == 25 else \
                    f"Standard (±{_spread}%)"
            lines.append(f"    Multiple range : ±{_spread}% ({_tier})")
        if vr.scenario_growth_rate is not None:
            lines.append(
                f"    EPS CAGR used  : {vr.scenario_growth_rate:.1f}%"
                " (bear=flat, base=+CAGR, bull=+130% CAGR)"
            )
        lines.append("")

        # ── Primary method — full bear/base/bull driver table ─────────────────
        # Shows BOTH the multiple and the earnings metric used in each scenario.
        # "What assumptions lead to this price target?" is answered here.
        pm = vr.scenario_primary_method
        if pm == "driver" and vr.driver_model_available:
            lines.append(f"    Primary Driver : Fundamental (revenue → margin → FCF → exit multiple)")
            lines.append(f"    {'':20s}  {'Bear':>9}  {'Base':>9}  {'Bull':>9}")
            lines.append(f"    {'':20s}  {'─'*9}  {'─'*9}  {'─'*9}")
            def _pct(v):
                return f"{v:+.0%}" if v is not None else "N/A"
            def _marg(v):
                return f"{v:.1%}" if v is not None else "N/A"
            def _mult(v):
                return f"{v:.1f}x" if v is not None else "N/A"
            def _fcfb(v):
                return f"${v/1e9:.2f}B" if v is not None else "N/A"
            lines.append(
                f"    {'Revenue growth':<20s}"
                f"  {_pct(vr.scenario_bear_rev_growth):>9}"
                f"  {_pct(vr.scenario_base_rev_growth):>9}"
                f"  {_pct(vr.scenario_bull_rev_growth):>9}"
            )
            lines.append(
                f"    {'Op margin':<20s}"
                f"  {_marg(vr.scenario_bear_op_margin):>9}"
                f"  {_marg(vr.scenario_base_op_margin):>9}"
                f"  {_marg(vr.scenario_bull_op_margin):>9}"
            )
            lines.append(
                f"    {'FCF conversion':<20s}"
                f"  {_marg(vr.scenario_bear_fcf_conv):>9}"
                f"  {_marg(vr.scenario_base_fcf_conv):>9}"
                f"  {_marg(vr.scenario_bull_fcf_conv):>9}"
            )
            lines.append(
                f"    {'Exit multiple':<20s}"
                f"  {_mult(vr.scenario_bear_exit_mult):>9}"
                f"  {_mult(vr.scenario_base_exit_mult):>9}"
                f"  {_mult(vr.scenario_bull_exit_mult):>9}"
            )
            if vr.scenario_bear_fwd_fcf is not None:
                lines.append(
                    f"    {'Fwd FCF':<20s}"
                    f"  {_fcfb(vr.scenario_bear_fwd_fcf):>9}"
                    f"  {_fcfb(vr.scenario_base_fwd_fcf):>9}"
                    f"  {_fcfb(vr.scenario_bull_fwd_fcf):>9}"
                )
            lines.append(
                f"    {'Implied price':<20s}"
                f"  {_fp(vr.bear_price):>9}"
                f"  {_fp(vr.base_price):>9}"
                f"  {_fp(vr.bull_price):>9}"
            )
            lines.append(
                f"    {'vs Current':<20s}"
                f"  {_upside(vr.bear_price, current_price):>9}"
                f"  {_upside(vr.base_price, current_price):>9}"
                f"  {_upside(vr.bull_price, current_price):>9}"
            )
            if vr.scenario_bear_label:
                lines.append(f"    Bear : {vr.scenario_bear_label}")
            if vr.scenario_base_label:
                lines.append(f"    Base : {vr.scenario_base_label}")
            if vr.scenario_bull_label:
                lines.append(f"    Bull : {vr.scenario_bull_label}")
            # ── Trend Impact on Valuation ─────────────────────────────────────
            if getattr(vr, "trend_impact_lines", None):
                lines.append(f"    Trend Impact on Valuation:")
                for _til in vr.trend_impact_lines:
                    # Map to user-facing language
                    if "Expanding" in _til or "expanding" in _til:
                        _suffix = " → supporting valuation"
                    elif "Deteriorating" in _til or "deteriorating" in _til:
                        _suffix = " → limiting upside"
                    elif "volatile" in _til or "Volatile" in _til:
                        _suffix = " → widening uncertainty band"
                    elif "neutral" in _til:
                        _suffix = ""
                    else:
                        _suffix = ""
                    lines.append(f"    • {_til}{_suffix}")
        elif pm == "P/E" and vr.scenario_bear_pe is not None:
            lines.append(f"    Primary Driver : P/E")
            lines.append(f"    {'':18s}  {'Bear':>9}  {'Base':>9}  {'Bull':>9}")
            lines.append(f"    {'':18s}  {'─'*9}  {'─'*9}  {'─'*9}")
            lines.append(
                f"    {'P/E multiple':<18s}"
                f"  {vr.scenario_bear_pe:>8.1f}x"
                f"  {vr.scenario_base_pe:>8.1f}x"
                f"  {vr.scenario_bull_pe:>8.1f}x"
            )
            lines.append(
                f"    {'EPS (1yr fwd)':<18s}"
                f"  {'${:.2f}'.format(vr.scenario_bear_eps):>9}"
                f"  {'${:.2f}'.format(vr.scenario_base_eps):>9}"
                f"  {'${:.2f}'.format(vr.scenario_bull_eps):>9}"
            )
            lines.append(
                f"    {'Implied price':<18s}"
                f"  {_fp(vr.pe_bear):>9}"
                f"  {_fp(vr.pe_base):>9}"
                f"  {_fp(vr.pe_bull):>9}"
            )
            lines.append(
                f"    {'vs Current':<18s}"
                f"  {_upside(vr.pe_bear, current_price):>9}"
                f"  {_upside(vr.pe_base, current_price):>9}"
                f"  {_upside(vr.pe_bull, current_price):>9}"
            )
        elif pm == "EV/EBITDA" and vr.scenario_bear_ev is not None:
            lines.append(f"    Primary Driver : EV/EBITDA")
            lines.append(f"    {'':18s}  {'Bear':>9}  {'Base':>9}  {'Bull':>9}")
            lines.append(f"    {'':18s}  {'─'*9}  {'─'*9}  {'─'*9}")
            lines.append(
                f"    {'EV/EBITDA mult':<18s}"
                f"  {vr.scenario_bear_ev:>8.1f}x"
                f"  {vr.scenario_base_ev:>8.1f}x"
                f"  {vr.scenario_bull_ev:>8.1f}x"
            )
            if vr.scenario_ev_ebitda_val is not None:
                _eb = vr.scenario_ev_ebitda_val
                lines.append(
                    f"    {'EBITDA (flat)':<18s}"
                    f"  {'${:.1f}B'.format(_eb/1e9):>9}"
                    f"  {'${:.1f}B'.format(_eb/1e9):>9}"
                    f"  {'${:.1f}B'.format(_eb/1e9):>9}"
                )
            lines.append(
                f"    {'Implied price':<18s}"
                f"  {_fp(vr.ev_bear):>9}"
                f"  {_fp(vr.ev_base):>9}"
                f"  {_fp(vr.ev_bull):>9}"
            )
            lines.append(
                f"    {'vs Current':<18s}"
                f"  {_upside(vr.ev_bear, current_price):>9}"
                f"  {_upside(vr.ev_base, current_price):>9}"
                f"  {_upside(vr.ev_bull, current_price):>9}"
            )
        elif pm == "P/S" and vr.scenario_bear_ps is not None:
            lines.append(f"    Primary Driver : P/S (unprofitable / early-stage)")
            lines.append(f"    {'':18s}  {'Bear':>9}  {'Base':>9}  {'Bull':>9}")
            lines.append(f"    {'':18s}  {'─'*9}  {'─'*9}  {'─'*9}")
            lines.append(
                f"    {'P/S multiple':<18s}"
                f"  {vr.scenario_bear_ps:>8.1f}x"
                f"  {vr.scenario_base_ps:>8.1f}x"
                f"  {vr.scenario_bull_ps:>8.1f}x"
            )
            if vr.scenario_ps_rev_per_share is not None:
                _rps = vr.scenario_ps_rev_per_share
                lines.append(
                    f"    {'Rev/share (flat)':<18s}"
                    f"  {'${:.2f}'.format(_rps):>9}"
                    f"  {'${:.2f}'.format(_rps):>9}"
                    f"  {'${:.2f}'.format(_rps):>9}"
                )
            lines.append(
                f"    {'Implied price':<18s}"
                f"  {_fp(vr.ps_bear):>9}"
                f"  {_fp(vr.ps_base):>9}"
                f"  {_fp(vr.ps_bull):>9}"
            )
            lines.append(
                f"    {'vs Current':<18s}"
                f"  {_upside(vr.ps_bear, current_price):>9}"
                f"  {_upside(vr.ps_base, current_price):>9}"
                f"  {_upside(vr.ps_bull, current_price):>9}"
            )

        # ── Supporting methods — base-case reference only ─────────────────────
        secondary = []
        if pm != "P/E" and vr.pe_base is not None and vr.scenario_base_pe is not None:
            secondary.append(
                f"P/E {vr.scenario_base_pe:.1f}x → {_fp(vr.pe_base).strip()}"
            )
        if pm != "EV/EBITDA" and vr.ev_base is not None and vr.scenario_base_ev is not None:
            secondary.append(
                f"EV/EBITDA {vr.scenario_base_ev:.1f}x → {_fp(vr.ev_base).strip()}"
            )
        if pm != "P/S" and vr.ps_base is not None and vr.scenario_base_ps is not None:
            secondary.append(
                f"P/S {vr.scenario_base_ps:.1f}x → {_fp(vr.ps_base).strip()}"
            )
        if secondary:
            lines.append(f"    Supporting methods (base case only): {' | '.join(secondary)}")

        # ── Aggregate ─────────────────────────────────────────────────────────
        if any(v is not None for v in [vr.bear_price, vr.base_price, vr.bull_price]):
            lines.append(f"    {'─'*18}  {'─'*9}  {'─'*9}  {'─'*9}")
            lines.append(
                f"    {'Aggregate':<18s}"
                f"  {_fp(vr.bear_price):>9}"
                f"  {_fp(vr.base_price):>9}"
                f"  {_fp(vr.bull_price):>9}"
            )
            lines.append(
                f"    {'vs Current':<18s}"
                f"  {_upside(vr.bear_price, current_price):>9}"
                f"  {_upside(vr.base_price, current_price):>9}"
                f"  {_upside(vr.bull_price, current_price):>9}"
            )
        if vr.upside_context:
            lines.append(f"    → {vr.upside_context}")
        if (
            growth_score is not None
            and growth_score < 40
            and vr.base_price is not None
            and current_price is not None
            and vr.base_price > current_price
        ):
            lines.append(
                "    → Upside driven more by multiple expansion than earnings growth"
                " — execution risk is elevated."
            )
        lines.append("")

        # ── Comparison of valuation methods (Part 3) ─────────────────────────
        _dash = "   —"
        _st_bear = getattr(scenario_tree, "bear_price",  None) if scenario_tree else None
        _st_base = getattr(scenario_tree, "base_price",  None) if scenario_tree else None
        _st_bull = getattr(scenario_tree, "bull_price",  None) if scenario_tree else None
        # Determine which supporting prices exist
        _pe_b  = vr.pe_base   if vr.pe_base  is not None else None
        _ev_b  = vr.ev_base   if vr.ev_base  is not None else None
        _ps_b  = vr.ps_base   if vr.ps_base  is not None else None
        _any_supporting = any(v is not None for v in [_pe_b, _ev_b, _ps_b, _st_base, analyst_target])
        if _any_supporting or (vr.bear_price is not None):
            def _mc(v):
                return f"${v:>7.0f}" if v is not None else "    —  "
            lines.append("  Comparison of Valuation Methods")
            lines.append("  ─────────────────────────────────────────────────────────────")
            lines.append(f"    {'':28s}  {'Bear':>8}  {'Base':>8}  {'Bull':>8}")
            lines.append(f"    {'':28s}  {'──────':>8}  {'──────':>8}  {'──────':>8}")
            lines.append(
                f"    {'Driver model (primary)':<28s}"
                f"  {_mc(vr.bear_price):>8}"
                f"  {_mc(vr.base_price):>8}"
                f"  {_mc(vr.bull_price):>8}"
            )
            if _pe_b is not None:
                lines.append(
                    f"    {'P/E × forward EPS':<28s}"
                    f"  {'    —  ':>8}  {_mc(_pe_b):>8}  {'    —  ':>8}"
                )
            if _ev_b is not None:
                lines.append(
                    f"    {'EV/EBITDA × EBITDA':<28s}"
                    f"  {'    —  ':>8}  {_mc(_ev_b):>8}  {'    —  ':>8}"
                )
            if _ps_b is not None:
                lines.append(
                    f"    {'P/S × revenue':<28s}"
                    f"  {'    —  ':>8}  {_mc(_ps_b):>8}  {'    —  ':>8}"
                )
            if scenario_tree is not None:
                lines.append(
                    f"    {'Scenario tree':<28s}"
                    f"  {_mc(_st_bear):>8}"
                    f"  {_mc(_st_base):>8}"
                    f"  {_mc(_st_bull):>8}"
                )
            if analyst_target is not None:
                lines.append(
                    f"    {'Analyst consensus':<28s}"
                    f"  {'    —  ':>8}  {_mc(analyst_target):>8}  {'    —  ':>8}"
                )
            lines += [
                "",
                "  Notes:",
                "    • Driver model is StockEval's primary — sizing and ACTION use this.",
                "    • Supporting methods shown for context; they don't drive the recommendation.",
            ]
            # Divergence note if supporting methods deviate
            _divs = [v for v in [_pe_b, _ev_b, _ps_b] if v is not None]
            if _divs and vr.base_price is not None and vr.base_price > 0:
                _max_div = max(abs(v - vr.base_price) / vr.base_price for v in _divs)
                if _max_div >= 0.25:
                    lines.append(
                        f"    • Supporting methods diverge ≥{_max_div:.0%} from driver base"
                        " — see Methodology Note above for explanation."
                    )
            lines.append("")

        # ── Probability Distribution ──────────────────────────────────────────
        _mc_vr = getattr(vr, "mc", None)
        if _mc_vr is not None:
            lines.append("    Probability Distribution")
            lines.append("    ─────────────────────────────────────────────────")
            lines.append(
                f"    {'Method':<22s}: {_mc_vr.method} · {_mc_vr.n_sims:,} simulations"
            )
            lines.append(
                f"    {'Expected Return':<22s}: {_mc_vr.mean_return:+.1%}"
                f"   (median {_mc_vr.median_return:+.1%})"
            )
            lines.append(
                f"    {'P5 / P95 (range)':<22s}: {_mc_vr.p5_return:+.1%} "
                f"→ {_mc_vr.p95_return:+.1%}"
            )
            lines.append(
                f"    {'P25 / P75':<22s}: {_mc_vr.p25_return:+.1%} "
                f"→ {_mc_vr.p75_return:+.1%}"
            )
            lines.append(
                f"    {'P(gain)':<22s}: {_mc_vr.prob_positive:.0%}   "
                f"P(>20% gain): {_mc_vr.prob_20_gain:.0%}"
            )
            lines.append(
                f"    {'P(loss)':<22s}: {_mc_vr.prob_loss:.0%}   "
                f"P(>20% loss): {_mc_vr.prob_loss_20:.0%}"
            )
            _ud_label = (
                "strongly right-skewed" if _mc_vr.upside_downside >= 3.0 else
                "right-skewed"          if _mc_vr.upside_downside >= 2.0 else
                "roughly symmetric"     if _mc_vr.upside_downside >= 0.8 else
                "left-skewed"
            )
            lines.append(
                f"    {'Upside/Downside':<22s}: {_mc_vr.upside_downside:.2f}x  ({_ud_label})"
            )
            lines.append("")

        # ── Distribution Insights ─────────────────────────────────────────────
        # Explains WHAT DRIVES the spread using the driver scenario inputs.
        # Only generated for driver-model stocks (bear/base/bull inputs available).
        _mc_for_insights = getattr(vr, "mc", None)
        if (
            _mc_for_insights is not None
            and getattr(vr, "driver_model_available", False)
            and getattr(vr, "scenario_bear_rev_growth", None) is not None
        ):
            _p5i   = _mc_for_insights.p5_return
            _p95i  = _mc_for_insights.p95_return
            _eri   = _mc_for_insights.mean_return
            _sprd  = _p95i - _p5i

            # Width characterisation
            _width_label = (
                "very wide (high uncertainty)"  if _sprd > 0.80 else
                "wide (elevated uncertainty)"   if _sprd > 0.50 else
                "moderate"                      if _sprd > 0.25 else
                "tight (low uncertainty)"
            )

            lines.append("    Distribution Insights")
            lines.append("    ─────────────────────────────────────────────────")
            lines.append(f"    Outcome spread {_sprd:.0%} (P5 to P95) — {_width_label}")

            # Identify primary spread drivers by relative scenario range
            _bear_rev = getattr(vr, "scenario_bear_rev_growth", None)
            _base_rev = getattr(vr, "scenario_base_rev_growth", None)
            _bull_rev = getattr(vr, "scenario_bull_rev_growth", None)
            _bear_mg  = getattr(vr, "scenario_bear_op_margin",  None)
            _base_mg  = getattr(vr, "scenario_base_op_margin",  None)
            _bull_mg  = getattr(vr, "scenario_bull_op_margin",  None)
            _bear_ex  = getattr(vr, "scenario_bear_exit_mult",  None)
            _base_ex  = getattr(vr, "scenario_base_exit_mult",  None)
            _bull_ex  = getattr(vr, "scenario_bull_exit_mult",  None)

            _driver_ranges: list[tuple[str, float]] = []
            if _base_rev and _base_rev != 0 and _bear_rev is not None and _bull_rev is not None:
                _driver_ranges.append(("revenue growth", abs(_bull_rev - _bear_rev) / abs(_base_rev)))
            if _base_mg and _base_mg != 0 and _bear_mg is not None and _bull_mg is not None:
                _driver_ranges.append(("operating margin", abs(_bull_mg - _bear_mg) / abs(_base_mg)))
            if _base_ex and _base_ex != 0 and _bear_ex is not None and _bull_ex is not None:
                _driver_ranges.append(("exit multiple", abs(_bull_ex - _bear_ex) / abs(_base_ex)))

            if _driver_ranges:
                _primary_driver = max(_driver_ranges, key=lambda x: x[1])
                lines.append(
                    f"    Primary spread driver: {_primary_driver[0]}"
                    f" (±{_primary_driver[1]*50:.0f}pp relative range)"
                )

            # Downside driver characterisation
            if _bear_mg is not None and _base_mg is not None:
                _mg_drop = (_base_mg - _bear_mg) * 100
                if _mg_drop >= 2.0:
                    lines.append(
                        f"    Downside driven by margin compression"
                        f" ({_base_mg:.1%} → {_bear_mg:.1%} in bear case)"
                    )
            if _bear_rev is not None and _base_rev is not None:
                _rev_drop = (_base_rev - _bear_rev) * 100
                if _rev_drop >= 5.0:
                    lines.append(
                        f"    Bear revenue assumption −{_rev_drop:.0f}pp vs base"
                        f" reflects growth risk"
                    )

            # Upside driver characterisation
            if _bull_mg is not None and _base_mg is not None:
                _mg_lift = (_bull_mg - _base_mg) * 100
                if _mg_lift >= 1.5:
                    lines.append(
                        f"    Upside driven by operating leverage"
                        f" ({_base_mg:.1%} → {_bull_mg:.1%} in bull case)"
                    )
            if _bull_ex is not None and _base_ex is not None and _base_ex > 0:
                _mult_lift = (_bull_ex - _base_ex) / _base_ex * 100
                if _mult_lift >= 10:
                    lines.append(
                        f"    Bull multiple re-rate of {_mult_lift:.0f}%"
                        f" contributes to upside"
                    )

            # Skew interpretation
            if _mc_for_insights.upside_downside >= 2.0:
                lines.append(
                    f"    Distribution is right-skewed ({_mc_for_insights.upside_downside:.1f}×)"
                    f" — upside scenarios are proportionally larger than downside"
                )
            elif _mc_for_insights.upside_downside < 0.8:
                lines.append(
                    f"    Distribution is left-skewed ({_mc_for_insights.upside_downside:.1f}×)"
                    f" — downside scenarios outweigh upside in magnitude"
                )
            lines.append("")

        # ── Methodology comparison note (Fix G) ───────────────────────────────
        # Flag when driver-model base diverges materially from the P/E supporting method
        _drv_base = vr.base_price
        _pe_base  = vr.pe_base
        if (
            _drv_base is not None and _pe_base is not None
            and _pe_base > 0 and current_price is not None and current_price > 0
        ):
            _meth_div = abs(_drv_base - _pe_base) / _pe_base
            if _meth_div >= 0.30:
                _dir = "above" if _drv_base > _pe_base else "below"
                lines.append(
                    f"    ⚠  Methodology note: driver model base (${_drv_base:.0f}) is"
                    f" {_meth_div:.0%} {_dir} the P/E-implied base (${_pe_base:.0f})."
                )
                lines.append(
                    "       Driver model uses a DCF-style FCF scenario tree; P/E method applies"
                    " a multiple to forward EPS.  Large divergence suggests FCF/earnings"
                    " conversion or growth assumptions differ materially — review both."
                )
                lines.append("")
            elif _meth_div >= 0.25:
                _dir = "above" if _drv_base > _pe_base else "below"
                lines.append(
                    f"    ℹ  Driver model base (${_drv_base:.0f}) is {_meth_div:.0%} {_dir}"
                    f" P/E-implied base (${_pe_base:.0f}) — earnings-to-FCF conversion"
                    " may explain the gap."
                )
                lines.append("")

        # ── FIX 1: Exit multiple cap note ─────────────────────────────────────
        if getattr(vr, "exit_mult_capped", False):
            _raw_m   = getattr(vr, "exit_mult_raw",            None)
            _mkt_m   = getattr(vr, "exit_mult_market_implied", None)
            _mkt_str = f"~{_mkt_m:.0f}x" if _mkt_m else "high"
            _raw_str = f"{_raw_m:.0f}x" if _raw_m else "above 60x"
            lines += [
                f"    ⚠  Exit multiple capped: model used 60x (ceiling) vs"
                f" market-implied {_mkt_str} EV/FCF.",
                "       The 60x cap reflects a typical steady-state business.",
                "       For high-growth or recovery names where the market prices in FCF",
                "       normalization, this cap produces conservative bear/base prices.",
                "       Consider analyst consensus and management FCF guidance alongside.",
                "",
            ]

        # ── FIX 2: FCF conversion normalisation note ──────────────────────────
        if getattr(vr, "fcf_conv_normalized", False):
            _ttm_c = getattr(vr, "fcf_conv_ttm",       None)
            _med_c = getattr(vr, "fcf_conv_5y_median",  None)
            _ttm_s = f"{_ttm_c:.0%}" if _ttm_c else "—"
            _med_s = f"{_med_c:.0%}" if _med_c else "—"
            lines += [
                f"    ⚠  FCF conversion normalized: TTM conversion ({_ttm_s}) is below"
                f" 5Y median ({_med_s}).",
                "       Base case uses the 5Y median to reflect steady-state expectations.",
                "       Bear case uses the TTM trough (elevated capex / M&A integration).",
                "       Bull case assumes recovery beyond median.",
                "",
            ]

        # ── FIX 3: Trend window disagreement note ─────────────────────────────
        if getattr(vr, "op_margin_window_disagree", False):
            _full_t   = getattr(vr, "op_margin_full_trend",   "")
            _recent_t = getattr(vr, "op_margin_recent_trend", "")
            if _full_t and _recent_t:
                lines += [
                    f"    ℹ  Margin trend: full-window trend is {_full_t.lower()},"
                    f" but recent 3-year trend is {_recent_t.lower()}.",
                    "       Valuation uses the recent trend direction for the base-case margin.",
                    "",
                ]

        # ── FIX 5: Driver vs analyst consensus divergence ─────────────────────
        _drv_b_for_div = getattr(vr, "base_price", None)
        if (
            _drv_b_for_div is not None and _drv_b_for_div > 0
            and analyst_target is not None and analyst_target > 0
            and getattr(vr, "driver_model_available", False)
        ):
            _cons_div = abs(_drv_b_for_div - analyst_target) / analyst_target
            if _cons_div >= 0.50:
                lines += [
                    f"    ⚠  Driver model base (${_drv_b_for_div:.0f}) diverges from"
                    f" analyst consensus (${analyst_target:.0f}) by {_cons_div:.0%}.",
                    "       Possible reasons: differing FCF normalisation assumptions,",
                    "       forward EPS trajectory, or exit multiple expectations.",
                    "       Treat driver model, consensus, and market price as three",
                    "       independent data points — no single one is authoritative.",
                    "",
                ]

        # ── PEG ───────────────────────────────────────────────────────────────
        if vr.peg_ratio is not None:
            _pe_shown = vr.scenario_base_pe or vr.scenario_pe_multiple
            _g_shown  = vr.eps_growth_rate
            _peg_formula = (
                f" ({_pe_shown:.1f}x ÷ {_g_shown:.1f}%)" if (_pe_shown and _g_shown) else ""
            )
            lines.append(f"    PEG Ratio : {vr.peg_ratio:.2f}x{_peg_formula}")
            lines.append(f"    → {vr.peg_interpretation}")
        elif vr.peg_interpretation:
            lines.append(f"    PEG : N/A — {vr.peg_interpretation}")

        lines.append("")
        return lines

    @staticmethod
    def _build_position_sizing_section(
        sc: "Scorecard",
        macro: dict,
        beta: "float | None" = None,
        beta_reliable: bool = True,
        pe_val: "float | None" = None,
        val_range: "ValuationRange | None" = None,
        price: "float | None" = None,
        factor_profile: object = None,
    ) -> "tuple[list[str], dict]":
        """
        Portfolio-manager position sizing framework.

        Rating derives from score/stance — never overridden by sizing constraints.
        Overrides reduce SIZE only; Buy-rated stocks always use active entry language.

            Strong Buy (score ≥ 80, Bullish)  → 3–5%   "Build / Full Position"
            Buy        (score 65–79, Bullish)  → 2–3%   "Initiate / Add"
            Hold       (Neutral stance)        → 0.5–1.5% "Track / Neutral"
            Sell       (Bearish stance)        → 0%     "Avoid / Exit"

        Hard overrides reduce SIZE only:
            P/E > 100x              → cap 1.5% (starter position — still Buy language)
            P/E > 50x OR PEG > 2.0 → cap 2%
            Momentum < 35           → force staged entry
            Weak momentum or expensive val → step down one increment (floor: 1.5% for Buy)
            Limited upside < 15%   → step down one increment
            ≥ 3 risk flags          → cap 2%
            Beta > 1.5              → step down one increment

        Returns (memo_lines, sizing_dict).
        sizing_dict keys:
            position_range, position_lo, position_hi, position_size
            entry_strategy, entry_detail, rationale
            conviction_tier, setup_quality, hard_cap_reason, rating
        """
        # Standard position increments — all outputs snap to these values.
        _INCREMENTS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

        def _snap(v: float) -> float:
            return min(_INCREMENTS, key=lambda x: abs(x - v))

        def _fmt(v: float) -> str:
            return f"{int(v)}%" if v == int(v) else f"{v:.1f}%"

        def _score(attr: str) -> "float | None":
            cat = getattr(sc, attr, None)
            return cat.score if (cat is not None and getattr(cat, "data_quality", "good") != "missing") else None

        regime         = macro.get("macro_regime") or "Unknown"
        recession_risk = macro.get("recession_risk_level") or "Unknown"
        flag_count     = len(sc.risk_flags)
        stance         = sc.stance
        overall_score  = sc.overall_score
        peg_ratio      = val_range.peg_ratio if val_range else None

        val_score = _score("valuation")
        mom_score = _score("momentum")

        # ── Sell: no position ─────────────────────────────────────────────────
        if stance == Stance.BEARISH:
            sizing = {
                "position_range":  "0%",
                "position_lo":     0.0,
                "position_hi":     0.0,
                "position_size":   0.0,
                "entry_strategy":  "No position",
                "entry_detail":    "Sell-rated — do not initiate. Revisit if valuation corrects or fundamentals improve materially.",
                "rationale":       "Downside risk outweighs potential upside at current prices.",
                "conviction_tier":     "none",
                "setup_quality":       "adverse",
                "hard_cap_reason":     None,
                "rating":              "Sell",
                "core_compounder_tag": None,
            }
            return (
                [
                    "  Position Sizing Guidance",
                    "  ────────────────────────",
                    "    Position Size    : 0%  (Sell — do not initiate)",
                    "    Entry Strategy   : No position",
                    "    Rationale        : Downside risk outweighs potential upside at current prices.",
                    "",
                ],
                sizing,
            )

        # ── Step 1: Base rating — score/stance driven, immutable ──────────────
        if stance == Stance.BULLISH and overall_score >= 80:
            base_rating = "Strong Buy"
        elif stance == Stance.BULLISH:
            base_rating = "Buy"
        else:
            base_rating = "Hold"  # Neutral stance

        # ── Step 2: Setup quality (momentum + valuation + macro) ─────────────
        if mom_score is not None:
            mom_q = "strong" if mom_score >= 65 else ("weak" if mom_score < 40 else "neutral")
        else:
            mom_q = "neutral"

        if val_score is not None:
            val_q = "cheap" if val_score >= 65 else ("expensive" if val_score < 40 else "fair")
        else:
            val_q = "fair"

        macro_weak       = regime == "Contraction" or recession_risk in ("Elevated", "High")
        macro_supportive = not macro_weak and recession_risk not in ("Moderate",)
        macro_q = "weak" if macro_weak else ("supportive" if macro_supportive else "neutral")

        positives = (mom_q == "strong") + (val_q in ("cheap", "fair")) + (macro_q == "supportive")
        negatives = (mom_q == "weak")   + (val_q == "expensive")        + (macro_q == "weak")

        if positives >= 2 and negatives == 0:
            setup = "strong"
        elif negatives >= 2 or (negatives >= 1 and positives == 0):
            setup = "weak"
        else:
            setup = "neutral"

        # ── Step 3: Base size — MC probability formula OR rating×setup table ────
        # When MC is available for Buy/Strong Buy stocks, the base size is driven
        # by the MC output distribution rather than the static lookup table.
        #
        # Formula:
        #   Risk_Adjusted_Score (RAS) = E[R] / |P5|
        #       — reward-to-downside ratio; >1 means upside exceeds downside mag.
        #   Conviction_Norm          = conviction_score / 100   (from MC shape)
        #   Raw_Size                 = RAS × Conviction_Norm × 5.0
        #       — anchored to 5% max; typical outputs land 1–4% for good risk/reward
        #   target = clamp(Raw_Size, BUY_FLOOR, rating_ceiling)
        #
        # Falls back to the rating×setup table when MC is absent (unprofitable
        # tickers that skip the driver model) or the rating is Hold/Sell.

        _BUY_FLOOR   = 0.0   # No artificial floor — let sizing math be honest
        _HOLD_FLOOR  = 0.0   # No artificial floor — Hold can size to 0%

        _size_table: dict[tuple[str, str], float] = {
            ("Strong Buy", "strong"):  5.0,
            ("Strong Buy", "neutral"): 4.0,
            ("Strong Buy", "weak"):    3.0,
            ("Buy",        "strong"):  3.0,
            ("Buy",        "neutral"): 2.0,
            ("Buy",        "weak"):    2.0,
            ("Hold",       "strong"):  1.5,
            ("Hold",       "neutral"): 1.0,
            ("Hold",       "weak"):    0.5,
        }

        _mc_pre          = getattr(val_range, "mc", None) if val_range else None
        _mc_profile_pre: "object | None" = None   # cached; reused by Step 4b
        _mc_base_method  = "table"                # audit trail for rationale

        if _mc_pre is not None and base_rating in ("Strong Buy", "Buy"):
            try:
                from analysis.monte_carlo import distribution_profile as _dp_fn
                _mc_profile_pre = _dp_fn(_mc_pre)

                _er_pre  = _mc_pre.mean_return      # E[R]
                _p5_pre  = _mc_pre.p5_return        # downside percentile
                _p95_pre = _mc_pre.p95_return       # upside percentile

                # Risk-Adjusted Score: E[R] per unit of downside magnitude
                # Floor the denominator at 1% so zero-downside stocks don't blow up.
                _downside_mag = abs(_p5_pre) if _p5_pre < 0 else 0.01
                _ras = _er_pre / _downside_mag
                _ras = max(0.05, min(2.5, _ras))   # clamp to sensible range

                # Conviction normalised from MC distribution shape [0, 1]
                _conv_norm = _mc_profile_pre.conviction_score / 100.0  # type: ignore[union-attr]

                # Core formula: RAS × conviction × max_portfolio_size
                _raw_size = _ras * _conv_norm * 5.0

                # Rating-tier ceiling: Strong Buy may size to 5%; Buy capped at 3%
                _rating_ceil = 5.0 if base_rating == "Strong Buy" else 3.0
                target = max(_BUY_FLOOR, min(_rating_ceil, _raw_size))

                _mc_base_method = "formula"
                print(
                    f"  [MC:formula] E[R]={_er_pre:+.1%} P5={_p5_pre:.1%}"
                    f" P95={_p95_pre:+.1%} RAS={_ras:.2f}"
                    f" conv={_conv_norm:.2f} raw={_raw_size:.2f}%"
                    f" → target={target:.2f}%"
                )
            except Exception as _fmla_err:
                print(f"  [MC:formula] skipped ({_fmla_err}) — using table")
                target = _size_table.get((base_rating, setup), 2.0)
        else:
            target = _size_table.get((base_rating, setup), 2.0)

        # ── Step 4: Hard overrides — size only, never rating ─────────────────
        hard_cap_reason: "str | None" = None
        force_staged = False
        size_reduced  = False

        # Extreme valuation cap
        if pe_val is not None and pe_val > 100:
            target = min(target, 1.5)
            hard_cap_reason = f"Capped due to extreme valuation (P/E {pe_val:.0f}x)"
            force_staged = True
            size_reduced  = True
        elif (pe_val is not None and pe_val > 50) or (peg_ratio is not None and peg_ratio > 2.0):
            target = min(target, 2.0)
            parts = []
            if pe_val is not None and pe_val > 50:
                parts.append(f"P/E {pe_val:.0f}x")
            if peg_ratio is not None and peg_ratio > 2.0:
                parts.append(f"PEG {peg_ratio:.1f}x")
            hard_cap_reason = f"Capped due to elevated valuation ({', '.join(parts)})"
            size_reduced    = True

        # Strongly negative momentum → staged entry
        if mom_score is not None and mom_score < 35:
            force_staged = True

        # Base-case upside
        base_upside_pct: "float | None" = None
        if val_range and val_range.base_price and price and price > 0:
            base_upside_pct = (val_range.base_price - price) / price * 100

        # Weak momentum / expensive valuation / limited upside → step down one increment
        # (Reduces size but preserves rating.  Floor enforced per rating band.)
        _floor = _BUY_FLOOR if base_rating in ("Strong Buy", "Buy") else _HOLD_FLOOR

        def _step_down(current: float) -> float:
            _snapped = _snap(current)
            _idx = _INCREMENTS.index(_snapped)
            return max(_INCREMENTS[max(0, _idx - 1)], _floor)

        if mom_q == "weak" and not hard_cap_reason:
            target = _step_down(target)
            hard_cap_reason = "Reduced due to weak momentum"
            size_reduced = True

        if val_q == "expensive" and not hard_cap_reason:
            target = _step_down(target)
            hard_cap_reason = "Capped due to premium valuation"
            size_reduced = True

        # MC override: when the full return distribution shows E[R] >= 15%, the
        # scenario base-case point estimate is too conservative to use as a cap.
        # Never say "limited upside" when the distribution is bullish.
        _mc_er_override = False
        _mc_pre_check = getattr(val_range, "mc", None) if val_range else None
        if _mc_pre_check is not None and _mc_pre_check.mean_return >= 0.15:
            _mc_er_override = True

        if (
            base_upside_pct is not None
            and base_upside_pct < 15.0
            and not hard_cap_reason
            and not _mc_er_override
        ):
            target = _step_down(target)
            hard_cap_reason = (
                f"Reduced: scenario base upside {base_upside_pct:.0f}% below hurdle"
                f" — distribution-based E[R] also below 15%"
            )
            size_reduced = True

        # Risk flags cap
        if flag_count >= 3:
            target = min(target, max(2.0, _floor))
            if hard_cap_reason is None:
                hard_cap_reason = f"Capped due to {flag_count} active risk flags"
                size_reduced = True

        # High-beta shave — step down one increment (skip when beta is unreliable)
        if beta_reliable and beta is not None and beta > 1.5:
            target = _step_down(target)

        # ── Step 4b: MC quality-adjusted tail caps ────────────────────────────
        # When the formula was used in Step 3, conviction is already IN the base
        # size — so this step's role is quality-adjusted P5 capping only.
        # Reuses _mc_profile_pre (computed in Step 3) to avoid duplicate work.
        mc_adjustment_reason: "str | None" = None
        _risk_class = None   # initialised here so Step 4c can always reference it
        _mc = getattr(val_range, "mc", None) if val_range else None
        if _mc is not None and base_rating in ("Strong Buy", "Buy"):
            try:
                from analysis.monte_carlo import (
                    distribution_profile as _dp,
                    classify_downside_risk as _cdr,
                )
                # Reuse cached profile from Step 3; fall back to fresh computation
                _profile    = _mc_profile_pre if _mc_profile_pre is not None else _dp(_mc)
                _prof_score = _score("profitability") or 50.0
                _hlth_score = _score("financial_health") or 50.0
                _flag_count = len(sc.risk_flags) if sc.risk_flags else 0
                _risk_class = _cdr(_mc, _prof_score, _hlth_score, _flag_count)
            except Exception:
                _profile    = None
                _risk_class = None

            if _profile is not None:
                _prev_target = target
                _ceil = 5.0 if base_rating == "Strong Buy" else 3.0

                # Apply continuous size adjustment.
                # Step-ups require: no prior cap AND a nonzero base (Kelly said
                # something — don't manufacture a position from nothing).
                _adj = _profile.size_adjustment
                if _adj > 0 and not hard_cap_reason and target > 0:
                    target = min(target + _adj, _ceil)
                elif _adj < 0:
                    target = max(target + _adj, _floor)

                # Quality-adjusted P5 tail caps — replace profile.size_cap with
                # risk_classification.size_cap when available; always override upward
                _eff_cap = (
                    _risk_class.size_cap
                    if _risk_class is not None
                    else _profile.size_cap
                )
                if _eff_cap < float("inf") and target > _eff_cap:
                    target = _eff_cap
                    if not hard_cap_reason:
                        if _risk_class is not None:
                            hard_cap_reason = (
                                f"Capped at {_fmt(_eff_cap)} — "
                                f"{_risk_class.cap_source}"
                            )
                        else:
                            _p5_pct = f"{_mc.p5_return * 100:.0f}%"
                            hard_cap_reason = (
                                f"Capped at {_fmt(_eff_cap)} — "
                                f"tail risk (P5≈{_p5_pct})"
                            )
                    size_reduced = True

                # Log conviction and risk classification for audit
                _rc_info = (
                    f" risktype={_risk_class.risk_type}"
                    f" qtier={_risk_class.quality_tier}"
                    f" effcap={_eff_cap}"
                    if _risk_class is not None else ""
                )
                print(
                    f"  [MC:profile] conviction={_profile.conviction_score:.0f}"
                    f" ({_profile.conviction_tier})"
                    f" return={_profile.return_score:.0f}"
                    f" skew={_profile.skew_score:.0f}"
                    f" risk={_profile.risk_score:.0f}"
                    f" iqr={_profile.iqr:.0%}"
                    f" width={_profile.width_tier}"
                    f" adj={_adj:+.2f}pp"
                    + _rc_info
                )

                # Rationale — prefer risk classification explanation when it
                # was the binding constraint (cap tightened or relaxed vs default)
                if target != _prev_target:
                    if (
                        _risk_class is not None
                        and _eff_cap != _profile.size_cap
                        and target == _eff_cap
                    ):
                        mc_adjustment_reason = _risk_class.explanation
                    else:
                        mc_adjustment_reason = _profile.rationale

        # ── Step 4b2: Explicit distribution-driven adjustments ───────────────
        # Applied AFTER conviction cap (Step 4b) so they act as specific
        # threshold gates on top of the formula-derived or quality-capped target.
        # Each rule is auditable, references E[R]/P5/P95, and sets a clear reason.
        if _mc is not None and base_rating in ("Strong Buy", "Buy"):
            _er_mc   = _mc.mean_return
            _p5_mc   = _mc.p5_return
            _p95_mc  = _mc.p95_return
            _range_mc = _p95_mc - _p5_mc
            _ceil_mc  = 5.0 if base_rating == "Strong Buy" else 3.0

            _spread_pp_mc = _range_mc * 100

            # Rule 1: High downside tail — P5 < -25% → reduce size significantly
            if _p5_mc < -0.25 and not hard_cap_reason:
                _prev_t = target
                target  = max(_step_down(_step_down(target)), _floor)
                if target != _prev_t:
                    hard_cap_reason = (
                        f"High downside tail (P5 {_p5_mc:.0%}) limits size"
                        f" — E[R] {_er_mc:+.0%}, P95 {_p95_mc:+.0%},"
                        f" {_spread_pp_mc:.0f}pp spread"
                    )
                    size_reduced = True
                    print(f"  [4b2:rule1] P5={_p5_mc:.1%} < -25% → {target:.2f}%")

            # Rule 2: Below-hurdle expected return — E[R] < 15% → reduce size
            elif _er_mc < 0.15 and not hard_cap_reason:
                _prev_t = target
                target  = _step_down(target)
                if target != _prev_t:
                    hard_cap_reason = (
                        f"Below-hurdle expected return (E[R] {_er_mc:+.0%})"
                        f" with {_spread_pp_mc:.0f}pp spread"
                        f" (P5 {_p5_mc:.0%} / P95 {_p95_mc:+.0%})"
                    )
                    size_reduced = True
                    print(f"  [4b2:rule2] E[R]={_er_mc:.1%} < 15% → {target:.2f}%")

            # Rule 3: Asymmetric upside with controlled downside → increase size
            elif _p95_mc > 0.50 and _p5_mc > -0.25 and not hard_cap_reason:
                _prev_t = target
                target  = min(target + 0.5, _ceil_mc)
                _ud_ratio_mc = _p95_mc / abs(_p5_mc) if _p5_mc != 0 else float("inf")
                if target != _prev_t:
                    mc_adjustment_reason = (
                        f"Asymmetric upside ({_ud_ratio_mc:.1f}× upside/downside)"
                        f" with controlled tail — E[R] {_er_mc:+.0%},"
                        f" P5 {_p5_mc:.0%} / P95 {_p95_mc:+.0%}"
                    )
                    print(f"  [4b2:rule3] P95={_p95_mc:.1%} P5={_p5_mc:.1%} ud={_ud_ratio_mc:.1f}× → {target:.2f}%")

            # Rule 4: Wide outcome distribution → uncertainty penalty
            if _range_mc > 0.70 and not hard_cap_reason and not size_reduced:
                _prev_t = target
                target  = _step_down(target)
                if target != _prev_t:
                    hard_cap_reason = (
                        f"Wide outcome distribution ({_spread_pp_mc:.0f}pp spread,"
                        f" P5 {_p5_mc:.0%} / P95 {_p95_mc:+.0%})"
                        f" — high dispersion warrants reduced sizing"
                    )
                    size_reduced = True
                    print(f"  [4b2:rule4] range={_range_mc:.1%} → {target:.2f}%")

        # ── Step 4c: Core Compounder minimum floor ───────────────────────────
        # High-quality businesses (top-quartile margins, strong balance sheet,
        # durable revenue model, no structural risk) must maintain at least
        # 1.5% even when momentum/valuation step-downs push below.
        # Overrides still apply: PE > 100x and structural risk disqualify.
        core_compounder_tag: "str | None" = None
        if base_rating in ("Strong Buy", "Buy"):
            try:
                from analysis.monte_carlo import classify_core_compounder as _ccc
                _grw_score = _score("growth")
                _rt = _risk_class.risk_type if _risk_class is not None else "mixed"
                _cc = _ccc(
                    profitability_score = _score("profitability") or 50.0,
                    health_score        = _score("financial_health") or 50.0,
                    growth_score        = _grw_score,
                    flag_count          = flag_count,
                    risk_type           = _rt,
                    pe_val              = pe_val,
                )
            except Exception:
                _cc = None

            if _cc is not None and _cc.is_core_compounder and _cc.floor_size > 0:
                if target < _cc.floor_size:
                    target = _cc.floor_size
                    _cc.floor_applied = True
                    core_compounder_tag = _cc.tag
                    # Promote rationale — floor overrides prior step-down reason
                    hard_cap_reason = None  # clear step-down cap; floor wins
                    mc_adjustment_reason = _cc.explanation
                    size_reduced = False    # floor raised it, not reduced

                print(
                    f"  [CC] is_core_compounder={_cc.is_core_compounder}"
                    f" floor={_cc.floor_size}%"
                    f" floor_applied={_cc.floor_applied}"
                    f" extreme_val_cap={_cc.extreme_val_cap}"
                )
            elif _cc is not None and not _cc.is_core_compounder:
                print(
                    f"  [CC] not qualified: {'; '.join(_cc.criteria_failed)}"
                )

        # ── Step 4d: Factor composite override ───────────────────────────────
        # Uses the cross-sectional 7-factor composite (regime + archetype
        # adjusted) to apply three tactical overlays:
        #
        #   i.  Momentum / profitability tension cap  — momentum z ≥ 75-score
        #       with profitability score < 45 → tactical cap at 2%.
        #   ii. Cyclicality recession cap — macro_score < 30 in a downturn
        #       regime → tactical cap at 1.5%.
        #   iii. Weak composite → force setup to "weak" (no size increase).
        #
        # None of these change the fundamental rating — sizes only.
        if factor_profile is not None:
            try:
                _fp_mom_score = getattr(factor_profile, "macro_score", None)   # reuse below
                _fp_composite = getattr(factor_profile, "composite_score", None)
                _fp_mom_z_score = (
                    50.0 + getattr(factor_profile, "momentum_z", 0.0) * 15.0
                )   # quick linear proxy (CDF not needed for threshold comparison)
                _fp_prof_score  = getattr(factor_profile, "profitability_score", None)
                _fp_macro_score = getattr(factor_profile, "macro_score", None)
                _fp_macro_regime_key = (
                    (macro or {}).get("macro_regime") or "Unknown"
                ).lower()
                _in_downturn = any(
                    kw in _fp_macro_regime_key
                    for kw in ("recession", "slowdown", "contraction", "late_cycle", "late cycle")
                )

                # i. Momentum / profitability tension
                if (
                    _fp_mom_z_score >= 75
                    and _fp_prof_score is not None
                    and _fp_prof_score < 45
                    and not hard_cap_reason
                ):
                    _new_cap = 2.0
                    if target > _new_cap:
                        target          = _new_cap
                        hard_cap_reason = (
                            f"Tactical cap at {_fmt(_new_cap)} — momentum "
                            f"(z-score={_fp_mom_z_score:.0f}) ahead of profitability "
                            f"({_fp_prof_score:.0f}/100)"
                        )
                        size_reduced = True
                    print(
                        f"  [4d:mom/prof] mom_score={_fp_mom_z_score:.0f}"
                        f" prof_score={_fp_prof_score:.0f}"
                        f" → cap applied={target > _new_cap or hard_cap_reason is not None}"
                    )

                # ii. Cyclicality recession cap
                if (
                    _fp_macro_score is not None
                    and _fp_macro_score < 30
                    and _in_downturn
                    and not hard_cap_reason
                ):
                    _new_cap = 1.5
                    if target > _new_cap:
                        target          = _new_cap
                        hard_cap_reason = (
                            f"Tactical cap at {_fmt(_new_cap)} — high cyclical "
                            f"exposure (macro_score={_fp_macro_score:.0f}/100) "
                            f"in downturn regime"
                        )
                        size_reduced = True
                    print(
                        f"  [4d:cyclical] macro_score={_fp_macro_score:.0f}"
                        f" in_downturn={_in_downturn}"
                    )

                # iii. Weak composite → force setup to weak (suppress size step-ups)
                if (
                    _fp_composite is not None
                    and _fp_composite < 38
                    and setup != "adverse"
                ):
                    setup = "weak"
                    print(
                        f"  [4d:composite] composite={_fp_composite:.0f}"
                        f" → setup forced to 'weak'"
                    )

            except Exception as _4d_err:
                print(f"  [4d] factor override skipped: {_4d_err}")

        # ── Step 4e: Scenario tree sizing gates ──────────────────────────────
        # Applies position caps and floors derived from the 4×3×3 scenario tree:
        #   · Wide dispersion (std > 40%)       → cap 1.5%
        #   · High downside mass (> 35%)        → cap 1.5%
        #   · High concentration (>70% in top3) → cap 2% (binary risk)
        #   · Worst-case return < −50%          → hard cap 1.5%
        #   · Asymmetric upside + low downside  → eligible for floor lift
        # None of these change the fundamental rating — sizes only.
        if factor_profile is not None or True:   # always run when scenario_tree available
            _st_arg = None
            # scenario_tree is captured in the outer _build_memo scope;
            # pass it in via the factor_profile closure workaround via caller kwarg.
            # It is injected as _scenario_tree_ref by the caller below.
            _st_arg = getattr(ReportingAgent._build_position_sizing_section, "_scenario_tree_ref", None)

        # Inline scenario tree gate using the closure variable scenario_tree_ref
        # (set by the caller immediately before calling this function)
        _st_sizing = getattr(ReportingAgent._build_position_sizing_section, "_scenario_tree_ref", None)
        if _st_sizing is not None:
            try:
                _std   = getattr(_st_sizing, "scenario_std",   None)
                _dm    = getattr(_st_sizing, "downside_mass",  None)
                _c3    = getattr(_st_sizing, "concentration_3", None)
                _wc    = getattr(_st_sizing, "worst_case",     None)
                _up    = getattr(_st_sizing, "upside_mass",    None)
                _wc_r  = getattr(_wc, "expected_return", None) if _wc is not None else None

                # Wide dispersion gate
                if _std is not None and _std > 0.40 and not hard_cap_reason:
                    _cap = 1.5
                    if target > _cap:
                        target = _cap
                        hard_cap_reason = (
                            f"Capped at {_fmt(_cap)} — wide scenario dispersion "
                            f"(std={_std:.0%})"
                        )
                        size_reduced = True
                    print(f"  [4e:dispersion] std={_std:.0%} → cap={_cap}%")

                # High downside mass gate
                if _dm is not None and _dm > 0.35 and not hard_cap_reason:
                    _cap = 1.5
                    if target > _cap:
                        target = _cap
                        hard_cap_reason = (
                            f"Capped at {_fmt(_cap)} — high probability of >−20% loss "
                            f"({_dm:.0%} downside mass)"
                        )
                        size_reduced = True
                    print(f"  [4e:downside_mass] dm={_dm:.0%} → cap={_cap}%")

                # Binary risk concentration gate
                if _c3 is not None and _c3 > 0.70 and not hard_cap_reason:
                    _cap = 2.0
                    if target > _cap:
                        target = _cap
                        hard_cap_reason = (
                            f"Capped at {_fmt(_cap)} — binary risk "
                            f"(top-3 scenarios = {_c3:.0%} of probability)"
                        )
                        size_reduced = True
                    print(f"  [4e:concentration] c3={_c3:.0%} → cap={_cap}%")

                # Worst-case return hard floor
                if _wc_r is not None and _wc_r < -0.50 and not hard_cap_reason:
                    _cap = 1.5
                    if target > _cap:
                        target = _cap
                        hard_cap_reason = (
                            f"Capped at {_fmt(_cap)} — extreme bear case "
                            f"({_wc_r:.0%} in worst scenario)"
                        )
                        size_reduced = True
                    print(f"  [4e:worst_case] wc_r={_wc_r:.0%} → cap={_cap}%")

                # Asymmetric upside floor lift (only when no prior cap applied)
                if (
                    _up is not None and _dm is not None
                    and _up > 0.40 and _dm < 0.12
                    and not hard_cap_reason
                    and base_rating in ("Strong Buy", "Buy")
                ):
                    _floor_lift = 1.5
                    if target < _floor_lift:
                        target = _floor_lift
                        mc_adjustment_reason = (
                            f"Floor lifted to {_fmt(_floor_lift)} — asymmetric upside "
                            f"({_up:.0%} probability of >+20% return vs "
                            f"{_dm:.0%} probability of >−20% loss)"
                        )
                    print(
                        f"  [4e:asym_upside] up={_up:.0%} dm={_dm:.0%}"
                        f" → floor_lift applied={target >= _floor_lift}"
                    )

            except Exception as _4e_err:
                print(f"  [4e] scenario tree gate skipped: {_4e_err}")

        # ── Step 4f: Divergence cap ───────────────────────────────────────────
        # When the scenario tree E[R] and the factor-model E[R] diverge by
        # > 12% the pipeline flags this as "significant".  Both models are
        # retained but sizing confidence is low — cap at 1.5% until the
        # analyst resolves the macro regime disagreement.
        _div_label = getattr(
            ReportingAgent._build_position_sizing_section, "_divergence_label_ref", None
        )
        _model_disagreement_flag: bool = False
        if _div_label == "significant":
            _model_disagreement_flag = True
            if not hard_cap_reason:
                _cap = 1.5
                if target > _cap:
                    target = _cap
                    hard_cap_reason = (
                        f"Capped at {_fmt(_cap)} — model disagreement detected: "
                        f"scenario E[R] and factor model E[R] diverge > 12 pp. "
                        f"Reduce conviction until models converge."
                    )
                    size_reduced = True
            print(f"  [4f:divergence] label={_div_label!r} → cap applied={size_reduced}")

        # ── Step 4g: Data coverage cap ────────────────────────────────────────
        # When the data integrity engine reports low data coverage (<70%),
        # we cannot trust the model outputs enough to take a full position.
        # Surface as an explicit sizing gate, not a silent signal reduction.
        _val_ref = getattr(ReportingAgent._build_position_sizing_section, "_validation_ref", None)
        if _val_ref is not None and not hard_cap_reason:
            try:
                _dcov         = getattr(_val_ref, "data_coverage",  1.0) or 1.0
                _dstatus      = getattr(_val_ref, "output_status",  "OK") or "OK"
                if _dcov < 0.70 or _dstatus == "LOW DATA CONFIDENCE":
                    _cap = 1.5
                    if target > _cap:
                        target = _cap
                        _cov_pct = f"{_dcov:.0%}"
                        hard_cap_reason = (
                            f"Capped at {_fmt(_cap)} — data coverage {_cov_pct} "
                            f"({_dstatus}). "
                            f"Too many key metrics unresolved to size confidently. "
                            f"Re-evaluate when data quality improves."
                        )
                        size_reduced = True
                    print(f"  [4g:data_coverage] cov={_dcov:.0%} status={_dstatus!r} → cap applied={size_reduced}")
            except Exception as _4g_err:
                print(f"  [4g] data coverage gate skipped: {_4g_err}")

        # ── Step 4h: Risk score gate ──────────────────────────────────────────
        # A low risk score (< 65/100) indicates structural financial risk,
        # business model fragility, or market risk that warrants a smaller
        # position regardless of upside conviction.
        # Applied as a 20% size reduction (not a hard cap) so conviction
        # still contributes but risk gets a vote.
        if not hard_cap_reason:
            try:
                _risk_cat   = getattr(sc, "risk", None)
                _risk_score = getattr(_risk_cat, "score", None) if _risk_cat else None
                if _risk_score is not None and _risk_score < 65:
                    _risk_reduction = 0.20 if _risk_score >= 50 else 0.35
                    _new_target = round(target * (1.0 - _risk_reduction), 2)
                    if _new_target < target:
                        target = _new_target
                        mc_adjustment_reason = (
                            f"Position reduced by {_risk_reduction:.0%} — "
                            f"risk score {_risk_score:.0f}/100 (< 65) indicates "
                            f"structural financial or business risk warrants smaller exposure."
                        )
                        size_reduced = True
                    print(f"  [4h:risk_score] score={_risk_score:.0f} → reduction={_risk_reduction:.0%} target={target:.2f}%")
            except Exception as _4h_err:
                print(f"  [4h] risk score gate skipped: {_4h_err}")

        # Snap to standard increment
        target = _snap(target)

        # ── Step 5: Entry strategy — Buy always uses active language ──────────
        # A "Buy" rated stock must NEVER use "Tracking position" language.
        # Build a brief distribution suffix when MC is available.
        _mc_ed = getattr(val_range, "mc", None) if val_range else None
        if _mc_ed is not None:
            _ed_spread = (_mc_ed.p95_return - _mc_ed.p5_return) * 100
            _ed_ud     = getattr(_mc_ed, "upside_downside", None)
            _ed_skew   = (
                "right-skewed" if _ed_ud is not None and _ed_ud >= 2.0 else
                "left-skewed"  if _ed_ud is not None and _ed_ud <  0.8 else
                "roughly symmetric"
            )
            _mc_suffix = (
                f" Distribution: E[R] {_mc_ed.mean_return:+.0%},"
                f" P5 {_mc_ed.p5_return:.0%} / P95 {_mc_ed.p95_return:+.0%}"
                f" ({_ed_spread:.0f}pp spread, {_ed_skew})."
            )
        else:
            _mc_suffix = ""

        if base_rating == "Hold":
            entry_strategy = "Tracking position"
            entry_detail   = "Monitor for improved fundamentals or a better entry point before committing capital."
        elif target == 0.0:
            # edge-case: hard overrides drove a non-Bearish stock to 0
            entry_strategy = "No position"
            entry_detail   = "Risk/reward does not support initiating at this time — revisit on improvement."
        elif size_reduced and target <= 2.0:
            # Buy-rated but constrained → Starter position language
            entry_strategy = "Staged entry"
            entry_detail   = f"Initiate a starter position; add to full Buy-sized allocation on confirmation.{_mc_suffix}"
        elif force_staged or setup == "weak":
            entry_strategy = "Staged entry"
            entry_detail   = f"Build in tranches as momentum and setup improve.{_mc_suffix}"
        elif base_rating == "Strong Buy" and setup == "strong":
            entry_strategy = "Full allocation"
            entry_detail   = f"Conviction and setup both strong — build to full size over 1–2 sessions.{_mc_suffix}"
        else:
            entry_strategy = "Staged entry"
            entry_detail   = f"Initiate and add as the thesis confirms across fundamental and technical signals.{_mc_suffix}"

        # ── Step 6: Rationale — ONE primary driver ────────────────────────────
        # When the MC formula drove sizing, always include the key distribution
        # statistics so the rationale is fully self-explaining.
        _mc_stats_suffix = ""
        if _mc is not None and _mc_base_method == "formula":
            _mc_stats_suffix = (
                f" (E[R]={_mc.mean_return:+.0%},"
                f" P5={_mc.p5_return:.0%},"
                f" P95={_mc.p95_return:+.0%})"
            )

        if hard_cap_reason:
            # hard_cap_reason is shown in the amber alert box in the UI.
            # rationale must be a complementary distribution-context statement —
            # not a copy of hard_cap_reason — so the same text doesn't appear twice.
            if _mc is not None:
                _er_r2  = _mc.mean_return
                _p5_r2  = _mc.p5_return
                _p95_r2 = _mc.p95_return
                _ud_r2  = getattr(_mc, "upside_downside", None)
                _spr_r2 = (_p95_r2 - _p5_r2) * 100
                if _ud_r2 is not None and _ud_r2 >= 2.0:
                    rationale = (
                        f"Distribution is right-skewed (E[R] {_er_r2:+.0%},"
                        f" {_ud_r2:.1f}× upside/downside, {_spr_r2:.0f}pp spread)"
                        f" — size constrained by tail risk, not lack of upside."
                    )
                elif _p5_r2 < -0.25:
                    if _er_r2 >= 0.10:
                        _er_qual = "is attractive"
                    elif _er_r2 >= 0.0:
                        _er_qual = "is roughly neutral"
                    else:
                        _er_qual = "indicates poor immediate risk/reward"
                    rationale = (
                        f"Expected return of {_er_r2:+.0%} {_er_qual};"
                        f" a {_p5_r2:.0%} downside tail (P5) justifies reduced allocation."
                    )
                elif _er_r2 < 0.15:
                    rationale = (
                        f"Below-hurdle E[R] ({_er_r2:+.0%}) with {_spr_r2:.0f}pp"
                        f" spread (P5 {_p5_r2:.0%} / P95 {_p95_r2:+.0%})."
                    )
                else:
                    rationale = (
                        f"Distribution: E[R] {_er_r2:+.0%},"
                        f" P5 {_p5_r2:.0%} / P95 {_p95_r2:+.0%} ({_spr_r2:.0f}pp spread)."
                    )
            else:
                rationale = "Position constrained per the sizing policy applied."
        elif mc_adjustment_reason:
            rationale = mc_adjustment_reason + "."
        elif _mc_base_method == "formula" and _mc is not None:
            # Formula-driven rationale: state the risk/reward using distribution shape
            _er_r   = _mc.mean_return
            _p5_r   = _mc.p5_return
            _p95_r  = _mc.p95_return
            _spr_r  = (_p95_r - _p5_r) * 100
            _ud_r   = getattr(_mc, "upside_downside", None)
            _ud_str = f", {_ud_r:.1f}× upside/downside" if _ud_r is not None else ""
            if _er_r >= 0.20 and _p5_r > -0.20:
                rationale = (
                    f"Favourable risk/reward: E[R] {_er_r:+.0%},"
                    f" P5 {_p5_r:.0%} / P95 {_p95_r:+.0%}"
                    f"{_ud_str}."
                )
            elif _er_r >= 0.15:
                rationale = (
                    f"Adequate expected return (E[R] {_er_r:+.0%})"
                    f" with {_spr_r:.0f}pp distribution spread"
                    f" (P5 {_p5_r:.0%} / P95 {_p95_r:+.0%})."
                )
            else:
                rationale = (
                    f"Below-hurdle expected return (E[R] {_er_r:+.0%})"
                    f" with high dispersion ({_spr_r:.0f}pp spread); position constrained."
                )
        elif base_rating == "Strong Buy" and setup == "strong":
            rationale = "Full size due to strong conviction and setup."
        elif base_rating == "Strong Buy" and setup == "neutral":
            rationale = "Reduced due to neutral near-term setup despite high-conviction fundamentals."
        elif base_rating == "Strong Buy" and setup == "weak":
            rationale = "Starter position — strong fundamentals offset by adverse near-term setup."
        elif base_rating == "Buy" and setup == "strong":
            rationale = "Full Buy allocation — solid risk/reward supports initiating a standard position."
        elif base_rating == "Buy" and setup == "neutral":
            rationale = "Standard Buy allocation — balanced risk/reward with no dominant near-term catalyst."
        elif base_rating == "Buy" and setup == "weak":
            rationale = "Starter Buy — initiate at reduced size; adverse near-term setup warrants staged entry."
        elif base_rating == "Hold":
            rationale = "Hold-rated — track only; await a catalyst before adding exposure."
        else:
            rationale = "Position sized per conviction tier and setup quality."

        conviction_tier = {
            "Strong Buy": "high",
            "Buy":        "medium",
            "Hold":       "hold",
        }.get(base_rating, "none")

        print(
            f"  [POS_SIZE] rating={base_rating!r} setup={setup} "
            f"mom={mom_q} val={val_q} macro={macro_q} "
            f"size_reduced={size_reduced} force_staged={force_staged} "
            f"target={_fmt(target)} strategy={entry_strategy!r}"
        )

        pos_range = _fmt(target)

        sizing = {
            "position_range":       pos_range,
            "position_lo":          target,
            "position_hi":          target,
            "position_size":        target,
            "entry_strategy":       entry_strategy,
            "entry_detail":         entry_detail,
            "rationale":            rationale,
            "conviction_tier":      conviction_tier,
            "setup_quality":        setup,
            "hard_cap_reason":      hard_cap_reason,
            "rating":               base_rating,
            "core_compounder_tag":  core_compounder_tag,
        }

        lines = [
            "  Position Sizing Guidance",
            "  ────────────────────────",
            f"    Position Size    : {pos_range}",
            f"    Entry Strategy   : {entry_strategy}",
            f"    Rationale        : {rationale}",
        ]
        # Show probability-formula calculation steps when MC drove sizing
        if _mc is not None and _mc_base_method == "formula":
            _er_ln  = _mc.mean_return
            _p5_ln  = _mc.p5_return
            _p95_ln = _mc.p95_return
            _dn_ln  = abs(_p5_ln) if _p5_ln < 0 else 0.01
            _ras_ln = max(0.05, min(2.5, _er_ln / _dn_ln))
            _cv_ln  = (_mc_profile_pre.conviction_score / 100.0  # type: ignore[union-attr]
                       if _mc_profile_pre is not None else 0.0)
            lines.append(
                f"    Sizing method    : Probability formula"
            )
            lines.append(
                f"      E[R]={_er_ln:+.1%}  P5={_p5_ln:.1%}  P95={_p95_ln:+.1%}"
            )
            lines.append(
                f"      RAS = E[R]/|P5| = {_ras_ln:.2f}  "
                f"Conviction = {_cv_ln:.0%}"
            )
            lines.append(
                f"      Raw size = {_ras_ln:.2f} × {_cv_ln:.0%} × 5% = "
                f"{_ras_ln * _cv_ln * 5.0:.1f}%  →  snapped to {pos_range}"
            )
        if _model_disagreement_flag:
            lines.append(
                "    ⚠  Model disagreement detected — scenario E[R] and factor model"
                " E[R] diverge > 12 pp. Conviction reduced until models converge."
            )
        lines.append("")
        return lines, sizing

    @staticmethod
    def _print_run_audit(
        ticker: str,
        stock_data: "StockData",
        findings: dict,
        pe_displayed: "float | None",
        ps_displayed: "float | None",
        ev_displayed: "float | None",
    ) -> None:
        """
        Print a structured audit log for every run.

        Covers:
          • requested / normalized ticker
          • price, shares_outstanding, market_cap (api vs computed)
          • P/E source and methodology
          • macro source, observation date, and value for each indicator
          • final normalized metric payload sent to the report
        """
        sep = "─" * 64
        print(f"\n  {sep}")
        print(f"  [RUN AUDIT] ticker={ticker!r}")
        print(f"  {sep}")

        # ── Profile / market cap ─────────────────────────────────────────────
        p         = stock_data.profile
        price     = stock_data.current_price
        shares    = stock_data.shares_outstanding
        mc_api    = stock_data.market_cap
        mc_cmp    = stock_data.market_cap_computed

        print(f"  [AUDIT] price_raw              = {price}")
        print(f"  [AUDIT] shares_outstanding_raw = {shares}")
        print(f"  [AUDIT] market_cap_api         = {mc_api}")
        print(f"  [AUDIT] market_cap_computed    = {mc_cmp}")

        if mc_api and mc_cmp and mc_cmp > 0:
            diff = abs(mc_api - mc_cmp) / mc_cmp * 100
            flag = "  *** MATERIAL DISCREPANCY ***" if diff > 10 else ""
            print(f"  [AUDIT] market_cap diff        = {diff:.1f}%{flag}")

        # ── Ratios source ────────────────────────────────────────────────────
        ratios = stock_data.latest_ratios
        pe_api = ratios.pe_ratio     if ratios else None
        ps_api = ratios.ps_ratio     if ratios else None
        ev_api = ratios.ev_to_ebitda if ratios else None

        print(f"  [AUDIT] pe_from_ratios         = {pe_api}  (period: {ratios.period if ratios else 'N/A'})")
        print(f"  [AUDIT] pe_displayed           = {pe_displayed}")
        if pe_api is not None and pe_displayed is not None:
            pe_diff = abs(pe_api - pe_displayed)
            flag = "  *** DISCREPANCY ***" if pe_diff > 2.0 else ""
            print(f"  [AUDIT] pe diff                = {pe_diff:.2f}{flag}")

        print(f"  [AUDIT] ps_from_ratios         = {ps_api}")
        print(f"  [AUDIT] ps_displayed           = {ps_displayed}")
        print(f"  [AUDIT] ev_from_ratios         = {ev_api}")
        print(f"  [AUDIT] ev_displayed           = {ev_displayed}")

        # ── Computed P/E cross-check ─────────────────────────────────────────
        inc = stock_data.latest_income
        _norm_m = (findings.get("fundamental") or {}).get("normalized_metrics")
        _currency_mismatch = getattr(_norm_m, "currency_mismatch", False) if _norm_m else False
        _fin_ccy = getattr(_norm_m, "financials_currency", "") if _norm_m else ""
        _px_ccy  = getattr(_norm_m, "price_currency", "") if _norm_m else ""
        if price and inc:
            sh = (mc_api / price) if (mc_api and price > 0) else None
            eps = inc.eps_diluted or inc.eps
            if eps is None and inc.net_income and sh and sh > 0:
                eps = inc.net_income / sh
            pe_computed = round(price / eps, 2) if (eps and eps > 0) else None
            print(f"  [AUDIT] pe_computed            = {pe_computed}  (current_price/annual_eps)")
            if pe_computed and pe_displayed:
                diff_pe = abs(pe_computed - pe_displayed) / pe_displayed * 100
                if _currency_mismatch and diff_pe > 15:
                    # Currency mismatch explains the divergence — price in _px_ccy, EPS in _fin_ccy.
                    # This is an expected artifact, not a data quality issue.
                    print(
                        f"  [AUDIT] pe_method diff         = {diff_pe:.1f}%"
                        f"  [suppressed: currency mismatch artifact"
                        f" — income stmt in {_fin_ccy}, price in {_px_ccy}]"
                    )
                else:
                    flag = "  *** METHODOLOGY DIFF ***" if diff_pe > 15 else ""
                    print(f"  [AUDIT] pe_method diff         = {diff_pe:.1f}%{flag}")

        # ── Macro snapshot with dates ────────────────────────────────────────
        macro = findings.get("macro", {})
        snap  = macro.get("snapshot", {})
        dates = macro.get("observation_dates", {})
        if snap:
            print(f"  [AUDIT] macro_regime           = {macro.get('macro_regime', 'N/A')}")
            print(f"  [AUDIT] macro_score            = {macro.get('macro_score', 'N/A')}")
            for k, v in snap.items():
                d = dates.get(k, "?")
                vstr = f"{v:.4f}" if isinstance(v, float) else "N/A"
                print(f"  [AUDIT] macro.{k:<22s} = {vstr}  (obs: {d})")

        # ── Final payload summary ────────────────────────────────────────────
        sector   = p.sector   if p else "N/A"
        industry = p.industry if p else "N/A"
        print(f"  [AUDIT] final_payload: sector={sector!r} industry={industry!r}"
              f"  price={price}  mktcap={mc_api}  PE={pe_displayed}  PS={ps_displayed}  EV={ev_displayed}")
        print(f"  {sep}\n")

    @staticmethod
    def _build_peer_comparison_section(pc: PeerComparison) -> list[str]:
        """
        Render the peer comparison table and insights.

        Layout:
          Peer Comparison
          ───────────────
            Ticker    P/E     P/S   Growth    PEG
            ──────    ─────   ─────  ──────   ─────
          ► TARGET   22.0x   5.0x   10.1%   2.18x
            PEER1    18.3x   3.2x    8.4%   2.18x
            ...

            • Insight 1
            • Insight 2
        """
        lines: list[str] = ["  Peer Comparison", "  ───────────────"]

        if not pc.has_peers:
            lines.append("    Peer comparison omitted — insufficient peer data.")
            lines.append("")
            return lines

        def _fv(v: "float | None", suffix: str = "") -> str:
            return f"{v:.1f}{suffix}" if v is not None else "N/A"

        # Header
        lines.append(
            f"    {'':1s}{'Ticker':<8s}  {'P/E':>6s}  {'P/S':>6s}  {'Growth':>7s}  {'PEG':>6s}"
        )
        lines.append(
            f"    {'':1s}{'──────':<8s}  {'─────':>6s}  {'─────':>6s}  {'──────':>7s}  {'─────':>6s}"
        )

        for row in pc.rows:
            marker = "►" if row.is_target else " "
            # Quality badge: shown for non-target peers only
            if not row.is_target and row.quality_score is not None:
                _conf  = getattr(row, "peer_confidence", "") or ""
                _proxy = getattr(row, "is_proxy", False)
                _badge = f"[~PROXY/{_conf}]" if _proxy else f"[{_conf}]"
                _qs    = f"{row.quality_score:.0f}"
            else:
                _badge = ""
                _qs    = ""

            lines.append(
                f"    {marker} {row.ticker:<8s}"
                f"  {_fv(row.pe, 'x'):>6s}"
                f"  {_fv(row.ps, 'x'):>6s}"
                f"  {_fv(row.growth_pct, '%'):>7s}"
                f"  {_fv(row.peg, 'x'):>6s}"
                + (f"  Q:{_qs:<3s} {_badge}" if _qs else "")
            )
            if not row.is_target and row.justification:
                lines.append(f"             → {row.justification}")

        lines.append("")

        if pc.insights:
            for insight in pc.insights:
                lines.append(f"    • {insight}")
            lines.append("")

        return lines

    @staticmethod
    def _build_peer_history_section(pc: "PeerComparison") -> list[str]:
        """
        Render a 5-year historical comparison table below the snapshot peer table.

        Organised by metric group; each group has one row per company.
        Only rendered when ≥1 row has historical data.

        Trend signal per row reuses classify_trend() from analysis/trend.py:
          ↑ Expanding  ↓ Deteriorating  → Stable  ⚠ Volatile
        """
        from analysis.trend import classify_trend

        rows = pc.rows if pc else []
        # Only render when at least one row has populated history
        if not any(r.historical for r in rows):
            return []

        lines: list[str] = []
        lines.append("  Peer Performance (5-Year)")
        lines.append("  ─────────────────────────")

        # Determine column headers from the widest historical set available
        max_periods = max((len(r.historical) for r in rows if r.historical), default=0)
        if max_periods == 0:
            return []

        col_labels = []
        for r in rows:
            if r.historical:
                col_labels = [h.label for h in r.historical]
                break

        def _pct(v: Optional[float], sign: bool = True) -> str:
            if v is None:
                return "  N/A"
            s = f"{v:+.1f}%" if sign else f"{v:.1f}%"
            return s

        def _mg(v: Optional[float]) -> str:
            if v is None:
                return "  N/A"
            return f"{v*100:.1f}%"

        def _trend(vals: list[Optional[float]]) -> str:
            clean = [v for v in vals if v is not None]
            if len(clean) < 2:
                return ""
            _, sig = classify_trend(clean)
            return sig

        # Metric groups: (title, attr, formatter, high_is_improving)
        _GROUPS = [
            ("Revenue Growth %",  "revenue_growth",  _pct,  True),
            ("EPS Growth %",      "eps_growth",       _pct,  True),
            ("EBITDA Growth %",   "ebitda_growth",    _pct,  True),
            ("Operating Margin",  "op_margin",        _mg,   True),
            ("Net Margin",        "net_margin",        _mg,   True),
            ("ROE",               "roe",               _mg,   True),
            ("ROIC",              "roic",              _mg,   True),
        ]

        # Column width constants
        _NAME_W = 12   # ticker width
        _COL_W  = 8    # value column width

        for group_title, attr, fmt, _ in _GROUPS:
            # Skip group if no row has any data for this metric
            has_data = any(
                any(getattr(h, attr) is not None for h in r.historical)
                for r in rows if r.historical
            )
            if not has_data:
                continue

            lines.append(f"\n  {group_title}")
            # Header row
            hdr = f"    {'':1s}{'Ticker':<{_NAME_W}s}"
            for lbl in col_labels:
                hdr += f"  {lbl:>{_COL_W}s}"
            hdr += f"  {'Trend':>5s}"
            lines.append(hdr)
            sep = f"    {'':1s}{'─'*_NAME_W}"
            for _ in col_labels:
                sep += f"  {'─'*_COL_W}"
            sep += f"  {'─'*5}"
            lines.append(sep)

            for row in rows:
                if not row.historical:
                    continue
                marker = "►" if row.is_target else " "
                ticker_str = (row.ticker or "")[:_NAME_W]
                line = f"    {marker} {ticker_str:<{_NAME_W}s}"

                vals: list[Optional[float]] = []
                for h in row.historical:
                    v = getattr(h, attr)
                    vals.append(v)
                    line += f"  {fmt(v):>{_COL_W}s}"

                # Pad missing periods
                for _ in range(max_periods - len(row.historical)):
                    line += f"  {'N/A':>{_COL_W}s}"

                sig = _trend(vals)
                line += f"  {sig:>5s}"
                lines.append(line)

        lines.append("")
        return lines

    # ── Pre-render enforcement layer ──────────────────────────────────────────
    #
    # Single-source-of-truth enforcement. Each section is owned by exactly one
    # module. This layer catches violations and either corrects them in-place
    # (peers, triggers) or raises loudly (memo structure — code bug territory).

    _REQUIRED_MEMO_SECTIONS = [
        "TOP TAKEAWAY",
        "INVESTMENT THESIS",
        "KEY RISKS",
        "WHAT WOULD CHANGE OUR VIEW",
        "FINAL VERDICT",
    ]

    # Patterns that indicate a trigger describes the CURRENT state rather
    # than a future condition.
    _BACKWARD_TRIGGER_PATTERNS = [
        re.compile(r"\bcurrently\b",                     re.I),
        re.compile(r"\bis already\b",                    re.I),
        re.compile(r"\bhas been\b",                      re.I),
        re.compile(r"\bcontinues? to\b",                 re.I),
        re.compile(r"\bpersists?\b",                     re.I),
        re.compile(r"\bremains? (weak|negative|poor)\b", re.I),
    ]

    @staticmethod
    def _enforce_report_sections(
        memo_result,             # MemoResult — mutated in place
        peer_cmp=None,           # PeerComparison | None — mutated in place when present
    ) -> None:
        """
        Pre-render enforcement. Raises RuntimeError for structural code bugs.
        Mutates peer_cmp.rows and memo_result.change_view to correct data violations.

        A. Peers    — remove any unintentional cross-archetype peer (skipped when
                      peer_cmp is None — peer engine returned no result)
        B. Triggers — remove any backward-looking trigger from change_view
        C. Memo     — RAISE if a required section is missing or 'Overview' present

        Rules:
        · PeerComparison  → PeerSelectionEngine ONLY (no sector/fallback peers)
        · WhatWouldChange → MemoEngine._change_view_bullets() ONLY
        · InvestmentMemo  → MemoEngine ONLY (no overrides, no appending)
        """
        # ── A. Peer enforcement (skipped when peer engine returned no result) ──
        # Unintentional cross-archetype peers have "limits direct comparability"
        # in their justification (set by justify_peer() in peer_selection_engine.py)
        # WITHOUT a [PROXY PEER] prefix.
        #
        # [PROXY PEER] peers are KEPT — they were selected intentionally by
        # _select_peers_with_relaxation() Step 4 (adjacent-archetype fallback)
        # and are already flagged for the reader.
        if peer_cmp is not None:
            bad_peers = [
                row for row in peer_cmp.rows
                if not row.is_target
                and "limits direct comparability" in (row.justification or "")
                and not (row.justification or "").startswith("[PROXY PEER]")
            ]
            if bad_peers:
                for row in bad_peers:
                    peer_cmp.rows.remove(row)
                print(
                    f"  [ENFORCE] PEER: removed {len(bad_peers)} cross-archetype peer(s): "
                    f"{[r.ticker for r in bad_peers]}"
                )
                # If no peers remain after removal, mark section empty
                if not any(not r.is_target for r in peer_cmp.rows):
                    peer_cmp.has_peers = False
                    print("  [ENFORCE] PEER: no valid peers remain — section will show 'No valid peers available'")
        else:
            print("  [ENFORCE] PEER: skipped — peer engine returned no result")

        # ── B. Trigger enforcement ────────────────────────────────────────────
        clean: list[str] = []
        removed_triggers: list[str] = []
        for trigger in memo_result.change_view:
            is_backward = any(
                pat.search(trigger) for pat in ReportingAgent._BACKWARD_TRIGGER_PATTERNS
            )
            if is_backward:
                removed_triggers.append(trigger[:70])
            else:
                clean.append(trigger)

        if removed_triggers:
            memo_result.change_view = clean
            print(
                f"  [ENFORCE] TRIGGER: removed {len(removed_triggers)} backward-looking trigger(s):\n"
                + "\n".join(f"    ✗ {t}" for t in removed_triggers)
            )

        # ── C. Memo structure + completeness enforcement ──────────────────────
        # MemoEngine is the sole owner of memo structure.
        # Section C is split into three tiers:
        #
        #   FATAL  — raise RuntimeError immediately; the run is broken.
        #            Required section missing OR 'Overview' in rendered text.
        #            These are code bugs in memo_engine.py, not data issues.
        #
        #   HARD   — raise RuntimeError; signals incomplete memo generation.
        #            Truncation (bullet/takeaway ends mid-sentence) or
        #            section under-populated (< 3 items).
        #
        #   WARN   — log and continue; memo is usable but exceeds targets.
        #            Word count > 250 (content target is ≤200 words).

        # Tier 1 FATAL: required sections present + no Overview injection
        rendered = memo_result.render()
        for section in ReportingAgent._REQUIRED_MEMO_SECTIONS:
            if section not in rendered:
                raise RuntimeError(
                    f"[ENFORCE] MEMO_STRUCTURE: required section '{section}' missing "
                    f"from MemoEngine output. MemoEngine is the sole owner of memo content — "
                    f"this is a code bug that must be fixed in analysis/memo_engine.py."
                )
        if "Overview" in rendered:
            raise RuntimeError(
                "[ENFORCE] MEMO_STRUCTURE: 'Overview' section found in MemoEngine output. "
                "MemoEngine must never produce an Overview section. "
                "Fix _top_takeaway() or _thesis_bullets() in analysis/memo_engine.py."
            )

        # Tiers 2+3: run MemoResult.validate() — separates HARD from WARN
        violations = memo_result.validate()
        hard_violations = [
            v for v in violations
            if v.startswith(("TRUNCATION:", "SECTION_COUNT:"))
        ]
        warn_violations = [
            v for v in violations
            if v.startswith("MEMO_LENGTH:")
        ]

        if hard_violations:
            raise RuntimeError(
                "[ENFORCE] MEMO_COMPLETENESS: memo failed completeness validation.\n"
                + "\n".join(f"  · {v}" for v in hard_violations)
                + "\nFix the relevant method in analysis/memo_engine.py."
            )

        if warn_violations:
            for v in warn_violations:
                print(f"  [ENFORCE] WARN: {v}")

    @staticmethod
    def _score_bar(score: float | None, width: int = 20) -> str:
        if score is None:
            return "[" + "?" * width + "]"
        filled = int(score / 100 * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    @staticmethod
    def _classify_stock_type(sc: "Scorecard") -> "tuple[str, str] | None":
        """
        Return (archetype, one-sentence description) based on category scores,
        or None when scores are too sparse to classify.
        """
        def _s(attr: str) -> "float | None":
            cat = getattr(sc, attr, None)
            return cat.score if cat and cat.data_quality != "missing" else None

        g = _s("growth")
        p = _s("profitability")
        v = _s("valuation")
        m = _s("momentum")

        if all(x is None for x in [g, p, v, m]):
            return None

        # Use 50 as neutral stand-in for missing individual scores
        g_ = g if g is not None else 50.0
        p_ = p if p is not None else 50.0
        v_ = v if v is not None else 50.0
        m_ = m if m is not None else 50.0

        if g_ >= 65 and p_ >= 65 and m_ >= 55:
            return "Compounder", "Consistent growth and strong margins suggest durable compounding potential."

        if p_ >= 65 and g_ < 45 and v_ >= 45:
            return "Defensive", "Reliable profitability with limited growth — prioritises capital preservation."

        if p_ < 45 and g_ >= 55:
            return "Turnaround", "Growth recovering ahead of margins — watch for a profitability inflection."

        if m_ >= 70 and g_ < 60:
            return "Momentum-driven", "Price momentum leads fundamental justification — susceptible to sentiment shifts."

        if v_ >= 65 and g_ < 40 and m_ < 45:
            return "Value Trap", "Cheap valuation without a near-term catalyst — discount may reflect structural headwinds."

        if max(g_, p_, m_) < 70 and min(g_, p_, m_) > 30:
            return "Cyclical", "No dominant quality signal — returns likely tied to macro and sector cycle."

        return None

    @staticmethod
    def _key_terms(text: str) -> set[str]:
        """Extract significant lowercase words (≥4 chars, non-stopword) from text."""
        _STOP = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "need", "used",
            "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "by",
            "from", "up", "about", "into", "through", "with", "this", "that",
            "its", "not", "nor", "so", "yet", "both", "either",
            "than", "then", "when", "where", "which", "who", "whom", "whose",
            "how", "what", "all", "each", "every", "more", "most",
            "other", "some", "such", "only", "same", "also", "well",
            "even", "still", "high", "low", "strong", "weak", "very", "just",
            "due", "over", "under", "above", "below", "rate", "data",
        }
        words = re.findall(r"[a-z]{4,}", text.lower())
        return {w for w in words if w not in _STOP}

    @staticmethod
    def _overlap_ratio(a: set[str], b: set[str]) -> float:
        """Return overlap as fraction of the smaller set (0.0–1.0)."""
        if not a or not b:
            return 0.0
        return len(a & b) / min(len(a), len(b))

    @staticmethod
    def _compute_growth_quality(stock_data: "StockData") -> "tuple[str, str] | None":
        """Return (label, one-sentence description) for growth quality or None if insufficient data."""
        stmts = stock_data.income_statements if stock_data else []
        if not stmts or len(stmts) < 3:
            return None

        # Collect ordered values (oldest → newest), skip None
        revenues = [s.revenue for s in reversed(stmts) if s.revenue and s.revenue > 0]
        eps_vals = [s.eps_diluted for s in reversed(stmts) if s.eps_diluted is not None]
        margins = [s.gross_profit_ratio for s in reversed(stmts) if s.gross_profit_ratio is not None]

        if len(revenues) < 2:
            return None

        # CAGR helper — returns None when either endpoint is non-positive because
        # (negative/positive)**fraction is complex in Python.
        def _cagr(start: float, end: float, periods: int) -> float | None:
            if start is None or start <= 0 or end is None or end <= 0 or periods < 1:
                return None
            try:
                return (end / start) ** (1.0 / periods) - 1.0
            except (ZeroDivisionError, ValueError):
                return None

        periods = len(revenues) - 1
        rev_cagr = _cagr(revenues[0], revenues[-1], periods)

        eps_cagr: float | None = None
        if len(eps_vals) >= 2 and eps_vals[0] and eps_vals[0] > 0:
            eps_cagr = _cagr(eps_vals[0], eps_vals[-1], len(eps_vals) - 1)

        # Margin trend: positive if last margin > first margin
        margin_improving: bool | None = None
        if len(margins) >= 2:
            margin_improving = margins[-1] > margins[0]

        if rev_cagr is None:
            return None

        rev_pct = rev_cagr * 100

        # Determine label
        eps_faster = (
            eps_cagr is not None and eps_cagr > rev_cagr * 1.1
        )
        eps_lagging = (
            eps_cagr is not None and eps_cagr < rev_cagr * 0.6
        )

        if rev_pct >= 15 and (eps_faster or margin_improving):
            label = "HIGH"
            desc = (
                f"Revenue growing at ~{rev_pct:.0f}% CAGR with "
                + ("earnings outpacing revenue — scalable model." if eps_faster
                   else "expanding margins — operational leverage evident.")
            )
        elif rev_pct >= 8 and not eps_lagging:
            label = "MEDIUM"
            margin_note = (
                "margins stable" if margin_improving is None
                else ("margins expanding" if margin_improving else "margins contracting")
            )
            desc = (
                f"Solid revenue CAGR of ~{rev_pct:.0f}% with {margin_note};"
                " growth is consistent but not exceptional."
            )
        elif rev_pct >= 8 and eps_lagging:
            label = "MEDIUM"
            desc = (
                f"Revenue growing at ~{rev_pct:.0f}% CAGR but earnings growth lags,"
                " suggesting reinvestment phase or margin pressure."
            )
        elif rev_pct >= 0:
            label = "LOW"
            desc = (
                f"Revenue CAGR of ~{rev_pct:.0f}% is below threshold;"
                " limited fundamental growth momentum."
            )
        else:
            label = "LOW"
            desc = f"Revenue declining at ~{abs(rev_pct):.0f}% CAGR — negative growth trend."

        return label, desc

    # ── Stage 4 helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _derive_outlook_action(
        sc: "Scorecard",
        price: "float | None",
        val_range: "ValuationRange | None",
        final_size: float = -1.0,
    ) -> "tuple[str, str, str | None]":
        """
        Return (outlook_label, action_label, why_str_or_None).

        OUTLOOK — describes the business thesis (score/stance-based).
        ACTION  — what to do right now, given current price vs fair value.
        WHY     — one-sentence explanation when OUTLOOK and ACTION diverge;
                  None when they are consistent (e.g. Bullish + BUY).

        final_size: position size as a decimal (0.0 = no position). When 0.0,
                    BUY/STAGED BUY/HOLD are vetoed in favour of WAIT so that
                    the ACTION label matches what the position sizing section
                    actually recommends.
        """
        from models.scorecard import Stance as _Stance

        score  = sc.overall_score
        stance = sc.stance
        conf   = sc.confidence

        # ── OUTLOOK ──
        if score >= 65:
            outlook_base = "Bullish"
        elif score >= 40:
            outlook_base = "Neutral"
        else:
            outlook_base = "Bearish"

        if conf >= 0.70:
            conf_tag = "high confidence"
        elif conf < 0.45:
            conf_tag = "low confidence — data gaps present"
        else:
            conf_tag = ""
        outlook = f"{outlook_base} — {conf_tag}" if conf_tag else outlook_base

        # ── Hard SELL ──
        if score < 40 or stance == _Stance.BEARISH:
            return outlook, "SELL", "Score/thesis is bearish — consider exiting or avoiding."

        # ── Price zone (P25 / P50 proxies) ──
        vr = val_range
        mc = getattr(vr, "mc", None) if vr else None
        if mc and getattr(mc, "p25_price", None) and getattr(mc, "median_price", None):
            p25 = mc.p25_price
            p50 = mc.median_price
        elif vr and vr.bear_price and vr.base_price:
            p25 = vr.bear_price * 1.15   # rough P25 proxy (bear ≈ P10, scale up)
            p50 = vr.base_price
        else:
            p25 = None
            p50 = None

        if price and p25 and price < p25:
            zone = "below_p25"
        elif price and p50 and price < p50:
            zone = "between_p25_p50"
        elif price and p50 and price >= p50:
            zone = "above_p50"
        else:
            zone = "unknown"

        p50_str = f"${p50:.0f}" if p50 else "fair value"
        p25_str = f"~${p25:.0f}" if p25 else "below fair value"

        _size_veto = (final_size >= 0.0 and final_size == 0.0)
        _size_veto_why = (
            "Long-term thesis intact, but risk/return metrics do not support "
            "initiating a position now. Revisit if expected return improves "
            "or downside tail narrows."
        )

        # ── Bullish ──
        if stance == _Stance.BULLISH:
            if zone == "below_p25":
                if _size_veto:
                    return outlook, "WAIT", _size_veto_why
                return outlook, "BUY", None
            if zone == "between_p25_p50":
                if _size_veto:
                    return outlook, "WAIT", _size_veto_why
                return outlook, "STAGED BUY", None
            # above P50 or unknown — WAIT
            why = (
                f"Long-term thesis is positive, but current price is at or above "
                f"{p50_str} (driver-model base case / P50). "
                f"Better entry expected on pullback to {p25_str}."
            )
            return outlook, "WAIT", why

        # ── Neutral ──
        if zone == "above_p50":
            why = (
                f"Neutral thesis — current price at or above {p50_str} "
                "(driver-model base). No favorable risk/reward for new entry."
            )
            return outlook, "WAIT", why
        if zone in ("below_p25", "between_p25_p50"):
            if _size_veto:
                return outlook, "WAIT", "Neutral thesis — risk/return metrics do not support entry."
            return outlook, "HOLD", None
        return outlook, "WAIT", "Neutral thesis — no clear entry signal from price zone."

    @staticmethod
    def _load_excel_summary(ticker: str) -> "dict | None":
        """
        Load data/excel_summaries/{TICKER}_excel.json if it exists.
        Returns None when file is absent (opt-in; silently skipped).
        Gracefully returns None on malformed JSON or OS errors.
        """
        from pathlib import Path
        import json as _json

        path = Path(f"data/excel_summaries/{ticker}_excel.json")
        if not path.exists():
            return None
        try:
            return _json.loads(path.read_text())
        except (ValueError, OSError):
            return None

    @staticmethod
    def _render_excel_reconciliation(
        excel: dict,
        vr: "ValuationRange | None",
        current_price: "float | None",
    ) -> list[str]:
        """
        Render the EXCEL RECONCILIATION section.
        All rows are optional — missing fields display a dash, never break.
        """
        lines: list[str] = []

        model_date = excel.get("model_date", "unknown date")
        lines += [
            "─" * 68,
            "  EXCEL RECONCILIATION",
            "─" * 68,
            f"  Excel model dated: {model_date}",
            "",
        ]

        col_w = 16   # Excel col width
        se_w  = 18   # StockEval col width
        dw    = 7    # Δ col width
        note_w = 20

        hdr = (
            f"    {'':26s}  {'Excel':>{col_w}}  {'StockEval':>{se_w}}"
            f"  {'Δ':>{dw}}  Note"
        )
        sep = (
            f"    {'':26s}  {'────────':>{col_w}}  {'──────────────':>{se_w}}"
            f"  {'───────':>{dw}}  ───────────────────"
        )
        lines += [hdr, sep]

        def _fmt_price(v: "float | None") -> str:
            return f"${v:.2f}" if isinstance(v, (int, float)) else "—"

        def _delta_note(excel_v: "float | None", se_v: "float | None", se_label: str = "") -> "tuple[str, str]":
            """Return (delta_str, note_str) for a pair of prices."""
            if excel_v is None or se_v is None or se_v == 0:
                return "—", se_label if se_label else ""
            d = (se_v - excel_v) / abs(excel_v)
            d_str = f"{d:+.0%}"
            if abs(d) <= 0.02:
                note = "✓ matches"
            elif abs(d) <= 0.10:
                note = f"close (Δ within 10%)"
            else:
                note = se_label if se_label else f"see methodology note"
            return d_str, note

        def _row(label: str, excel_v: "float | None", se_v: "float | None",
                 se_label: str = "", forced_note: str = "") -> str:
            e_str = _fmt_price(excel_v)
            s_str = (_fmt_price(se_v) + (f" ({se_label})" if se_label else "")) if se_v is not None else "—"
            if forced_note:
                d_str, note = "—", forced_note
            else:
                d_str, note = _delta_note(excel_v, se_v)
            return (
                f"    {label:<26s}  {e_str:>{col_w}}  {s_str:>{se_w}}"
                f"  {d_str:>{dw}}  {note}"
            )

        # Current price
        excel_price = excel.get("current_price_excel")
        lines.append(_row("Current price", excel_price, current_price,
                          forced_note="Live intraday" if excel_price and current_price else ""))

        # 2026 fair value (Excel has 5Y midpoint; StockEval driver base is 1Y)
        fv_2026     = excel.get("fair_value_2026")
        fv_range    = excel.get("fair_value_2026_range") or []
        fv_low      = fv_range[0] if len(fv_range) > 0 else None
        fv_high     = fv_range[1] if len(fv_range) > 1 else None
        se_base     = vr.base_price if vr else None
        fv_range_str = f"[${fv_low:.0f}–${fv_high:.0f}]" if fv_low and fv_high else ""
        e_fv_str    = (f"${fv_2026:.2f} {fv_range_str}".strip()) if fv_2026 else "—"
        lines.append(
            f"    {'2026 Fair Value':<26s}  {e_fv_str:>{col_w}}  {'not computed':>{se_w}}"
            f"  {'—':>{dw}}  Excel uses 5Y midpoint"
        )

        # DCF
        dcf         = excel.get("dcf_2026")
        pe_base_se  = vr.pe_base if vr else None
        d_str, note = _delta_note(dcf, pe_base_se)
        if dcf and pe_base_se:
            note = "P/E approx of DCF"
        lines.append(_row("2026 DCF", dcf, pe_base_se, "P/E", forced_note=note if note != "—" else ""))

        # Intrinsic value
        iv          = excel.get("intrinsic_2026")
        lines.append(_row("2026 Intrinsic value", iv, pe_base_se, "P/E",
                          forced_note="Same P/E approximation" if iv else ""))

        # Driver model base
        driver_base = vr.base_price if vr else None
        lines.append(
            f"    {'Driver scenario base':<26s}  {'—':>{col_w}}  {_fmt_price(driver_base):>{se_w}}"
            f"  {'—':>{dw}}  SE primary"
        )

        # Forward EPS fields
        eps_2026 = excel.get("expected_eps_2026")
        lines.append(
            f"    {'Forward EPS 2026':<26s}  {_fmt_price(eps_2026):>{col_w}}  {'—':>{se_w}}"
            f"  {'—':>{dw}}  SE doesn't compute fwd EPS"
        )
        eps_2027 = excel.get("expected_eps_2027")
        if eps_2027:
            lines.append(
                f"    {'Forward EPS 2027':<26s}  {_fmt_price(eps_2027):>{col_w}}  {'—':>{se_w}}"
                f"  {'—':>{dw}}  SE doesn't compute fwd EPS"
            )

        # Beta
        excel_beta = excel.get("beta")
        from models.stock_data import StockData as _SD   # noqa: F401 (type-only use at runtime)
        # We don't have beta directly here; show Excel value only
        if excel_beta is not None:
            lines.append(
                f"    {'Beta':<26s}  {excel_beta:>{col_w}.2f}  {'from stock profile':>{se_w}}"
                f"  {'—':>{dw}}  verify in stock header"
            )

        # WACC
        wacc = excel.get("wacc")
        if wacc is not None:
            lines.append(
                f"    {'WACC':<26s}  {wacc*100:>{col_w}.1f}%  {'not used':>{se_w}}"
                f"  {'—':>{dw}}  SE uses exit multiple, not WACC"
            )

        # Avg P/E
        avg_pe = excel.get("average_pe_ratio")
        pe_range = excel.get("pe_ratio_range") or []
        if avg_pe is not None:
            pe_range_str = f"[{pe_range[0]:.0f}x–{pe_range[1]:.0f}x]" if len(pe_range) >= 2 else ""
            lines.append(
                f"    {'Avg P/E (Excel)':<26s}  {avg_pe:>{col_w}.1f}x  {'see peer table':>{se_w}}"
                f"  {'—':>{dw}}  {pe_range_str}"
            )

        lines.append("")
        return lines
