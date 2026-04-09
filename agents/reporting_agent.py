"""
ReportingAgent

Compiles all agent findings into the final investment memo and score summary.
This is the terminal agent — it produces the human-readable output.
"""
from __future__ import annotations

import re

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
                reasoning="Data unavailable", data_quality="missing"
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
        signal_conf, signal_explanation = compute_signal_confidence(_all_cats)
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
        market_findings = findings.get("market", {}).get("findings", {})
        analyst_note = market_findings.get("analyst", {}).get("note", "")
        pt_note = market_findings.get("price_target", {}).get("note", "")
        sector_note = market_findings.get("sector", {}).get("note", "")

        sentiment_findings = findings.get("sentiment", {}).get("findings", {})
        sentiment_note = sentiment_findings.get("note", "")

        # ── Run audit log ─────────────────────────────────────────────────────
        # Printed unconditionally so every run leaves a clear trail of raw
        # inputs → displayed values.  Discrepancies are flagged with ***.
        self._print_run_audit(ticker, stock_data, findings, pe_val, ps_val, ev_val)

        lines: list[str] = []

        # Header
        lines += [
            "=" * 68,
            f"  INVESTMENT SNAPSHOT — {ticker}",
            "=" * 68,
            "",
        ]

        # Company info + key metrics
        lines += [
            f"  Company       : {company}",
            f"  Sector        : {sector}  |  Industry: {industry}",
            f"  Market Cap    : {mktcap}",
            f"  Current Price : ${price:.2f}" if price else "  Current Price : N/A",
            f"  P/E Ratio     : {pe_str}",
            f"  P/S Ratio     : {ps_str}",
            f"  EV/EBITDA     : {ev_str}",
            "",
            "─" * 68,
            f"  Overall Score : {sc.overall_score:.0f} / 100",
            f"  Stance        : {sc.stance.value.upper()}",
            *(
                [f"  Stock Type    : {_st[0]} — {_st[1]}"]
                if (_st := self._classify_stock_type(sc)) else []
            ),
            f"  Confidence    : {sc.confidence:.0%}",
            *(
                [f"  {'':16s}  {sc.confidence_explanation}"]
                if sc.confidence_explanation else []
            ),
            "─" * 68,
            "",
            "  Category Subscores (score × weight = contribution):",
        ]

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
            lines.append("  Bullish Factors:")
            for f in bullish_filtered[:4]:
                lines.append(f"    + {f}")
            lines.append("")

        # Build term set from shown bullish factors so Key Drivers can skip
        # anything already stated there or in Strengths reasoning.
        _shown_terms: set[str] = _strength_terms.copy()
        for _f in bullish_filtered[:4]:
            _shown_terms |= self._key_terms(_f)

        if bearish_filtered:
            lines.append("  Bearish Factors:")
            for f in bearish_filtered[:4]:
                lines.append(f"    - {f}")
            lines.append("")

        # Risk flags
        if sc.risk_flags:
            lines.append("  Risk Flags:")
            for flag in sc.risk_flags[:5]:
                lines.append(f"    ⚠  {flag}")
            lines.append("")

        # What would change the view
        lines.append("  What Would Change This View:")
        for item in sc.what_would_change_view[:4]:
            lines.append(f"    → {item}")
        lines.append("")

        # Key drivers — filter out any that restate Strengths or Bullish Factors
        _drivers_deduped = [
            d for d in sc.key_drivers
            if self._overlap_ratio(self._key_terms(d), _shown_terms) < _OVERLAP_THRESH
        ]
        lines.append("  Key Drivers:")
        for d in _drivers_deduped[:3]:
            lines.append(f"    • {d}")
        lines.append("")

        # Macro regime context
        lines += self._build_macro_section(macro_findings)

        # Analyst context
        if analyst_note or pt_note or sector_note:
            lines.append("  Market Context:")
            for note in [sector_note, analyst_note, pt_note]:
                if note:
                    lines.append(f"    {note}")
            lines.append("")

        # Investment Memo
        lines += [
            "─" * 68,
            "  INVESTMENT MEMO",
            "─" * 68,
            "",
        ]

        lines.append("  Overview")
        lines.append(f"  --------")
        desc = (p.description[:280] + "...") if p and p.description and len(p.description) > 280 else (p.description if p else "No description available.")
        lines.append(f"  {desc}")
        lines.append("")

        lines.append("  Strengths")
        lines.append("  ---------")
        if strong_cats:
            for cat in strong_cats:
                score_obj = getattr(sc, cat)
                lines.append(f"  {cat.replace('_',' ').title()} ({score_obj.score:.0f}/100): {score_obj.reasoning}")
        else:
            lines.append("  No standout strengths — balanced profile with no clear differentiator.")
        lines.append("")

        lines.append("  Risks")
        lines.append("  -----")
        if weak_cats:
            for cat in weak_cats:
                score_obj = getattr(sc, cat)
                lines.append(f"  {cat.replace('_',' ').title()} ({score_obj.score:.0f}/100): {score_obj.reasoning}")
        if sc.risk_flags:
            for flag in sc.risk_flags[:3]:
                lines.append(f"  • {flag}")
        if not weak_cats and not sc.risk_flags:
            lines.append("  No material risks identified at this time.")
        lines.append("")

        lines.append("  Valuation View")
        lines.append("  --------------")
        _peg_display = f"{val_range.peg_ratio:.2f}x" if (val_range and val_range.peg_ratio is not None) else "N/A"
        lines.append(f"  P/E: {pe_str}  |  P/S: {ps_str}  |  EV/EBITDA: {ev_str}  |  PEG: {_peg_display}")
        if val_range and val_range.peg_ratio is not None and val_range.peg_interpretation:
            lines.append(f"  {val_range.peg_interpretation}")
        if sc.valuation and sc.valuation.data_quality != "missing":
            # Emit the reasoning — this will already include the PEG/PE tension
            # narrative if the valuation scorer detected a contradiction.
            lines.append(f"  {sc.valuation.reasoning}")
            # Show the tension factor first if present (it's the last factor appended)
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

        _growth_score = (
            sc.growth.score
            if sc.growth and sc.growth.data_quality != "missing"
            else None
        )
        lines += self._build_valuation_range_section(val_range, price, growth_score=_growth_score)

        # ── Peer comparison ────────────────────────────────────────────────────
        try:
            _peg    = val_range.peg_ratio       if val_range else None
            _growth = val_range.eps_growth_rate if val_range else None
            peer_cmp = build_peer_comparison(
                target_ticker=ticker,
                target_pe=pe_val,
                target_ps=ps_val,
                target_growth=_growth,
                target_peg=_peg,
                sector=sector,
                industry=industry,
                target_mkt_cap=stock_data.market_cap if stock_data else None,
            )
            lines += self._build_peer_comparison_section(peer_cmp)
        except Exception as _exc:
            print(f"  [REPORT] peer comparison failed — {_exc}")

        lines.append("  Final Verdict")
        lines.append("  -------------")
        verdict = self._write_verdict(sc)
        lines.append(f"  {verdict}")
        _kt = self._write_key_tension(sc)
        if _kt:
            lines.append(f"  Key Tension  : {_kt}")
        lines.append("")

        _beta = getattr(stock_data.profile, "beta", None) if stock_data and stock_data.profile else None
        lines += self._build_position_sizing_section(sc, macro_findings, beta=_beta)

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
        regime   = macro.get("macro_regime") or macro.get("regime", "")
        risk     = macro.get("recession_risk_level") or macro.get("recession_risk", "")

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
            f"  (regime={regime!r}, recession_risk={risk!r}, delta={delta:+.2f})"
        )

    def _build_macro_section(self, macro: dict) -> list[str]:
        """
        Return a compact, investment-oriented macro section as a list of lines.
        Caller appends these directly into the memo line list.
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
                "    Macro data unavailable — no LEI-based macro overlay was applied."
            )
            lines.append("")
            return lines

        # Header line: regime | score | recession risk
        score_str = f"{score:.0f}/100" if score is not None else "N/A"
        lines.append(
            f"    Regime: {regime}  |  Score: {score_str}  |  Recession Risk: {risk}"
        )

        # Sector tilt
        if tilt:
            lines.append(f"    Sector Tilt: {tilt}")

        # Tailwind / neutral / headwind verdict + confidence modifier
        verdict = self._macro_verdict(regime, risk)
        lines.append(f"    {verdict}")

        # Key factors (up to 2 each), stripped of trailing detail clauses, one per line
        for f in bullish[:2]:
            lines.append(f"    + {f.split(' — ')[0]}")
        for f in bearish[:2]:
            lines.append(f"    ✕ {f.split(' — ')[0]}")

        lines.append("")
        return lines

    @staticmethod
    def _macro_verdict(regime: str, recession_risk: str) -> str:
        """One-line PM-style macro verdict."""
        if regime in ("Expansion", "Recovery"):
            base = "Macro is a tailwind — supportive backdrop"
        elif regime == "Slowdown":
            base = "Macro is softening — proceed with selectivity"
        else:  # Contraction
            base = "Macro is a headwind — positioning should be defensive"

        if recession_risk in ("Elevated", "High"):
            return f"{base}; recession risk is {recession_risk.lower()}."
        return f"{base}."

    @staticmethod
    def _build_valuation_range_section(
        vr: "ValuationRange | None",
        current_price: "float | None",
        growth_score: "float | None" = None,
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
        lines: list[str] = ["  Valuation Range", "  ───────────────"]

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
        if pm == "P/E" and vr.scenario_bear_pe is not None:
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
        sc: Scorecard,
        macro: dict,
        beta: float | None = None,
    ) -> list[str]:
        """
        Derive a position sizing framework from stance, confidence, macro regime,
        recession risk, risk flags, beta (volatility), and valuation score.

        Tier (from stance + confidence + macro + flags) sets the base max size.
        Beta and valuation multipliers then scale it up or down.
        A hard cap of 10% is applied last.

        Base max by tier:  conviction=8%  moderate=5%  cautious=3%  avoid=0%
        Beta multiplier  : >1.5 → ×0.80  |  <0.8 → ×1.10  |  else ×1.0
        Val  multiplier  : score<45 → ×0.80  |  score>70 → ×1.10  |  else ×1.0
        Hard cap         : 10% regardless of multipliers
        """
        regime         = macro.get("macro_regime") or macro.get("regime", "Unknown")
        recession_risk = macro.get("recession_risk_level") or macro.get("recession_risk", "Unknown")
        flag_count     = len(sc.risk_flags)
        stance         = sc.stance
        confidence     = sc.confidence
        val_score      = (
            sc.valuation.score
            if sc.valuation and sc.valuation.data_quality != "missing"
            else None
        )

        # ── Base tier ──────────────────────────────────────────────────────────
        if stance == Stance.BEARISH:
            tier = "avoid"
        elif stance == Stance.BULLISH and confidence >= 0.75:
            tier = "conviction"
        elif stance == Stance.BULLISH:
            tier = "moderate"
        else:
            tier = "cautious"

        # ── Macro step-down ────────────────────────────────────────────────────
        macro_headwind = regime == "Contraction" or recession_risk in ("Elevated", "High")
        _order = ["conviction", "moderate", "cautious", "avoid"]
        if macro_headwind and tier != "avoid":
            tier = _order[min(_order.index(tier) + 1, len(_order) - 1)]

        # ── Weak fundamental step-down ─────────────────────────────────────────
        # Growth < 40 or momentum < 40 cap conviction → moderate regardless of macro.
        growth_score_ps = (
            sc.growth.score if sc.growth and sc.growth.data_quality != "missing" else None
        )
        momentum_score_ps = (
            sc.momentum.score if sc.momentum and sc.momentum.data_quality != "missing" else None
        )
        fundamentals_weak = (
            (growth_score_ps is not None and growth_score_ps < 40)
            or (momentum_score_ps is not None and momentum_score_ps < 40)
        )
        if fundamentals_weak and tier == "conviction":
            tier = "moderate"

        # ── Risk flag cap ──────────────────────────────────────────────────────
        if flag_count >= 3 and tier == "conviction":
            tier = "moderate"

        if tier == "avoid":
            rationale = "Bearish stance — no position recommended until the view improves."
            return [
                "  Position Sizing Guidance",
                "  ────────────────────────",
                f"    {rationale}",
                "",
            ]

        # ── Base max size ──────────────────────────────────────────────────────
        base_max = {"conviction": 8.0, "moderate": 5.0, "cautious": 3.0}[tier]

        # ── Beta (volatility) multiplier ───────────────────────────────────────
        beta_mult = 1.0
        beta_note = ""
        if beta is not None:
            if beta > 1.5:
                beta_mult = 0.80
                beta_note = f"high beta ({beta:.1f}×) caps upside"
            elif beta < 0.8:
                beta_mult = 1.10
                beta_note = f"low beta ({beta:.1f}×) supports steadier sizing"

        # ── Valuation multiplier ───────────────────────────────────────────────
        val_mult = 1.0
        val_note = ""
        if val_score is not None:
            if val_score < 45:
                val_mult = 0.80
                val_note = "stretched valuation limits upside"
            elif val_score > 70:
                val_mult = 1.10
                val_note = "attractive valuation supports fuller sizing"

        # ── Apply multipliers and hard cap ─────────────────────────────────────
        # Standard hard cap is 8%; only an unmodified conviction tier can reach
        # 10% (exceptional case where no risk overlays are active).
        standard_cap = 8.0
        exceptional_cap = 10.0
        cap = exceptional_cap if (tier == "conviction" and beta_mult == 1.0 and val_mult == 1.0 and not macro_headwind and flag_count < 3) else standard_cap
        max_pct = min(base_max * beta_mult * val_mult, cap)

        # ── Round to nearest 0.5%, derive starter and add-on ──────────────────
        def _r(v: float) -> float:
            return round(v * 2) / 2

        def _fmt(v: float) -> str:
            return f"{int(v)}%" if v == int(v) else f"{v:.1f}%"

        max_r       = _r(max_pct)
        starter_lo  = max(_r(max_pct * 0.35), 0.5)
        starter_hi  = max(_r(max_pct * 0.50), starter_lo + 0.5)
        addon_max   = max(_r(max_pct * 0.75), starter_hi + 0.5)

        starter_str = f"{_fmt(starter_lo)}–{_fmt(starter_hi)}"
        addon_str   = f"up to {_fmt(addon_max)}"
        max_str     = _fmt(max_r)

        # ── Rationale ─────────────────────────────────────────────────────────
        modifiers: list[str] = []
        if macro_headwind:
            modifiers.append(f"macro headwinds ({regime})")
        if beta_note:
            modifiers.append(beta_note)
        if val_note:
            modifiers.append(val_note)
        if flag_count >= 3:
            modifiers.append(f"{flag_count} risk flags")
        elif flag_count > 0:
            modifiers.append(f"{flag_count} risk flag{'s' if flag_count > 1 else ''}")
        if growth_score_ps is not None and growth_score_ps < 40:
            modifiers.append(f"weak growth ({growth_score_ps:.0f}/100)")
        if momentum_score_ps is not None and momentum_score_ps < 40:
            modifiers.append(f"weak momentum ({momentum_score_ps:.0f}/100)")

        has_fundamental_weakness = fundamentals_weak or flag_count > 0
        _conf_label = (
            "high confidence" if confidence >= 0.70
            else "moderate confidence" if confidence >= 0.50
            else "lower confidence"
        )

        if not modifiers:
            rationale = (
                f"Strong {stance.value.lower()} case with {_conf_label}"
                " and no material risk overlays — supports a fuller allocation range."
            )
        elif has_fundamental_weakness:
            mod_str = "; ".join(modifiers)
            rationale = (
                f"{stance.value} thesis ({_conf_label})"
                f" — position with caution given: {mod_str}."
            )
        else:
            mod_str = "; ".join(modifiers)
            rationale = (
                f"{stance.value} thesis ({_conf_label}) scaled by: {mod_str}."
            )

        return [
            "  Position Sizing Guidance",
            "  ────────────────────────",
            f"    Starter position : {starter_str}  of portfolio",
            f"    Add-on trigger   : {addon_str}  if conviction improves",
            f"    Max suggested    : {max_str}  of portfolio",
            f"    {rationale}",
            "",
        ]

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
        if price and inc:
            sh = (mc_api / price) if (mc_api and price > 0) else None
            eps = inc.eps_diluted or inc.eps
            if eps is None and inc.net_income and sh and sh > 0:
                eps = inc.net_income / sh
            pe_computed = round(price / eps, 2) if (eps and eps > 0) else None
            print(f"  [AUDIT] pe_computed            = {pe_computed}  (current_price/annual_eps)")
            if pe_computed and pe_displayed:
                diff_pe = abs(pe_computed - pe_displayed) / pe_displayed * 100
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
            lines.append(
                f"    {marker} {row.ticker:<8s}"
                f"  {_fv(row.pe, 'x'):>6s}"
                f"  {_fv(row.ps, 'x'):>6s}"
                f"  {_fv(row.growth_pct, '%'):>7s}"
                f"  {_fv(row.peg, 'x'):>6s}"
            )

        lines.append("")

        if pc.insights:
            for insight in pc.insights:
                lines.append(f"    • {insight}")
            lines.append("")

        return lines

    def _write_verdict(self, sc: Scorecard) -> str:
        stance = sc.stance.value
        score  = sc.overall_score
        conf   = sc.confidence

        def _s(attr: str) -> float | None:
            cat = getattr(sc, attr, None)
            return cat.score if cat and cat.data_quality != "missing" else None

        growth_s = _s("growth")
        val_s    = _s("valuation")
        mom_s    = _s("momentum")
        prof_s   = _s("profitability")

        if stance == "Bullish":
            if growth_s is not None and growth_s >= 65 and mom_s is not None and mom_s >= 65:
                body = (
                    "Growth and momentum are both well above average,"
                    " providing dual confirmation of the bullish setup."
                )
            elif val_s is not None and val_s >= 65 and growth_s is not None and growth_s >= 55:
                body = (
                    "Attractive valuation combined with improving growth"
                    " creates a favourable risk/reward at current prices."
                )
            elif val_s is not None and val_s >= 65:
                body = (
                    "The valuation case is compelling; upside is available"
                    " if the growth trajectory holds."
                )
            elif mom_s is not None and mom_s < 45:
                # Explicitly name the fundamental/momentum split instead of silently blending
                fund_s_list = [s for s in [growth_s, prof_s, val_s] if s is not None]
                fund_avg = sum(fund_s_list) / len(fund_s_list) if fund_s_list else None
                if fund_avg is not None:
                    body = (
                        f"Fundamentals (avg {fund_avg:.0f}/100) are driving the bullish view"
                        f" despite weak momentum ({mom_s:.0f}/100)."
                        " Momentum carries 10% weight; consider waiting for price confirmation."
                    )
                else:
                    body = (
                        "Weak momentum is a near-term concern, but fundamentals"
                        " are driving the overall bullish stance."
                    )
            else:
                body = (
                    "The fundamental setup is constructive, though the margin"
                    " of conviction is moderate — size accordingly."
                )
            return f"Score {score:.0f}/100, {conf:.0%} confidence — Bullish. {body}"

        elif stance == "Bearish":
            if growth_s is not None and growth_s < 40:
                body = (
                    f"Weak growth ({growth_s:.0f}/100) is the primary headwind;"
                    " the fundamental profile does not justify the current price."
                )
            elif val_s is not None and val_s < 40:
                body = (
                    "Valuation appears stretched relative to the fundamental outlook"
                    " — risk/reward is not compelling at current levels."
                )
            elif mom_s is not None and mom_s < 40:
                body = (
                    "Deteriorating momentum signals near-term selling pressure;"
                    " revisit when technical conditions stabilise."
                )
            else:
                body = (
                    "Risk factors outweigh near-term catalysts at current prices."
                    " Revisit if the conditions listed above materially improve."
                )
            return f"Score {score:.0f}/100, {conf:.0%} confidence — Bearish. {body}"

        else:  # Neutral
            if val_s is not None and val_s >= 60 and growth_s is not None and growth_s < 45:
                body = (
                    "Valuation looks supportive, but weak growth"
                    " limits the upside case to a re-rating scenario."
                )
            elif growth_s is not None and growth_s >= 60 and val_s is not None and val_s < 45:
                body = (
                    "Growth quality is solid, but current pricing"
                    " compresses the margin of safety — patience is warranted."
                )
            elif (
                growth_s is not None and growth_s < 45
                and mom_s is not None and mom_s < 45
            ):
                body = (
                    "Neither growth nor momentum provides a near-term catalyst;"
                    " the setup does not favour initiating a position at this stage."
                )
            elif prof_s is not None and prof_s >= 65:
                body = (
                    "Strong profitability provides a floor, but without a clear growth"
                    " or valuation catalyst the risk/reward is roughly in balance."
                )
            else:
                body = (
                    "No single driver is strong enough to shift the stance;"
                    " the risk/reward is roughly in balance at current prices."
                )
            return f"Score {score:.0f}/100, {conf:.0%} confidence — Neutral/Hold. {body}"

    @staticmethod
    def _score_bar(score: float | None, width: int = 20) -> str:
        if score is None:
            return "[" + "?" * width + "]"
        filled = int(score / 100 * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    @staticmethod
    def _write_key_tension(sc: "Scorecard") -> "str | None":
        """
        Return a one-sentence framing of the core investment tension, or None
        when scores are too sparse to identify a meaningful tradeoff.
        Evaluated in priority order — the first match wins.
        """
        def _s(attr: str) -> "float | None":
            cat = getattr(sc, attr, None)
            return cat.score if cat and cat.data_quality != "missing" else None

        g  = _s("growth")
        p  = _s("profitability")
        v  = _s("valuation")
        m  = _s("momentum")
        fh = _s("financial_health")

        # Valuation vs growth direction
        if v is not None and g is not None:
            if v < 40 and g >= 65:
                return "Premium valuation vs strong growth profile."
            if v >= 65 and g < 40:
                return "Attractive valuation vs limited growth runway."
            if v < 40 and g < 45:
                return "Stretched valuation vs weak fundamental justification."

        # Growth execution vs margin delivery
        if g is not None and p is not None and g >= 65 and p < 45:
            return "High growth ambition vs unproven margin delivery."

        # Momentum vs valuation
        if m is not None and v is not None and m >= 70 and v < 40:
            return "Price momentum vs stretched valuation at current levels."

        # Quality vs near-term sentiment
        if p is not None and m is not None and p >= 65 and m < 45:
            return "Quality business profile vs weak near-term price action."

        # Growth vs balance sheet
        if fh is not None and g is not None and fh < 45 and g >= 60:
            return "Growth trajectory vs balance sheet vulnerability."

        # Solid fundamentals, no market conviction
        if g is not None and m is not None and g >= 60 and m < 45:
            return "Solid fundamentals vs weak near-term market conviction."

        return None

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

        # CAGR helper
        def _cagr(start: float, end: float, periods: int) -> float | None:
            if start <= 0 or end is None or periods < 1:
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
