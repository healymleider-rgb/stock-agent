"""
qxo_scenario_run.py
===================
One-shot analysis: feed the QXO Excel base case into the valuation system's
scenario tree and Monte Carlo engines.

Inputs are taken entirely from the extracted Excel JSON.  No live data fetch.

Reference price (CURRENT_PRICE) is a required input — set to the actual
market price before interpreting return distributions.  The default below
is $12.00 for illustration only.

Run:
    python3 -m analysis.qxo_scenario_run
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.scenario_tree import build_scenario_tree
from analysis.monte_carlo   import (
    run_monte_carlo,
    GrowthDistParams,
    MultipleDistParams,
)

# ─── Excel base case (from extracted JSON) ───────────────────────────────────
EXCEL = {
    "revenue_growth":      0.3662,
    "margin_current":      0.168,
    "margin_target":       0.38,
    "eps_y1":              0.40,
    "eps_y2":              0.71,
    "eps_y3":              1.02,
    "pt_base":             14.74,
    "pt_low":              1.56,
    "pt_high":             29.32,
    "implied_pe":          36.85,
    "pe_low":              3.90,
    "pe_high":             73.29,
}

# ─── Reference price (SET THIS TO ACTUAL MARKET PRICE) ───────────────────────
# Returns in all outputs are relative to this price.
# Default 12.00 implies ~+23% upside to the base PT of $14.74.
CURRENT_PRICE = 12.00

N_SIMS        = 10_000
MACRO_REGIME  = "base"          # current macro environment
HORIZON_YEARS = 1               # 1-year horizon to 2026 PT


# ─── Build scenario tree ──────────────────────────────────────────────────────
def run_scenario_tree():
    """
    Run the Markov-based scenario tree.

    Note: the system's internal probability tables are calibrated for
    established, positive-EPS companies.  For a turnaround like QXO
    (negative → positive EPS inflection), these outputs will
    understate integration-failure risk.  Use the MC with custom
    GrowthDistParams as the primary distribution estimate.
    """
    tree = build_scenario_tree(
        macro_regime   = MACRO_REGIME,
        earnings_trend = "accelerating",   # negative → positive = maximum acceleration
        current_pe     = EXCEL["implied_pe"],
        base_eps       = EXCEL["eps_y1"],
        current_price  = CURRENT_PRICE,
        factor_profile = None,
        hrl_result     = None,
    )
    return tree


# ─── Build MC with acquisition-distorted / binary-outcome parameters ─────────
def run_mc_binary():
    """
    Custom GrowthDistParams that explicitly model the binary acquisition
    integration outcome rather than a standard normal distribution.

    Architecture
    ------------
    metric_current = EXCEL eps_y1 = $0.40
        The MC treats Year 1 EPS as the starting metric.  g = 0 means
        the base case ($0.40) is achieved exactly.  g < 0 means a miss;
        g < -1.0 means EPS goes negative again.

    GrowthDistParams (acquisition-distorted, fat-tailed)
    ──────────────────────────────────────────────────────
    growth_mean  = 0.0       Centered on base ($0.40 hit exactly)
    sigma_down   = 0.65      Wide left tail: 1σ-down EPS = $0.14
                             Reflects: integration drag, cost overrun, demand miss
    sigma_up     = 0.40      Narrower right tail: 1σ-up EPS = $0.56
                             Reflects: synergy beat, faster margin recovery
    shock_prob   = 0.22      22% P(integration failure shock)
                             Anchored to binary outcome prior (not macro shock)
    shock_mean   = -0.80     Shock EPS = $0.40 × (1 − 0.80) ≈ $0.08
                             Models: integration delay of 4–6 quarters

    MultipleDistParams (mean-reverting Beta, very wide dispersion)
    ──────────────────────────────────────────────────────────────
    low          = 3.90      Excel low (from PT=$1.56 / EPS=$0.40)
    high         = 73.29     Excel high (from PT=$29.32 / EPS=$0.40)
    current      = 36.85     Excel implied P/E
    fair         = 26.0      Sector-adjusted fair P/E for building distribution
                             (historical mid-cycle for industrials/distribution)
    mr_speed     = 0.35      Moderate reversion toward fair value
    concentration= 2.5       Very low → very wide Beta; reflects binary uncertainty
    correlation_rho = 0.65   High: earnings beats → multiple expansion (and vice versa)
    """
    gp = GrowthDistParams(
        growth_mean  = 0.0,
        sigma_down   = 0.65,
        sigma_up     = 0.40,
        shock_prob   = 0.22,
        shock_mean   = -0.80,
        shock_std    = 0.25,
        quality_tier = "cyclical",
        macro_regime = MACRO_REGIME,
    )

    mp = MultipleDistParams(
        low              = EXCEL["pe_low"],    # 3.90
        high             = EXCEL["pe_high"],   # 73.29
        current          = EXCEL["implied_pe"], # 36.85
        fair             = 26.0,
        mr_speed         = 0.35,
        concentration    = 2.5,
        rate_adj         = -0.04,
        correlation_rho  = 0.65,
        corr_sensitivity = 0.10,
        quality_tier     = "cyclical",
        macro_regime     = MACRO_REGIME,
    )

    mc = run_monte_carlo(
        current_price   = CURRENT_PRICE,
        metric_current  = EXCEL["eps_y1"],     # Year 1 forward EPS = $0.40
        growth_mean     = 0.0,
        growth_std      = 0.525,               # midpoint of sigma_down/sigma_up
        multiple_bear   = EXCEL["pe_low"],
        multiple_base   = EXCEL["implied_pe"],
        multiple_bull   = EXCEL["pe_high"],
        horizon_years   = HORIZON_YEARS,
        n_sims          = N_SIMS,
        method          = "P/E",
        growth_params   = gp,
        multiple_params = mp,
    )
    return mc, gp, mp


# ─── Scenario assumptions table (Bear / Base / Bull) ─────────────────────────
def scenario_assumptions():
    """
    Map Excel inputs to Bear / Base / Bull scenario rows.
    All returns are relative to CURRENT_PRICE.
    """
    cp = CURRENT_PRICE
    rows = {
        "Bear": {
            "eps_y1":              0.10,
            "eps_driver":          "Integration drag; margin stays at ~12–15%",
            "multiple":            10.0,
            "multiple_rationale":  "Failed turnaround re-rated to distressed/value",
            "price_target_1yr":    round(0.10 * 10.0, 2),
            "return_vs_ref":       round((0.10 * 10.0 / cp - 1) * 100, 1),
            "margin_y1":           0.13,
            "prob_weight":         0.30,
        },
        "Base": {
            "eps_y1":              EXCEL["eps_y1"],
            "eps_driver":          "Excel model on schedule; margin recovers toward 22–25%",
            "multiple":            EXCEL["implied_pe"],
            "multiple_rationale":  "Market prices the turnaround at growth-company P/E",
            "price_target_1yr":    EXCEL["pt_base"],
            "return_vs_ref":       round((EXCEL["pt_base"] / cp - 1) * 100, 1),
            "margin_y1":           0.22,
            "prob_weight":         0.48,
        },
        "Bull": {
            "eps_y1":              0.60,
            "eps_driver":          "Faster synergy capture; margin toward 30%+",
            "multiple":            58.0,
            "multiple_rationale":  "Re-rating to premium growth multiple on execution",
            "price_target_1yr":    round(0.60 * 58.0, 2),
            "return_vs_ref":       round((0.60 * 58.0 / cp - 1) * 100, 1),
            "margin_y1":           0.28,
            "prob_weight":         0.22,
        },
    }
    return rows


# ─── Print output ─────────────────────────────────────────────────────────────
def print_results():
    print("\n" + "=" * 70)
    print("  QXO — PROBABILISTIC VALUATION FRAMEWORK")
    print(f"  Excel base case fed into scenario tree + Monte Carlo")
    print(f"  Reference price: ${CURRENT_PRICE:.2f}  |  Horizon: {HORIZON_YEARS}yr  |  N={N_SIMS:,}")
    print("=" * 70)

    # ── Scenario assumptions ───────────────────────────────────────────────────
    rows = scenario_assumptions()
    print("\n── SCENARIO ASSUMPTIONS (Bear / Base / Bull) ───────────────────────")
    header = f"  {'Scenario':<8}  {'EPS Y1':>8}  {'P/E':>8}  {'PT ($)':>8}  {'Return':>8}  {'Margin':>8}  {'Prob':>6}"
    print(header)
    print("  " + "-" * 64)
    for name, r in rows.items():
        print(
            f"  {name:<8}  ${r['eps_y1']:>7.2f}  {r['multiple']:>7.1f}x"
            f"  ${r['price_target_1yr']:>7.2f}  {r['return_vs_ref']:>+7.1f}%"
            f"  {r['margin_y1']:>7.1%}  {r['prob_weight']:>5.0%}"
        )

    # ── Scenario tree ─────────────────────────────────────────────────────────
    print("\n── SCENARIO TREE (Markov / system model) ───────────────────────────")
    print("  NOTE: system tables calibrated for positive-EPS companies.")
    print("  EPS inflection risk is understated here — see MC below for primary.\n")
    tree = run_scenario_tree()
    if tree is not None:
        print(f"  Leaves:               {len(tree.leaves)}")
        print(f"  Weighted E[R]:        {tree.weighted_return:+.1%}")
        print(f"  Scenario std:         {tree.scenario_std:.1%}")
        print(f"  VaR (P5 leaf):        {tree.var_95:.1%}")
        print(f"  Shock prob:           {tree.shock_prob:.1%}")
        print(f"  Downside mass (<−20%):{tree.downside_mass:.1%}")
        print(f"  Upside mass (>+20%):  {tree.upside_mass:.1%}")
        if tree.best_case:
            bc = tree.best_case
            print(f"  Bull case:  {bc.label:<38}  ${bc.target_price:>6.2f}  ({bc.expected_return:+.0%})")
        if tree.worst_case:
            wc = tree.worst_case
            print(f"  Bear case:  {wc.label:<38}  ${wc.target_price:>6.2f}  ({wc.expected_return:+.0%})")
        print(f"\n  Top leaves by probability:")
        sorted_leaves = sorted(tree.leaves, key=lambda l: l.probability, reverse=True)
        for l in sorted_leaves[:6]:
            print(
                f"    {l.probability:>5.1%}  {l.label:<42}"
                f"  ${l.target_price:>6.2f}  ({l.expected_return:+.0%})"
            )
    else:
        print("  Scenario tree could not be built (likely: no current_price).")

    # ── Monte Carlo ────────────────────────────────────────────────────────────
    print("\n── MONTE CARLO (custom binary/fat-tailed distribution) ──────────────")
    print("  PRIMARY distribution estimate for this turnaround/acquisition bet.\n")
    mc, gp, mp = run_mc_binary()

    print(f"  GrowthDistParams (acquisition-distorted):")
    print(f"    growth_mean  = {gp.growth_mean:+.2f}   (centered on base EPS ${EXCEL['eps_y1']:.2f})")
    print(f"    sigma_down   = {gp.sigma_down:.2f}    (1σ-down EPS = ${EXCEL['eps_y1'] * (1 - gp.sigma_down):.2f})")
    print(f"    sigma_up     = {gp.sigma_up:.2f}    (1σ-up  EPS = ${EXCEL['eps_y1'] * (1 + gp.sigma_up):.2f})")
    print(f"    shock_prob   = {gp.shock_prob:.0%}   (integration failure)")
    print(f"    shock_mean   = {gp.shock_mean:.2f}   (EPS in shock ≈ ${EXCEL['eps_y1'] * (1 + gp.shock_mean):.2f})")
    print(f"\n  MultipleDistParams:")
    print(f"    range        = {mp.low:.1f}x – {mp.high:.1f}x")
    print(f"    current      = {mp.current:.1f}x   fair = {mp.fair:.1f}x")
    print(f"    concentration= {mp.concentration:.1f}  (2.5 = very wide / binary)")
    print(f"    corr_rho     = {mp.correlation_rho:.2f}   (earnings beat → multiple expansion)")

    print(f"\n  Return distribution ({N_SIMS:,} paths, horizon={HORIZON_YEARS}yr):")
    print(f"    Mean return:     {mc.mean_return:+.1%}")
    print(f"    Median return:   {mc.median_return:+.1%}")
    print(f"    P5  (downside):  {mc.p5_return:+.1%}   →  ${mc.p5_price:.2f}")
    print(f"    P25:             {mc.p25_return:+.1%}   →  ${mc.p25_price:.2f}")
    print(f"    P75:             {mc.p75_return:+.1%}   →  ${mc.p75_price:.2f}")
    print(f"    P95 (upside):    {mc.p95_return:+.1%}   →  ${mc.p95_price:.2f}")
    print(f"    Skewness:        {mc.skewness:+.2f}")

    print(f"\n  Outcome probabilities:")
    print(f"    P(gain):         {mc.prob_positive:.0%}")
    print(f"    P(>20% gain):    {mc.prob_20_gain:.0%}")
    print(f"    P(loss):         {mc.prob_loss:.0%}")
    print(f"    P(>20% loss):    {mc.prob_loss_20:.0%}")

    print(f"\n  Distribution shape: {mc.upside_skew_label}")
    print(f"  Upside/downside:   {mc.upside_downside:.2f}x  (>2.5 = right-skewed)")
    print(f"  Risk label:        {mc.risk_label}")
    print(f"  Half-Kelly:        {mc.kelly_fraction:.1%}")

    # ── MC parameter ranges summary ───────────────────────────────────────────
    print("\n── MONTE CARLO INPUT RANGES (summary) ──────────────────────────────")
    print(f"  {'Parameter':<28}  {'Bear':>10}  {'Base':>10}  {'Bull':>10}")
    print("  " + "-" * 64)
    rows_mc = [
        ("EPS Y1 (metric)",    f"${EXCEL['eps_y1']*(1+gp.shock_mean):.2f}", f"${EXCEL['eps_y1']:.2f}",  f"${EXCEL['eps_y1']*(1+gp.sigma_up):.2f}"),
        ("EPS growth (around base)", f"{gp.shock_mean:.0%}",                f"{gp.growth_mean:+.0%}",   f"+{gp.sigma_up:.0%}"),
        ("sigma_down / sigma_up",    f"σ↓={gp.sigma_down:.2f}",            f"—",                        f"σ↑={gp.sigma_up:.2f}"),
        ("Shock prob",               f"{gp.shock_prob:.0%}",                f"—",                        f"0%"),
        ("Exit multiple",            f"{mp.low:.1f}x",                       f"{mp.current:.1f}x",       f"{mp.high:.1f}x"),
        ("Multiple fair value",      f"—",                                   f"{mp.fair:.1f}x",          f"—"),
        ("Margin assumption",        "12–15%",                               "~22–25%",                  "28–35%"),
        ("P/E concentration",        f"β conc={mp.concentration:.1f}",      f"(very wide)",             f"—"),
        ("Growth-multiple ρ",        f"—",                                   f"{mp.correlation_rho:.2f}", f"—"),
    ]
    for r in rows_mc:
        print(f"  {r[0]:<28}  {r[1]:>10}  {r[2]:>10}  {r[3]:>10}")

    # ── Acquisition distortion note ───────────────────────────────────────────
    print("\n── ACQUISITION DISTORTION FLAGS ────────────────────────────────────")
    flags = [
        "Revenue CAGR of 36.6% includes acquired-entity consolidation. "
        "Organic growth is lower — do not use 36.6% as an organic run-rate.",

        "Margin compressed 38% → 16.8% post-acquisition. Distribution is "
        "bimodal: integration success (→ 35%+) vs. drag (→ 12–15% floor). "
        "The GrowthDistParams left tail (σ_down=0.65) models this explicitly.",

        "EPS turning from −$0.633 → +$0.40 is a single-year discrete jump. "
        "The shock_prob=22% component models the risk that the inflection "
        "does NOT occur in Year 1 as modeled.",

        "Model applies constant 36.85× P/E across all 3 forecast years. "
        "This is a specific assumption — not a neutral multiple; "
        "the MultipleDistParams fair value is anchored at 26× (sector median).",

        "Earnings × multiple correlation (ρ=0.65) is elevated because "
        "turnaround names are especially vulnerable to multiple compression "
        "on execution misses and re-rating on beats.",
    ]
    for i, f in enumerate(flags, 1):
        words = f.split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 > 64:
                print(f"  [{i}] {line}" if not line.startswith(" ") else f"      {line}")
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            print(f"  [{i}] {line}" if not (line.startswith("[")) else f"      {line}")

    print("\n" + "=" * 70 + "\n")
    return tree, mc


if __name__ == "__main__":
    print_results()
