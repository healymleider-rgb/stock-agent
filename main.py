"""
Stock Evaluation Engine — entry point.

Usage
─────
  python main.py AAPL
  python main.py MSFT --verbose
  python main.py NVDA --no-memo
  python main.py TSLA --history       # show past evaluations
"""
from __future__ import annotations

import argparse
import sys
import time

from colorama import Fore, Style, init as colorama_init

from agents.orchestrator_agent import OrchestratorAgent
from config import Config
from memory.evaluation_memory import EvaluationMemory
from models.scorecard import Scorecard, Stance
from models.state import EvaluationState
from utils.helpers import format_large_number
from utils.logger import logger

colorama_init(autoreset=True)


# ── CLI argument parsing ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Agentic stock evaluation engine powered by Financial Modeling Prep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL
  python main.py NVDA --verbose
  python main.py TSLA --no-memo
  python main.py MSFT --history
        """,
    )
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print the orchestrator's step-by-step reasoning log",
    )
    parser.add_argument(
        "--no-memo",
        action="store_true",
        help="Skip the investment memo section",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Show past evaluations for this ticker from memory",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save this evaluation to memory",
    )
    return parser.parse_args()


# ── Color helpers ──────────────────────────────────────────────────────────────

def _color_score(score: float) -> str:
    if score >= 70:
        return Fore.GREEN + f"{score:.0f}" + Style.RESET_ALL
    elif score >= 50:
        return Fore.YELLOW + f"{score:.0f}" + Style.RESET_ALL
    else:
        return Fore.RED + f"{score:.0f}" + Style.RESET_ALL


def _color_stance(stance: Stance) -> str:
    if stance == Stance.BULLISH:
        return Fore.GREEN + stance.value.upper() + Style.RESET_ALL
    elif stance == Stance.BEARISH:
        return Fore.RED + stance.value.upper() + Style.RESET_ALL
    else:
        return Fore.YELLOW + stance.value.upper() + Style.RESET_ALL


# ── History display ────────────────────────────────────────────────────────────

def show_history(ticker: str, memory: EvaluationMemory) -> None:
    history = memory.get_history(ticker, limit=10)
    if not history:
        print(f"\nNo past evaluations found for {ticker}.")
        return
    print(f"\n{'─'*60}")
    print(f"  Past evaluations for {ticker}  ({len(history)} records)")
    print(f"{'─'*60}")
    for entry in history:
        ts = entry.get("timestamp", "")[:19].replace("T", " ")
        score = entry.get("overall_score", "N/A")
        stance = entry.get("stance", "N/A")
        conf = entry.get("confidence", 0)
        print(f"  {ts}  Score: {score:.0f}  Stance: {stance}  Conf: {conf:.0%}")
    print()


# ── Data verification ──────────────────────────────────────────────────────────

def _print_data_verification(ticker: str, state: EvaluationState, scorecard: Scorecard) -> None:
    """
    Print a quick table of ticker-specific raw values immediately after
    evaluation so it's easy to confirm real data is being used, not defaults.
    """
    sd = state.stock_data
    ratios = sd.latest_ratios
    fv = lambda x: f"{x:.1f}" if x is not None else "N/A"

    pe_val  = ratios.pe_ratio     if ratios else None
    ps_val  = ratios.ps_ratio     if ratios else None
    ev_val  = ratios.ev_to_ebitda if ratios else None
    val_src = "api_ratios"

    if pe_val is None or ps_val is None or ev_val is None:
        _inc = sd.latest_income
        _bal = sd.latest_balance
        _pr  = sd.current_price
        _mc  = sd.market_cap
        _sh  = (_mc / _pr) if (_mc and _pr and _pr > 0) else None

        if pe_val is None and _pr and _inc and _sh:
            _eps = _inc.eps_diluted or _inc.eps
            if _eps is None and _inc.net_income and _sh > 0:
                _eps = _inc.net_income / _sh
            if _eps and _eps > 0:
                pe_val = round(_pr / _eps, 2)

        if ps_val is None and _mc and _inc and _inc.revenue and _inc.revenue > 0:
            ps_val = round(_mc / _inc.revenue, 2)

        if ev_val is None and _mc and _inc and _inc.ebitda and _inc.ebitda > 0:
            _debt = (_bal.total_debt or 0.0) if _bal else 0.0
            _cash = (_bal.cash_and_equivalents or 0.0) if _bal else 0.0
            _ev   = _mc + _debt - _cash
            if _ev > 0:
                ev_val = round(_ev / _inc.ebitda, 2)

        val_src = "derived_metrics" if (pe_val or ps_val or ev_val) else "none"

    print(f"  [REPORT DEBUG] valuation source = {val_src!r}")

    print(f"\n  {'─'*52}")
    print(f"  {Fore.CYAN}Data Verification — {ticker}{Style.RESET_ALL}")
    print(f"  {'─'*52}")
    print(f"  Company       : {sd.profile.company_name if sd.profile else 'N/A'}")
    price_str = f"${sd.current_price:.2f}" if sd.current_price is not None else "N/A"
    print(f"  Current Price : {price_str}")
    print(f"  Market Cap    : {format_large_number(sd.market_cap)}")
    print(f"  P/E Ratio     : {fv(pe_val)}")
    print(f"  P/S Ratio     : {fv(ps_val)}")
    print(f"  EV/EBITDA     : {fv(ev_val)}")

    # Category scores — flag "missing" data so defaults don't look like real scores
    _cat_attrs = ["valuation", "growth", "profitability", "financial_health", "momentum", "risk"]
    score_parts = []
    for attr in _cat_attrs:
        cat = getattr(scorecard, attr, None)
        short = attr[:3].title()
        if cat is None or cat.data_quality == "missing":
            score_parts.append(f"{short}={Fore.RED}N/A{Style.RESET_ALL}")
        else:
            score_parts.append(f"{short}={_color_score(cat.score)}")
    print(f"  Scores        : {', '.join(score_parts)}")
    print(f"  {'─'*52}\n")


# ── Main evaluation flow ──────────────────────────────────────────────────────

def run_evaluation(ticker: str, args: argparse.Namespace) -> int:
    memory = EvaluationMemory()

    if args.history:
        show_history(ticker, memory)
        return 0

    # Check for a recent cached evaluation (optional — currently always re-runs)
    last = memory.get_last_evaluation(ticker)
    if last:
        last_ts = last.get("timestamp", "")[:10]
        print(f"  Note: previous evaluation found ({last_ts}) — running fresh analysis.")

    print(f"\n{Fore.CYAN}  Evaluating {ticker}...{Style.RESET_ALL}")
    print(f"  Agents online: orchestrator, data, fundamental, technical, market, sentiment, risk, reporting\n")

    start_ts = time.time()
    orchestrator = OrchestratorAgent()

    try:
        scorecard, memo, state = orchestrator.evaluate(ticker)
    except EnvironmentError as exc:
        print(f"\n{Fore.RED}Configuration error: {exc}{Style.RESET_ALL}")
        return 1
    except Exception as exc:
        logger.exception("Evaluation failed for %s: %s", ticker, exc)
        print(f"\n{Fore.RED}Evaluation failed: {exc}{Style.RESET_ALL}")
        return 1

    elapsed = time.time() - start_ts

    # ── Verbose: data verification + reasoning log ────────────────────────────
    if args.verbose:
        _print_data_verification(ticker, state, scorecard)
        print(f"\n{Fore.CYAN}  ── Reasoning Log ──{Style.RESET_ALL}")
        for line in state.reasoning_log:
            print(f"  {line}")
        print()
        print(f"  API endpoints called: {len(orchestrator.api_call_log)}")
        for ep in orchestrator.api_call_log:
            print(f"    {ep}")
        print()

    # ── Print memo ─────────────────────────────────────────────────────────────
    if not args.no_memo:
        print(memo)
    else:
        # Compact score summary
        _print_compact(scorecard)

    print(f"  Completed in {elapsed:.1f}s  |  Iterations: {state.iteration}  |  Confidence: {scorecard.confidence:.0%}\n")

    # ── Persist to memory ──────────────────────────────────────────────────────
    if not args.no_save:
        cat = scorecard.category_scores_dict()
        memory.record_evaluation(ticker, {
            "overall_score": scorecard.overall_score,
            "stance": scorecard.stance.value,
            "confidence": scorecard.confidence,
            "valuation": cat.get("valuation"),
            "growth": cat.get("growth"),
            "profitability": cat.get("profitability"),
            "financial_health": cat.get("financial_health"),
            "momentum": cat.get("momentum"),
            "risk": cat.get("risk"),
            "bullish_factors": scorecard.bullish_factors,
            "bearish_factors": scorecard.bearish_factors,
            "risk_flags": scorecard.risk_flags,
        })
        print(f"  Evaluation saved to memory/{ticker}.json")

    return 0


def _print_compact(sc: Scorecard) -> None:
    print(f"\n  Ticker: {sc.ticker}")
    print(f"  Overall Score: {_color_score(sc.overall_score)} / 100")
    print(f"  Stance:        {_color_stance(sc.stance)}")
    print(f"  Confidence:    {sc.confidence:.0%}\n")

    _cat_attrs = ["valuation", "growth", "profitability", "financial_health", "momentum", "risk"]
    for attr in _cat_attrs:
        cat = getattr(sc, attr, None)
        label = attr.replace("_", " ").title().ljust(18)
        if cat is None or cat.data_quality == "missing":
            print(f"    {label}  N/A (no data)")
        else:
            print(f"    {label}  {_color_score(cat.score)}")
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        Config.validate()
    except EnvironmentError as exc:
        print(f"\n{Fore.RED}Startup error: {exc}{Style.RESET_ALL}")
        print("  Copy .env.example to .env and add your FMP_API_KEY.\n")
        sys.exit(1)

    args = parse_args()
    sys.exit(run_evaluation(args.ticker.upper(), args))


if __name__ == "__main__":
    main()
