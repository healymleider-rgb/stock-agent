#!/usr/bin/env python3
"""
Batch-evaluate a list of tickers and generate PDF reports.

Usage:
  python3 scripts/batch_evaluate.py --tickers TTD PYPL AXON --output ./reports
  python3 scripts/batch_evaluate.py --tickers-file tickers.txt --output ./reports

Output:
  {output}/AAPL_2026-05-05.pdf
  {output}/MSFT_2026-05-05.pdf
  ...
  {output}/_index.csv   (one row per ticker: ticker, score, action, sizing, etc.)
  {output}/_summary.md  (human-readable, grouped by BUY/STAGED BUY/HOLD/WAIT/SELL/FAILED)

Notes:
  - The report page accepts ?job_id=xxx to load pre-computed results; no re-evaluation.
  - Playwright waits for [data-report-ready] attribute before capturing the PDF.
  - Evaluations run in parallel (--max-parallel); PDF generation is sequential.
"""

import argparse
import csv
import re
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

from boss_actions import EXCEL_ACTIONS

_ACTION_RE = re.compile(r'^\s*ACTION\s*:\s*(BUY|STAGED BUY|HOLD|WAIT|SELL)\b', re.MULTILINE)

API_BASE      = "http://localhost:8000"
FRONTEND_BASE = "http://localhost:3000"


# ── Evaluation helpers ────────────────────────────────────────────────────────

def submit_evaluation(ticker: str, force_override: bool = False,
                       force_justification: str = "") -> str:
    """Submit a ticker for evaluation. Returns job_id."""
    payload: dict = {"ticker": ticker}
    if force_override:
        payload["force_validation_override"] = True
        payload["force_justification"] = force_justification or "batch run"
    r = requests.post(f"{API_BASE}/api/evaluate", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["job_id"]


def wait_for_completion(job_id: str, timeout_s: int = 120) -> dict:
    """Poll until job is complete/blocked/failed. Returns full job dict."""
    start = time.time()
    while time.time() - start < timeout_s:
        time.sleep(2)
        r = requests.get(f"{API_BASE}/api/jobs/{job_id}", timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("status") in ("complete", "qualified", "blocked", "failed", "error"):
            return data
    raise TimeoutError(f"Job {job_id} did not finish in {timeout_s}s")


def evaluate_one_ticker(ticker: str, force_override: bool = False) -> dict:
    """Evaluate one ticker. Returns summary dict (no PDF yet)."""
    try:
        job_id = submit_evaluation(
            ticker,
            force_override=force_override,
            force_justification="batch run — data quality review deferred",
        )
        job_data = wait_for_completion(job_id, timeout_s=120)

        status = job_data.get("status", "?")
        result = job_data.get("result") or {}
        si     = result.get("stock_info") or {}
        sc     = result.get("scorecard") or {}

        # ACTION is in the memo text: "  ACTION   : HOLD (…)"
        memo   = result.get("memo", "")
        _m     = _ACTION_RE.search(memo)
        action = _m.group(1) if _m else (
            (result.get("scorecard") or {}).get("position_sizing", {}).get("rating")
            or "UNKNOWN"
        )

        row = {
            "ticker":              ticker,
            "status":              status,
            "job_id":              job_id,
            "score":               sc.get("overall_score", "N/A"),
            "action":              action,
            "stance":              sc.get("stance", ""),
            "current_price":       si.get("current_price"),
            "market_cap":          si.get("market_cap"),
            "revenue_growth_yoy":  si.get("revenue_growth_yoy"),
            "eps_growth_yoy":      si.get("eps_growth_yoy"),
            "recommended_now_pct": (sc.get("position_sizing") or {}).get("position_size"),
            "pdf_path": "",
            "error":    None,
        }

        if status not in ("complete", "qualified"):
            # Blocked or failed — still usable for PDF (blocked can be overridden)
            vlog = result.get("validation_log") or {}
            failures = []
            for b in (vlog.get("blocks") or []):
                if not b.get("passed"):
                    failures += b.get("failures") or []
            row["error"] = "; ".join(failures[:3]) or job_data.get("error", "")

        return row

    except Exception as exc:
        return {
            "ticker": ticker, "status": "error", "error": str(exc),
            "job_id": None, "score": None, "action": None,
            "stance": None, "current_price": None, "market_cap": None,
            "revenue_growth_yoy": None, "eps_growth_yoy": None,
            "recommended_now_pct": None, "pdf_path": "",
        }


# ── PDF rendering ─────────────────────────────────────────────────────────────

def render_pdf(ticker: str, job_id: str, output_dir: Path, context) -> Path:
    """
    Render the report page for a ticker as a PDF using Playwright.

    Navigates to /report/{ticker}?job_id={job_id} — the frontend loads the
    pre-computed result directly without starting a new evaluation.
    Waits for [data-report-ready] to appear before capturing the PDF.
    """
    page = context.new_page()
    try:
        url = f"{FRONTEND_BASE}/report/{ticker}?job_id={job_id}"
        # domcontentloaded is enough — we'll wait for the app's own ready signal
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)

        # Wait for the report to finish rendering (replaces networkidle)
        page.wait_for_selector("[data-report-ready]", timeout=30_000)

        # Small settling pause for any chart/animation renders
        page.wait_for_timeout(1_500)

        today     = datetime.now().strftime("%Y-%m-%d")
        out_path  = output_dir / f"{ticker}_{today}.pdf"
        page.pdf(
            path=str(out_path),
            format="Letter",
            print_background=True,
            margin={
                "top":    "0.5in",
                "bottom": "0.5in",
                "left":   "0.5in",
                "right":  "0.5in",
            },
        )
        return out_path
    finally:
        page.close()


# ── Output helpers ────────────────────────────────────────────────────────────

def write_index(rows: list, output_dir: Path) -> None:
    index_path = output_dir / "_index.csv"
    fieldnames = [
        "ticker", "status", "score", "action", "stance",
        "current_price", "market_cap",
        "revenue_growth_yoy", "eps_growth_yoy",
        "recommended_now_pct", "pdf_path", "error",
    ]
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"  Index: {index_path}")


def write_summary(rows: list, output_dir: Path) -> None:
    summary_path = output_dir / "_summary.md"

    groups: dict[str, list] = {
        "BUY": [], "STAGED BUY": [], "HOLD": [],
        "WAIT": [], "SELL": [], "FAILED": [],
    }
    for r in rows:
        if r.get("status") not in ("complete", "qualified", "blocked"):
            groups["FAILED"].append(r)
            continue
        action = (r.get("action") or "UNKNOWN").upper().strip()
        if action in groups:
            groups[action].append(r)
        else:
            groups.setdefault("OTHER", []).append(r)

    lines: list[str] = []
    today = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    lines += [
        "# Batch Evaluation Summary",
        f"_{today} — {len(rows)} tickers_",
        "",
        "## Counts",
        "",
    ]
    for action in ["BUY", "STAGED BUY", "HOLD", "WAIT", "SELL", "FAILED"]:
        count = len(groups.get(action, []))
        if count:
            lines.append(f"- **{action}**: {count}")
    lines.append("")

    for action in ["BUY", "STAGED BUY", "HOLD", "WAIT", "SELL"]:
        group = groups.get(action, [])
        if not group:
            continue
        lines.append(f"## {action} ({len(group)})")
        lines.append("")
        group_sorted = sorted(
            group,
            key=lambda r: (r.get("score") if isinstance(r.get("score"), (int, float)) else -1),
            reverse=True,
        )
        lines += [
            "| Ticker | Score | Price | Market Cap | Rev Growth YoY | EPS Growth YoY | Recommended Now |",
            "|--------|-------|-------|------------|----------------|----------------|-----------------|",
        ]
        for r in group_sorted:
            ticker  = r.get("ticker", "?")
            score   = r.get("score", "—")
            if isinstance(score, (int, float)):
                score = f"{score:.0f}"
            price   = r.get("current_price", "—")
            if isinstance(price, (int, float)):
                price = f"${price:,.2f}"
            mc      = r.get("market_cap", "—")
            if isinstance(mc, (int, float)):
                mc = f"${mc/1e9:.2f}B" if mc > 1e9 else f"${mc/1e6:.0f}M"
            rev_g   = r.get("revenue_growth_yoy", "—")
            if isinstance(rev_g, (int, float)):
                rev_g = f"{rev_g:+.1f}%"
            eps_g   = r.get("eps_growth_yoy", "—")
            if isinstance(eps_g, (int, float)):
                eps_g = f"{eps_g:+.1f}%"
            elif eps_g is None:
                eps_g = "N/A"
            now     = r.get("recommended_now_pct", "—")
            if isinstance(now, (int, float)):
                now = f"{now:.1f}%"
            lines.append(f"| {ticker} | {score} | {price} | {mc} | {rev_g} | {eps_g} | {now} |")
        lines.append("")

    if groups.get("FAILED"):
        lines += [f"## FAILED ({len(groups['FAILED'])})", ""]
        lines += ["| Ticker | Status | Error |", "|--------|--------|-------|"]
        for r in groups["FAILED"]:
            error = (r.get("error") or "")[:120].replace("|", "\\|")
            lines.append(f"| {r.get('ticker','?')} | {r.get('status','?')} | {error} |")
        lines.append("")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Summary: {summary_path}")


def write_comparison(rows: list, output_dir: Path) -> None:
    """Write 3-column comparison: Ticker | Excel Action | StockEval Action, alphabetical."""
    comp_path = output_dir / "_comparison.csv"
    stockeval: dict[str, str] = {
        r["ticker"]: (r.get("action") or "—").upper().strip()
        for r in rows
        if r.get("ticker")
    }
    all_tickers = sorted(set(EXCEL_ACTIONS) | set(stockeval))
    with open(comp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Ticker", "Excel Action", "StockEval Action"])
        for ticker in all_tickers:
            excel = EXCEL_ACTIONS.get(ticker, "—")
            se    = stockeval.get(ticker, "—")
            writer.writerow([ticker, excel, se])
    print(f"  Comparison: {comp_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate tickers and generate PDF reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tickers", nargs="+", help="Tickers to evaluate")
    parser.add_argument("--tickers-file", help="File with one ticker per line")
    parser.add_argument("--output", default="./reports", help="Output folder")
    parser.add_argument("--max-parallel", type=int, default=5,
                        help="Max concurrent evaluations (default 5)")
    parser.add_argument("--skip-pdf", action="store_true",
                        help="Skip PDF generation — only run evaluations")
    parser.add_argument("--force-override", action="store_true",
                        help="Pass force_validation_override=true for all tickers")
    args = parser.parse_args()

    # Resolve ticker list
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    elif args.tickers_file:
        with open(args.tickers_file) as f:
            tickers = [
                line.strip().upper()
                for line in f
                if line.strip() and not line.startswith("#")
            ]
    else:
        parser.error("Must provide --tickers or --tickers-file")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBatch evaluation — {len(tickers)} tickers, {args.max_parallel} parallel")
    print(f"Output: {output_dir.absolute()}\n")

    # ── Phase 1: parallel evaluations ────────────────────────────────────────
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = {
            ex.submit(evaluate_one_ticker, t, args.force_override): t
            for t in tickers
        }
        for i, fut in enumerate(as_completed(futures), 1):
            ticker = futures[fut]
            row    = fut.result()
            results.append(row)
            status = row.get("status", "?")
            flag   = "✓" if status in ("complete", "qualified") else ("⚠" if status == "blocked" else "✗")
            score  = row.get("score")
            score_str = f" score={score:.0f}" if isinstance(score, (int, float)) else ""
            action = row.get("action") or ""
            print(f"  [{i:>3}/{len(tickers)}] {flag} {ticker:<8} {status:<10}{score_str}  {action}")

    # ── Phase 2: sequential PDF rendering ────────────────────────────────────
    renderable = [r for r in results if r.get("job_id") and r.get("status") in ("complete", "qualified", "blocked")]
    if not args.skip_pdf and renderable:
        print(f"\nGenerating PDFs for {len(renderable)} tickers…")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(
                viewport={"width": 1280, "height": 1600},
            )
            for i, row in enumerate(renderable, 1):
                try:
                    pdf_path = render_pdf(row["ticker"], row["job_id"], output_dir, context)
                    row["pdf_path"] = pdf_path.name
                    print(f"  [{i:>3}/{len(renderable)}] PDF  {row['ticker']:<8} → {pdf_path.name}")
                except PWTimeout:
                    row["error"] = (row.get("error") or "") + " | PDF: page timed out"
                    print(f"  [{i:>3}/{len(renderable)}] TIMEOUT  {row['ticker']}")
                except Exception as exc:
                    row["error"] = (row.get("error") or "") + f" | PDF: {exc}"
                    print(f"  [{i:>3}/{len(renderable)}] PDF FAIL {row['ticker']}: {exc}")
            context.close()
            browser.close()
    elif args.skip_pdf:
        print("\nPDF generation skipped (--skip-pdf).")

    # ── Write outputs ─────────────────────────────────────────────────────────
    print()
    write_index(results, output_dir)
    write_summary(results, output_dir)
    write_comparison(results, output_dir)

    success = sum(1 for r in results if r.get("status") in ("complete", "qualified"))
    blocked = sum(1 for r in results if r.get("status") == "blocked")
    failed  = sum(1 for r in results if r.get("status") not in ("complete", "qualified", "blocked"))
    pdfs    = sum(1 for r in results if r.get("pdf_path"))

    print(f"""
Results: {success} complete  {blocked} blocked  {failed} failed  |  {pdfs} PDFs generated
Reports: {output_dir.absolute()}
""")
    subprocess.run(["open", str(output_dir.absolute())], check=False)


if __name__ == "__main__":
    main()
