"""
LAYER 1 — EXCEL EXTRACTOR
Run this on any valuation Excel file matching the standard template.
Output: standardized input block, ready to paste into Claude or pipe
        into analysis.layer2_processor for fully automated processing.

Usage:
    python extract_inputs.py MSFT_26.xlsx
    python extract_inputs.py *.xlsx          # batch all files
    python extract_inputs.py MSFT_26.xlsx --json   # output JSON instead of key-value
    python extract_inputs.py MSFT_26.xlsx --process  # Layer 1 + Layer 2 in one shot
"""

import sys
import json
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import pandas as pd
except ImportError:
    print("ERROR: pip install pandas openpyxl")
    sys.exit(1)


def extract(filepath: str) -> dict:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data    = pd.read_excel(filepath, sheet_name="Data",         header=None)
    metrics = pd.read_excel(filepath, sheet_name="Metrics",      header=None)
    proj    = pd.read_excel(filepath, sheet_name="Projections",  header=None)
    summary = pd.read_excel(filepath, sheet_name="Summary Sheet",header=None)

    # ── helpers ──────────────────────────────────────────────────────────────
    def get_summary(label):
        for _, row in summary.iterrows():
            if str(row[0]).strip() == label:
                v = row[1]
                return None if (v != v or str(v) == "nan") else v
        return None

    def parse_price_range(s):
        try:
            s = str(s).replace("$", "").replace(",", "")
            parts = [x.strip() for x in s.split("-")]
            return float(parts[1]), float(parts[0])   # low, high
        except Exception:
            return None, None

    def parse_fair_value(s):
        try:
            return float(str(s).replace("$", "").replace(",", ""))
        except Exception:
            return None

    def parse_pe_range(s):
        try:
            s = str(s).replace(",", "")
            parts = [x.strip() for x in s.split("-")]
            return float(parts[0]), float(parts[1])
        except Exception:
            return None, None

    def safe_float(v, decimals=4):
        try:
            f = float(v)
            if f != f:     # NaN
                return None
            return round(f, decimals)
        except Exception:
            return None

    def find_data_row(label_fragment):
        for _, row in data.iterrows():
            if label_fragment.lower() in str(row[0]).lower():
                return row
        return None

    # ── ticker from cell A1 ──────────────────────────────────────────────────
    raw_ticker = str(data.iloc[0, 0]).strip()
    if "(" in raw_ticker and ")" in raw_ticker:
        ticker = raw_ticker.split("(")[-1].replace(")", "").strip()
    else:
        ticker = path.stem.split("_")[0].upper()

    # ── summary sheet pulls ───────────────────────────────────────────────────
    current_price       = safe_float(get_summary("Current Price"))
    wacc                = safe_float(get_summary("WACC"))
    avg_pe              = safe_float(get_summary("Average P/E Ratio"), 2)
    beta                = safe_float(get_summary("Beta (5Y Monthly)"), 2)
    proj_growth         = safe_float(get_summary("Projected Growth"))
    eps_2026            = safe_float(get_summary("2026"))
    eps_2027            = safe_float(get_summary("2027"))
    eps_2028            = safe_float(get_summary("2028"))
    base_2026           = parse_fair_value(get_summary("2026 Fair Value"))
    low_2026, high_2026 = parse_price_range(get_summary("2026 Expected Range"))
    pe_low, pe_high     = parse_pe_range(get_summary("P/E Ratio Range"))

    # ── data sheet pulls ─────────────────────────────────────────────────────
    gm_row       = find_data_row("Gross Profit Margins")
    nm_row       = find_data_row("Net Earnings / Total Revenue")
    gross_margin = safe_float(gm_row[1]) if gm_row is not None else None
    net_margin   = safe_float(nm_row[1]) if nm_row is not None else None

    # ── metrics sheet EPS (current year) ────────────────────────────────────
    eps_current = None
    for i, row in metrics.iterrows():
        if "Earnings Per Share" in str(row[0]):
            for j in range(i + 2, min(i + 5, len(metrics))):
                v = safe_float(metrics.iloc[j][1])
                if v is not None:
                    eps_current = v
                    break
            break

    return {
        "TICKER":                   ticker,
        "CURRENT_PRICE":            current_price,
        "EPS_CURRENT":              eps_current,
        "EPS_2026":                 eps_2026,
        "EPS_2027":                 eps_2027,
        "EPS_2028":                 eps_2028,
        "PRICE_TARGET_2026_BASE":   base_2026,
        "PRICE_TARGET_2026_LOW":    round(low_2026,  2) if low_2026  else None,
        "PRICE_TARGET_2026_HIGH":   round(high_2026, 2) if high_2026 else None,
        "AVG_PE_RATIO":             avg_pe,
        "PE_RANGE_LOW":             round(pe_low,  2)   if pe_low    else None,
        "PE_RANGE_HIGH":            round(pe_high, 2)   if pe_high   else None,
        "PROJECTED_REVENUE_GROWTH": proj_growth,
        "GROSS_MARGIN_CURRENT":     gross_margin,
        "NET_MARGIN_CURRENT":       net_margin,
        "WACC":                     wacc,
        "BETA":                     beta,
    }


def to_keyvalue(d: dict) -> str:
    return "\n".join(
        f"{k}: {v if v is not None else 'null'}"
        for k, v in d.items()
    )


def _save_json(data: dict | list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2))
    print(f"  saved → {out_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Layer 1: extract standardized inputs from valuation Excel models."
    )
    parser.add_argument("files", nargs="+", help="Excel file(s) to process")
    parser.add_argument("--json",    action="store_true", help="Output as JSON key-value block")
    parser.add_argument("--process", action="store_true",
                        help="Run Layer 2 processor immediately after extraction")
    parser.add_argument("--save",    action="store_true",
                        help="Save JSON output to data/layer2/<TICKER>_layer2.json")
    parser.add_argument("--outdir",  default="data/layer2",
                        help="Output directory for --save (default: data/layer2)")
    args = parser.parse_args()

    results = []
    for f in args.files:
        try:
            d = extract(f)
            results.append(d)
            ticker = d.get("TICKER", Path(f).stem.upper())

            if args.process or args.save:
                from analysis.layer2_processor import process_input_block
                out = process_input_block(d)

                if args.save:
                    out_path = Path(args.outdir) / f"{ticker}_layer2.json"
                    _save_json(out, out_path)

                print(json.dumps(out, indent=2))

            elif args.json:
                out = d
                if args.save:
                    out_path = Path(args.outdir) / f"{ticker}_inputs.json"
                    _save_json(out, out_path)
                print(json.dumps(out, indent=2))

            else:
                print(f"# ── {ticker} ──────────────────────────────────")
                print(to_keyvalue(d))
                print()

        except Exception as e:
            print(f"ERROR processing {f}: {e}", file=sys.stderr)

    if args.json and not args.process and not args.save and len(results) > 1:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
