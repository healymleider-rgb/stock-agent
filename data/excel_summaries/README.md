# Excel Summaries — Reconciliation Data

Place a file named `{TICKER}_excel.json` here to enable the **EXCEL RECONCILIATION**
section in the StockEval memo for that ticker.

The section is **opt-in**: if no file exists for a ticker, no section is rendered
and no placeholder appears. Copy `TTD_excel.example.json` as a starting template.

---

## File naming

```
data/excel_summaries/TTD_excel.json
data/excel_summaries/AAPL_excel.json
```

Ticker must match exactly (uppercase) what is passed to the evaluation endpoint.

---

## JSON schema

All fields are optional except `ticker` and `model_date`.
Missing fields render as `—` in the table; they never break the report.

```json
{
  "ticker": "TTD",
  "model_date": "2026-04-29",

  "current_price_excel": 23.68,

  "fair_value_2026": 120.43,
  "fair_value_2026_range": [50.51, 205.68],

  "dcf_2026": 42.81,
  "dcf_2026_range": [23.85, 76.65],

  "intrinsic_2026": 43.30,
  "intrinsic_2026_range": [31.46, 64.68],

  "expected_eps_2026": 1.27,
  "expected_eps_2027": 1.49,
  "expected_eps_2028": 1.69,

  "beta": 1.17,
  "wacc": 0.0835,
  "projected_growth": 0.0999,

  "average_pe_ratio": 94.83,
  "pe_ratio_range": [39.77, 161.95]
}
```

### Field reference

| Field | Type | Description |
|---|---|---|
| `ticker` | string | Must match eval ticker exactly |
| `model_date` | string | ISO date of the Excel model (YYYY-MM-DD) |
| `current_price_excel` | number | Price used in the Excel model (compared to live price) |
| `fair_value_2026` | number | Excel 5-year midpoint fair value |
| `fair_value_2026_range` | [lo, hi] | Bear/bull range from Excel |
| `dcf_2026` | number | DCF intrinsic value (discounted FCF at WACC) |
| `dcf_2026_range` | [lo, hi] | DCF scenario range |
| `intrinsic_2026` | number | P/E × forward EPS intrinsic value |
| `intrinsic_2026_range` | [lo, hi] | Intrinsic value range |
| `expected_eps_2026/27/28` | number | Forward EPS estimates |
| `beta` | number | Beta used in the Excel model |
| `wacc` | number | WACC as decimal (e.g. 0.0835 = 8.35%) |
| `projected_growth` | number | Revenue or earnings growth assumption |
| `average_pe_ratio` | number | Average P/E used for valuation |
| `pe_ratio_range` | [lo, hi] | P/E range used |

---

## How the reconciliation table works

StockEval compares its computed values to the Excel model:

- **Δ ≤ 2%** → "✓ matches"
- **Δ ≤ 10%** → "close (Δ within 10%)"
- **Δ > 10%** → auto-note explaining the methodology difference

Fields that StockEval doesn't compute (e.g. forward EPS, WACC, 5Y fair value)
are shown with `—` in the StockEval column and a plain-language note explaining why.
