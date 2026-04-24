# StockEval Schema Library

`ticker_analysis_v1.schema.json` — Solo mode output contract. Validated by `validate_example.py`.

---

## Ticker-agnostic contract

`ticker_analysis_v1.schema.json` is valid for any equity ticker, not just NFLX.
Three reference examples cover distinct ticker profiles:

- `examples/nflx_ticker_analysis_v1.json` — Buy thesis, deep value (current price below P25)
- `examples/msft_ticker_analysis_v1.json` — Buy thesis, fair value (current price between P25 and P50)
- `examples/axon_ticker_analysis_v1.json` — Hold/Wait thesis, 0% recommended_now (edge case: tests schema generalization beyond profitable tickers)

Production outputs for AAPL, GOOGL, or any other ticker follow the same schema. No
ticker-specific code paths exist — the walker, classifier, and validator are structurally
agnostic.

### Where ticker identity appears

Three places carry ticker-specific content: the `ticker` field (pattern `^[A-Z]{1,6}$`);
`source` strings that template the ticker, e.g., `"company_filing_10Q_{TICKER}_{period}"`;
and `DerivedValue.formula` strings that embed numeric values as literals. Formulas must be
regenerated when upstream data changes; `upstream.stockeval_layer2_hash` detects when
that has happened.

### Acceptance bar

Every engine output must satisfy:

| Rule | Requirement |
|------|-------------|
| N1 | `passed=True`, `corrections=0` — every `formula` evaluates to `value` within ±0.001 or ±0.5% |
| F1 | `passed=True`, `stale_block_count=0` — equity `current_price.vintage` within 26h of `as_of` |
| S1 | `passed=True`, `unsourced_count=0` — every numeric value carries `source`, `derived: true`, or inherited source |

`unsourced_count > 0` means the engine has a schema gap for that field. Fix the engine,
not the data.

### Implementation narrowings (discovered 2026-04-22)

**`_parse_iso8601`** — Date-only strings like `"2026-04-16"` are treated as midnight UTC,
overstating vintage age by up to 24h. Acceptable for daily-stamped brokerage data;
revisit if intraday vintages become the norm.

**`_walk_price_objects`** — F1 yields only dicts whose key contains `"price"`, correctly
excluding fundamentals (EPS, shares, multiples) from freshness classification. Would miss
price fields named `"quote"`, `"last"`, or `"mark"`. Pipeline uses `"price"` consistently;
flag if future data sources introduce alternatives.

### Block coverage by layer

Blocks are scoped to the artifact they were designed for:

| Layer | Blocks | Artifact |
|-------|--------|----------|
| StockEval Layer 2 (input) | 1–7 | Freshness, market cap triangle, P/E basis, scenario anchoring, macro integrity, FCF guidance, execution coherence |
| Portfolio Engine Solo Mode (derived) | 8–10 | Formula-value consistency, price vintage freshness, source attribution |
| Portfolio Engine Context Mode (future) | 11+ | Entry price coherence, position size bounds, weight sum, marginal effect consistency |

Blocks failing on the wrong layer are expected behavior, not a gap.

### Multiples convention

P/E uses TTM diluted EPS; P/S uses TTM revenue; EV/EBITDA uses TTM EBITDA. PEG uses the
EPS growth rate from StockEval peer comparison data — not forward consensus — because
using a different growth source across tickers breaks cross-ticker PEG comparability. `DerivedValue.formula` records the exact denominators used, making the convention
auditable per-ticker.
