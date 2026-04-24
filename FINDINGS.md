# StockEval Pipeline Findings

Issues surfaced by the pre-report validation gate that originate upstream
of validation. These are real data or pipeline bugs; they're not caused
by the gate and cannot be fixed within it. Each finding includes repro
steps and the block that catches it.

---

## Finding 1 — Market cap API/computed mismatch [RESOLVED 2026-04-24]

**Discovered:** 2026-04-22
**Block that catches it:** Block 2 (Market Cap Triangle)
**Severity:** Moderate — shipping a mismatched market cap misrepresents
position sizing calculations.

**Symptom:**
On live evaluations of NFLX, MSFT, and AXON today, Block 2 flagged a
2.7–2.8% discrepancy between `market_cap_b.api` (from FMP) and `price ×
shares`. Example: NFLX API=$394.10B vs computed=$405.13B, Δ=2.7%,
exceeds 0.5% tolerance.

**Repro:**
```
POST /api/evaluate {"ticker": "NFLX"}
→ job_status: blocked
→ validation_log.blocks[1].failures:
    ["market_cap_b mismatch: API $394.10B vs computed $405.13B (Δ=2.7% > 0.5% tolerance)"]
```

**Likely causes:**
- FMP `market_cap` field is updated on a different cadence than price
  and shares (possibly EOD vs intraday)
- Shares outstanding changed (buyback, issuance) after FMP cached the
  `market_cap` field
- FMP is using float shares instead of total diluted shares

**Current gate behavior:**
`market_cap_b.auth` is set to `price × shares` (computed). This is the
correct preference — computed is ground truth when price and shares are live.

**Proposed fix:**
No immediate action needed at the gate level. Long-term options:
1. Stop calling FMP's `market_cap` field; always compute locally from
   `price × shares_diluted`
2. Treat the mismatch as a data-source warning, not a report blocker —
   widen the tolerance or downgrade to a correction
3. Widen the tolerance to 3% if the discrepancy is chronic across tickers

**Owner:** [unknown — fill in]

**Resolution (2026-04-24):** Took option 1 — stopped reading FMP's `market_cap` field
entirely. `build_snapshot` now sets `auth_mktcap_b = comp_mktcap_b` (price × shares)
unconditionally. `_b2_market_cap_triangle` was rewritten to verify price × shares > 0
rather than comparing API vs computed. Commit SHA: pending.

---

## Finding 2 — MacroLEIAgent not populating snapshot [RESOLVED 2026-04-24]

**Discovered:** 2026-04-22
**Block that catches it:** Block 5 (Macro Series Integrity)
**Severity:** High — four macro indicators (CLI, jobless claims, housing
starts, manufacturing employment) are absent from every snapshot,
indicating the agent that should populate them is either not running or
failing silently.

**Symptom:**
`macro_findings["lei_snapshot"]` is missing keys or all values are None.
Block 5 emits:

```
Macro indicators missing from snapshot:
  ['cli', 'jobless_claims', 'housing_starts', 'manuf_employ']
  — these should be populated by MacroLEIAgent
```

**Repro:**
```
POST /api/evaluate {"ticker": "AXON"}
→ job_status: blocked
→ validation_log.blocks[4].failures:
    ["Macro indicators missing from snapshot: [...]"]
```
Observed on every ticker evaluated on 2026-04-22.

**Likely causes:**
- MacroLEIAgent is not being invoked in the evaluation pipeline
- It's called but failing silently (caught exception, empty dict returned)
- The FRED/OECD data source is unreachable and fallback to cached values
  is not working
- The agent is populating a different key than what `build_snapshot` reads
  (check `lei_snapshot` vs `macro_indicators` or similar)

**Current gate behavior:**
Correctly reports the gap. Report is blocked unless a manual override is
provided via `force_validation_override=true` + `force_justification`.

**Proposed fix:**
1. Confirm MacroLEIAgent is invoked in `orchestrator_agent.py`
2. Add structured logging inside MacroLEIAgent to trace where it exits
   on empty output
3. If the upstream data source is unreachable, add a `macro.stale_mode`
   flag so the gate can distinguish "agent broken" from "data stale but
   last-known-good applied" — the latter should not block

**Owner:** [unknown — fill in]

**Resolution (2026-04-24):** Two-part fix in `build_snapshot`:
1. Field name: was reading `macro_findings.get("lei_snapshot")`, should be `"snapshot"` —
   the key `MacroLEIAgent` actually writes to.
2. Key names: three FRED key mismatches corrected (`"cli"` → `"oecd_cli"`,
   `"manuf_employ"` → `"mfg_employment"`, `"yield_spread"` → `"yield_spread_10y2y"`).

Additionally, `_b5_macro_integrity` was updated to exempt `cli` from the "missing
indicator" failure when `cli_stale=True` — the OECD CLI series on FRED has not been
updated since 2024-01-01 (844 days stale as of fix date), so `MacroLEIAgent` legitimately
nulls it via `StaleMacroError`. The other three indicators (jobless_claims, housing_starts,
mfg_employment) remain required. Commit SHA: pending.

---

## Finding tracking

When a finding is resolved:
1. Fix it in the upstream code (not in `validation_gate.py`).
2. Delete the finding from this file.
3. Note the resolution date and the fixing commit in git log.
4. Run the full validation suite against a fresh ticker evaluation to
   confirm the gate no longer flags the issue.
