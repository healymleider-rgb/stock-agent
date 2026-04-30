// ── Core analysis types ───────────────────────────────────────────────────────

export interface TrendResult {
  revenue_growth: string;      // "Expanding" | "Deteriorating" | "Stable" | "Volatile"
  op_margin: string;
  net_margin: string;
  roe: string;
  roic: string;
  revenue_growth_sig: string;  // "↑" | "↓" | "→" | "⚠"
  op_margin_sig: string;
  net_margin_sig: string;
  roe_sig: string;
  roic_sig: string;
  growth_adj: number;
  profitability_adj: number;
  valuation_margin_adj: number;
  valuation_rev_adj: number;
  confidence_penalty: number;
}

export interface HistoricalYear {
  label: string;               // "Current", "FY-1", "FY-2", …
  fiscal_year: string;         // e.g. "2024"
  revenue_growth: number | null; // percentage, e.g. 15.5
  eps_growth: number | null;
  op_margin: number | null;    // percentage, e.g. 25.3
  net_margin: number | null;
  roe: number | null;
  roic: number | null;
  ebitda_growth: number | null;
}

export interface CategoryScore {
  name: string;
  score: number;        // 0-100
  weight: number;       // contribution weight
  factors: string[];
  reasoning: string;
  data_quality: "good" | "partial" | "missing";
}

export interface PositionSizing {
  position_range: string;           // e.g. "3%" (single snapped value)
  position_lo: number;              // same as position_hi (single point)
  position_hi: number;              // target size for bar visualisation
  position_size: number;            // canonical target (0/1/1.5/2/3/4/5)
  entry_strategy:
    | "Full allocation"
    | "Staged entry"
    | "Tracking position"
    | "No position";
  entry_detail: string;             // one-sentence entry guidance
  rationale: string;                // single primary driver
  conviction_tier: "high" | "medium" | "hold" | "none";
  setup_quality: "strong" | "neutral" | "weak" | "adverse";
  hard_cap_reason: string | null;       // populated when a hard cap was applied
  rating: "Strong Buy" | "Buy" | "Hold" | "Sell"; // derived from score/stance, not size
  core_compounder_tag: string | null;   // "Core Compounder Allocation" or null
}

export interface Scorecard {
  ticker: string;
  overall_score: number;
  stance: "Bullish" | "Neutral" | "Bearish";
  confidence: number;   // 0-1
  categories: {
    valuation: CategoryScore | null;
    growth: CategoryScore | null;
    profitability: CategoryScore | null;
    financial_health: CategoryScore | null;
    momentum: CategoryScore | null;
    risk: CategoryScore | null;
  };
  bullish_factors: string[];
  bearish_factors: string[];
  key_drivers: string[];
  what_would_change_view: string[];
  risk_flags: string[];
  position_sizing: PositionSizing | Record<string, never>;
}

export interface MCSim {
  n_sims: number;
  horizon_years: number;
  method: string;            // "driver" (driver-chain MC)
  growth_mean: number;       // decimal, e.g. 0.12
  growth_std: number;        // decimal
  // Return distribution
  mean_return: number;
  median_return: number;
  p5_return: number;
  p25_return: number;
  p75_return: number;
  p95_return: number;
  skewness: number;
  // Probabilities (0–1)
  prob_positive: number;
  prob_20_gain: number;
  prob_loss: number;
  prob_loss_20: number;
  // Price distribution
  mean_price: number;
  p5_price: number;
  p25_price: number;
  median_price: number;
  p75_price: number;
  p95_price: number;
  // Sizing signals
  kelly_fraction: number;
  upside_downside: number;
}

export interface ValuationRange {
  available: boolean;
  // P/E scenarios — implied prices
  pe_bear?: number | null;
  pe_base?: number | null;
  pe_bull?: number | null;
  // EV scenarios — implied prices
  ev_bear?: number | null;
  ev_base?: number | null;
  ev_bull?: number | null;
  // P/S scenarios — implied prices
  ps_bear?: number | null;
  ps_base?: number | null;
  ps_bull?: number | null;
  // P/E scenario multiples (displayed in scenario table as "Nx")
  pe_bear_mult?: number | null;
  pe_base_mult?: number | null;
  pe_bull_mult?: number | null;
  // EV/EBITDA scenario multiples
  ev_bear_mult?: number | null;
  ev_base_mult?: number | null;
  ev_bull_mult?: number | null;
  // P/S scenario multiples
  ps_bear_mult?: number | null;
  ps_base_mult?: number | null;
  ps_bull_mult?: number | null;
  // Aggregate price targets
  bear_price?: number | null;
  base_price?: number | null;
  bull_price?: number | null;
  // Context
  upside_context?: string | null;
  peg_ratio?: number | null;
  peg_interpretation?: string | null;
  eps_growth_rate?: number | null;
  data_quality?: string | null;
  // Scenario driver fields — EPS per scenario (P/E method; varies bear/base/bull)
  scenario_bear_eps?: number | null;
  scenario_base_eps?: number | null;
  scenario_bull_eps?: number | null;
  // Primary method and EBITDA/rev-per-share drivers (flat across scenarios)
  scenario_primary_method?: string | null;
  scenario_growth_rate?: number | null;
  scenario_ev_ebitda_val?: number | null;
  scenario_ps_rev_per_share?: number | null;
  // Monte Carlo probabilistic distribution (null when inputs insufficient)
  mc?: MCSim | null;
  // Driver-based scenario model fields (populated when driver_model_available = true)
  driver_model_available?: boolean;
  scenario_bear_rev_growth?: number | null;
  scenario_base_rev_growth?: number | null;
  scenario_bull_rev_growth?: number | null;
  scenario_bear_op_margin?: number | null;
  scenario_base_op_margin?: number | null;
  scenario_bull_op_margin?: number | null;
  scenario_bear_fcf_conv?: number | null;
  scenario_base_fcf_conv?: number | null;
  scenario_bull_fcf_conv?: number | null;
  scenario_bear_exit_mult?: number | null;
  scenario_base_exit_mult?: number | null;
  scenario_bull_exit_mult?: number | null;
  scenario_bear_fwd_fcf?: number | null;
  scenario_base_fwd_fcf?: number | null;
  scenario_bull_fwd_fcf?: number | null;
  scenario_bear_label?: string;
  scenario_base_label?: string;
  scenario_bull_label?: string;
  trend_impact_lines?: string[];
}

// ── Validation Log (pre-report validation gate) ───────────────────────────────

export interface ValidationBlock {
  id:          number;
  name:        string;
  passed:      boolean;
  failures:    string[];
  corrections: { field: string; old: string; new: string; reason: string }[];
}

export interface ValidationLog {
  status:      string;           // "CLEAR TO GENERATE" | "BLOCKED (...)" | "QUALIFIED"
  is_clear:    boolean;
  qualified:   boolean;          // clear but with active overrides
  blocks:      ValidationBlock[];
  overrides:   { block: number; reason_code: string; justification: string }[];
  text:        string;           // full formatted log (canonical format)
}

export interface PeerRow {
  ticker: string;
  company_name?: string | null;
  market_cap?: number | null;
  is_target?: boolean;
  // Valuation
  pe?: number | null;
  ps?: number | null;
  ev_ebitda?: number | null;
  peg?: number | null;
  eps_growth?: number | null;          // decimal, e.g. 0.125 = 12.5%
  // Growth (percentages, e.g. 15.5 = 15.5%)
  revenue_growth?: number | null;
  ebitda_growth?: number | null;
  // Profitability (ratios 0-1)
  gross_margin?: number | null;
  operating_margin?: number | null;
  net_margin?: number | null;
  roe?: number | null;
  roic?: number | null;
  // Financial health
  debt_equity?: number | null;
  current_ratio?: number | null;
  interest_coverage?: number | null;
  // Market / risk
  beta?: number | null;
  // 5-year historical data (newest first: index 0 = Current)
  historical?: HistoricalYear[];
}

export interface PeerComparison {
  has_peers: boolean;
  rows: PeerRow[];
  // 5-element array: [valuation, growth, profitability, financial_health, market]
  // Empty string means no insight for that section.
  insights: string[];
  // Selection quality metadata
  peer_level: 1 | 2 | 3;           // 1 = direct, 2 = relaxed, 3 = proxy (adjacent arch)
  section_label: string;            // display title for the section
  proxy_note: string;               // non-empty only when peer_level === 3
  // Peer-relative trend insight bullets (up to 5)
  peer_trend_insights?: string[];
}

export interface MacroData {
  available: boolean;
  // Core fields
  macro_regime?: string;
  macro_score?: number;
  recession_risk_level?: string;
  sector_tilt?: string;
  bullish_macro_factors?: string[];
  bearish_macro_factors?: string[];
  data_coverage?: number;
  // Phase 1 LEI fields — null when FRED unavailable or data window too short
  cycle_phase?: string | null;
  lei_trend?: string | null;
  yield_spread_trend?: string | null;
  // Pre-computed narrative from MacroLEIAgent
  reasoning_summary?: string;
  // Observation dates keyed by indicator name (for staleness display)
  observation_dates?: Record<string, string>;
}

export interface StockInfo {
  company_name: string;
  sector: string;
  industry: string;
  description: string;
  beta: number | null;
  current_price: number | null;
  market_cap: number | null;
  pe_ratio: number | null;
  ps_ratio: number | null;
  ev_ebitda: number | null;
  pb_ratio: number | null;
  dividend_yield: number | null;
  roe: number | null;
  roa: number | null;
  gross_margin: number | null;
  net_margin: number | null;
  debt_to_equity: number | null;
  current_ratio: number | null;
  // TTM fundamentals — stored values that drive all ratio calculations
  // P/E = price / ttm_eps (verifiable), FCF drives the scenario tree base
  ttm_eps: number | null;
  ttm_eps_source: string | null;
  ttm_fcf: number | null;
  // Data dates — for transparency / staleness display
  price_date: string | null;         // ISO YYYY-MM-DD when quote price was sampled
  fundamentals_date: string | null;  // most recent financial statement filing date
  // Shares provenance — from SEC EDGAR via NormalizedMetrics
  shares_outstanding: number | null;
  shares_source: string | null;
  shares_filing_date: string | null;
  shares_filing_url: string | null;
  // Metric provenance — used to show basis labels (e.g. "P/E (TTM)")
  _sources?: {
    price?: string;
    market_cap?: string;
    pe_ratio?: string;
    ps_ratio?: string;
    ev_ebitda?: string;
    ttm_eps?: string;
    ttm_fcf?: string;
  };
}

// ── API response types ────────────────────────────────────────────────────────

export interface EvaluateResponse {
  ticker: string;
  scorecard: Scorecard;
  stock_info: StockInfo;
  valuation_range: ValuationRange;
  peer_comparison: PeerComparison;
  macro: MacroData;
  memo: string;
  reasoning_log: string[];
  evaluated_at: string;
  stock_type_label: string | null;
  stock_type_desc: string | null;
  key_tension: string | null;
  trends?: TrendResult | null;
  validation_log?: ValidationLog | null;
}

export interface JobStatus {
  status: "pending" | "running" | "complete" | "error";
  progress: number;     // 0-100
  step: string;
  result?: EvaluateResponse;
  error?: string;
}

export interface StartEvaluationResponse {
  job_id: string;
}

// ── History ───────────────────────────────────────────────────────────────────

export interface HistoryEntry {
  timestamp: string;
  overall_score: number;
  stance: "Bullish" | "Neutral" | "Bearish";
  confidence: number;
  actual_return_30d: number | null;
  actual_return_90d: number | null;
}
