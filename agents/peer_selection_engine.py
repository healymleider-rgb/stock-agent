"""
PeerSelectionEngine
===================
Adaptive peer selection and ranking based on business model archetype.

Pipeline
--------
1. classify_archetype()
   SIC fast-path → industry string match → revenue model inference → fallback.
   Checks peer_overrides.json first for manual corrections.

2. get_adaptive_bands()
   Returns archetype-specific market cap bands (lo/hi multipliers).
   Tight for Banks; very wide for Fintech and Exchanges.

3. apply_structural_filters()
   Hard constraints that survive archetype matching:
   · Revenue similarity  (±50% of target, using mktcap/PS proxy)
   · Margin similarity   (±20pp gross margin)
   · Geographic preference  (same region scores higher; doesn't hard-exclude)

4. rank_peers()
   Weighted Euclidean distance on archetype-specific metrics.
   Lower distance = better economic comparability.
   Falls back to completeness scoring when key metrics are absent.

Auto-expand
-----------
If <3 peers pass structural filters after Pass 1, the market cap band is
expanded once (lo_expanded / hi_expanded) and the filter re-runs.

Integration (analysis/peer_comparison.py)
------------------------------------------
  from agents.peer_selection_engine import PeerSelectionEngine

  engine    = PeerSelectionEngine()
  archetype = engine.classify(sector=sector, industry=industry,
                               gross_margin=target_gross_margin,
                               operating_margin=target_operating_margin)
  bands     = engine.get_adaptive_bands(archetype)
  # ... existing candidate pool build with bands.lo / bands.hi ...
  ranked    = engine.rank_peers(target_row, usable_rows, archetype)
  ranked    = engine.apply_structural_filters(target_row, ranked)
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional


# ── Archetype enum ─────────────────────────────────────────────────────────────

class Archetype(IntEnum):
    BANKS              = 1
    ASSET_MANAGERS     = 2
    INSURANCE          = 3
    EXCHANGES          = 4   # exchanges + payment networks + clearing
    FINANCIAL_DATA     = 5   # financial data & analytics providers
    INVESTMENT_BANKS   = 6   # investment banks + broker-dealers
    FINTECH            = 7   # lending platforms, neobanks, BNPL
    OTHER              = 8


ARCHETYPE_NAMES: dict[Archetype, str] = {
    Archetype.BANKS:            "Banks",
    Archetype.ASSET_MANAGERS:   "Asset Managers",
    Archetype.INSURANCE:        "Insurance",
    Archetype.EXCHANGES:        "Exchanges / Market Infrastructure",
    Archetype.FINANCIAL_DATA:   "Financial Data & Analytics",
    Archetype.INVESTMENT_BANKS: "Investment Banks / Broker Dealers",
    Archetype.FINTECH:          "Fintech Platforms",
    Archetype.OTHER:            "Other",
}


# ── Adaptive band configuration ────────────────────────────────────────────────

@dataclass
class BandConfig:
    """Market cap multiplier bands for a given archetype."""
    lo:          float   # normal pass — lower bound (multiplier of target mktcap)
    hi:          float   # normal pass — upper bound
    lo_expanded: float   # auto-expand pass (triggered when <3 peers survive)
    hi_expanded: float
    # Structural constraint: peer revenue must be within [rev_lo, rev_hi] × target revenue
    # Derived via mktcap/PS proxy. Set to None to skip.
    rev_lo:      Optional[float] = 0.35
    rev_hi:      Optional[float] = 3.0
    # Gross margin similarity: abs(peer_gm - target_gm) must be ≤ this (in ratio units, e.g. 0.20 = 20pp)
    margin_tol:  Optional[float] = 0.20


# Decision tree: archetype → BandConfig
#
# Banks:         Tight — NIM, credit risk, deposit base are highly size-sensitive.
#                JPM vs HBAN are fundamentally different businesses.
# Asset Managers: Moderate — AUM economics scale but fee compression hits large players earlier.
# Insurance:     Moderate — Reserve dynamics differ by size, but combined ratio is universal.
# Exchanges:     Wide — Few pure-play exchanges exist; can't be too restrictive.
# Financial Data: Wide — MSCI, SPGI, Morningstar span a wide mktcap range but share ARR dynamics.
# Inv. Banks:    Moderate — Deal leverage and balance sheet size matter but less than in banking.
# Fintech:       Very wide — High-growth fintechs span 10-100x size ranges in the same archetype.
# Other:         Moderate — Default bands, no strong structural reason for tighter constraint.

_BAND_TABLE: dict[Archetype, BandConfig] = {
    Archetype.BANKS: BandConfig(
        lo=0.25, hi=4.0,
        lo_expanded=0.15, hi_expanded=7.0,
        rev_lo=0.25, rev_hi=4.0,
        margin_tol=None,   # margins differ structurally (NIM-driven); skip margin filter
    ),
    Archetype.ASSET_MANAGERS: BandConfig(
        lo=0.20, hi=5.0,
        lo_expanded=0.10, hi_expanded=8.0,
        rev_lo=0.25, rev_hi=4.0,
        margin_tol=0.25,
    ),
    Archetype.INSURANCE: BandConfig(
        lo=0.20, hi=5.0,
        lo_expanded=0.10, hi_expanded=8.0,
        rev_lo=0.20, rev_hi=5.0,
        margin_tol=None,   # combined ratio varies widely by line of business
    ),
    Archetype.EXCHANGES: BandConfig(
        lo=0.10, hi=10.0,
        lo_expanded=0.05, hi_expanded=15.0,
        rev_lo=0.15, rev_hi=7.0,
        margin_tol=0.25,
    ),
    Archetype.FINANCIAL_DATA: BandConfig(
        lo=0.10, hi=10.0,
        lo_expanded=0.05, hi_expanded=15.0,
        rev_lo=0.15, rev_hi=7.0,
        margin_tol=0.25,
    ),
    Archetype.INVESTMENT_BANKS: BandConfig(
        lo=0.20, hi=5.0,
        lo_expanded=0.10, hi_expanded=8.0,
        rev_lo=0.20, rev_hi=5.0,
        margin_tol=None,   # trading vs. advisory revenue mix varies; skip
    ),
    Archetype.FINTECH: BandConfig(
        lo=0.05, hi=12.0,
        lo_expanded=0.02, hi_expanded=20.0,
        rev_lo=0.10, rev_hi=10.0,
        margin_tol=0.30,   # wide — early-stage vs mature fintech margins diverge
    ),
    Archetype.OTHER: BandConfig(
        lo=0.20, hi=5.0,
        lo_expanded=0.10, hi_expanded=8.0,
        rev_lo=0.25, rev_hi=4.0,
        margin_tol=0.20,
    ),
}


# ── Archetype-specific distance weights ───────────────────────────────────────
#
# Keys must match fields on PeerRow (or _DM).
# Each archetype weights the metrics most central to its economic model.
# Weights sum to 1.0; missing metrics are skipped (weight redistributed).

_WEIGHTS: dict[Archetype, dict[str, float]] = {
    Archetype.BANKS: {
        "roe":              0.25,
        "net_margin":       0.20,
        "debt_equity":      0.15,   # proxy for leverage — banks with high leverage differ fundamentally
        "revenue_growth":   0.20,
        "operating_margin": 0.20,
    },
    Archetype.ASSET_MANAGERS: {
        "revenue_growth":   0.30,
        "operating_margin": 0.30,
        "roe":              0.20,
        "net_margin":       0.20,
    },
    Archetype.INSURANCE: {
        "roe":              0.30,
        "net_margin":       0.25,
        "operating_margin": 0.25,
        "revenue_growth":   0.20,
    },
    Archetype.EXCHANGES: {
        "operating_margin": 0.30,
        "revenue_growth":   0.30,
        "gross_margin":     0.20,
        "net_margin":       0.20,
    },
    Archetype.FINANCIAL_DATA: {
        "gross_margin":     0.25,
        "operating_margin": 0.25,
        "revenue_growth":   0.30,
        "net_margin":       0.20,
    },
    Archetype.INVESTMENT_BANKS: {
        "roe":              0.30,
        "operating_margin": 0.25,
        "revenue_growth":   0.25,
        "net_margin":       0.20,
    },
    Archetype.FINTECH: {
        "revenue_growth":   0.35,
        "gross_margin":     0.25,
        "operating_margin": 0.25,
        "net_margin":       0.15,
    },
    Archetype.OTHER: {
        "gross_margin":     0.25,
        "operating_margin": 0.25,
        "revenue_growth":   0.25,
        "net_margin":       0.25,
    },
}


# ── Archetype description tables (for peer justification) ─────────────────────

_ARCHETYPE_REVENUE_TYPE: dict[Archetype, str] = {
    Archetype.BANKS:            "net interest income",
    Archetype.ASSET_MANAGERS:   "AUM-based management fee",
    Archetype.INSURANCE:        "underwriting premium and investment income",
    Archetype.EXCHANGES:        "transaction and data fee",
    Archetype.FINANCIAL_DATA:   "recurring subscription / ARR",
    Archetype.INVESTMENT_BANKS: "advisory and capital markets fee",
    Archetype.FINTECH:          "take rate on transaction volume",
    Archetype.OTHER:            "diversified revenue",
}

_ARCHETYPE_ECON_DRIVER: dict[Archetype, str] = {
    Archetype.BANKS:            "rate sensitivity and credit cycle",
    Archetype.ASSET_MANAGERS:   "market beta and flow retention",
    Archetype.INSURANCE:        "combined ratio and reserve adequacy",
    Archetype.EXCHANGES:        "ADTV and capture rate",
    Archetype.FINANCIAL_DATA:   "NRR and pricing power",
    Archetype.INVESTMENT_BANKS: "deal flow and wallet share",
    Archetype.FINTECH:          "credit loss and user monetization",
    Archetype.OTHER:            "segment revenue mix",
}

_ARCHETYPE_KEY_METRIC: dict[Archetype, str] = {
    Archetype.BANKS:            "NIM expansion and loan loss provisioning",
    Archetype.ASSET_MANAGERS:   "fee rate compression vs. AUM growth",
    Archetype.INSURANCE:        "loss ratio and float deployment",
    Archetype.EXCHANGES:        "fee-per-contract and ADTV sensitivity",
    Archetype.FINANCIAL_DATA:   "NRR and ARR per seat",
    Archetype.INVESTMENT_BANKS: "fee backlog and M&A cycle exposure",
    Archetype.FINTECH:          "take rate compression and unit economics",
    Archetype.OTHER:            "segment EBIT contribution",
}

# Cross-archetype: (target_arch, peer_arch) → shared characteristic / divergence note
#
# Coverage: every pair that appears in _ADJACENT (peer_comparison.py) must have
# an entry here so justify_peer() produces a specific sentence, not a generic fallback.
# Pairs covered: EXCHANGES ↔ FINANCIAL_DATA, EXCHANGES ↔ INVESTMENT_BANKS.
_CROSS_ARCH_SHARED: dict[tuple[int, int], str] = {
    # Exchanges / Market Infrastructure ↔ Financial Data & Analytics
    (Archetype.EXCHANGES,        Archetype.FINANCIAL_DATA):   "recurring data subscription revenue",
    (Archetype.FINANCIAL_DATA,   Archetype.EXCHANGES):        "transaction-linked and volume-sensitive revenue",
    # Exchanges / Market Infrastructure ↔ Investment Banks / Broker Dealers
    (Archetype.EXCHANGES,        Archetype.INVESTMENT_BANKS): "capital markets activity and fee capture on transaction flow",
    (Archetype.INVESTMENT_BANKS, Archetype.EXCHANGES):        "market structure fee revenue and volume-driven economics",
    # Investment Banks ↔ Asset Managers (pre-existing, kept for completeness)
    (Archetype.INVESTMENT_BANKS, Archetype.ASSET_MANAGERS):   "AUM and fee-revenue overlap",
    (Archetype.ASSET_MANAGERS,   Archetype.INVESTMENT_BANKS): "capital markets cycle exposure",
    # Fintech ↔ Banks (pre-existing)
    (Archetype.FINTECH,          Archetype.BANKS):            "credit and deposit economics",
    (Archetype.BANKS,            Archetype.FINTECH):          "digital lending and rate sensitivity",
}

_CROSS_ARCH_DIVERGENCE: dict[tuple[int, int], str] = {
    # Exchanges / Market Infrastructure ↔ Financial Data & Analytics
    (Archetype.EXCHANGES,        Archetype.FINANCIAL_DATA):   "subscription stability vs. transaction cyclicality limits direct comparability",
    (Archetype.FINANCIAL_DATA,   Archetype.EXCHANGES):        "volume dependence vs. ARR stability limits direct comparability",
    # Exchanges / Market Infrastructure ↔ Investment Banks / Broker Dealers
    (Archetype.EXCHANGES,        Archetype.INVESTMENT_BANKS): "balance sheet risk vs. exchange float economics limits direct comparability",
    (Archetype.INVESTMENT_BANKS, Archetype.EXCHANGES):        "deal-cycle revenue volatility vs. transaction fee stability limits direct comparability",
    # Investment Banks ↔ Asset Managers (pre-existing)
    (Archetype.INVESTMENT_BANKS, Archetype.ASSET_MANAGERS):   "balance sheet risk vs. fee-only model limits direct comparability",
    (Archetype.ASSET_MANAGERS,   Archetype.INVESTMENT_BANKS): "deal-cycle volatility vs. AUM stability limits direct comparability",
    # Fintech ↔ Banks (pre-existing)
    (Archetype.FINTECH,          Archetype.BANKS):            "regulatory capital structure limits direct comparability",
    (Archetype.BANKS,            Archetype.FINTECH):          "balance sheet intermediation vs. platform economics limits direct comparability",
}


def _scale_label(mkt_cap: float) -> str:
    if mkt_cap >= 100e9:
        return "mega-cap"
    if mkt_cap >= 10e9:
        return "large-cap"
    if mkt_cap >= 2e9:
        return "mid-cap"
    return "small-cap"


# ── Geographic region detection ────────────────────────────────────────────────

class GeoRegion(str):
    US     = "US"
    DEV    = "DEV"    # developed ex-US (EU, UK, Japan, Canada, Australia)
    EM     = "EM"     # emerging markets

_DEV_EXCHANGES = {
    "NYSE", "NASDAQ", "TSX", "LSE", "EURONEXT", "XETRA",
    "JPX", "ASX", "SIX", "OMX", "OSE", "OSLO",
}
_EM_EXCHANGES = {
    "BSE", "NSE", "HKEX", "SSE", "SZSE", "KSE", "BOVESPA",
    "JSE", "MOEX", "IDX", "SET", "BM&F", "SGX",
}
_US_COUNTRIES = {"US", "United States"}
_EM_COUNTRIES = {
    "China", "India", "Brazil", "Russia", "South Africa",
    "Indonesia", "Mexico", "Thailand", "Turkey", "Egypt",
    "Vietnam", "Philippines", "Malaysia", "Argentina", "Colombia",
}


def detect_geo(exchange: str, country: str) -> str:
    exch  = (exchange or "").upper().replace("-", "").replace(" ", "")
    cntry = (country or "").strip()
    if cntry in _US_COUNTRIES:
        return GeoRegion.US
    if any(exch.startswith(e.replace("-", "").replace(" ", "")) for e in _US_EXCHANGES):
        return GeoRegion.US
    if cntry in _EM_COUNTRIES:
        return GeoRegion.EM
    if any(exch.startswith(e.replace("-", "").replace(" ", "")) for e in _EM_EXCHANGES):
        return GeoRegion.EM
    return GeoRegion.DEV

_US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX", "BATS", "CBOE"}


# ── Classification result ──────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    archetype:  Archetype
    method:     str          # "override" | "sic" | "industry_match" | "revenue_model" | "fallback"
    confidence: float        # 0.0–1.0
    note:       str = ""


# ── SIC → archetype fast-path ─────────────────────────────────────────────────
# SIC ranges from US SIC manual. These are high-confidence mappings.

_SIC_RANGES: list[tuple[int, int, Archetype]] = [
    (6020, 6029, Archetype.BANKS),              # state commercial banks
    (6035, 6036, Archetype.BANKS),              # savings institutions
    (6110, 6159, Archetype.BANKS),              # federal-chartered credit + mortgage
    (6160, 6163, Archetype.BANKS),              # mortgage bankers
    (6200, 6211, Archetype.INVESTMENT_BANKS),   # security brokers, dealers
    (6221, 6221, Archetype.EXCHANGES),          # commodity contracts
    (6282, 6282, Archetype.ASSET_MANAGERS),     # investment advisory
    (6311, 6321, Archetype.INSURANCE),          # life, A&H, fire insurance
    (6324, 6324, Archetype.INSURANCE),          # hospital/medical service plans
    (6331, 6399, Archetype.INSURANCE),          # fire, casualty, other
    (6411, 6411, Archetype.INSURANCE),          # insurance agents/brokers
]


def _classify_by_sic(sic: Optional[int]) -> Optional[Archetype]:
    if sic is None:
        return None
    for lo, hi, arch in _SIC_RANGES:
        if lo <= sic <= hi:
            return arch
    return None


# ── Industry string → archetype ────────────────────────────────────────────────

_INDUSTRY_PATTERNS: list[tuple[re.Pattern, Archetype, float]] = [
    # Pattern, archetype, confidence
    # Banks
    (re.compile(r"bank|savings institution|thrift|credit union|mortgage bank", re.I),
     Archetype.BANKS, 0.90),
    # Asset Managers
    (re.compile(r"asset management|investment management|fund management|wealth management|"
                r"investment advisor|portfolio management", re.I),
     Archetype.ASSET_MANAGERS, 0.90),
    # Insurance
    (re.compile(r"insurance|reinsurance|surety|title insurance|annuity", re.I),
     Archetype.INSURANCE, 0.90),
    # Exchanges / Market Infrastructure
    (re.compile(r"exchange|clearinghouse|clearing house|securities exchange|"
                r"payment network|card network|settlement", re.I),
     Archetype.EXCHANGES, 0.90),
    # Financial Data & Analytics
    (re.compile(r"financial data|market data|financial analytics|financial information|"
                r"index provider|benchmark|credit rating|research.*financ", re.I),
     Archetype.FINANCIAL_DATA, 0.90),
    # Investment Banks / Broker Dealers
    (re.compile(r"investment bank|capital markets|broker.?dealer|securities brokerage|"
                r"prime broker|underwriting", re.I),
     Archetype.INVESTMENT_BANKS, 0.85),
    # Fintech Platforms
    (re.compile(r"fintech|neobank|digital bank|digital lending|buy now pay|"
                r"lending platform|payment processing|digital payment|"
                r"money transfer|remittance|cryptocurrency exchange", re.I),
     Archetype.FINTECH, 0.85),
    # Diversified financial (lower confidence — could be banks or asset managers)
    (re.compile(r"diversified financial|financial services|financial holding", re.I),
     Archetype.OTHER, 0.50),
]


def _classify_by_industry(industry: str) -> Optional[tuple[Archetype, float]]:
    """Return (archetype, confidence) or None if no pattern matches."""
    for pattern, arch, conf in _INDUSTRY_PATTERNS:
        if pattern.search(industry):
            return arch, conf
    return None


# ── Revenue model inference ────────────────────────────────────────────────────
# Uses available financial ratios to disambiguate when industry string is vague.

def _classify_by_revenue_model(
    gross_margin:      Optional[float],
    operating_margin:  Optional[float],
    net_margin:        Optional[float],
    debt_equity:       Optional[float],
    sector:            str,
) -> Optional[tuple[Archetype, float]]:
    """
    Infer archetype from margin profile when industry label is ambiguous.

    Rules (in priority order):
    · Very high gross margin (>0.70) + recurring revenue profile → Financial Data or Exchanges
    · Very high leverage (D/E > 8) + financial sector → Banks
    · Thin gross margin (<0.30) + financial sector → Banks or Insurance
    · Moderate gross margin (0.40–0.70) + financial sector → Asset Managers or Fintech
    """
    if sector.lower() not in ("financial services", "financials", "finance"):
        return None

    gm  = gross_margin
    dm  = debt_equity

    if gm is not None:
        if gm > 0.70:
            # High gross margin typical of software-like financial data or exchanges
            return Archetype.FINANCIAL_DATA, 0.55
        if gm < 0.30:
            # Low gross margin suggests intermediary (bank-like) or insurance
            if dm is not None and dm > 5.0:
                return Archetype.BANKS, 0.60
            return Archetype.INSURANCE, 0.50
        if 0.40 <= gm <= 0.70:
            # Moderate gross margin → asset manager or fintech
            if operating_margin is not None and operating_margin > 0.30:
                return Archetype.ASSET_MANAGERS, 0.55
            return Archetype.FINTECH, 0.50

    # High leverage in financial sector → bank
    if dm is not None and dm > 8.0:
        return Archetype.BANKS, 0.60

    return None


# ── Overrides ──────────────────────────────────────────────────────────────────

_OVERRIDES_PATH = Path(__file__).parent.parent / "data" / "peer_overrides.json"

def _load_overrides() -> dict[str, dict]:
    if _OVERRIDES_PATH.exists():
        try:
            with open(_OVERRIDES_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

_OVERRIDES: dict[str, dict] = _load_overrides()


def _classify_by_override(ticker: str) -> Optional[ClassificationResult]:
    entry = _OVERRIDES.get(ticker.upper())
    if entry is None:
        return None
    arch_id  = entry.get("archetype", 8)
    reason   = entry.get("reason", "manual override")
    return ClassificationResult(
        archetype  = Archetype(arch_id),
        method     = "override",
        confidence = 1.0,
        note       = reason,
    )


# ── Main classifier ────────────────────────────────────────────────────────────

class PeerSelectionEngine:
    """
    Adaptive peer filtering and ranking engine.

    Usage
    -----
    engine = PeerSelectionEngine()

    # Step 1 — classify
    result = engine.classify(
        ticker="COIN", sector="Financial Services", industry="Crypto Exchange",
        gross_margin=0.87, operating_margin=0.20, net_margin=0.15,
        debt_equity=None,
    )

    # Step 2 — get adaptive market cap bands
    bands = engine.get_adaptive_bands(result.archetype)
    # → Use bands.lo / bands.hi in _build_candidate_pool screener call

    # Step 3 — structural filter (call after usable_rows collected)
    filtered = engine.apply_structural_filters(target_row, usable_rows, result.archetype,
                                               target_mkt_cap, target_ps)

    # Step 4 — rank by economic similarity
    ranked = engine.rank_peers(target_row, filtered, result.archetype)

    # Step 5 — auto-expand if too few peers
    if len(ranked) < 3:
        expanded_bands = engine.get_adaptive_bands(result.archetype, expanded=True)
        # rebuild candidate pool with expanded bands, then re-filter/re-rank
    """

    def classify(
        self,
        ticker:           str = "",
        sector:           str = "",
        industry:         str = "",
        gross_margin:     Optional[float] = None,
        operating_margin: Optional[float] = None,
        net_margin:       Optional[float] = None,
        debt_equity:      Optional[float] = None,
        sic:              Optional[int] = None,
    ) -> ClassificationResult:
        """
        Classify a company into a business model archetype.

        Decision tree
        -------------
        0. Check manual overrides first (always wins)
        1. SIC code fast-path (high confidence for regulated entities)
        2. Industry string pattern match
        3. Revenue model inference from margin profile
        4. Fallback: Archetype.OTHER
        """
        # Step 0: overrides
        if ticker:
            override = _classify_by_override(ticker)
            if override is not None:
                return override

        # Step 1: SIC fast-path
        sic_arch = _classify_by_sic(sic)
        if sic_arch is not None:
            return ClassificationResult(
                archetype=sic_arch, method="sic", confidence=0.95,
                note=f"SIC={sic}",
            )

        # Step 2: industry string
        if industry:
            match = _classify_by_industry(industry)
            if match is not None:
                arch, conf = match
                return ClassificationResult(
                    archetype=arch, method="industry_match", confidence=conf,
                    note=f"industry={industry!r}",
                )

        # Step 3: revenue model inference
        rev_match = _classify_by_revenue_model(
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            debt_equity=debt_equity,
            sector=sector,
        )
        if rev_match is not None:
            arch, conf = rev_match
            return ClassificationResult(
                archetype=arch, method="revenue_model", confidence=conf,
                note=f"gm={gross_margin} om={operating_margin} de={debt_equity}",
            )

        # Step 4: fallback
        return ClassificationResult(
            archetype=Archetype.OTHER, method="fallback", confidence=0.30,
            note="no signal resolved",
        )

    # ── Adaptive bands ─────────────────────────────────────────────────────────

    def get_adaptive_bands(
        self,
        archetype: Archetype,
        expanded:  bool = False,
    ) -> BandConfig:
        """Return the BandConfig for the given archetype."""
        cfg = _BAND_TABLE.get(archetype, _BAND_TABLE[Archetype.OTHER])
        if expanded:
            # Return a view with expanded bands (don't mutate original)
            return BandConfig(
                lo           = cfg.lo_expanded,
                hi           = cfg.hi_expanded,
                lo_expanded  = cfg.lo_expanded,
                hi_expanded  = cfg.hi_expanded,
                rev_lo       = cfg.rev_lo * 0.5 if cfg.rev_lo else None,
                rev_hi       = cfg.rev_hi * 2.0 if cfg.rev_hi else None,
                margin_tol   = (cfg.margin_tol * 1.5) if cfg.margin_tol else None,
            )
        return cfg

    # ── Structural filtering ───────────────────────────────────────────────────

    def apply_structural_filters(
        self,
        target_row:    "PeerRow",  # type: ignore[name-defined]  # avoids circular import
        candidates:    list,
        archetype:     Archetype,
        target_mkt_cap: Optional[float] = None,
        target_ps:     Optional[float] = None,
        target_geo:    str = GeoRegion.US,
        expanded:      bool = False,
    ) -> list:
        """
        Apply structural constraints and geographic preference.

        Structural filters (hard):
        · Revenue similarity: peer_rev in [cfg.rev_lo × target_rev, cfg.rev_hi × target_rev]
        · Margin similarity: |peer_gm − target_gm| ≤ cfg.margin_tol
          (both filters only applied when target values are available)

        Geographic preference (soft — penalises but does not hard-exclude):
        · Same region → priority boost (sort key bonus)
        · US-listed always preferred for US targets
        """
        cfg = self.get_adaptive_bands(archetype, expanded=expanded)

        # Estimate target revenue via mktcap / PS
        target_revenue: Optional[float] = None
        if target_mkt_cap and target_ps and target_ps > 0:
            target_revenue = target_mkt_cap / target_ps
        elif hasattr(target_row, "market_cap") and hasattr(target_row, "ps"):
            mc = getattr(target_row, "market_cap", None)
            ps = getattr(target_row, "ps", None)
            if mc and ps and ps > 0:
                target_revenue = mc / ps

        target_gm = getattr(target_row, "gross_margin", None)

        passed: list = []
        excluded_rev   = 0
        excluded_margin = 0

        for row in candidates:
            # ── Revenue similarity filter ────────────────────────────────────
            if cfg.rev_lo is not None and target_revenue is not None:
                peer_mc = getattr(row, "market_cap", None)
                peer_ps = getattr(row, "ps", None)
                if peer_mc and peer_ps and peer_ps > 0:
                    peer_revenue = peer_mc / peer_ps
                    rev_ratio    = peer_revenue / target_revenue
                    if not (cfg.rev_lo <= rev_ratio <= cfg.rev_hi):
                        excluded_rev += 1
                        continue

            # ── Margin similarity filter ─────────────────────────────────────
            if cfg.margin_tol is not None and target_gm is not None:
                peer_gm = getattr(row, "gross_margin", None)
                if peer_gm is not None:
                    if abs(peer_gm - target_gm) > cfg.margin_tol:
                        excluded_margin += 1
                        continue

            passed.append(row)

        if excluded_rev or excluded_margin:
            print(
                f"  [PSE] structural filter: {len(candidates)} in → {len(passed)} pass "
                f"(rev_excl={excluded_rev} margin_excl={excluded_margin})"
            )

        # ── Geographic soft preference — sort US-matching peers higher ───────
        def _geo_key(row) -> int:
            peer_geo = getattr(row, "_geo", None) or GeoRegion.US
            if target_geo == GeoRegion.US:
                return 0 if peer_geo == GeoRegion.US else (1 if peer_geo == GeoRegion.DEV else 2)
            if target_geo == GeoRegion.DEV:
                return 0 if peer_geo == GeoRegion.DEV else (1 if peer_geo == GeoRegion.US else 2)
            return 0 if peer_geo == GeoRegion.EM else 1

        # Stable sort: maintain economic similarity order but prefer same region
        passed.sort(key=_geo_key)
        return passed

    # ── Peer ranking ───────────────────────────────────────────────────────────

    def rank_peers(
        self,
        target_row:  "PeerRow",  # type: ignore[name-defined]
        candidates:  list,
        archetype:   Archetype,
    ) -> list:
        """
        Rank candidates by weighted Euclidean distance from the target.

        Lower distance = more economically similar.
        Falls back to completeness scoring when key archetype metrics are all absent.
        """
        if not candidates:
            return []

        weights = _WEIGHTS.get(archetype, _WEIGHTS[Archetype.OTHER])
        metric_names = list(weights.keys())

        # Build value vectors — collect all non-None values per metric for normalisation
        all_vals: dict[str, list[float]] = {m: [] for m in metric_names}
        all_rows = [target_row] + candidates
        for row in all_rows:
            for m in metric_names:
                v = getattr(row, m, None)
                if v is not None:
                    all_vals[m].append(v)

        def _normalise(val: Optional[float], vals: list[float]) -> Optional[float]:
            if val is None or not vals:
                return None
            mn, mx = min(vals), max(vals)
            if mx == mn:
                return 0.5
            return (val - mn) / (mx - mn)

        def _distance(target_vec: dict[str, Optional[float]],
                      cand_vec:   dict[str, Optional[float]]) -> tuple[float, int]:
            """Return (distance, metrics_used). Lower distance = better match."""
            total_w  = 0.0
            sq_sum   = 0.0
            used     = 0
            for m, w in weights.items():
                t = target_vec.get(m)
                c = cand_vec.get(m)
                if t is None or c is None:
                    continue
                sq_sum  += w * (t - c) ** 2
                total_w += w
                used    += 1
            if total_w == 0:
                return float("inf"), 0
            # Normalise by actual weight sum (handle missing metrics gracefully)
            return math.sqrt(sq_sum / total_w), used

        # Vectorise target
        target_vec: dict[str, Optional[float]] = {
            m: _normalise(getattr(target_row, m, None), all_vals[m])
            for m in metric_names
        }

        scored: list[tuple[float, int, object]] = []
        for row in candidates:
            cand_vec = {
                m: _normalise(getattr(row, m, None), all_vals[m])
                for m in metric_names
            }
            dist, used = _distance(target_vec, cand_vec)
            scored.append((dist, -used, row))   # sort: asc dist, desc metrics_used

        scored.sort(key=lambda x: (x[0], x[1]))

        print(
            f"  [PSE] ranked {len(scored)} candidates (archetype={ARCHETYPE_NAMES[archetype]}):"
        )
        for dist, neg_used, row in scored[:6]:
            print(
                f"    {getattr(row, 'ticker', '?')}: dist={dist:.4f} "
                f"metrics_used={-neg_used}"
            )

        return [row for _, _, row in scored]

    # ── Peer justification ────────────────────────────────────────────────────

    def justify_peer(
        self,
        target_ticker:    str,
        target_archetype: Archetype,
        peer_ticker:      str,
        peer_name:        str,
        peer_archetype:   Archetype,
        peer_mkt_cap:     float = 0.0,
    ) -> str:
        """
        Generate a one-sentence justification for why peer_ticker is a relevant
        comparable for target_ticker.

        Always references:
          (a) business model / revenue type
          (b) economic driver
          (c) key value metric

        Same-archetype peers use the base template.
        Cross-archetype peers note the shared characteristic and the divergence.
        """
        display = peer_name or peer_ticker
        scale   = _scale_label(peer_mkt_cap) if peer_mkt_cap else "comparable"

        rev_type    = _ARCHETYPE_REVENUE_TYPE.get(target_archetype, "diversified revenue")
        econ_driver = _ARCHETYPE_ECON_DRIVER.get(target_archetype, "core business economics")
        key_metric  = _ARCHETYPE_KEY_METRIC.get(target_archetype, "segment EBIT contribution")

        if peer_archetype == target_archetype:
            return (
                f"{display} shares {target_ticker}'s {rev_type} model and "
                f"{econ_driver} sensitivity, providing a direct read on how the "
                f"market prices {key_metric} at {scale} scale."
            )

        # Cross-archetype template
        key        = (int(target_archetype), int(peer_archetype))
        shared     = _CROSS_ARCH_SHARED.get(key, econ_driver)
        divergence = _CROSS_ARCH_DIVERGENCE.get(
            key, "business model mix limits direct comparability"
        )
        return (
            f"{display}'s {shared} provides a valuation anchor for "
            f"{target_ticker}'s {econ_driver} economics, though "
            f"{divergence}."
        )

    # ── Simplified peer ranking (relaxation cascade Steps 3–4) ───────────────

    def rank_peers_simplified(
        self,
        target_row:  "PeerRow",  # type: ignore[name-defined]
        candidates:  list,
    ) -> list:
        """
        Rank candidates using only {revenue_growth, operating_margin, gross_margin}.
        Used when full archetype weights cannot produce a meaningful ranking (sparse data).

        Weights: revenue_growth=0.40, operating_margin=0.35, gross_margin=0.25.
        Falls back to completeness-score order when all three metrics are absent.
        """
        if not candidates:
            return []

        SIMPLE_METRICS: dict[str, float] = {
            "revenue_growth":   0.40,
            "operating_margin": 0.35,
            "gross_margin":     0.25,
        }

        # Collect all non-None values per metric for min-max normalisation
        all_vals: dict[str, list[float]] = {m: [] for m in SIMPLE_METRICS}
        for row in [target_row] + candidates:
            for m in SIMPLE_METRICS:
                v = getattr(row, m, None)
                if v is not None:
                    all_vals[m].append(v)

        def _norm(val: Optional[float], vals: list[float]) -> Optional[float]:
            if val is None or not vals:
                return None
            mn, mx = min(vals), max(vals)
            return 0.5 if mx == mn else (val - mn) / (mx - mn)

        def _dist(t_vec: dict, c_vec: dict) -> tuple[float, int]:
            sq, tw, used = 0.0, 0.0, 0
            for m, w in SIMPLE_METRICS.items():
                t = t_vec.get(m)
                c = c_vec.get(m)
                if t is None or c is None:
                    continue
                sq  += w * (t - c) ** 2
                tw  += w
                used += 1
            if tw == 0:
                return float("inf"), 0
            return math.sqrt(sq / tw), used

        t_vec = {m: _norm(getattr(target_row, m, None), all_vals[m]) for m in SIMPLE_METRICS}

        scored: list[tuple[float, int, object]] = []
        for row in candidates:
            c_vec = {m: _norm(getattr(row, m, None), all_vals[m]) for m in SIMPLE_METRICS}
            dist, used = _dist(t_vec, c_vec)
            scored.append((dist, -used, row))

        scored.sort(key=lambda x: (x[0], x[1]))
        return [row for _, _, row in scored]

    # ── Convenience: full pipeline ─────────────────────────────────────────────

    def select_peers(
        self,
        target_row:      "PeerRow",  # type: ignore[name-defined]
        candidates:      list,
        ticker:          str = "",
        sector:          str = "",
        industry:        str = "",
        gross_margin:    Optional[float] = None,
        operating_margin: Optional[float] = None,
        net_margin:      Optional[float] = None,
        debt_equity:     Optional[float] = None,
        sic:             Optional[int] = None,
        target_mkt_cap:  Optional[float] = None,
        target_ps:       Optional[float] = None,
        target_geo:      str = GeoRegion.US,
        max_peers:       int = 4,
        min_peers:       int = 3,
    ) -> tuple[list, ClassificationResult]:
        """
        Full pipeline: classify → filter → rank → auto-expand if needed.

        Returns (selected_peers, ClassificationResult).
        selected_peers has at most max_peers items.
        """
        classification = self.classify(
            ticker=ticker, sector=sector, industry=industry,
            gross_margin=gross_margin, operating_margin=operating_margin,
            net_margin=net_margin, debt_equity=debt_equity, sic=sic,
        )
        archetype = classification.archetype

        print(
            f"  [PSE] {ticker}: archetype={ARCHETYPE_NAMES[archetype]!r} "
            f"method={classification.method!r} confidence={classification.confidence:.2f}"
        )

        # Structural filter (normal bands)
        filtered = self.apply_structural_filters(
            target_row, candidates, archetype,
            target_mkt_cap=target_mkt_cap, target_ps=target_ps,
            target_geo=target_geo, expanded=False,
        )

        # Auto-expand if too few peers
        if len(filtered) < min_peers and len(candidates) > len(filtered):
            print(
                f"  [PSE] only {len(filtered)} peers after normal filter "
                f"— auto-expanding structural constraints"
            )
            filtered = self.apply_structural_filters(
                target_row, candidates, archetype,
                target_mkt_cap=target_mkt_cap, target_ps=target_ps,
                target_geo=target_geo, expanded=True,
            )
            if not filtered:
                # Last resort: use all candidates
                filtered = candidates

        # Rank
        ranked = self.rank_peers(target_row, filtered, archetype)
        return ranked[:max_peers], classification


# ── Singleton for import convenience ──────────────────────────────────────────

_engine: Optional[PeerSelectionEngine] = None

def get_engine() -> PeerSelectionEngine:
    global _engine
    if _engine is None:
        _engine = PeerSelectionEngine()
    return _engine
