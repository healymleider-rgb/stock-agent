"""
MemoEngine
==========
Institutional-quality investment memo generation.

Design principles
-----------------
· Synthesize, don't transcribe — bullets are constructed from structured
  inputs, not trimmed from verbose category reasoning strings.
· Specificity over generality — every sentence must contribute a claim
  that is falsifiable or at least specific to this company's data profile.
· Tension-forward — the memo exists to convey the core tradeoff, not to
  summarise what data is available.
· Language discipline — banned phrases, passive-to-active substitutions,
  and per-section word budgets are enforced mechanically.
· Forward-looking only — change-view triggers describe future states
  (what has NOT yet happened), never the current situation restated.

Output target: ≤200 words for the Investment Memo section (Top Takeaway
through Verdict). The data section above the memo is not counted.

Integration
-----------
    from analysis.memo_engine import MemoEngine, MemoInput

    engine = MemoEngine()
    result = engine.build(MemoInput(
        scorecard   = sc,
        company     = "Visa Inc.",
        sector      = "Financial Services",
        industry    = "Credit Services",
        macro       = macro_findings,
        pe          = 28.4,
        ps          = 14.2,
        ev_ebitda   = 22.1,
        price       = 272.30,
    ))
    memo_text = result.render()
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ── Language quality rules ─────────────────────────────────────────────────────
#
# Applied to EVERY string produced by the engine before it is returned.
# Each entry is (pattern, replacement) where pattern is a compiled regex.
# Applied in order — be careful about overlapping patterns.

_FILLER_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bit is worth noting that\b", re.I),  ""),
    (re.compile(r"\bit should be noted that\b", re.I),  ""),
    (re.compile(r"\bone important factor is\b", re.I),  ""),
    (re.compile(r"\boverall[,]?\s+", re.I),             ""),
    (re.compile(r"\bgenerally speaking[,]?\s*", re.I),  ""),
    (re.compile(r"\bwith that said[,]?\s*", re.I),      ""),
    (re.compile(r"\bthat being said[,]?\s*", re.I),     ""),
    (re.compile(r"\bmoreover[,]?\s+", re.I),            ""),
    (re.compile(r"\bfurthermore[,]?\s+", re.I),         ""),
    (re.compile(r"\badditionally[,]?\s+", re.I),        "Additionally "),
    (re.compile(r"\bthis is indicative of\b", re.I),    "indicating"),
    (re.compile(r"\bdemonstrates the company'?s\b", re.I), "reflects"),
    (re.compile(r"\bprovides a strong foundation for\b", re.I), "supports"),
    (re.compile(r"\bshows strong\b", re.I),             "shows solid"),
    (re.compile(r"\bvery strong\b", re.I),              "strong"),
    (re.compile(r"\bvery weak\b", re.I),                "weak"),
    (re.compile(r"\brelatively\s+", re.I),              ""),
    (re.compile(r"\bcurrently\s+", re.I),               ""),
    (re.compile(r"\bis currently\b", re.I),             "is"),
    (re.compile(r"\bhas been\s+", re.I),                "is "),
    (re.compile(r"\bhas shown\b", re.I),                "shows"),
    (re.compile(r"\bthe company\s+(?=has|is|does|shows|reports)", re.I), ""),
    # Catch metric-fragment bullets that survived the _make_bullet() label strip.
    # Pattern: "high-quality business: gross margin …" produced by old factor-prefix logic.
    # Replacement is a complete institutional sentence so the bullet always reads cleanly.
    (
        re.compile(r"\bhigh[- ]quality business:?\s*gross margin\b[^.!?]*", re.I),
        "high-quality business with sector-leading margins and strong returns on capital",
    ),
    (re.compile(r"\s{2,}"),                             " "),   # collapse whitespace
]

# Change-view reframe logic lives in _reframe_trigger() below.

# Words that must NOT be the last word of a sentence.
# Ending on these signals a mid-clause cut, not a complete thought.
_DANGLING_WORDS: frozenset[str] = frozenset({
    # coordinating conjunctions
    "and", "but", "or", "nor", "yet", "so",
    # subordinating conjunctions
    "if", "as", "than", "because", "while", "since", "although", "unless",
    # prepositions
    "of", "with", "to", "for", "in", "on", "at", "by", "from",
    "into", "about", "through", "between", "against", "toward",
    # articles / determiners — never end a clause
    "a", "an", "the",
    # relative pronouns
    "which", "that", "who", "whose", "when", "where",
})

# Regex for detecting competitive-positioning signals in bullish_factors.
# Used by _thesis_bullets() to guarantee a competitive-positioning bullet
# is attempted even when no category score directly covers it.
_COMPETITIVE_KEYWORDS: re.Pattern = re.compile(
    r"\b(moat|market share|network effect|switching cost|pricing power"
    r"|scale advantage|competitive position|market leader|dominant"
    r"|barriers to entry|category leader|platform advantage)\b",
    re.I,
)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class MemoInput:
    """All inputs needed to generate a memo."""
    scorecard:    object          # models.scorecard.Scorecard
    company:      str
    sector:       str
    industry:     str
    macro:        dict = field(default_factory=dict)
    pe:           Optional[float] = None
    ps:           Optional[float] = None
    ev_ebitda:    Optional[float] = None
    price:        Optional[float] = None
    # ── Peer comparison context (Peer Comparison Enforcer, requirement #4) ─────
    # peer_medians: flat dict of {metric: value} for the comparable universe.
    #   Keys: "pe", "ps", "ev_ebitda", "gross_margin", "revenue_growth"
    #   e.g. {"pe": 24.1, "ps": 5.2, "ev_ebitda": 18.0}
    # peer_rows:    list of per-peer dicts for percentile computation.
    #   e.g. [{"ticker": "MSFT", "pe": 31.2, "ps": 11.4}, ...]
    # When peer_medians is populated, valuation and growth bullets are formatted
    # as "[Metric]: subject [X] vs peer median [Y] — [interpretation]".
    peer_medians: dict = field(default_factory=dict)
    peer_rows:    list = field(default_factory=list)
    # ACTION from _derive_outlook_action ("BUY", "STAGED BUY", "WAIT", "HOLD", "SELL").
    # When set, TOP TAKEAWAY and FINAL VERDICT derive their rating language from
    # this value rather than the raw score/stance rating string.
    action:       str  = ""


@dataclass
class MemoResult:
    """Fully rendered memo sections."""
    top_takeaway:      str
    thesis_bullets:    list[str]
    risk_bullets:      list[str]
    change_view:       list[str]
    verdict:           str
    key_tension:       Optional[str] = None
    word_count:        int = 0
    _locked:           bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def content_word_count(self) -> int:
        """Word count of the 4 required memo sections only.

        Excludes verdict, key_tension, section headers, and separators.
        Verdict and key_tension are rendered separately in the UI and are not
        subject to the 120–180-word constraint.
        """
        parts = [
            self.top_takeaway,
            " ".join(self.thesis_bullets),
            " ".join(self.risk_bullets),
            " ".join(self.change_view),
        ]
        return len(" ".join(filter(None, parts)).split())

    def __setattr__(self, name: str, value) -> None:
        # Underscore-prefixed attributes are internal engine metadata and are
        # always settable — only public fields are frozen after lock().
        if getattr(self, "_locked", False) and not name.startswith("_"):
            raise AttributeError(
                f"[MEMO LOCK] MemoResult is immutable after lock() — "
                f"attempted to set '{name}'."
            )
        object.__setattr__(self, name, value)

    def lock(self) -> "MemoResult":
        """Freeze the memo object — no further field mutations allowed after this point."""
        object.__setattr__(self, "_locked", True)
        return self

    def render(self) -> str:
        """
        Render the memo as a plain-text block.
        Sections are separated by blank lines.
        Target: ≤200 words.
        """
        lines: list[str] = []

        lines += ["  INVESTMENT MEMO", "  ───────────────", ""]

        # TOP TAKEAWAY — rendered as a flowing paragraph (not sentence-per-line)
        lines += ["  TOP TAKEAWAY", "  ────────────"]
        lines.append(f"  {self.top_takeaway.strip()}")
        lines.append("")

        # INVESTMENT THESIS
        lines += ["  INVESTMENT THESIS", "  ─────────────────"]
        for b in self.thesis_bullets:
            lines.append(f"    • {b}")
        lines.append("")

        # KEY RISKS
        lines += ["  KEY RISKS", "  ─────────"]
        for b in self.risk_bullets:
            lines.append(f"    • {b}")
        lines.append("")

        # WHAT WOULD CHANGE OUR VIEW
        lines += ["  WHAT WOULD CHANGE OUR VIEW", "  ──────────────────────────"]
        for t in self.change_view:
            lines.append(f"    → {t}")
        lines.append("")

        # KEY TENSION
        if self.key_tension:
            lines.append(f"  KEY TENSION: {self.key_tension}")
            lines.append("")

        # FINAL VERDICT
        lines += ["  FINAL VERDICT", "  ─────────────"]
        lines.append(f"  {self.verdict}")

        text = "\n".join(lines)
        return text

    def validate(self) -> list[str]:
        """
        Return a list of violation strings. Empty list means the memo is clean.

        Hard violations (trigger regeneration):
          OVERVIEW         — rendered text contains the word 'Overview'
          TRUNCATION       — any field ends without terminal punctuation, ends with a
                             dangling conjunction/preposition, or ends on a single
                             non-article letter (all indicate a mid-sentence cut)
          SECTION_COUNT    — any of thesis, risks, change_view has fewer than 3 items
          STRUCTURE_MISSING— a required section header is absent from the rendered memo
          REQUIRED_RISKS   — valuation, momentum, or macro risk bullet is missing
          DUPLICATE_PHRASE — a 4+ consecutive-word sequence appears in 2+ bullets
          MEMO_OVERFLOW    — content word count exceeds 180 (hard cap; 4 sections only)

        Soft violations (logged, do not trigger regeneration):
          MEMO_TOO_SHORT   — content word count below 120 (cannot be fixed by tightening)
        """
        rendered = self.render()
        violations: list[str] = []

        _terminal = re.compile(r"[.!?]$")

        # OVERVIEW — MemoEngine must never emit this heading
        if "Overview" in rendered:
            violations.append(
                "OVERVIEW: rendered memo contains 'Overview' — "
                "remove from _top_takeaway() or _thesis_bullets() in memo_engine.py"
            )

        # TRUNCATION — comprehensive completeness check per text field
        def _check_completeness(label: str, text: str) -> None:
            stripped = text.strip()
            if not stripped:
                return
            # 1. Terminal punctuation missing
            if not _terminal.search(stripped):
                violations.append(
                    f"TRUNCATION: {label} ends without terminal punctuation — "
                    f"'{stripped[-60:]}'"
                )
                return   # remaining checks apply to the pre-punct text
            # 2. Dangling conjunction/preposition before terminal punctuation
            #    e.g. "...strong growth and." or "...risk of."
            body = re.sub(r"[.!?]+$", "", stripped)
            last_word = body.split()[-1].lower().rstrip(".,;:") if body.split() else ""
            if last_word in _DANGLING_WORDS:
                violations.append(
                    f"TRUNCATION: {label} ends with dangling word '{last_word}' — "
                    f"'{stripped[-60:]}'"
                )
            # 3. Single-character last word that is not a valid sentence-final word
            #    Catches "...P/E of 36x l." (l = truncated "leaves")
            elif len(last_word) == 1 and last_word not in {"a", "i"}:
                violations.append(
                    f"TRUNCATION: {label} ends with suspicious single-character word '{last_word}' — "
                    f"'{stripped[-60:]}'"
                )

        checks: list[tuple[str, str]] = [("top_takeaway", self.top_takeaway)]
        if self.verdict:
            checks.append(("verdict", self.verdict))
        for label, text in checks:
            _check_completeness(label, text)

        all_bullets = (
            [("thesis", b) for b in self.thesis_bullets]
            + [("risks",  b) for b in self.risk_bullets]
            + [("change_view", b) for b in self.change_view]
        )
        for section, bullet in all_bullets:
            _check_completeness(section, bullet)

        # SECTION_COUNT — minimum bullets per section
        min_bullets = 3
        for name, lst in [
            ("thesis_bullets", self.thesis_bullets),
            ("risk_bullets",   self.risk_bullets),
            ("change_view",    self.change_view),
        ]:
            if len(lst) < min_bullets:
                violations.append(
                    f"SECTION_COUNT: {name} has {len(lst)} item(s) — minimum is {min_bullets}"
                )

        # STRUCTURE_MISSING — all required section headers must be present
        for header in ("TOP TAKEAWAY", "INVESTMENT THESIS", "KEY RISKS", "WHAT WOULD CHANGE", "FINAL VERDICT"):
            if header not in rendered:
                violations.append(
                    f"STRUCTURE_MISSING: section '{header}' not found in rendered memo"
                )

        # REQUIRED_RISKS — valuation, momentum, and business model bullets must always appear.
        # These are unconditionally emitted by _risk_bullets(); this check guards
        # against accidental removal or deduplication swallowing them.
        rendered_lower = rendered.lower()
        for risk_type, keywords in (
            ("valuation",       ("valuation risk", "valuation watch")),
            ("momentum",        ("momentum risk",)),
            ("business model",  ("business model risk", "model risk", "competitive risk", "disruption risk")),
        ):
            if not any(kw in rendered_lower for kw in keywords):
                violations.append(
                    f"REQUIRED_RISKS: '{risk_type}' risk bullet missing from Key Risks section"
                )

        # DUPLICATE_PHRASE — detect repeated 4+ consecutive-word sequences across bullets.
        # One phrase appearing in two different bullets signals either redundancy or a
        # copy-paste artefact from agent-provided inputs.
        def _four_grams(text: str) -> set[str]:
            ws = text.lower().split()
            return {" ".join(ws[i:i + 4]) for i in range(len(ws) - 3)}

        all_bullets = self.thesis_bullets + self.risk_bullets + self.change_view
        seen_grams: set[str] = set()
        for bullet in all_bullets:
            grams = _four_grams(bullet)
            overlap = grams & seen_grams
            if overlap:
                violations.append(
                    f"DUPLICATE_PHRASE: repeated 4-gram across bullets — '{next(iter(overlap))}'"
                )
                break   # one violation is enough to trigger regeneration
            seen_grams.update(grams)

        # CONTRADICTION — Buy/Strong Buy rating coexisting with weak category scores.
        # These are SOFT violations: the rating originates in the scoring engine and
        # cannot be overridden here, but the calling code must surface the tension.
        #
        # Rule:
        #   1–2 categories below 60 → CONTRADICTION_WATCH (log only)
        #   3+  categories below 60 → CONTRADICTION (requires forward-looking justification
        #       in the narrative or a rating downgrade upstream)
        _schema = getattr(self, "_schema", {})
        _rating_val = _schema.get("rating", "")
        if _rating_val in ("Buy", "Strong Buy"):
            _cat_scores = _schema.get("category_scores", {})
            _low_cats = sorted(
                cat for cat, score in _cat_scores.items()
                if score is not None and score < 60
            )
            if len(_low_cats) > 2:
                violations.append(
                    f"CONTRADICTION: {_rating_val} rating with {len(_low_cats)} categories "
                    f"below 60 ({', '.join(_low_cats)}) — "
                    f"add explicit forward-looking justification to the verdict or downgrade upstream"
                )
            elif _low_cats:
                violations.append(
                    f"CONTRADICTION_WATCH: {_rating_val} rating — "
                    f"{', '.join(_low_cats)} score{'s' if len(_low_cats) > 1 else ''} below 60; "
                    f"one-sentence explanation in verdict is sufficient"
                )

        # Word count checks — 120–180 word hard window for the 4 core sections.
        # (verdict and key_tension are excluded from content_word_count.)
        _cwc = self.content_word_count
        if _cwc > 180:
            # HARD — triggers regeneration with a tighter budget
            violations.append(
                f"MEMO_OVERFLOW: {_cwc} content words — "
                f"exceeds 180-word hard cap for the 4 required sections"
            )
        elif _cwc < 120:
            # SOFT — too short to fix by tightening budgets; logged only
            violations.append(
                f"MEMO_TOO_SHORT: {_cwc} content words — "
                f"below 120-word minimum (cannot be fixed by regeneration)"
            )

        return violations

    def to_dict(self) -> dict:
        """
        Serialise for API response embedding.

        Schema-first: `quantitative_schema` contains every number that appears
        in the narrative — the API consumer can verify internal consistency by
        checking that no narrative number is absent from the schema block.

        `risk_flags_structured` parallels `risk_bullets` with the five required
        metadata fields per Risk Flag Discipline: trigger, threshold, scenario,
        probability, impact.
        """
        return {
            # ── Prose sections ───────────────────────────────────────────────
            "top_takeaway":   self.top_takeaway,
            "thesis_bullets": self.thesis_bullets,
            "risk_bullets":   self.risk_bullets,
            "change_view":    self.change_view,
            "verdict":        self.verdict,
            "key_tension":    self.key_tension,
            "word_count":     self.word_count,
            # ── Schema block (requirement #1: schema-first) ──────────────────
            # Single source of truth for all quantitative values in the narrative.
            # All derived values (% upside, PEG, spreads) must be computed from
            # these base fields by the consumer — never introduced in the prose.
            "quantitative_schema": getattr(self, "_schema", {}),
            # ── Structured risk flags (requirement #5: risk flag discipline) ─
            # Parallel to risk_bullets; each entry has the five required fields:
            # trigger, threshold, scenario, probability, impact.
            "risk_flags_structured": getattr(self, "_risk_meta", []),
        }


# ── Engine ─────────────────────────────────────────────────────────────────────

class MemoEngine:
    """
    Converts structured scorecard + market data into an institutional memo.

    Call build() as the entry point; all other methods are helpers.
    build() runs a regeneration loop (up to 3 budget tiers) to produce a memo
    that passes validate().  Instance state (_sentence_budget etc.) is reset
    per attempt — do not rely on it between build() calls.
    """

    # Budget tiers for the regeneration loop (sentence_budget, bullet_budget,
    # max_bullets, include_tension).  Tried in order; first clean pass wins.
    #
    # Sizing rationale (target: 120–180 content words across 4 sections):
    #   Takeaway  : 2–3 sentences × sentence_budget words  ≈ 28–42 w
    #   Thesis    : max_bullets × bullet_budget words        ≈ 40–56 w
    #   Risks     : max_bullets × bullet_budget words        ≈ 40–56 w
    #   Change view: max_bullets × bullet_budget words       ≈ 40–56 w
    # At normal tier: ~120–170 words. Tighter/minimal tiers reduce from there.
    _BUDGETS: list[tuple] = [
        (14, 14, 4, False),  # normal  — targets ~140–170 words
        (12, 12, 4, False),  # tighter — targets ~120–150 words
        (10, 10, 3, False),  # minimal — targets ~100–130 words (MEMO_TOO_SHORT possible)
    ]

    def __init__(self) -> None:
        # Defaults match the normal (first) budget tier.
        # build() resets these at the start of each attempt.
        self._sentence_budget: int  = 28
        self._bullet_budget:   int  = 22
        self._max_bullets:     int  = 5
        self._include_tension: bool = True

    # ── Public entry point ─────────────────────────────────────────────────────

    def build(self, inp: MemoInput) -> MemoResult:
        sc  = inp.scorecard
        mac = inp.macro or {}

        # ── PASS 1: Category synthesis ────────────────────────────────────────
        # Produces {category: {score, tier, weight, evidence}} for every scored
        # category.  Pass 2 (text generation) reads from this dict only —
        # narrative claims are therefore traceable to a single authoritative source.
        synthesis = self._build_synthesis(sc)

        # Soft violation prefixes — these do NOT block acceptance and do NOT
        # trigger regeneration. All other violations are hard.
        #   MEMO_TOO_SHORT       — word count too low to fix by tightening
        #   CONTRADICTION        — rating/score tension; cannot be fixed by regenerating
        #   CONTRADICTION_WATCH  — minor rating/score tension; informational only
        # "CONTRADICTION" (no colon) matches both "CONTRADICTION:" and "CONTRADICTION_WATCH:"
        _SOFT_PREFIXES = ("MEMO_TOO_SHORT:", "CONTRADICTION")

        best_result:     Optional[MemoResult] = None
        best_hard_count: int                  = 999

        for attempt, (sb, bb, mb, include_tension) in enumerate(self._BUDGETS):
            self._sentence_budget = sb
            self._bullet_budget   = bb
            self._max_bullets     = mb
            self._include_tension = include_tension

            # ── PASS 2: Text generation ───────────────────────────────────────
            # All generation methods receive the Pass 1 synthesis where possible.
            # _thesis_bullets uses synthesis for weight-ordered category bullets.
            # _risk_bullets uses peer_medians for peer-relative valuation framing.
            takeaway    = self._top_takeaway(sc, inp.company, mac, inp.pe,
                                              action=getattr(inp, "action", ""))
            thesis      = self._thesis_bullets(sc, synthesis)
            risks       = self._risk_bullets(
                sc, mac, inp.pe, inp.ps, inp.ev_ebitda,
                peer_medians=inp.peer_medians,
                peer_rows=inp.peer_rows,
            )
            change_view = self._change_view_bullets(sc)
            verdict     = self._verdict(sc, action=getattr(inp, "action", ""))
            tension     = self._key_tension(sc) if self._include_tension else None

            result = MemoResult(
                top_takeaway   = takeaway,
                thesis_bullets = thesis,
                risk_bullets   = risks,
                change_view    = change_view,
                verdict        = verdict,
                key_tension    = tension,
            )
            result.word_count = result.content_word_count

            # ── Schema-first: attach quantitative snapshot to result ──────────
            # validate() reads _schema for the contradiction check; to_dict()
            # exposes it to the API consumer as the source-of-truth for all
            # numbers that appear in the narrative.
            _cat_scores = {
                cat: _cat_score(sc, cat)
                for cat in ("profitability", "financial_health", "growth",
                            "valuation", "momentum")
            }
            result._schema = {                                    # type: ignore[attr-defined]
                "overall_score":   getattr(sc, "overall_score", None),
                "rating":          _rating_str(sc),
                "category_scores": _cat_scores,
                "pe":              inp.pe,
                "ps":              inp.ps,
                "ev_ebitda":       inp.ev_ebitda,
                "price":           inp.price,
                # Pass 1 outputs — every narrative claim is traceable to these
                # weight/tier/evidence triples (Two-Pass Narrative, requirement #2).
                "synthesis":       synthesis,
                # Peer context snapshot (Peer Comparison Enforcer, requirement #4).
                # Preserved here so the consumer can verify peer-relative claims.
                "peer_medians":    inp.peer_medians or {},
            }

            # ── Risk flag metadata (Risk Flag Discipline, requirement #5) ────
            # One structured entry per required bullet: trigger, threshold,
            # scenario, probability, impact.  Agent-sourced supplemental bullets
            # do not get entries (their metadata is unknown at engine level).
            _v   = _cat_scores.get("valuation")
            _mom = _cat_scores.get("momentum")
            _mac = mac.get("macro_score")
            result._risk_meta = [                                 # type: ignore[attr-defined]
                {
                    "label":       "Valuation risk",
                    "trigger":     "multiple compression or earnings miss vs. consensus expectations",
                    "threshold":   (
                        f"P/E > {inp.pe:.0f}x with growth deceleration > 5%" if inp.pe
                        else "multiples above sector median with any fundamental miss"
                    ),
                    "scenario":    "bear: de-rating to sector median; base: multiple stable; bull: re-rating on sustained beats",
                    "probability": "high" if (_v is not None and _v < 40) else "medium",
                    "impact":      "high",
                },
                {
                    "label":       "Momentum risk",
                    "trigger":     "sustained price decline or break of key technical level",
                    "threshold":   "close below 200-day moving average on above-average volume for 3+ sessions",
                    "scenario":    "bear: -10% additional drawdown; base: consolidation; bull: momentum re-acceleration",
                    "probability": "high" if (_mom is not None and _mom < 45) else "low" if (_mom is not None and _mom >= 70) else "medium",
                    "impact":      "medium",
                },
                {
                    "label":       "Macro risk",
                    "trigger":     "recession onset, Fed tightening above expectations, or credit spread widening",
                    "threshold":   "ISM Manufacturing < 47 for 3 consecutive months OR 10Y yield > 5.5%",
                    "scenario":    "bear: -20% sector-wide de-rating; base: -5% multiple compression; bull: no impact",
                    "probability": "high" if (_mac is not None and _mac < 45) else "medium",
                    "impact":      "medium",
                },
                {
                    "label":       "Business model risk",
                    "trigger":     "competitive displacement, pricing-power erosion, or structural margin compression",
                    "threshold":   "gross margin decline > 300 bps over 2 quarters OR market share loss > 200 bps",
                    "scenario":    "bear: -25% on fundamental re-rating; base: -10% on multiple compression; bull: irrelevant",
                    "probability": "medium",
                    "impact":      "high",
                },
            ]

            violations = result.validate()
            hard = [v for v in violations if not any(v.startswith(p) for p in _SOFT_PREFIXES)]

            if not hard:
                if attempt > 0:
                    print(f"  [MEMO] regeneration succeeded on attempt {attempt + 1}")
                return result

            # Track the attempt with the fewest hard violations as fallback.
            # On tie, prefer the later attempt (tighter budget = shorter = safer).
            if len(hard) <= best_hard_count:
                best_result     = result
                best_hard_count = len(hard)

            print(
                f"  [MEMO] attempt {attempt + 1} failed validation "
                f"(budget sb={sb}/bb={bb}/mb={mb}/tension={include_tension}): "
                + "; ".join(hard[:3])
            )

        # All budget tiers exhausted — return the attempt with the fewest hard violations
        print(
            f"  [MEMO] WARNING: all budget tiers exhausted — "
            f"returning best-effort memo ({best_hard_count} hard violation(s))"
        )
        return best_result  # type: ignore[return-value]  # guaranteed non-None after loop

    # ── Top Takeaway ───────────────────────────────────────────────────────────

    def _top_takeaway(
        self,
        sc:      object,
        company: str,
        macro:   dict,
        pe:      Optional[float],
        action:  str = "",
    ) -> str:
        """
        2–3 sentences. Total target: ≤45 words.

        S1 — business character (what kind of business; quality + growth read)
        S2 — valuation/price tension (is it worth acting on now?)
        S3 — rating + single binding constraint (what determines position size)
        """
        g   = _cat_score(sc, "growth")
        p   = _cat_score(sc, "profitability")
        fh  = _cat_score(sc, "financial_health")
        v   = _cat_score(sc, "valuation")
        mom = _cat_score(sc, "momentum")
        rating = _rating_str(sc)

        # S1 — business character
        quality = _avg(p, fh)
        if quality is not None and quality >= 72 and g is not None and g >= 65:
            s1 = (
                f"{company} is a high-quality, high-growth business —"
                " quality fundamentals and growth underpin the thesis."
            )
        elif quality is not None and quality >= 65 and g is not None and g >= 55:
            s1 = (
                f"{company} delivers consistent profitability"
                " with growth sufficient to sustain the investment case."
            )
        elif quality is not None and quality >= 65 and (g is None or g < 50):
            s1 = (
                f"{company} is a quality-led story —"
                " solid margins, but growth is the limiting factor."
            )
        elif g is not None and g >= 70 and (quality is None or quality < 60):
            s1 = (
                f"{company} is in a high-growth phase —"
                " margin delivery is the key execution risk."
            )
        elif quality is not None and quality < 45:
            s1 = (
                f"{company} faces fundamental pressure —"
                " weak margins and balance sheet stress narrow the margin of safety."
            )
        else:
            s1 = (
                f"{company} presents a mixed fundamental profile —"
                f" the {rating} rating reflects a balanced assessment."
            )

        # S2 — valuation/price tension
        s2 = ""
        if v is not None and g is not None:
            if v < 40 and g >= 65:
                s2 = (
                    "Valuation is elevated, but strong growth provides support —"
                    " execution consistency is the watchpoint."
                )
            elif v >= 65 and g >= 55:
                s2 = "Valuation is attractive relative to the growth profile, offering asymmetric upside."
            elif v < 40 and g < 45:
                s2 = (
                    "Elevated valuation and weak growth compress the margin of safety;"
                    " a catalyst is needed to justify current prices."
                )
            elif v >= 65 and g < 45:
                s2 = "The discount is real, but re-rating requires a growth catalyst, not just multiple compression."
        elif v is not None and v < 40:
            s2 = f"Valuation is stretched{f' at {pe:.1f}x P/E' if pe else ''}; risk/reward is skewed to the downside."
        elif v is not None and v >= 65:
            s2 = f"Valuation is a clear positive{f' at {pe:.1f}x P/E' if pe else ''} — the stock screens cheap."
        elif mom is not None and mom < 45:
            s2 = "Price action is a near-term headwind; a staged entry reduces timing risk."

        # S3 — rating + constraint
        conf = getattr(sc, "confidence", 0.5)
        conf_label = "high" if conf >= 0.70 else "moderate" if conf >= 0.50 else "limited"

        # Identify the binding constraint
        weak_cats = [
            (lbl, score)
            for lbl, score in [
                ("stretched valuation", v),
                ("weak momentum", mom),
                ("limited growth", g),
                ("margin pressure", p),
                ("balance sheet stress", fh),
            ]
            if score is not None and score < 45
        ]
        if action == "BUY":
            if weak_cats:
                constraint = weak_cats[0][0]
                s3 = f"Rating: Buy — price in strong entry zone; {constraint} is the key risk to monitor."
            else:
                s3 = f"Rating: Buy with {conf_label} conviction — price below fair value, fundamentals support the thesis."
        elif action == "STAGED BUY":
            if weak_cats:
                constraint = weak_cats[0][0]
                s3 = f"Rating: Staged Buy — build gradually; {constraint} warrants a measured entry."
            else:
                s3 = f"Rating: Staged Buy — price in the entry zone; accumulate on further weakness."
        elif action == "WAIT":
            s3 = "Rating: Wait — long-term thesis intact; price at or above fair value, await a pullback."
        elif action == "HOLD":
            s3 = "Rating: Hold — fundamentals support the thesis; no immediate entry catalyst at current price."
        elif action == "SELL":
            s3 = "Rating: Sell — thesis broken or price materially above fair value."
        elif weak_cats:
            constraint = weak_cats[0][0]
            s3 = f"Rating: {rating} with {conf_label} conviction — {constraint} caps near-term upside."
        else:
            s3 = f"Rating: {rating} with {conf_label} conviction — fundamentals support current positioning."

        sentences = [s for s in [s1, s2, s3] if s]
        # Cap each sentence using the current sentence budget.
        # _ensure_terminal_punct guarantees every compressed sentence ends cleanly —
        # this is the primary guard against TRUNCATION violations in top_takeaway.
        capped = [
            _ensure_terminal_punct(_compress_to_n_words(s, self._sentence_budget))
            for s in sentences
        ]
        return _clean(" ".join(capped))

    # ── Investment Thesis ──────────────────────────────────────────────────────

    def _thesis_bullets(self, sc: object, synthesis: dict) -> list[str]:
        """
        3–5 bullets. Each bullet must:
          · Be specific to this company's score profile
          · Start with a quality/growth/moat/structural driver
          · Include an implication clause (separated by " — ")
          · Contain ≤bullet_budget words
          · Not repeat information from another bullet

        Sources (in priority order):
          1. Category-derived bullets from Pass 1 synthesis, ordered by importance weight
             (Two-Pass Narrative, requirement #2 — highest-weight insight leads)
          2. Bullish factors (sc.bullish_factors)
          3. Key drivers (sc.key_drivers)
        """
        bullets: list[str] = []
        seen: set[str] = set()

        def _add(text: str) -> bool:
            b = _make_bullet(text, self._bullet_budget)
            key = b[:50].lower()
            if key not in seen and len(b) > 15:
                seen.add(key)
                bullets.append(b)
                return True
            return False

        # Tier 1: category-derived bullets ordered by Pass 1 importance weight.
        # Highest-weight categories lead the thesis — this is the key Two-Pass
        # mechanism: conflicts between categories are resolved by weight rank,
        # not by the order categories happen to appear in _THESIS_CATEGORY_RULES.
        _cat_impl_map  = {rule[0]: rule[2] for rule in _THESIS_CATEGORY_RULES}
        _cat_threshold = {rule[0]: rule[1] for rule in _THESIS_CATEGORY_RULES}

        ordered_cats = sorted(
            synthesis.items(),
            key=lambda kv: kv[1].get("weight", 0.0),
            reverse=True,
        )
        for cat_name, syn in ordered_cats:
            score     = syn.get("score")
            tier      = syn.get("tier", "missing")
            threshold = _cat_threshold.get(cat_name, 65)
            if score is None or score < threshold or tier not in ("very_strong", "strong", "moderate"):
                continue
            cat_obj    = getattr(sc, cat_name, None)
            impl_map   = _cat_impl_map.get(cat_name, {})
            bullet     = _derive_category_bullet(score, cat_obj, impl_map)
            if bullet:
                _add(bullet)

        # Tier 1b: competitive positioning — scan bullish factors for moat/positioning signals.
        # Runs before the general bullish-factor sweep to guarantee at least one
        # competitive-positioning bullet when signals are present.
        if len(bullets) < self._max_bullets:
            for f in getattr(sc, "bullish_factors", [])[:8]:
                if _COMPETITIVE_KEYWORDS.search(f):
                    if _add(f):
                        break   # one competitive bullet is sufficient

        # Tier 2: bullish factors from agents
        for f in getattr(sc, "bullish_factors", [])[:5]:
            if len(bullets) >= self._max_bullets:
                break
            _add(f)

        # Tier 3: key drivers
        for d in getattr(sc, "key_drivers", [])[:3]:
            if len(bullets) >= self._max_bullets:
                break
            _add(d)

        # Fallback: guarantee SECTION_COUNT minimum (3 bullets).
        # In production, bullish_factors and key_drivers from agents fill most
        # gaps; these fallbacks only fire on sparse scorecards with few strong
        # categories and no agent-sourced bullets.
        overall = getattr(sc, "overall_score", 50)
        _FALLBACKS = [
            # Index 0: fired first when zero bullets exist
            (
                "Composite fundamental profile is constructive — above-average scores across key dimensions."
                if overall >= 65 else
                "Scorecard supports the current rating on a balanced read of available evidence."
            ),
            # Index 1 and 2: neutral summary bullets that avoid implying strength
            # not present in the scores (used when category bullets are few)
            "Risk/reward assessment reflects a mixed fundamental profile with select category strengths.",
            "Investment case depends on sustaining the strongest-scoring categories over the cycle.",
        ]
        for _fb in _FALLBACKS:
            if len(bullets) >= 3:
                break
            _add(_fb)

        return bullets[:self._max_bullets]

    # ── Key Risks ──────────────────────────────────────────────────────────────

    def _risk_bullets(
        self,
        sc:          object,
        macro:       dict,
        pe:          Optional[float],
        ps:          Optional[float],
        ev_ebitda:   Optional[float],
        peer_medians: Optional[dict] = None,
        peer_rows:    Optional[list] = None,
    ) -> list[str]:
        """
        3–5 bullets. REQUIRED (always emitted regardless of score):
          · Valuation risk — active framing if v < 55; latent-risk framing otherwise.
            When peer_medians is provided, formats as:
            "[Metric]: subject [X] vs peer median [Y] — [interpretation]"
            (Peer Comparison Enforcer, requirement #4)
          · Momentum risk  — directional framing based on mom score band
          · Macro risk     — regime/recession-level framing; benign fallback if macro OK

        Supplemental (emitted when categories are weak):
          · Fundamental risks from weak categories (growth, margin, balance sheet)
          · Agent risk_flags and bearish_factors

        Each bullet: ≤bullet_budget words, active voice, specific.
        """
        peer_medians = peer_medians or {}
        peer_rows    = peer_rows    or []
        bullets: list[str] = []
        seen: set[str] = set()

        def _add(text: str) -> bool:
            b = _make_bullet(text, self._bullet_budget)
            key = b[:50].lower()
            if key not in seen and len(b) > 10:
                seen.add(key)
                bullets.append(b)
                return True
            return False

        v   = _cat_score(sc, "valuation")
        mom = _cat_score(sc, "momentum")
        g   = _cat_score(sc, "growth")
        p   = _cat_score(sc, "profitability")
        fh  = _cat_score(sc, "financial_health")

        # ── REQUIRED: valuation risk — always emitted ─────────────────────────
        # Language is risk-framed even when valuation score is strong (a rich
        # multiple is itself a risk to the thesis if the growth story falters).
        #
        # Peer Comparison Enforcer (#4): when peer_medians is populated, format
        # the bullet as "[Metric]: subject [X] vs peer median [Y] — [interpretation]"
        # and include percentile position when peer_rows is available.
        if v is not None and v < 55:
            # Active risk — multiples already stretched.
            # Peer-relative format: "[Metric]: subject [X] vs median [Y] — [interpretation]"
            # ≤14 words; percentile label stored in schema not prose (keeps budget).
            if pe is not None and "pe" in peer_medians:
                pm_pe = peer_medians["pe"]
                prem  = ((pe - pm_pe) / pm_pe * 100) if pm_pe else 0.0
                # Percentile label omitted from prose to stay within word budget;
                # consumer computes it from peer_rows in quantitative_schema.
                if prem > 0:
                    _add(f"Valuation risk — P/E: {pe:.0f}x vs median {pm_pe:.0f}x ({prem:.0f}% premium); de-rating risk is elevated.")
                else:
                    _add(f"Valuation risk — P/E: {pe:.0f}x vs median {pm_pe:.0f}x; any miss amplifies de-rating risk.")
            elif ps is not None and "ps" in peer_medians:
                pm_ps = peer_medians["ps"]
                _add(f"Valuation risk — P/S: {ps:.1f}x vs median {pm_ps:.1f}x; stretched multiple amplifies downside.")
            elif ev_ebitda is not None and "ev_ebitda" in peer_medians:
                pm_ev = peer_medians["ev_ebitda"]
                _add(f"Valuation risk — EV/EBITDA: {ev_ebitda:.0f}x vs median {pm_ev:.0f}x; margin of safety is compressed.")
            elif pe is not None and pe > 30:
                _add(f"Valuation risk — P/E of {pe:.0f}x leaves limited downside protection if growth disappoints.")
            elif ps is not None and ps > 8:
                _add(f"Valuation risk — P/S of {ps:.1f}x prices in sustained outperformance; re-rating exposure is high.")
            elif ev_ebitda is not None and ev_ebitda > 20:
                _add(f"Valuation risk — EV/EBITDA of {ev_ebitda:.0f}x compresses the margin of safety.")
            else:
                _add("Valuation risk — multiples are stretched relative to fundamental support; downside risk is asymmetric.")
        else:
            # Valuation not currently stretched — frame as a latent risk
            if pe is not None and "pe" in peer_medians:
                pm_pe = peer_medians["pe"]
                _add(f"Valuation risk — P/E: {pe:.0f}x vs median {pm_pe:.0f}x; any miss would compress the multiple.")
            elif pe is not None:
                _add(f"Valuation risk — P/E of {pe:.0f}x leaves limited room for a growth miss; multiple contraction would amplify any fundamental disappointment.")
            else:
                _add("Valuation risk — any de-rating from current levels would compress risk/reward materially.")

        # ── REQUIRED: momentum risk — always emitted ──────────────────────────
        if mom is not None and mom < 50:
            _add("Momentum risk — negative price trend signals market scepticism; further downside before stabilisation is possible.")
        elif mom is not None and mom >= 70:
            _add("Momentum risk — elevated price momentum increases whipsaw exposure; a trend break would accelerate downside.")
        else:
            _add("Momentum risk — mixed price action warrants a staged entry to manage timing and sequencing risk.")

        # ── REQUIRED: macro risk — always emitted ─────────────────────────────
        rec_risk  = macro.get("recession_risk_level") or macro.get("recession_risk", "")
        mac_score = macro.get("macro_score")
        if rec_risk in ("Elevated", "High") or (mac_score is not None and mac_score < 45):
            _add("Macro risk — elevated recession probability and tightening financial conditions are headwinds.")
        elif macro.get("macro_regime", "") in ("Stagflation", "Late Cycle", "Contraction"):
            _add(f"Macro risk — {macro.get('macro_regime', 'late-cycle')} regime raises cyclical vulnerability.")
        else:
            _add("Macro risk — deterioration in leading indicators or a rate spike would pressure valuations.")

        # ── REQUIRED: business model risk — always emitted ────────────────────
        # Scan bearish factors for competitive/structural signals first;
        # fall back to a generic model-risk statement.
        _BMODEL_KWORDS = re.compile(
            r"competi|disrupt|substitut|platform|customer.concentrat|"
            r"regulation|revenue.model|business.model|market.share|pricing.power",
            re.I,
        )
        _bmodel_emitted = False
        for _bf in getattr(sc, "bearish_factors", [])[:6]:
            if _BMODEL_KWORDS.search(_bf):
                _add(f"Business model risk — {_bf.lstrip('Bearish:').lstrip('Risk:').strip()}")
                _bmodel_emitted = True
                break
        if not _bmodel_emitted:
            # Derive from profitability/growth weakness as a proxy for model pressure
            if p is not None and p < 50 and g is not None and g < 50:
                _add("Business model risk — weak margins and decelerating growth suggest structural competitive pressure.")
            elif p is not None and p < 50:
                _add("Business model risk — margin pressure relative to peers may indicate structural competitive disadvantage.")
            else:
                _add("Business model risk — competitive disruption or pricing-power erosion could compress long-term margins.")

        # ── Fundamental risks from weak categories ────────────────────────────
        if g is not None and g < 45:
            _add("Growth risk — decelerating revenue trajectory limits earnings power and multiple support.")

        if p is not None and p < 45:
            _add("Margin risk — profitability below sector benchmarks constrains free cash flow and limits financial flexibility.")

        if fh is not None and fh < 45:
            _add("Balance sheet risk — elevated leverage or thin liquidity buffer amplifies downside in a stress scenario.")

        # ── Agent-sourced risk flags ────────────────────────────────────────────
        for flag in getattr(sc, "risk_flags", [])[:3]:
            if len(bullets) >= 5:
                break
            _add(flag)

        # ── Bearish factors ────────────────────────────────────────────────────
        for f in getattr(sc, "bearish_factors", [])[:3]:
            if len(bullets) >= 5:
                break
            _add(f)

        # Fallback
        if not bullets:
            bullets.append("No material risk flags identified from available data — monitor earnings delivery.")

        return bullets[:self._max_bullets]

    # ── What Would Change Our View ─────────────────────────────────────────────

    def _change_view_bullets(self, sc: object) -> list[str]:
        """
        3–5 forward-looking triggers.

        Rules:
        · Every trigger must describe a FUTURE state that has NOT yet occurred.
        · If the current state already matches the trigger, reframe as failure/continuation.
        · Supplement sparse agent-provided triggers with category-specific defaults.
        · Each trigger: ≤18 words.
        """
        v   = _cat_score(sc, "valuation")
        mom = _cat_score(sc, "momentum")
        g   = _cat_score(sc, "growth")
        p   = _cat_score(sc, "profitability")
        fh  = _cat_score(sc, "financial_health")

        agent_triggers = list(getattr(sc, "what_would_change_view", []))
        filtered = [_reframe_trigger(t, v, mom, g) for t in agent_triggers[:4]]

        seen: set[str] = set()
        bullets: list[str] = []

        def _add(text: str) -> None:
            key = text[:50].lower()
            if key not in seen and len(text) > 10:
                seen.add(key)
                bullets.append(text)

        for t in filtered:
            _add(t)

        # Supplement with category-specific defaults where needed

        # Bull → Bear flips: if currently bullish, what would break the thesis?
        stance = _stance_str(sc)
        if stance == "Bullish":
            if len(bullets) < 3:
                if g is not None and g >= 65:
                    _add("Material growth deceleration below consensus for two consecutive quarters.")
                if p is not None and p >= 65:
                    _add("Significant margin compression signalling pricing power deterioration.")
                if mom is not None and mom >= 55:
                    _add("Sustained break below the 200-day moving average on above-average volume.")
                _add("Sector re-rating driven by a macro regime shift or rate spike.")

        # Bear → Bull flips: if currently bearish/neutral, what would improve the view?
        elif stance in ("Bearish", "Neutral"):
            if len(bullets) < 3:
                if g is not None and g < 55:
                    _add("Revenue acceleration to at-or-above market growth rate sustained over two quarters.")
                if v is not None and v < 45:
                    _add("Meaningful de-rating to within 10% of the sector median multiple.")
                if mom is not None and mom < 45:
                    _add("Price reclaims the 200-day moving average with improving breadth.")
                if fh is not None and fh < 45:
                    _add("Meaningful debt reduction or equity raise that restores balance sheet flexibility.")

        # Universal triggers
        if len(bullets) < 3:
            _add("Earnings guidance revision that materially alters the long-term growth trajectory.")

        return [_make_bullet(b, self._bullet_budget) for b in bullets[:4]]

    # ── Verdict ────────────────────────────────────────────────────────────────

    def _verdict(self, sc: object, action: str = "") -> str:
        """
        One crisp sentence. Pattern:
        "[Primary strength] supports [rating], but [key constraint] warrants [action/monitoring]."

        Target: ≤25 words.
        """
        v   = _cat_score(sc, "valuation")
        g   = _cat_score(sc, "growth")
        p   = _cat_score(sc, "profitability")
        fh  = _cat_score(sc, "financial_health")
        mom = _cat_score(sc, "momentum")
        rating = _rating_str(sc)

        # Primary strength
        strengths = [
            (lbl, score)
            for lbl, score in [
                ("strong margins", p),
                ("balance sheet resilience", fh),
                ("above-average growth", g),
                ("positive momentum", mom),
                ("attractive valuation", v),
            ]
            if score is not None and score >= 65
        ]
        # Primary concern
        concerns = [
            (lbl, score)
            for lbl, score in [
                ("stretched valuation", v),
                ("weak momentum", mom),
                ("limited growth", g),
                ("margin pressure", p),
            ]
            if score is not None and score < 45
        ]

        s_lbl = strengths[0][0] if strengths else "quality fundamentals"
        c_lbl = concerns[0][0] if concerns else "risk management"

        if action == "BUY":
            if strengths and concerns:
                return _clean(
                    f"{s_lbl.capitalize()} and attractive entry price support the Buy —"
                    f" {c_lbl} is the key risk; size accordingly."
                )
            elif strengths:
                return _clean(
                    f"{s_lbl.capitalize()} supports the Buy — price is below fair value"
                    f" with no dominant constraint on position size."
                )
            else:
                return _clean(
                    f"Price below fair value supports the Buy — monitor {c_lbl}"
                    f" as the primary near-term risk."
                )
        elif action == "STAGED BUY":
            if concerns:
                return _clean(
                    f"Staged entry is warranted — {c_lbl} limits conviction;"
                    f" build the position gradually as the thesis confirms."
                )
            else:
                return _clean(
                    f"{s_lbl.capitalize()} supports a staged entry —"
                    f" accumulate gradually; price is in the entry zone."
                )
        elif action == "HOLD":
            if strengths:
                return _clean(
                    f"{s_lbl.capitalize()} supports holding the position —"
                    f" no immediate entry catalyst; maintain and monitor."
                )
            else:
                return _clean(
                    "Fundamentals support holding — no clear entry or exit signal at current price."
                )
        elif action == "SELL":
            if concerns:
                return _clean(
                    f"{c_lbl.capitalize()} is the dominant risk — the thesis is broken"
                    f" or price is materially above fair value; consider exiting."
                )
            else:
                return _clean(
                    "Thesis is broken or price is materially above fair value — consider exiting."
                )
        elif action == "WAIT" and strengths:
            return _clean(
                f"{s_lbl.capitalize()} supports the long-term thesis —"
                f" price at or above fair value; waiting for a better entry."
            )
        elif action == "WAIT":
            return _clean(
                "Long-term thesis is intact — price at or above fair value;"
                " waiting for a pullback before initiating."
            )
        elif strengths and concerns:
            s_lbl = strengths[0][0]
            c_lbl = concerns[0][0]
            return _clean(
                f"{s_lbl.capitalize()} supports the {rating} —"
                f" {c_lbl} is the binding constraint on position size."
            )
        elif strengths:
            s_lbl  = strengths[0][0]
            s2_lbl = strengths[1][0] if len(strengths) > 1 else "consistent execution"
            return _clean(
                f"{s_lbl.capitalize()} and {s2_lbl} support the {rating} —"
                f" conviction is driven by quality fundamentals and asymmetric upside."
            )
        elif concerns:
            c_lbl = concerns[0][0]
            return _clean(
                f"{c_lbl.capitalize()} is the dominant risk — the {rating} rating"
                f" reflects constrained risk/reward at current prices."
            )
        else:
            return _clean(
                f"Balanced fundamental profile supports the {rating} —"
                f" no single dominant driver; risk/reward is symmetric."
            )

    # ── Key Tension ────────────────────────────────────────────────────────────

    def _key_tension(self, sc: object) -> Optional[str]:
        """
        One sentence framing the core investment tradeoff.
        Returns None when no meaningful tension is detectable.
        """
        g   = _cat_score(sc, "growth")
        p   = _cat_score(sc, "profitability")
        v   = _cat_score(sc, "valuation")
        mom = _cat_score(sc, "momentum")
        fh  = _cat_score(sc, "financial_health")

        # Priority order — first match wins
        if v is not None and g is not None:
            if v < 40 and g >= 65:
                return "Premium valuation vs. strong growth profile — execution is the determining factor."
            if v >= 65 and g < 40:
                return "Attractive valuation vs. limited near-term growth runway — value trap risk exists."
            if v < 40 and g < 45:
                return "Stretched valuation vs. weak fundamental support — risk/reward is unfavourable."
        if g is not None and p is not None and g >= 65 and p < 45:
            return "High growth ambition vs. unproven margin delivery."
        if mom is not None and v is not None and mom >= 70 and v < 40:
            return "Price momentum vs. stretched valuation — momentum may fade before valuation re-rates."
        if p is not None and mom is not None and p >= 65 and mom < 45:
            return "Quality business vs. weak near-term price action — patience required."
        if fh is not None and g is not None and fh < 45 and g >= 60:
            return "Strong growth trajectory vs. balance sheet vulnerability — leverage risk is under-appreciated."
        if g is not None and mom is not None and g >= 60 and mom < 45:
            return "Solid fundamentals vs. weak near-term market conviction."
        return None

    # ── Pass 1: Category synthesis ─────────────────────────────────────────────

    def _build_synthesis(self, sc: object) -> dict:
        """
        Pass 1 of the Two-Pass Narrative (requirement #2).

        For each scored category, produces:
          score    — raw float from the scorecard (or None if data quality is 'missing')
          tier     — "very_strong" | "strong" | "moderate" | "weak" | "very_weak" | "missing"
          weight   — relative importance in narrative synthesis [0.0–1.0].
                     Higher-weight categories take narrative priority in Pass 2.
                     Weight is derived from score tier, so it reflects fundamental
                     quality rather than arbitrary ordering.
          evidence — one-sentence factual summary (traceability anchor).

        Pass 2 generation methods read from this dict only.  Every narrative
        claim must be traceable to a tier/weight/evidence triple here.

        Weight derivation rationale:
          very_strong (≥85) → 0.90  strongest signal, leads the thesis
          strong      (≥70) → 0.70  clear positive, secondary supporting role
          moderate    (≥65) → 0.50  marginal positive, appears only if needed
          weak        (≥50) → 0.30  below threshold — not used for thesis bullets
          very_weak   (<50) → 0.15  strong negative — drives risk bullets instead
          missing            → 0.00  no data
        """
        cats = ("profitability", "financial_health", "growth", "valuation", "momentum")
        synthesis: dict = {}
        for cat in cats:
            score = _cat_score(sc, cat)
            if score is None:
                synthesis[cat] = {
                    "score": None, "tier": "missing", "weight": 0.0, "evidence": "",
                }
                continue
            if score >= 85:   tier, weight = "very_strong", 0.90
            elif score >= 70: tier, weight = "strong",      0.70
            elif score >= 65: tier, weight = "moderate",    0.50
            elif score >= 50: tier, weight = "weak",        0.30
            else:             tier, weight = "very_weak",   0.15
            synthesis[cat] = {
                "score":    score,
                "tier":     tier,
                "weight":   weight,
                "evidence": _synthesis_evidence(cat, tier),
            }
        return synthesis


# ── Category thesis bullet derivation ─────────────────────────────────────────
#
# Each rule: (category_name, min_score_to_trigger, implication_map)
# implication_map: {score_tier: implication_clause}
# score_tier boundaries: "very_strong" = ≥85, "strong" = ≥70, "moderate" = ≥65

_THESIS_CATEGORY_RULES: list[tuple] = [
    (
        "profitability", 65,
        {
            "very_strong": "scalable business model with sector-leading margins and strong returns on invested capital",
            "strong":      "above-average margins reflect durable pricing power and support high-quality earnings compounding",
            "moderate":    "profitability above sector median reflects operating discipline and supports reinvestment at acceptable returns",
        },
    ),
    (
        "financial_health", 65,
        {
            "very_strong": "balance sheet strength supports sustained reinvestment, capital returns, and downside protection through the cycle",
            "strong":      "strong cash flow and balance sheet resilience support capital returns and limit downside exposure",
            "moderate":    "healthy financials provide flexibility to invest through the cycle and weather earnings volatility",
        },
    ),
    (
        "growth", 65,
        {
            "very_strong": "above-consensus growth trajectory creates a durable compounding case and sustained re-rating potential",
            "strong":      "solid growth profile supports multiple expansion and rewards patient capital if execution holds",
            "moderate":    "revenue growth and earnings power are sufficient to sustain the investment thesis",
        },
    ),
    (
        "momentum", 65,
        {
            "very_strong": "technically constructive setup — price action, trend, and breadth all confirm the fundamental view",
            "strong":      "positive price momentum reduces timing risk and provides a constructive near-term backdrop for entry",
            "moderate":    "constructive technical setup presents no meaningful price-based headwind and supports the fundamental thesis",
        },
    ),
    (
        "valuation", 65,
        {
            "very_strong": "undemanding valuation suggests the market is pricing in more risk than the fundamentals support",
            "strong":      "current valuation provides a meaningful margin of safety and improves the asymmetry of risk/reward",
            "moderate":    "reasonable valuation supports the risk/reward case and lowers the bar for fundamental delivery",
        },
    ),
]


def _synthesis_evidence(cat: str, tier: str) -> str:
    """
    One-sentence factual summary for a category at a given tier.

    Used as the traceability anchor in Pass 1 synthesis output.  Every Pass 2
    narrative claim about a category must be consistent with this evidence string.
    Kept deliberately terse — this is a structured data field, not prose.
    """
    _EVIDENCE: dict[str, dict[str, str]] = {
        "profitability": {
            "very_strong": "Gross and operating margins are sector-leading with strong ROIC.",
            "strong":      "Margins are above sector median, supporting high-quality earnings.",
            "moderate":    "Profitability is at or slightly above sector levels.",
            "weak":        "Margins are below sector benchmarks, limiting reinvestment capacity.",
            "very_weak":   "Margins are materially below peers, signalling structural challenges.",
        },
        "financial_health": {
            "very_strong": "Balance sheet is fortress-grade with ample liquidity and low leverage.",
            "strong":      "Leverage is manageable and cash generation supports capital returns.",
            "moderate":    "Financial health is adequate; no near-term stress signals.",
            "weak":        "Leverage is elevated or liquidity is thin, limiting flexibility.",
            "very_weak":   "Balance sheet stress is material; debt service capacity is constrained.",
        },
        "growth": {
            "very_strong": "Revenue and earnings growth significantly exceed sector consensus.",
            "strong":      "Growth is solid and above sector median on key metrics.",
            "moderate":    "Growth is positive but broadly in line with sector.",
            "weak":        "Growth is decelerating or trailing sector peers.",
            "very_weak":   "Revenue and earnings are contracting or deeply below peers.",
        },
        "valuation": {
            "very_strong": "Multiples are well below sector medians on all primary metrics.",
            "strong":      "Valuation is below sector average and offers a margin of safety.",
            "moderate":    "Valuation is broadly in line with sector; no significant premium.",
            "weak":        "Multiples are above sector average, limiting downside protection.",
            "very_weak":   "Valuation is materially stretched; risk/reward is unfavourable.",
        },
        "momentum": {
            "very_strong": "Price action, trend, and breadth are all constructive and aligned.",
            "strong":      "Positive price momentum and upward trend are intact.",
            "moderate":    "Price action is mixed but not presenting a meaningful headwind.",
            "weak":        "Price momentum is negative; market is pricing in fundamental risk.",
            "very_weak":   "Price trend is sharply negative with deteriorating breadth.",
        },
    }
    return _EVIDENCE.get(cat, {}).get(tier, "")


def _derive_category_bullet(
    score:           float,
    cat_obj:         object,
    implication_map: dict[str, str],
) -> Optional[str]:
    """
    Synthesise a thesis bullet from a category score and its implications map.
    Returns the tier-matched narrative implication as a complete sentence.
    Never surfaces raw factor metrics — insights only, no data dumps.
    """
    # Tier
    if score >= 85:   tier = "very_strong"
    elif score >= 70: tier = "strong"
    else:             tier = "moderate"

    implication = implication_map.get(tier, "")

    # Return the narrative implication directly — avoid metric-fragment bullets
    # (e.g. "48.5% gross margin — …") which read as data dumps, not investment insights.
    return _clean(f"{implication.capitalize()}.") if implication else None


# ── Text processing helpers ────────────────────────────────────────────────────

def _cat_score(sc: object, attr: str) -> Optional[float]:
    cat = getattr(sc, attr, None)
    if cat is None:
        return None
    if getattr(cat, "data_quality", "good") == "missing":
        return None
    return getattr(cat, "score", None)


def _avg(*args: Optional[float]) -> Optional[float]:
    vals = [a for a in args if a is not None]
    return sum(vals) / len(vals) if vals else None


def _stance_str(sc: object) -> str:
    stance = getattr(sc, "stance", None)
    if stance is None:
        return "Neutral"
    v = getattr(stance, "value", str(stance))
    return v   # "Bullish" / "Neutral" / "Bearish"


def _rating_str(sc: object) -> str:
    score  = getattr(sc, "overall_score", 50)
    stance = _stance_str(sc)
    if stance == "Bullish" and score >= 80:
        return "Strong Buy"
    if stance == "Bullish":
        return "Buy"
    if stance == "Neutral":
        return "Hold"
    return "Sell"



def _compress_to_n_words(text: str, n: int) -> str:
    """
    Truncate to ≤n words, always ending on a complete sentence or clause boundary.

    Priority:
    1. If text fits within n words — return as-is.
    2. Prefer the last sentence-ending punctuation (. ! ?) within the truncated
       window, provided it falls in the second half (avoids cutting after 2 words).
    3. Fall back to the last clause boundary (, and but which that).
    4. Last resort — return the n-word truncation with any trailing dangling
       conjunctions/prepositions stripped (caller appends terminal punctuation).
    """
    words = text.split()
    if len(words) <= n:
        return text
    truncated = " ".join(words[:n])
    midpoint  = len(truncated) // 2

    # 1. Prefer sentence-ending punctuation in the second half of the window
    for punct in (".", "!", "?"):
        idx = truncated.rfind(punct)
        if idx > midpoint:
            return truncated[:idx + 1]

    # 1b. Em-dash / semicolon — natural prose break, higher priority than comma.
    # These are the primary structural separators in institutional sentence templates.
    for em in (" — ", " – ", "; "):
        idx = truncated.rfind(em)
        if idx > midpoint:
            candidate = truncated[:idx]
            cand_words = candidate.split()
            while cand_words and cand_words[-1].lower().rstrip(".,;:") in _DANGLING_WORDS:
                cand_words.pop()
            if cand_words:
                return " ".join(cand_words)

    # 2. Clause boundary (does not add punctuation — caller handles it)
    for stop in (", ", " and ", " but ", " which ", " that "):
        idx = truncated.rfind(stop)
        if idx > 0:
            candidate = truncated[:idx]
            # Strip trailing dangling words from the candidate too
            cand_words = candidate.split()
            while cand_words and cand_words[-1].lower().rstrip(".,;:") in _DANGLING_WORDS:
                cand_words.pop()
            if cand_words:
                return " ".join(cand_words)

    # 3. No boundary found — trim trailing dangling words before returning
    result_words = list(words[:n])
    while result_words and result_words[-1].lower().rstrip(".,;:!?") in _DANGLING_WORDS:
        result_words.pop()
    # Guard: always return at least half the requested budget
    if len(result_words) < max(1, n // 2):
        result_words = list(words[:max(1, n // 2)])
    return " ".join(result_words)


def _make_bullet(text: str, budget: int = 22) -> str:
    """
    Convert a raw text string to a clean, ≤budget-word memo bullet.

    Steps:
    1. Strip label prefixes ("Valuation: ", "Risk: ", etc.)
    2. Extract the first sentence if multi-sentence.
    3. Apply language quality rules.
    4. Enforce ≤budget word cap.
    5. Capitalise first word; ensure ends with period.
    """
    # Strip category label prefix — handles both plain ("Profitability: ") and
    # hyphenated ("High-quality business: ") formats.
    text = re.sub(r"^[A-Z][\w][\w\s-]*:\s*", "", text).strip()
    # Take first sentence
    first = re.split(r"(?<=[.!?])\s+", text)[0]
    # Language rules
    cleaned = _clean(first)
    # Word cap
    capped  = _compress_to_n_words(cleaned, budget)
    # Capitalise
    if capped:
        capped = capped[0].upper() + capped[1:]
    # End with period
    if capped and not capped.endswith((".", "!", "?")):
        capped += "."
    return capped


def _ensure_terminal_punct(text: str) -> str:
    """
    Guarantee text ends with terminal punctuation (. ! ?).

    Applied to individual sentences after `_compress_to_n_words` in contexts
    where `_make_bullet` is not called (e.g. `_top_takeaway` sentence joining).
    Also strips any trailing dangling conjunction/preposition before adding
    the period so we never produce "...strong growth and." or similar.
    """
    t = text.strip()
    if not t:
        return t
    # Strip trailing dangling words before adding period
    words = t.rstrip(".!?").split()
    while words and words[-1].lower().rstrip(".,;:") in _DANGLING_WORDS:
        words.pop()
    t = " ".join(words) if words else t.rstrip(".!?")
    if not t:
        return text.strip()
    if not t.endswith((".", "!", "?")):
        t += "."
    return t


def _clean(text: str) -> str:
    """Apply all language quality rules to a string."""
    for pattern, replacement in _FILLER_SUBS:
        text = pattern.sub(replacement, text)
    # Collapse double spaces
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def _reframe_trigger(
    trigger:     str,
    val_score:   Optional[float],
    mom_score:   Optional[float],
    growth_score: Optional[float],
) -> str:
    """
    Ensure a change-view trigger is forward-looking.
    Detects conditions already present and reframes with failure/continuation language.
    """
    t = trigger

    # Momentum already broken
    if mom_score is not None and mom_score < 45:
        t = re.sub(r"(?i)\bbreakdown?\s+below\b", "failure to reclaim", t)
        t = re.sub(r"(?i)\bbreaks?\s+below\b",    "failure to reclaim", t)
        t = re.sub(
            r"(?i)\bbreakdown?\b(?!\s+reclaim)",
            "failure to reclaim key support",
            t,
        )

    # Valuation already expensive
    if val_score is not None and val_score < 40:
        t = re.sub(r"(?i)multiple expansion",     "further multiple compression", t)
        t = re.sub(r"(?i)valuation expansion",    "further valuation deterioration", t)

    # Growth already decelerating
    if growth_score is not None and growth_score < 40:
        if re.search(r"growth decelerat|revenue declin", t, re.I):
            if not re.search(r"\bacceler|\bfurther\b|\bcontinued\b", t, re.I):
                t = "Continued " + t[0].lower() + t[1:]

    return t
