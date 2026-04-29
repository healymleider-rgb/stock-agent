"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import { evaluate } from "@/lib/api";
import type { EvaluateResponse, CategoryScore, MCSim, PeerRow as PeerRowType, TrendResult, HistoricalYear, ValuationRange, MacroData, ValidationLog } from "@/lib/types";
import {
  cn,
  formatLargeNumber,
  formatPrice,
  formatRatio,
  formatMultiple,
  stanceColor,
  scoreColor,
  scoreLabel,
  categoryLabel,
  safeFixed,
} from "@/lib/utils";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

import {
  ArrowLeft,
  Download,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  CheckCircle2,
  XCircle,
  ArrowRight,
  Dot,
  ChevronDown,
  ChevronUp,
  Globe,
} from "lucide-react";

// ── Portfolio snapshot persistence ────────────────────────────────────────────
// Saved to localStorage so the portfolio builder can read all evaluated stocks.

const _PORTFOLIO_LS_KEY = "stockeval_portfolio_snapshots";

interface PortfolioSnapshotData {
  ticker:          string;
  company_name:    string;
  sector:          string;
  current_price:   number;
  beta:            number;
  evaluated_at:    string;
  stance:          string;
  overall_score:   number;
  mom_score:       number;
  risk_score:      number;
  val_score:       number;
  prof_score:      number;
  fin_score:       number;
  mc_available:    boolean;
  p5:  number; p25: number; p50: number; p75: number; p95: number;
  expected_return: number;
  rr_ratio:        number;
  prob_positive:   number;
}

function _savePortfolioSnapshot(ticker: string, ev: EvaluateResponse): void {
  if (typeof window === "undefined") return;
  const mc   = ev.valuation_range.mc;
  const si   = ev.stock_info;
  const sc   = ev.scorecard;
  const cats = sc.categories;
  const snap: PortfolioSnapshotData = {
    ticker,
    company_name:    si.company_name              ?? ticker,
    sector:          si.sector                    ?? "Unknown",
    current_price:   si.current_price             ?? 0,
    beta:            si.beta                      ?? 1.0,
    evaluated_at:    ev.evaluated_at,
    stance:          sc.stance,
    overall_score:   sc.overall_score,
    mom_score:       cats.momentum?.score         ?? 50,
    risk_score:      cats.risk?.score             ?? 50,
    val_score:       cats.valuation?.score        ?? 50,
    prof_score:      cats.profitability?.score    ?? 50,
    fin_score:       cats.financial_health?.score ?? 50,
    mc_available:    !!(mc && mc.p5_price > 0 && mc.p95_price > 0),
    p5:  mc?.p5_price     ?? 0,
    p25: mc?.p25_price    ?? 0,
    p50: mc?.median_price ?? 0,
    p75: mc?.p75_price    ?? 0,
    p95: mc?.p95_price    ?? 0,
    expected_return: mc?.mean_return     ?? 0,
    rr_ratio:        mc?.upside_downside ?? 1.0,
    prob_positive:   mc?.prob_positive   ?? 0.5,
  };
  try {
    const existing: Record<string, PortfolioSnapshotData> = JSON.parse(
      localStorage.getItem(_PORTFOLIO_LS_KEY) || "{}"
    );
    existing[ticker] = snap;
    localStorage.setItem(_PORTFOLIO_LS_KEY, JSON.stringify(existing));
  } catch { /* storage quota or parse error — fail silently */ }
}

// ── Narrative sanitizer ───────────────────────────────────────────────────────
// Removes contradictory phrases from Python-generated text so all sections
// align with the execution state (the single source of truth).
// Only replaces high-confidence patterns — never rewrites legitimate analysis.

function sanitizeNarrative(text: string, ds: DecisionSummary): string {
  if (!text) return text;

  const isHighConvictionBuy = ds.executionStatus === 'BUY NOW' && ds.conviction === 'High';
  const isBuyAction  = ds.executionStatus === 'BUY NOW' || ds.executionStatus === 'STAGED BUY';
  const isWait       = ds.executionStatus === 'WAIT';
  const isReduceNow  = ds.executionStatus === 'TRIM' || ds.executionStatus === 'EXIT';

  let t = text;

  // 1. "aggressive full entry" — replace regardless of state (removed from engine but may be in Python memo)
  t = t.replace(/\baggressive full entry\b/gi,
    isBuyAction ? 'staged entry' : isWait ? 'no entry until confirmation' : 'reduce exposure');

  if (!isHighConvictionBuy) {
    // 2. "high conviction buy" / "buy with high conviction"
    const highConvReplacement =
      isWait       ? 'positive long-term thesis — await entry confirmation' :
      isReduceNow  ? 'long-term thesis intact — current action is to reduce' :
      ds.executionStatus === 'STAGED BUY' && ds.conviction === 'Low'
                   ? 'positive long-term thesis; staged accumulation recommended given current risk factors'
                   : 'positive long-term thesis; recommend staged accumulation';

    t = t.replace(/\bhigh[- ]conviction buy(?:ing)?\b/gi,  highConvReplacement);
    t = t.replace(/\bbuy with high conviction\b/gi,         highConvReplacement);

    // 3. "build a full position now" / "build full position"
    const fullPosReplacement = isWait
      ? 'wait for confirmation before initiating a position'
      : 'initiate a partial position gradually';
    t = t.replace(/\bbuild(?:ing)?\s+(?:a\s+)?full\s+position\s+now\b/gi, fullPosReplacement);
    t = t.replace(/\bbuild(?:ing)?\s+(?:a\s+)?full\s+position\b/gi,       fullPosReplacement);

    // 4. "enter immediately" when execution is not BUY NOW
    if (ds.executionStatus !== 'BUY NOW') {
      t = t.replace(/\benter immediately\b/gi, isWait ? 'wait for entry confirmation' : 'initiate gradually');
    }

    // 5. "scale aggressively" / "aggressive accumulation"
    t = t.replace(/\bscale aggressively\b/gi,    'scale in gradually');
    t = t.replace(/\baggressive accumulation\b/gi, 'staged accumulation');
  }

  // 6. For WAIT: replace action-oriented entry prose with hold-and-wait language
  if (isWait) {
    t = t.replace(/\benter now\b/gi, 'wait for entry confirmation');

    // "Attractive entry — a starter position at current is justified..." contradicts WAIT
    t = t.replace(
      /Attractive entry\s*—[^.]+\./i,
      `Hold at 0%. Entry trigger: ${ds.buyTrigger} Starter: ${ds.starterPct ?? 'position on trigger'}.`
    );

    // "Current price is the ideal entry; initiate..." contradicts WAIT (strong buy zone)
    t = t.replace(
      /Current price is the ideal entry;[^.]+\./i,
      `Hold at 0%. Entry trigger: ${ds.buyTrigger}`
    );
  }

  return t;
}

// ── Validation Log card ───────────────────────────────────────────────────────

function ValidationLogCard({ vlog }: { vlog: ValidationLog }) {
  const allPassed   = vlog.blocks.every(b => b.passed);
  const corrections = vlog.blocks.flatMap(b => b.corrections);

  const headerBg = allPassed
    ? (vlog.qualified ? '#eff6ff' : '#f0fdf4')
    : '#fef2f2';
  const headerBorder = allPassed
    ? (vlog.qualified ? '#bfdbfe' : '#bbf7d0')
    : '#fecaca';
  const headerDot = allPassed
    ? (vlog.qualified ? '#2563eb' : '#16a34a')
    : '#dc2626';
  const statusLabel = vlog.status;

  return (
    <Card className="border-slate-200 shadow-sm animate-fade-in no-print">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-base font-semibold text-slate-800">
            Pre-Report Validation Log
          </CardTitle>
          <span
            className="text-xs font-bold px-2.5 py-1 rounded-full"
            style={{
              backgroundColor: headerBg,
              border: `1px solid ${headerBorder}`,
              color: headerDot,
            }}
          >
            {statusLabel}
          </span>
        </div>
        <p className="text-xs text-slate-400 mt-1">
          7-block pre-render audit · runs before every report
        </p>
      </CardHeader>
      <CardContent className="px-6 pb-6 space-y-4">
        {/* Block grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {vlog.blocks.map(b => (
            <div
              key={b.block_id}
              className="flex items-start gap-2 p-2.5 rounded-lg border"
              style={{
                backgroundColor: b.passed ? '#f8fafc' : '#fef2f2',
                borderColor: b.passed ? '#e2e8f0' : '#fecaca',
              }}
            >
              <span
                className="mt-0.5 w-4 h-4 rounded-full shrink-0 flex items-center justify-center text-[9px] font-bold"
                style={{
                  backgroundColor: b.passed ? '#16a34a' : '#dc2626',
                  color: '#fff',
                }}
              >
                {b.passed ? '✓' : '✗'}
              </span>
              <div className="min-w-0">
                <p className="text-xs font-semibold text-slate-700 leading-tight">
                  Block {b.block_id} — {b.name}
                </p>
                {!b.passed && b.failures.map((f, i) => (
                  <p key={i} className="text-[11px] text-red-600 leading-snug mt-0.5">{f}</p>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Corrections */}
        {corrections.length > 0 && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-3">
            <p className="text-xs font-semibold text-amber-700 mb-2 uppercase tracking-wider">
              Corrections Applied
            </p>
            <div className="space-y-1">
              {corrections.map((c, i) => (
                <p key={i} className="text-[11px] text-amber-800 leading-snug">
                  <span className="font-mono">{c.field}</span>
                  {': '}
                  <span className="line-through text-amber-500">{c.old}</span>
                  {' → '}
                  <span className="font-semibold">{c.new}</span>
                  <span className="text-amber-600"> ({c.reason})</span>
                </p>
              ))}
            </div>
          </div>
        )}

        {/* Overrides */}
        {vlog.overrides.length > 0 && (
          <div className="rounded-lg border border-blue-200 bg-blue-50 p-3">
            <p className="text-xs font-semibold text-blue-700 mb-2 uppercase tracking-wider">
              Active Overrides
            </p>
            {vlog.overrides.map((o, i) => (
              <p key={i} className="text-[11px] text-blue-800 leading-snug">
                Block {o.block}: <span className="font-mono">{o.reason_code}</span>
                {' — '}{o.justification}
              </p>
            ))}
          </div>
        )}

        {/* Raw log (collapsed) */}
        <details className="text-[10px] text-slate-400">
          <summary className="cursor-pointer select-none hover:text-slate-600 font-medium">
            View raw validation log
          </summary>
          <pre className="mt-2 p-2 bg-slate-50 rounded border border-slate-200 whitespace-pre-wrap font-mono text-[9px] text-slate-500 leading-relaxed">
            {vlog.text}
          </pre>
        </details>
      </CardContent>
    </Card>
  );
}

// ── Score ring component ──────────────────────────────────────────────────────

function ScoreRing({ score }: { score: number }) {
  const pct = Math.max(0, Math.min(100, score));
  const colors = scoreColor(pct);
  const conic = `conic-gradient(from -90deg, ${colors.ring} ${pct}%, #e2e8f0 ${pct}% 100%)`;

  return (
    <div
      className="score-ring w-36 h-36 mx-auto"
      style={{ background: conic }}
    >
      <div className="w-28 h-28 rounded-full bg-white flex flex-col items-center justify-center m-4 shadow-sm">
        <span className={cn("text-3xl font-bold tabular-nums", colors.text)}>
          {Math.round(pct)}
        </span>
        <span className="text-xs text-slate-400 font-medium">/ 100</span>
      </div>
    </div>
  );
}

// ── Category bars ─────────────────────────────────────────────────────────────

/** PDF-safe hex color for a 0-100 score bar fill */
function barHex(score: number): string {
  return score >= 65 ? '#16a34a' : score >= 45 ? '#d97706' : '#dc2626';
}

function CategoryBar({ cat, name }: { cat: CategoryScore | null; name: string }) {
  if (!cat) return null;
  // Momentum gets its own dedicated bar with stronger PDF guarantees
  if (name === 'momentum') return <MomentumBar cat={cat} />;
  const colors = scoreColor(cat.score);
  const hex    = barHex(cat.score);
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-700">{categoryLabel(name)}</span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">{scoreLabel(cat.score)}</span>
          <span className={cn("text-sm font-bold tabular-nums", colors.text)}>
            {Math.round(cat.score)}
          </span>
        </div>
      </div>
      {/* Track + fill — fully inline-styled for PDF safety */}
      <div style={{ height: '8px', borderRadius: '9999px', overflow: 'hidden', backgroundColor: '#f1f5f9', WebkitPrintColorAdjust: 'exact' }}>
        <div style={{ height: '100%', width: `${cat.score}%`, borderRadius: '9999px', backgroundColor: hex, WebkitPrintColorAdjust: 'exact' }} />
      </div>
      {cat.reasoning && (
        <p className="text-xs text-slate-400 leading-relaxed">{cat.reasoning}</p>
      )}
    </div>
  );
}

// ── Dedicated Momentum bar — rich descriptor + PDF-safe inline styles ─────────

function MomentumBar({ cat }: { cat: CategoryScore }) {
  const score      = Math.round(Math.max(0, Math.min(100, cat.score)));
  const isStrong   = score >= 70;
  const isNeutral  = score >= 40;
  const barColor   = isStrong ? '#16a34a' : isNeutral ? '#eab308' : '#dc2626';
  const pillBg     = isStrong ? '#dcfce7' : isNeutral ? '#fef9c3' : '#fee2e2';
  const textColor  = isStrong ? '#15803d' : isNeutral ? '#a16207' : '#b91c1c';
  const descriptor = isStrong ? 'Strong'  : isNeutral ? 'Neutral' : 'Weak';

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-700">Momentum</span>
        <div className="flex items-center gap-2">
          <span
            className="text-xs font-semibold px-1.5 py-0.5 rounded"
            style={{ backgroundColor: pillBg, color: textColor }}
          >
            {descriptor}
          </span>
          <span className="text-sm font-bold tabular-nums" style={{ color: textColor }}>
            {score}/100
          </span>
        </div>
      </div>
      {/* Fully inline-styled — no Tailwind dependency in PDF path */}
      <div style={{
        height: '8px', backgroundColor: '#f1f5f9',
        borderRadius: '9999px', overflow: 'hidden',
        WebkitPrintColorAdjust: 'exact',
      }}>
        <div style={{
          height: '100%', width: `${score}%`,
          backgroundColor: barColor, borderRadius: '9999px',
          WebkitPrintColorAdjust: 'exact',
        }} />
      </div>
      {cat.reasoning && (
        <p className="text-xs text-slate-400 leading-relaxed">{cat.reasoning}</p>
      )}
    </div>
  );
}

// ── Loading skeleton ──────────────────────────────────────────────────────────

function ReportSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <Skeleton className="h-14 w-full rounded-none" />
      <div className="max-w-7xl mx-auto px-6 space-y-6">
        <Skeleton className="h-48 w-full rounded-xl" />
        <div className="grid grid-cols-3 gap-6">
          <Skeleton className="h-64 rounded-xl" />
          <Skeleton className="col-span-2 h-64 rounded-xl" />
        </div>
        <Skeleton className="h-96 w-full rounded-xl" />
      </div>
    </div>
  );
}

// ── Main report page ──────────────────────────────────────────────────────────

export default function ReportPage() {
  const params = useParams();
  const router = useRouter();
  const ticker = (params?.ticker as string)?.toUpperCase() ?? "";

  const [data, setData] = useState<EvaluateResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [progress, setProgress] = useState(0);
  const [step, setStep] = useState("Starting evaluation…");
  const [error, setError] = useState<string | null>(null);
  const [showMemo, setShowMemo] = useState(false);

  const load = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await evaluate(ticker, (s, p) => {
        setStep(s);
        setProgress(p);
      });
      setData(result);
      _savePortfolioSnapshot(ticker, result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Evaluation failed.");
    } finally {
      setIsLoading(false);
    }
  }, [ticker]);

  useEffect(() => {
    if (ticker) load();
  }, [ticker, load]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-white">
        <div className="h-14 bg-white border-b border-slate-200 flex items-center px-6 gap-4">
          <button onClick={() => router.push("/")} className="text-slate-400 hover:text-slate-700 transition-colors">
            <ArrowLeft className="w-4 h-4" />
          </button>
          <span className="font-mono font-semibold text-slate-900">{ticker}</span>
          <span className="text-slate-400 text-sm">{step}</span>
        </div>
        <div className="max-w-2xl mx-auto pt-16 px-6 space-y-4">
          <div className="text-center space-y-3">
            <p className="text-slate-500 text-sm">{step}</p>
            <Progress value={progress} className="h-1.5" />
            <p className="text-xs text-slate-400">{progress}% complete</p>
          </div>
        </div>
        <div className="max-w-7xl mx-auto px-6 mt-10">
          <ReportSkeleton />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-white flex flex-col items-center justify-center px-6 gap-6">
        <Alert variant="destructive" className="max-w-xl">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Evaluation Failed</AlertTitle>
          <AlertDescription className="text-xs break-all mt-1">{error}</AlertDescription>
        </Alert>
        <div className="flex gap-3">
          <button
            onClick={load}
            className="px-5 py-2.5 bg-slate-900 text-white rounded-lg text-sm font-semibold hover:bg-slate-700 transition-colors"
          >
            Retry
          </button>
          <button
            onClick={() => router.push("/")}
            className="px-5 py-2.5 bg-slate-100 text-slate-700 rounded-lg text-sm font-semibold hover:bg-slate-200 transition-colors"
          >
            Go back
          </button>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const { scorecard, stock_info, valuation_range, macro, memo } = data;
  const trends: TrendResult | null = data.trends ?? null;

  // ── Data staleness flags ─────────────────────────────────────────────────────
  // Price: flag if quote date is > 5 days before evaluation (delayed / stale feed)
  // Fundamentals: flag if most recent statement is > 400 days old (>1 annual cycle)
  const _evalMs = new Date(data.evaluated_at).getTime();
  const isStalePrice = stock_info.price_date != null
    ? (_evalMs - new Date(stock_info.price_date).getTime()) / 86_400_000 > 5
    : false;
  const isStaleFundamentals = stock_info.fundamentals_date != null
    ? (_evalMs - new Date(stock_info.fundamentals_date).getTime()) / 86_400_000 > 400
    : false;
  // Sanitize peer rows once — converts impossible 0.0 sentinels to null
  const peer_comparison = {
    ...data.peer_comparison,
    rows: data.peer_comparison.rows.map(sanitizePeerRow),
  };
  const stance = scorecard.stance;
  const sc = stanceColor(stance);
  const overallColors = scoreColor(scorecard.overall_score);
  const categories = scorecard.categories;
  // Pre-computed scores used by entry strategy + momentum bar (component-level scope)
  const momScore  = categories.momentum?.score  ?? 50;
  const riskScore = categories.risk?.score      ?? 50;

  // Macro-adjusted MC distribution — applies regime × sector sensitivity to P5–P95
  const macroAdj: MacroAdjustment | null =
    valuation_range.mc && macro.available && stock_info.current_price != null
      ? computeMacroAdjustment(
          valuation_range.mc,
          macro,
          stock_info.sector,
          stock_info.beta ?? 1.0,
          stock_info.current_price,
        )
      : null;
  const displayMC: MCSim | null | undefined = macroAdj
    ? buildAdjustedMC(valuation_range.mc!, macroAdj, stock_info.current_price ?? 0)
    : valuation_range.mc;

  // Page-level decision summary — used by Final Verdict + Investment Memo sections
  // All sections inherit the same thesis/execution/conviction state from this single source.
  // Guard: requires valid valuation range; returns null when insufficient data.
  const pageLevelDS: DecisionSummary | null = (() => {
    if (
      !valuation_range.available ||
      !stock_info.current_price  ||
      valuation_range.bear_price == null ||
      valuation_range.base_price == null ||
      valuation_range.bull_price == null
    ) return null;
    const lv = computeEntryLevels(
      stock_info.current_price,
      valuation_range.bear_price,
      valuation_range.base_price,
      valuation_range.bull_price,
      momScore, riskScore,
      displayMC,
    );
    return computeDecisionSummary(lv, stock_info.current_price, momScore, riskScore, displayMC, stance);
  })();

  // Investor-facing stance labels
  const STANCE_LABEL: Record<string, string> = { Bullish: "Buy", Neutral: "Hold", Bearish: "Sell" };
  const STANCE_DESC: Record<string, string> = {
    Bullish: "Attractive risk/reward with upside catalysts",
    Neutral: "Balanced risk/reward, no strong catalyst",
    Bearish: "Downside risk outweighs upside",
  };
  const displayStance = STANCE_LABEL[stance] ?? stance;
  const isStagedOrLowConv =
    pageLevelDS?.executionStatus === 'STAGED BUY' || pageLevelDS?.conviction === 'Low';
  // Execution-aware stance description
  const stanceDesc = (isStagedOrLowConv && stance === 'Bullish')
    ? "Attractive long-term risk/reward; staged entry recommended"
    : (STANCE_DESC[stance] ?? "");
  // Headline label — clarifies long-term thesis vs current execution in scorecard/verdict contexts
  const headlineStance = (isStagedOrLowConv && displayStance === 'Buy')
    ? "Buy (long-term thesis)"
    : displayStance;

  // Extract sections from Python memo (── SECTION ── format)
  const _memoSect = (pat: string) =>
    memo.match(
      new RegExp(
        `${pat}[─\\s]*\\n+([\\s\\S]+?)(?=\\n[ \\t]*[A-Z][^\\n]*\\n[ \\t]*─{2,}|$)`,
        "i"
      )
    )?.[1]?.trim() ?? "";

  // Strip any residual header-like first line from verdictText (e.g. "Buy — 72/100 | …")
  // that duplicates the JSX-rendered scorecard header above it.
  // Also strip ASCII/Unicode dash-only separator lines (e.g. "  ----------" or "  ────────").
  const _cleanVerdict = (raw: string): string => {
    const lines = raw.split("\n");
    const filtered = lines.filter(
      (l) =>
        !/^\s*(Buy|Hold|Sell)\s*[—–|]/.test(l) &&
        !/^\s*[-─═]{4,}\s*$/.test(l)
    );
    return filtered.join("\n").trim();
  };

  // Word-boundary truncation — avoids cutting mid-word in callout strings.
  const _truncate = (s: string, max: number): string => {
    if (s.length <= max) return s;
    const cut = s.slice(0, max);
    const lastSpace = cut.lastIndexOf(" ");
    return (lastSpace > max * 0.6 ? cut.slice(0, lastSpace) : cut) + "…";
  };

  const topTakeaway = _memoSect("TOP TAKEAWAY").slice(0, 600);
  const verdictText  = _cleanVerdict(_memoSect("(?:FINAL )?VERDICT")).slice(0, 600);

  // Parse structured bullets from MemoEngine sections
  const _memoBullets = (sectionName: string): string[] => {
    const raw = _memoSect(sectionName);
    if (!raw) return [];
    return raw
      .split("\n")
      .map((l) => l.replace(/^\s*[•→\-]\s*/, "").trim())
      .filter(Boolean);
  };
  const investmentThesisBullets = _memoBullets("INVESTMENT THESIS");
  const keyRisksBullets         = _memoBullets("KEY RISKS");

  // ── System-text guard (used in callout and cover page) ──────────────────
  const _isSystemText = (s: string) =>
    /inconclusive|not yet available|not configured|not evaluated|data not|insufficient data/i.test(s);
  const keyRisk   = scorecard.bearish_factors.find((f) => !_isSystemText(f)) ?? "";
  const keyDriver = scorecard.key_drivers.find((d) => !_isSystemText(d))
                 ?? scorecard.bullish_factors.find((f) => !_isSystemText(f))
                 ?? "";

  // ── One-line investment callout (data-driven, PM-style) ──────────────────
  const callout = (() => {
    // Priority 1: first sentence of TOP TAKEAWAY from the memo (already analytic)
    // Reject if it contains system/data-absence language that should never reach the user.
    const _hasSysText = (s: string) =>
      /inconclusive|not yet available|not configured|not evaluated|data not|insufficient data|unavailable|defaulting/i.test(s);
    if (topTakeaway) {
      const firstSentence = topTakeaway.split(/\.\s+/)[0].trim();
      if (firstSentence.length >= 30 && firstSentence.length <= 220 && !_hasSysText(firstSentence)) {
        return firstSentence.endsWith(".") ? firstSentence : firstSentence + ".";
      }
    }
    // Priority 2: construct from category scores + key factors
    const score    = scorecard.overall_score;
    const valScore = categories.valuation?.score   ?? 50;
    const profScore= categories.profitability?.score?? 50;
    const momScore = categories.momentum?.score    ?? 50;

    // Quality descriptor
    const qualLabel = profScore >= 70 ? "high-quality business"
                    : profScore <= 40 ? "business under margin pressure"
                    : "solid business";
    // "Attractive" reserved for valScore ≥ 70 (genuine discount); 40–70 is "fair value"
    const valLabel  = valScore >= 70 ? "at an attractive valuation"
                    : valScore <= 40 ? "at a stretched valuation"
                    : "at fair value";
    const momLabel  = momScore >= 60 ? "improving momentum"
                    : momScore <= 40 ? "weak near-term momentum"
                    : "mixed momentum";

    const _cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

    if (score >= 70 && valScore >= 60) {
      const risk = keyRisk
        ? ` — watch: ${_truncate(keyRisk.replace(/^[A-Z][a-z]+:\s*/, ""), 65)}`
        : "";
      // Remove "Fundamentally strong" — qualLabel already conveys quality
      return `${_cap(qualLabel)} ${valLabel}${risk}.`;
    }
    if (score >= 70 && valScore < 50) {
      const dep = keyDriver
        ? _truncate(keyDriver.replace(/^[A-Z][a-z]+:\s*/, ""), 65).toLowerCase()
        : "continued execution";
      return `High-quality business at a premium multiple — upside contingent on ${dep}.`;
    }
    if (score >= 55 && momScore >= 60) {
      return `${_cap(qualLabel)} with ${momLabel} ${valLabel} — risk/reward is increasingly favourable.`;
    }
    if (score >= 50) {
      return `${_cap(qualLabel)} ${valLabel}; no strong near-term catalyst to shift the balance.`;
    }
    const headwind = keyRisk
      ? _truncate(keyRisk.replace(/^[A-Z][a-z]+:\s*/, ""), 65).toLowerCase()
      : "multiple headwinds";
    return `Risk/reward is unfavourable — ${headwind}; monitor for stabilisation before entry.`;
  })();

  // ── Sanitized narrative — execution-aligned versions of Python-generated text ──
  // sanitizeNarrative() removes phrases that contradict the current execution state.
  // Original text is preserved where execution allows the language.
  const sanitizedCallout      = pageLevelDS ? sanitizeNarrative(callout,      pageLevelDS) : callout;
  const sanitizedVerdictText  = pageLevelDS ? sanitizeNarrative(verdictText,  pageLevelDS) : verdictText;
  const sanitizedTopTakeaway  = pageLevelDS ? sanitizeNarrative(topTakeaway,  pageLevelDS) : topTakeaway;

  return (
    <div className="min-h-screen bg-slate-50">
      {/* ── Sticky top bar ─────────────────────────────────────── */}
      <div className="sticky top-0 z-50 bg-white/95 backdrop-blur border-b border-slate-200 no-print">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center gap-4">
          <button
            onClick={() => router.push("/")}
            className="text-slate-400 hover:text-slate-700 transition-colors shrink-0"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>

          <div className="flex items-center gap-2 min-w-0">
            <span className="font-mono font-bold text-slate-900 text-lg shrink-0">{ticker}</span>
            {stock_info.company_name && (
              <span className="text-sm text-slate-500 truncate hidden sm:block">
                — {stock_info.company_name}
              </span>
            )}
          </div>

          <div className="flex items-center gap-2 ml-auto shrink-0">
            {stock_info.sector && (
              <Badge variant="secondary" className="hidden sm:flex text-xs">
                {stock_info.sector}
              </Badge>
            )}
            {/* Thesis badge */}
            <Badge
              variant={stance === "Bullish" ? "bullish" : stance === "Bearish" ? "bearish" : "neutral"}
            >
              {displayStance}
            </Badge>
            {/* Execution badge — distinguishes current action from long-term thesis */}
            {pageLevelDS && (
              <>
                <ArrowRight className="w-3 h-3 text-slate-300 shrink-0 hidden sm:block" />
                <span
                  className="hidden sm:inline-flex text-xs font-bold px-2.5 py-0.5 rounded-full"
                  style={{ backgroundColor: pageLevelDS.executionBg, color: pageLevelDS.executionFg, WebkitPrintColorAdjust: 'exact' } as React.CSSProperties}
                >
                  {pageLevelDS.executionStatus}
                </span>
              </>
            )}
            <div
              className={cn(
                "flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-bold tabular-nums",
                overallColors.bg, overallColors.text
              )}
            >
              {Math.round(scorecard.overall_score)}/100
            </div>
            <button
              onClick={() => router.push("/portfolio")}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-colors no-print"
            >
              Portfolio
            </button>
            <button
              onClick={() => window.print()}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-600 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors no-print"
            >
              <Download className="w-3 h-3" />
              PDF
            </button>
          </div>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-6">

        {/* ── Section 1: Company header ───────────────────────── */}
        <Card className="border-slate-200 shadow-sm animate-fade-in no-print">
          <CardContent className="p-6">
            <div className="flex flex-col lg:flex-row gap-6">
              {/* Left: company info */}
              <div className="flex-1 space-y-3">
                <div>
                  <div className="flex items-start gap-3 flex-wrap">
                    <h1 className="text-2xl font-bold text-slate-900">
                      {stock_info.company_name || ticker}
                    </h1>
                    {data.stock_type_label && (
                      <Badge variant="secondary" className="mt-1">
                        {data.stock_type_label}
                      </Badge>
                    )}
                  </div>
                  <p className="text-sm text-slate-500 mt-1">
                    {[stock_info.industry, stock_info.sector].filter(Boolean).join(" · ")}
                  </p>
                  {data.stock_type_desc && (
                    <p className="text-xs text-slate-400 mt-1 italic">{data.stock_type_desc}</p>
                  )}
                </div>
                {/* Investment callout — long-term thesis */}
                <div className="mt-3 px-3 py-2.5 bg-slate-900 rounded-lg">
                  <p className="text-sm font-medium text-white leading-snug">{sanitizedCallout}</p>
                </div>
                {/* Execution action row — distinguishes current action from long-term thesis */}
                {pageLevelDS && (
                  <div className="flex items-center gap-2 flex-wrap mt-1.5">
                    <span className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">Action</span>
                    <span
                      className="text-xs font-bold px-2 py-0.5 rounded-full"
                      style={{ backgroundColor: pageLevelDS.thesisRatingBg, color: pageLevelDS.thesisRatingFg } as React.CSSProperties}
                    >
                      Thesis: {pageLevelDS.thesisRating}
                    </span>
                    <ArrowRight className="w-3 h-3 text-slate-300 shrink-0" />
                    <span
                      className="text-xs font-bold px-2 py-0.5 rounded-full"
                      style={{ backgroundColor: pageLevelDS.executionBg, color: pageLevelDS.executionFg } as React.CSSProperties}
                    >
                      {pageLevelDS.executionStatus}
                    </span>
                    <span className="text-[11px] text-slate-400">
                      · {pageLevelDS.conviction} conviction · {pageLevelDS.targetPct} target
                    </span>
                  </div>
                )}
                {stock_info.description && (
                  <p className="text-sm text-slate-600 leading-relaxed line-clamp-3 mt-2">
                    {stock_info.description}
                  </p>
                )}
                <p className="text-xs text-slate-400 mt-1">
                  Generated:{" "}
                  {new Date(data.evaluated_at).toLocaleDateString("en-US", {
                    month: "long",
                    day: "numeric",
                    year: "numeric",
                  })}{" "}
                  at{" "}
                  {new Date(data.evaluated_at).toLocaleTimeString("en-US", {
                    hour: "numeric",
                    minute: "2-digit",
                  })}
                </p>
              </div>

              <Separator orientation="vertical" className="hidden lg:block h-auto" />

              {/* Right: key stats */}
              <div className="lg:w-80 shrink-0">
                <div className="grid grid-cols-2 gap-3">
                  <StatCell label="Price" value={formatPrice(stock_info.current_price)} large />
                  <StatCell label="Market Cap" value={formatLargeNumber(stock_info.market_cap)} large />
                  <StatCell label={stock_info._sources?.pe_ratio?.includes('ttm') ? 'P/E (TTM)' : 'P/E'} value={formatMultiple(stock_info.pe_ratio)} />
                  <StatCell label="P/S" value={formatMultiple(stock_info.ps_ratio)} />
                  <StatCell label="EV/EBITDA" value={formatMultiple(stock_info.ev_ebitda)} />
                  <StatCell label="Beta" value={safeFixed(stock_info.beta, 2)} />
                  <StatCell label="Gross Margin" value={formatRatio(stock_info.gross_margin)} />
                  <StatCell label="Net Margin" value={formatRatio(stock_info.net_margin)} />
                </div>
                {(stock_info.price_date || stock_info.fundamentals_date) && (
                  <p className="text-xs text-slate-400 mt-2 flex flex-wrap items-center gap-x-1">
                    {stock_info.price_date && (
                      <>
                        <span>Price: {stock_info.price_date}</span>
                        {isStalePrice && (
                          <span className="px-1 py-0.5 bg-amber-50 text-amber-600 rounded text-[10px] font-medium">stale</span>
                        )}
                      </>
                    )}
                    {stock_info.price_date && stock_info.fundamentals_date && <span>·</span>}
                    {stock_info.fundamentals_date && (
                      <>
                        <span>Financials: {stock_info.fundamentals_date}</span>
                        {isStaleFundamentals && (
                          <span className="px-1 py-0.5 bg-amber-50 text-amber-600 rounded text-[10px] font-medium">stale</span>
                        )}
                      </>
                    )}
                    <span>· FMP</span>
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* ── Section 2: Score overview ───────────────────────── */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-fade-in no-print">
          {/* Score ring */}
          <Card className="border-slate-200 shadow-sm">
            <CardContent className="p-6 flex flex-col items-center gap-4">
              <ScoreRing score={scorecard.overall_score} />

              <div className="flex items-center gap-2 text-center">
                <div className={cn("w-2 h-2 rounded-full", sc.dot)} />
                <span className={cn("font-semibold text-lg", sc.text)}>{headlineStance}</span>
              </div>

              <p className="text-sm text-slate-500">
                {Math.round(scorecard.confidence * 100)}% confidence
              </p>

              {data.key_tension && (
                <p className="text-xs text-slate-400 italic text-center leading-relaxed border-t border-slate-100 pt-3 w-full">
                  {data.key_tension}
                </p>
              )}
            </CardContent>
          </Card>

          {/* Category bars */}
          <Card className="md:col-span-2 border-slate-200 shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-semibold text-slate-700">Category Breakdown</CardTitle>
            </CardHeader>
            <CardContent className="p-6 pt-0 space-y-4">
              {Object.entries(categories).map(([key, cat]) => (
                <CategoryBar key={key} cat={cat} name={key} />
              ))}
            </CardContent>
          </Card>
        </div>

        {/* ── Section 3: Tabbed analysis ──────────────────────── */}
        <div className="animate-fade-in no-print">
          <Tabs defaultValue="fundamentals">
            <TabsList className="bg-white border border-slate-200 p-1 no-print">
              <TabsTrigger value="fundamentals">Fundamentals</TabsTrigger>
              <TabsTrigger value="valuation">Valuation</TabsTrigger>
              <TabsTrigger value="peers">Peers</TabsTrigger>
              <TabsTrigger value="macro">Macro</TabsTrigger>
            </TabsList>

            {/* ── Fundamentals tab ─────────────────── */}
            <TabsContent value="fundamentals" className="mt-4 space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <FactorCard
                  title="Supporting Factors"
                  items={scorecard.bullish_factors}
                  variant="bullish"
                />
                <FactorCard
                  title="Risk Factors"
                  items={scorecard.bearish_factors.filter(f => !/inconclusive|not yet available|not configured|not evaluated|signal inconclusive/i.test(f))}
                  variant="bearish"
                />
              </div>

              {scorecard.risk_flags.length > 0 && (
                <Card className="border-amber-200 shadow-sm">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-semibold text-amber-700 flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4" />
                      Risk Flags
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-6 pb-5">
                    <ul className="space-y-2">
                      {scorecard.risk_flags.map((flag, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-amber-800">
                          <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0 text-amber-500" />
                          {flag}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {scorecard.what_would_change_view.length > 0 && (
                  <Card className="border-slate-200 shadow-sm">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-semibold text-slate-700">
                        What Would Change This View
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="px-6 pb-5">
                      <ul className="space-y-2">
                        {scorecard.what_would_change_view.map((item, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                            <ArrowRight className="w-3.5 h-3.5 mt-0.5 shrink-0 text-slate-400" />
                            {item}
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                )}

                {scorecard.key_drivers.length > 0 && (
                  <Card className="border-slate-200 shadow-sm">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-semibold text-slate-700">
                        Key Drivers
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="px-6 pb-5">
                      <ul className="space-y-2">
                        {scorecard.key_drivers.map((item, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                            <Dot className="w-4 h-4 mt-0.5 shrink-0 text-slate-400" />
                            {item}
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                )}
              </div>
              {/* Trend Summary */}
              <TrendSummaryCard trends={trends} />
            </TabsContent>

            {/* ── Valuation tab ────────────────────── */}
            <TabsContent value="valuation" className="mt-4 space-y-4">
              {/* Key valuation ratios */}
              <Card className="border-slate-200 shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-semibold text-slate-700">Current Valuation</CardTitle>
                </CardHeader>
                <CardContent className="px-6 pb-5">
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
                    <StatCell label={stock_info._sources?.pe_ratio?.includes('ttm') ? 'P/E (TTM)' : 'P/E'} value={formatMultiple(stock_info.pe_ratio)} large />
                    <StatCell label="P/S" value={formatMultiple(stock_info.ps_ratio)} large />
                    <StatCell label="EV/EBITDA" value={formatMultiple(stock_info.ev_ebitda)} large />
                    <StatCell
                      label="PEG"
                      value={formatMultiple(valuation_range?.peg_ratio)}
                      large
                    />
                  </div>
                  {valuation_range?.peg_interpretation && (
                    <p className="text-sm text-slate-600 border-t border-slate-100 pt-3">
                      {valuation_range.peg_interpretation}
                    </p>
                  )}
                  {valuation_range?.eps_growth_rate != null && (
                    <p className="text-xs text-slate-400 mt-1">
                      EPS Growth Rate: {(valuation_range.eps_growth_rate * 100).toFixed(1)}%
                    </p>
                  )}
                  {(stock_info.ttm_eps != null || stock_info.ttm_fcf != null) && (
                    <div className="flex gap-6 mt-3 pt-3 border-t border-slate-100">
                      {stock_info.ttm_eps != null && (
                        <div>
                          <p className="text-xs text-slate-400 font-medium">EPS (TTM)</p>
                          <p className="text-sm font-semibold tabular-nums text-slate-900">
                            ${stock_info.ttm_eps.toFixed(2)}
                          </p>
                        </div>
                      )}
                      {stock_info.ttm_fcf != null && (
                        <div>
                          <p className="text-xs text-slate-400 font-medium">FCF (TTM)</p>
                          <p className="text-sm font-semibold tabular-nums text-slate-900">
                            {formatLargeNumber(stock_info.ttm_fcf)}
                          </p>
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Valuation range table */}
              {valuation_range?.available && !displayMC && (
                <Card className="border-slate-200 shadow-sm overflow-hidden">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-semibold text-slate-700">
                      Scenario Analysis
                      <span className="ml-2 text-[11px] font-normal text-slate-400">
                        Supporting — primary valuation: Monte Carlo distribution
                      </span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-0">
                    <Table>
                      <TableHeader>
                        <TableRow className="bg-slate-50">
                          <TableHead className="pl-6 font-semibold text-slate-600">Metric</TableHead>
                          <TableHead className="text-center text-red-600 font-semibold">Bear</TableHead>
                          <TableHead className="text-center text-blue-600 font-semibold">Base</TableHead>
                          <TableHead className="text-center text-green-600 font-semibold">Bull</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        <ValuationRow
                          label={valuation_range.scenario_primary_method === 'P/E' ? 'P/E ★' : 'P/E'}
                          bear={valuation_range.pe_bear_mult}
                          base={valuation_range.pe_base_mult}
                          bull={valuation_range.pe_bull_mult}
                        />
                        {/* EPS sub-row — only for P/E since EPS varies per scenario */}
                        {valuation_range.scenario_bear_eps != null && (
                          <TableRow className="text-slate-400">
                            <TableCell className="pl-10 text-xs italic">EPS (fwd)</TableCell>
                            <TableCell className="text-center text-xs tabular-nums">{valuation_range.scenario_bear_eps != null ? `$${valuation_range.scenario_bear_eps.toFixed(2)}` : '—'}</TableCell>
                            <TableCell className="text-center text-xs tabular-nums">{valuation_range.scenario_base_eps != null ? `$${valuation_range.scenario_base_eps.toFixed(2)}` : '—'}</TableCell>
                            <TableCell className="text-center text-xs tabular-nums">{valuation_range.scenario_bull_eps != null ? `$${valuation_range.scenario_bull_eps.toFixed(2)}` : '—'}</TableCell>
                          </TableRow>
                        )}
                        <ValuationRow
                          label={valuation_range.scenario_primary_method === 'EV/EBITDA' ? 'EV/EBITDA ★' : 'EV/EBITDA'}
                          bear={valuation_range.ev_bear_mult}
                          base={valuation_range.ev_base_mult}
                          bull={valuation_range.ev_bull_mult}
                        />
                        <ValuationRow
                          label={valuation_range.scenario_primary_method === 'P/S' ? 'P/S ★' : 'P/S'}
                          bear={valuation_range.ps_bear_mult}
                          base={valuation_range.ps_base_mult}
                          bull={valuation_range.ps_bull_mult}
                        />
                        <TableRow className="bg-slate-50 font-semibold">
                          <TableCell className="pl-6 font-semibold text-slate-700">Target Price</TableCell>
                          <TableCell className="text-center text-red-600 font-bold">
                            {formatPrice(valuation_range.bear_price)}
                          </TableCell>
                          <TableCell className="text-center text-blue-600 font-bold">
                            {formatPrice(valuation_range.base_price)}
                          </TableCell>
                          <TableCell className="text-center text-green-600 font-bold">
                            {formatPrice(valuation_range.bull_price)}
                          </TableCell>
                        </TableRow>
                        {stock_info.current_price && (
                          <TableRow>
                            <TableCell className="pl-6 text-slate-500 text-xs">vs Current ({formatPrice(stock_info.current_price)})</TableCell>
                            <TableCell className="text-center text-xs">
                              {valuation_range.bear_price != null
                                ? formatUpside(valuation_range.bear_price, stock_info.current_price)
                                : "—"}
                            </TableCell>
                            <TableCell className="text-center text-xs">
                              {valuation_range.base_price != null
                                ? formatUpside(valuation_range.base_price, stock_info.current_price)
                                : "—"}
                            </TableCell>
                            <TableCell className="text-center text-xs">
                              {valuation_range.bull_price != null
                                ? formatUpside(valuation_range.bull_price, stock_info.current_price)
                                : "—"}
                            </TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                    {valuation_range.upside_context && (
                      <div className="px-6 py-3 border-t border-slate-100">
                        <p className="text-xs text-slate-500">{valuation_range.upside_context}</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* ── Distribution-Based Valuation (primary) ───────────── */}
              {valuation_range.mc && (
                <MCCard
                  mc={valuation_range.mc}
                  currentPrice={stock_info.current_price}
                />
              )}

              {/* ── Return Distribution Chart ─────────────────────────── */}
              {valuation_range.mc && (
                <ReturnDistributionChart mc={valuation_range.mc} />
              )}

              {/* ── Valuation Range Bar (MC-anchored when available) ───── */}
              <ValuationRangeBar
                bearPrice={valuation_range.bear_price}
                basePrice={valuation_range.base_price}
                bullPrice={valuation_range.bull_price}
                currentPrice={stock_info.current_price}
                mc={displayMC}
              />

              {/* ── Macro-Adjusted Valuation ──────────────────────────── */}
              {macroAdj && stock_info.current_price != null && (
                <MacroAdjustedValuationCard
                  adj={macroAdj}
                  currentPrice={stock_info.current_price}
                />
              )}

              {/* ── Entry Strategy & Price Levels ─────────────────────── */}
              <EntryStrategyCard
                vr={valuation_range}
                currentPrice={stock_info.current_price}
                momScore={momScore}
                riskScore={riskScore}
                mc={displayMC}
                stance={stance}
              />

              {/* ── Driver-based scenario model (reference) ──────────── */}
              <DriverModelCard vr={valuation_range} currentPrice={stock_info.current_price} />

            </TabsContent>

            {/* ── Peers tab ────────────────────────── */}
            <TabsContent value="peers" className="mt-4 space-y-4">
              {!peer_comparison.has_peers ? (
                <Card className="border-slate-200 shadow-sm">
                  <CardContent className="p-12 text-center">
                    <Globe className="w-8 h-8 text-slate-300 mx-auto mb-3" />
                    <p className="text-slate-500 text-sm">No peer comparison data available for this ticker.</p>
                  </CardContent>
                </Card>
              ) : (
                <div className="space-y-5">
                  {peer_comparison.peer_level === 3 && peer_comparison.proxy_note && (
                    <div className="flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
                      <span className="mt-0.5 shrink-0 font-semibold">Proxy Peers</span>
                      <span>{peer_comparison.proxy_note}</span>
                    </div>
                  )}
                  {/* 1. Valuation */}
                  <PeerSection
                    title="Valuation"
                    insight={peer_comparison.insights[0] ?? ""}
                    rows={peer_comparison.rows}
                    columns={[
                      { header: "P/E", key: "pe", format: formatMultiple, isValuation: true },
                      { header: "P/S", key: "ps", format: formatMultiple, isValuation: true },
                      { header: "EV/EBITDA", key: "ev_ebitda", format: formatMultiple, isValuation: true },
                      { header: "PEG", key: "peg", format: formatMultiple, isValuation: true },
                    ]}
                  />
                  {/* 2. Growth */}
                  <PeerSection
                    title="Growth"
                    insight={peer_comparison.insights[1] ?? ""}
                    rows={peer_comparison.rows}
                    columns={[
                      { header: "Rev Growth", key: "revenue_growth", format: fmtPctRaw, highIsGood: true },
                      { header: "EPS CAGR", key: "eps_growth", format: (v) => v != null ? fmtPctRaw(v * 100) : "—", highIsGood: true },
                      { header: "EBITDA Growth", key: "ebitda_growth", format: fmtPctRaw, highIsGood: true },
                    ]}
                  />
                  {/* 3. Profitability */}
                  <PeerSection
                    title="Profitability"
                    insight={peer_comparison.insights[2] ?? ""}
                    rows={peer_comparison.rows}
                    columns={[
                      { header: "Gross Margin", key: "gross_margin", format: fmtPct, highIsGood: true },
                      { header: "Op. Margin", key: "operating_margin", format: fmtPct, highIsGood: true },
                      { header: "Net Margin", key: "net_margin", format: fmtPct, highIsGood: true },
                      { header: "ROE", key: "roe", format: fmtPct, highIsGood: true },
                      { header: "ROIC", key: "roic", format: fmtPct, highIsGood: true },
                    ]}
                  />
                  {/* 4. Financial Health */}
                  <PeerSection
                    title="Financial Health"
                    insight={peer_comparison.insights[3] ?? ""}
                    rows={peer_comparison.rows}
                    columns={[
                      { header: "D/E", key: "debt_equity", format: (v) => v != null ? `${v.toFixed(2)}x` : "—", highIsGood: false },
                      { header: "Current Ratio", key: "current_ratio", format: (v) => v != null ? `${v.toFixed(2)}x` : "—", highIsGood: true },
                      { header: "Int. Coverage", key: "interest_coverage", format: (v) => v != null ? `${v.toFixed(1)}x` : "—", highIsGood: true },
                    ]}
                  />
                  {/* 5. Market / Risk */}
                  <PeerSection
                    title="Market & Risk"
                    insight={peer_comparison.insights[4] ?? ""}
                    rows={peer_comparison.rows}
                    columns={[
                      { header: "Beta", key: "beta", format: (v) => v != null ? v.toFixed(2) : "—", highIsGood: false },
                    ]}
                  />

                  {/* 6. 5-Year Historical */}
                  <PeerHistoricalTable rows={peer_comparison.rows} peerTrendInsights={peer_comparison.peer_trend_insights} />

                </div>
              )}
            </TabsContent>

            {/* ── Macro tab ────────────────────────── */}
            <TabsContent value="macro" className="mt-4 space-y-4">
              {!macro.available ? (
                <Card className="border-slate-200 shadow-sm">
                  <CardContent className="p-12 text-center">
                    <p className="text-slate-500 text-sm">Macro data not available.</p>
                  </CardContent>
                </Card>
              ) : (
                <>
                  <Card className="border-slate-200 shadow-sm">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-semibold text-slate-700">Macro Regime</CardTitle>
                    </CardHeader>
                    <CardContent className="px-6 pb-5 space-y-4">
                      {/* Macro header — labeled rows */}
                      <dl className="grid grid-cols-[140px_1fr] gap-x-2 gap-y-1.5 text-sm">
                        {macro.macro_regime && (
                          <>
                            <dt className="text-slate-500">Macro Regime</dt>
                            <dd className="font-semibold text-slate-800">{macro.macro_regime}</dd>
                          </>
                        )}
                        {macro.cycle_phase && macro.cycle_phase !== 'unknown' && (
                          <>
                            <dt className="text-slate-500">Cycle Phase</dt>
                            <dd className="capitalize text-slate-700">{macro.cycle_phase} Cycle</dd>
                          </>
                        )}
                        {macro.recession_risk_level && (
                          <>
                            <dt className="text-slate-500">Recession Risk</dt>
                            <dd className={cn(
                              "font-semibold",
                              macro.recession_risk_level.toLowerCase().includes("high") ? "text-red-700" :
                              macro.recession_risk_level.toLowerCase().includes("low") ? "text-green-700" :
                              "text-amber-700"
                            )}>{macro.recession_risk_level}</dd>
                          </>
                        )}
                      </dl>

                      {/* Score bar */}
                      {macro.macro_score != null && (
                        <div className="space-y-1.5">
                          <div className="flex justify-between text-xs text-slate-500">
                            <span>Macro Score</span>
                            <span className="font-semibold">{Math.round(macro.macro_score)}/100</span>
                          </div>
                          <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                            <div
                              className={cn(
                                "h-full rounded-full",
                                scoreColor(macro.macro_score).bar
                              )}
                              style={{ width: `${macro.macro_score}%` }}
                            />
                          </div>
                        </div>
                      )}

                      {/* LEI trend signals */}
                      {(macro.lei_trend || macro.yield_spread_trend) && (
                        <div className="flex flex-wrap gap-4 text-xs text-slate-500 border-t border-slate-100 pt-3">
                          {macro.lei_trend && (
                            <span>
                              <span className="font-semibold text-slate-600">CLI: </span>
                              {macro.lei_trend}
                            </span>
                          )}
                          {macro.yield_spread_trend && (
                            <span>
                              <span className="font-semibold text-slate-600">Curve: </span>
                              {macro.yield_spread_trend}
                            </span>
                          )}
                        </div>
                      )}

                      {/* LEI interpretation narrative */}
                      {macro.reasoning_summary && (
                        <p className="text-sm text-slate-600 border-t border-slate-100 pt-3 leading-relaxed">
                          {macro.reasoning_summary}
                        </p>
                      )}

                      {/* Sector tilt */}
                      {macro.sector_tilt && (
                        <p className="text-sm text-slate-600 border-t border-slate-100 pt-3">
                          <span className="font-semibold text-slate-700">Sector Tilt: </span>
                          {macro.sector_tilt}
                        </p>
                      )}
                      {macro.observation_dates && Object.keys(macro.observation_dates).length > 0 && (
                        <div className="border-t border-slate-100 pt-3">
                          <p className="text-xs text-slate-400 font-medium mb-1.5">Indicator Observation Dates</p>
                          <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-0.5 text-xs text-slate-500">
                            {Object.entries(macro.observation_dates).map(([indicator, date]) => (
                              <React.Fragment key={indicator}>
                                <dt>{indicator}</dt>
                                <dd className="tabular-nums text-right">{date}</dd>
                              </React.Fragment>
                            ))}
                          </dl>
                        </div>
                      )}
                    </CardContent>
                  </Card>

                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <FactorCard
                      title="Macro Tailwinds"
                      items={macro.bullish_macro_factors || []}
                      variant="bullish"
                    />
                    <FactorCard
                      title="Macro Headwinds"
                      items={macro.bearish_macro_factors || []}
                      variant="bearish"
                    />
                  </div>
                </>
              )}
            </TabsContent>
          </Tabs>
        </div>

        {/* ── Print-only: 8-page institutional report ─────────── */}
        <div className="only-print">

          {/* PAGE 1 — Cover */}
          <div className="print-page">
            <p className="text-xs text-slate-400 mb-4">
              Generated:{" "}
              {new Date(data.evaluated_at).toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" })}{" "}
              at{" "}
              {new Date(data.evaluated_at).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })}
            </p>
            <div className="flex items-start gap-8 mb-5">
              <div className="flex-1">
                <h1 className="text-3xl font-bold text-slate-900">{stock_info.company_name || ticker}</h1>
                <div className="flex items-center gap-2 mt-1 flex-wrap text-sm text-slate-500">
                  <span className="font-mono font-semibold">{ticker}</span>
                  {stock_info.sector && <><span>·</span><span>{stock_info.sector}</span></>}
                  {stock_info.industry && <><span>·</span><span>{stock_info.industry}</span></>}
                </div>
                {data.stock_type_label && (
                  <span className="inline-block mt-1.5 px-2 py-0.5 text-xs bg-slate-100 text-slate-600 rounded">
                    {data.stock_type_label}
                  </span>
                )}
                {/* Investment callout — long-term thesis */}
                <div className="mt-3 px-3 py-2 bg-slate-900 rounded">
                  <p className="text-sm font-medium text-white leading-snug">{sanitizedCallout}</p>
                </div>
                {/* Thesis / Execution row — print cover */}
                {pageLevelDS && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap', marginTop: '6px' }}>
                    <span style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Action</span>
                    <span style={{ fontSize: '10px', fontWeight: 700, padding: '1px 7px', borderRadius: '9999px', backgroundColor: pageLevelDS.thesisRatingBg, color: pageLevelDS.thesisRatingFg, WebkitPrintColorAdjust: 'exact' }}>
                      Thesis: {pageLevelDS.thesisRating}
                    </span>
                    <span style={{ fontSize: '9px', color: '#94a3b8' }}>→</span>
                    <span style={{ fontSize: '10px', fontWeight: 700, padding: '1px 7px', borderRadius: '9999px', backgroundColor: pageLevelDS.executionBg, color: pageLevelDS.executionFg, WebkitPrintColorAdjust: 'exact' }}>
                      {pageLevelDS.executionStatus}
                    </span>
                    <span style={{ fontSize: '9px', color: '#94a3b8' }}>
                      · {pageLevelDS.conviction} conviction · {pageLevelDS.targetPct} target
                    </span>
                  </div>
                )}
                {data.key_tension && (
                  <div className="mt-3 p-2.5 bg-slate-50 border-l-4 border-slate-300 rounded-r">
                    <p className="text-xs text-slate-600 italic">{data.key_tension}</p>
                  </div>
                )}
              </div>
              <div className="flex flex-col items-center gap-2 shrink-0">
                <ScoreRing score={scorecard.overall_score} />
                <p className={cn("text-2xl font-bold", sc.text)}>{headlineStance}</p>
                <p className="text-xs text-slate-400 text-center max-w-[160px]">{stanceDesc}</p>
                <p className="text-sm text-slate-500">{Math.round(scorecard.confidence * 100)}% confidence</p>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-4 mb-5">
              <StatCell label="Price" value={formatPrice(stock_info.current_price)} large />
              <StatCell label="Market Cap" value={formatLargeNumber(stock_info.market_cap)} large />
              <StatCell label="P/E" value={formatMultiple(stock_info.pe_ratio)} large />
              <StatCell label="P/S" value={formatMultiple(stock_info.ps_ratio)} large />
              <StatCell label="EV/EBITDA" value={formatMultiple(stock_info.ev_ebitda)} large />
              <StatCell label="Beta" value={safeFixed(stock_info.beta, 2)} large />
            </div>
            <div className="space-y-2 border-t border-slate-100 pt-4">
              {scorecard.bullish_factors[0] && (
                <div className="flex items-start gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-500 mt-0.5 shrink-0" />
                  <p className="text-sm text-slate-700"><span className="font-semibold text-slate-800 text-xs mr-1.5">Working</span>{scorecard.bullish_factors[0]}</p>
                </div>
              )}
              {keyRisk && (
                <div className="flex items-start gap-2">
                  <XCircle className="w-4 h-4 text-red-400 mt-0.5 shrink-0" />
                  <p className="text-sm text-slate-700"><span className="font-semibold text-slate-800 text-xs mr-1.5">Key Risk</span>{keyRisk}</p>
                </div>
              )}
              {scorecard.what_would_change_view[0] && (
                <div className="flex items-start gap-2">
                  <ArrowRight className="w-4 h-4 text-slate-400 mt-0.5 shrink-0" />
                  <p className="text-sm text-slate-700"><span className="font-semibold text-slate-800 text-xs mr-1.5">Watch Items</span>{scorecard.what_would_change_view[0]}</p>
                </div>
              )}
            </div>
          </div>

          {/* PAGE 2 — Category Breakdown */}
          <div className="print-page print-page-break">
            <h2 className="text-xl font-bold text-slate-900 mb-6 pb-2 border-b border-slate-200">Category Breakdown</h2>
            <div className="space-y-5">
              {Object.entries(categories).map(([key, cat]) => (
                <CategoryBar key={key} cat={cat} name={key} />
              ))}
            </div>
          </div>

          {/* PAGE 3 — Fundamental Analysis */}
          <div className="print-page print-page-break">
            <h2 className="text-xl font-bold text-slate-900 mb-6 pb-2 border-b border-slate-200">Fundamental Analysis</h2>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <FactorCard title="Supporting Factors" items={scorecard.bullish_factors} variant="bullish" />
                <FactorCard title="Risk Factors" items={scorecard.bearish_factors.filter(f => !/inconclusive|not yet available|not configured|not evaluated|signal inconclusive/i.test(f))} variant="bearish" />
              </div>
              {scorecard.risk_flags.length > 0 && (
                <Card className="border-amber-200 shadow-sm">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-semibold text-amber-700 flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4" /> Risk Flags
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="px-6 pb-5">
                    <ul className="space-y-2">
                      {scorecard.risk_flags.map((flag, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-amber-800">
                          <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0 text-amber-500" /> {flag}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              )}
              <div className="grid grid-cols-2 gap-4">
                {scorecard.what_would_change_view.length > 0 && (
                  <Card className="border-slate-200 shadow-sm">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-semibold text-slate-700">What Would Change This View</CardTitle>
                    </CardHeader>
                    <CardContent className="px-6 pb-5">
                      <ul className="space-y-2">
                        {scorecard.what_would_change_view.map((item, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                            <ArrowRight className="w-3.5 h-3.5 mt-0.5 shrink-0 text-slate-400" /> {item}
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                )}
                {scorecard.key_drivers.length > 0 && (
                  <Card className="border-slate-200 shadow-sm">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm font-semibold text-slate-700">Key Drivers</CardTitle>
                    </CardHeader>
                    <CardContent className="px-6 pb-5">
                      <ul className="space-y-2">
                        {scorecard.key_drivers.map((item, i) => (
                          <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                            <Dot className="w-4 h-4 mt-0.5 shrink-0 text-slate-400" /> {item}
                          </li>
                        ))}
                      </ul>
                    </CardContent>
                  </Card>
                )}
              </div>
              {/* Trend Summary */}
              <TrendSummaryCard trends={trends} />
            </div>
          </div>

          {/* PAGE 4 — Valuation */}
          <div className="print-page print-page-break">
            <h2 className="text-xl font-bold text-slate-900 mb-6 pb-2 border-b border-slate-200">Valuation</h2>
            <div className="space-y-4">
              <Card className="border-slate-200 shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-semibold text-slate-700">Current Valuation</CardTitle>
                </CardHeader>
                <CardContent className="px-6 pb-5">
                  <div className="grid grid-cols-4 gap-4 mb-4">
                    <StatCell label={stock_info._sources?.pe_ratio?.includes('ttm') ? 'P/E (TTM)' : 'P/E'} value={formatMultiple(stock_info.pe_ratio)} large />
                    <StatCell label="P/S" value={formatMultiple(stock_info.ps_ratio)} large />
                    <StatCell label="EV/EBITDA" value={formatMultiple(stock_info.ev_ebitda)} large />
                    <StatCell label="PEG" value={formatMultiple(valuation_range?.peg_ratio)} large />
                  </div>
                  {valuation_range?.peg_interpretation && (
                    <p className="text-sm text-slate-600 border-t border-slate-100 pt-3">{valuation_range.peg_interpretation}</p>
                  )}
                </CardContent>
              </Card>
              {valuation_range?.available && !displayMC && (
                <Card className="border-slate-200 shadow-sm overflow-hidden">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-semibold text-slate-700">
                      Scenario Analysis
                      {valuation_range.scenario_primary_method && (
                        <span className="ml-2 text-xs font-normal text-slate-400">(Primary driver: {valuation_range.scenario_primary_method} ★)</span>
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="p-0">
                    <Table>
                      <TableHeader>
                        <TableRow className="bg-slate-50">
                          <TableHead className="pl-6 font-semibold text-slate-600">Metric</TableHead>
                          <TableHead className="text-center text-red-600 font-semibold">Bear</TableHead>
                          <TableHead className="text-center text-blue-600 font-semibold">Base</TableHead>
                          <TableHead className="text-center text-green-600 font-semibold">Bull</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        <ValuationRow label={valuation_range.scenario_primary_method === 'P/E' ? 'P/E ★' : 'P/E'} bear={valuation_range.pe_bear_mult} base={valuation_range.pe_base_mult} bull={valuation_range.pe_bull_mult} />
                        {valuation_range.scenario_bear_eps != null && (
                          <TableRow className="text-slate-400">
                            <TableCell className="pl-10 text-xs italic">EPS (fwd)</TableCell>
                            <TableCell className="text-center text-xs tabular-nums">{valuation_range.scenario_bear_eps != null ? `$${valuation_range.scenario_bear_eps.toFixed(2)}` : '—'}</TableCell>
                            <TableCell className="text-center text-xs tabular-nums">{valuation_range.scenario_base_eps != null ? `$${valuation_range.scenario_base_eps.toFixed(2)}` : '—'}</TableCell>
                            <TableCell className="text-center text-xs tabular-nums">{valuation_range.scenario_bull_eps != null ? `$${valuation_range.scenario_bull_eps.toFixed(2)}` : '—'}</TableCell>
                          </TableRow>
                        )}
                        <ValuationRow label={valuation_range.scenario_primary_method === 'EV/EBITDA' ? 'EV/EBITDA ★' : 'EV/EBITDA'} bear={valuation_range.ev_bear_mult} base={valuation_range.ev_base_mult} bull={valuation_range.ev_bull_mult} />
                        <ValuationRow label={valuation_range.scenario_primary_method === 'P/S' ? 'P/S ★' : 'P/S'} bear={valuation_range.ps_bear_mult} base={valuation_range.ps_base_mult} bull={valuation_range.ps_bull_mult} />
                        <TableRow className="bg-slate-50 font-semibold">
                          <TableCell className="pl-6 font-semibold text-slate-700">Target Price</TableCell>
                          <TableCell className="text-center text-red-600 font-bold">{formatPrice(valuation_range.bear_price)}</TableCell>
                          <TableCell className="text-center text-blue-600 font-bold">{formatPrice(valuation_range.base_price)}</TableCell>
                          <TableCell className="text-center text-green-600 font-bold">{formatPrice(valuation_range.bull_price)}</TableCell>
                        </TableRow>
                        {stock_info.current_price && (
                          <TableRow>
                            <TableCell className="pl-6 text-slate-500 text-xs">vs Current ({formatPrice(stock_info.current_price)})</TableCell>
                            <TableCell className="text-center text-xs">{valuation_range.bear_price != null ? formatUpside(valuation_range.bear_price, stock_info.current_price) : "—"}</TableCell>
                            <TableCell className="text-center text-xs">{valuation_range.base_price != null ? formatUpside(valuation_range.base_price, stock_info.current_price) : "—"}</TableCell>
                            <TableCell className="text-center text-xs">{valuation_range.bull_price != null ? formatUpside(valuation_range.bull_price, stock_info.current_price) : "—"}</TableCell>
                          </TableRow>
                        )}
                      </TableBody>
                    </Table>
                    {valuation_range.upside_context && (
                      <div className="px-6 py-3 border-t border-slate-100">
                        <p className="text-xs text-slate-500">{valuation_range.upside_context}</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}
              {/* ── Valuation Range Bar */}
              <ValuationRangeBar
                bearPrice={valuation_range.bear_price}
                basePrice={valuation_range.base_price}
                bullPrice={valuation_range.bull_price}
                currentPrice={stock_info.current_price}
                mc={displayMC}
              />

              {/* ── Macro-Adjusted Valuation */}
              {macroAdj && stock_info.current_price != null && (
                <MacroAdjustedValuationCard
                  adj={macroAdj}
                  currentPrice={stock_info.current_price}
                />
              )}

              {/* ── Entry Strategy & Price Levels */}
              <EntryStrategyCard
                vr={valuation_range}
                currentPrice={stock_info.current_price}
                momScore={momScore}
                riskScore={riskScore}
                mc={displayMC}
                stance={stance}
              />

              {/* Driver-based scenario model */}
              <DriverModelCard vr={valuation_range} currentPrice={stock_info.current_price} />
              {/* Return Distribution Chart */}
              {valuation_range.mc && (
                <ReturnDistributionChart mc={valuation_range.mc} />
              )}
            </div>
          </div>

          {/* PAGE 5 — Peer Comparison (5 sections) */}
          <div className="print-page print-page-break">
            <h2 className="text-xl font-bold text-slate-900 mb-5 pb-2 border-b border-slate-200">
              {peer_comparison.section_label ?? "Peer Comparison"}
            </h2>
            {peer_comparison.peer_level === 3 && peer_comparison.proxy_note && (
              <p className="mb-4 text-xs text-amber-700 italic">{peer_comparison.proxy_note}</p>
            )}
            {!peer_comparison.has_peers ? (
              <p className="text-sm text-slate-500">No peer comparison data available.</p>
            ) : (
              <div className="space-y-5">
                <PeerSection
                  title="Valuation"
                  insight={peer_comparison.insights[0] ?? ""}
                  rows={peer_comparison.rows}
                  isPrint
                  columns={[
                    { header: "P/E", key: "pe", format: formatMultiple, isValuation: true },
                    { header: "P/S", key: "ps", format: formatMultiple, isValuation: true },
                    { header: "EV/EBITDA", key: "ev_ebitda", format: formatMultiple, isValuation: true },
                    { header: "PEG", key: "peg", format: formatMultiple, isValuation: true },
                  ]}
                />
                <PeerSection
                  title="Growth"
                  insight={peer_comparison.insights[1] ?? ""}
                  rows={peer_comparison.rows}
                  isPrint
                  columns={[
                    { header: "Rev Growth", key: "revenue_growth", format: fmtPctRaw, highIsGood: true },
                    { header: "EPS CAGR", key: "eps_growth", format: (v) => v != null ? fmtPctRaw(v * 100) : "—", highIsGood: true },
                    { header: "EBITDA Growth", key: "ebitda_growth", format: fmtPctRaw, highIsGood: true },
                  ]}
                />
                <PeerSection
                  title="Profitability"
                  insight={peer_comparison.insights[2] ?? ""}
                  rows={peer_comparison.rows}
                  isPrint
                  columns={[
                    { header: "Gross Margin", key: "gross_margin", format: fmtPct, highIsGood: true },
                    { header: "Op. Margin", key: "operating_margin", format: fmtPct, highIsGood: true },
                    { header: "Net Margin", key: "net_margin", format: fmtPct, highIsGood: true },
                    { header: "ROE", key: "roe", format: fmtPct, highIsGood: true },
                    { header: "ROIC", key: "roic", format: fmtPct, highIsGood: true },
                  ]}
                />
                <PeerSection
                  title="Financial Health"
                  insight={peer_comparison.insights[3] ?? ""}
                  rows={peer_comparison.rows}
                  isPrint
                  columns={[
                    { header: "D/E", key: "debt_equity", format: (v) => v != null ? `${v.toFixed(2)}x` : "—", highIsGood: false },
                    { header: "Current Ratio", key: "current_ratio", format: (v) => v != null ? `${v.toFixed(2)}x` : "—", highIsGood: true },
                    { header: "Int. Coverage", key: "interest_coverage", format: (v) => v != null ? `${v.toFixed(1)}x` : "—", highIsGood: true },
                  ]}
                />
                <PeerSection
                  title="Market & Risk"
                  insight={peer_comparison.insights[4] ?? ""}
                  rows={peer_comparison.rows}
                  isPrint
                  columns={[
                    { header: "Beta", key: "beta", format: (v) => v != null ? v.toFixed(2) : "—", highIsGood: false },
                  ]}
                />
                {/* 5-year historical performance */}
                <PeerHistoricalTable rows={peer_comparison.rows} peerTrendInsights={peer_comparison.peer_trend_insights} />
              </div>
            )}
          </div>

          {/* PAGE 6 — Macro Environment */}
          <div className="print-page print-page-break">
            <h2 className="text-xl font-bold text-slate-900 mb-6 pb-2 border-b border-slate-200">Macro Environment</h2>
            {!macro.available ? (
              <p className="text-sm text-slate-500">Macro data not available.</p>
            ) : (
              <div className="space-y-4">
                <Card className="border-slate-200 shadow-sm">
                  <CardContent className="px-6 py-5 space-y-4">
                    <dl className="grid grid-cols-[160px_1fr] gap-x-2 gap-y-2 text-sm">
                      {macro.macro_regime && (
                        <><dt className="text-slate-500">Macro Regime</dt><dd className="font-semibold text-slate-800">{macro.macro_regime}</dd></>
                      )}
                      {macro.cycle_phase && macro.cycle_phase !== 'unknown' && (
                        <><dt className="text-slate-500">Cycle Phase</dt><dd className="capitalize text-slate-700">{macro.cycle_phase} Cycle</dd></>
                      )}
                      {macro.recession_risk_level && (
                        <><dt className="text-slate-500">Recession Risk</dt>
                        <dd className={cn("font-semibold",
                          macro.recession_risk_level.toLowerCase().includes("high") ? "text-red-700" :
                          macro.recession_risk_level.toLowerCase().includes("low") ? "text-green-700" : "text-amber-700"
                        )}>{macro.recession_risk_level}</dd></>
                      )}
                      {macro.sector_tilt && (
                        <><dt className="text-slate-500">Sector Tilt</dt><dd className="text-slate-700">{macro.sector_tilt}</dd></>
                      )}
                    </dl>
                    {macro.macro_score != null && (
                      <div className="space-y-1.5">
                        <div className="flex justify-between text-xs text-slate-500">
                          <span>Macro Score</span>
                          <span className="font-semibold">{Math.round(macro.macro_score)}/100</span>
                        </div>
                        <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                          <div className={cn("h-full rounded-full", scoreColor(macro.macro_score).bar)} style={{ width: `${macro.macro_score}%` }} />
                        </div>
                      </div>
                    )}
                    {macro.reasoning_summary && (
                      <p className="text-sm text-slate-600 border-t border-slate-100 pt-3 leading-relaxed">{macro.reasoning_summary}</p>
                    )}
                  </CardContent>
                </Card>
                <div className="grid grid-cols-2 gap-4">
                  <FactorCard title="Macro Tailwinds" items={macro.bullish_macro_factors || []} variant="bullish" />
                  <FactorCard title="Macro Headwinds" items={macro.bearish_macro_factors || []} variant="bearish" />
                </div>
              </div>
            )}
          </div>

          {/* PAGE 7 — Final Verdict / Position Sizing */}
          <div className="print-page print-page-break">
            <h2 className="text-xl font-bold text-slate-900 mb-6 pb-2 border-b border-slate-200">Final Verdict &amp; Position Sizing</h2>
            <div className="grid grid-cols-2 gap-4">
              <Card className="border-slate-200 shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-semibold text-slate-700">Verdict</CardTitle>
                </CardHeader>
                <CardContent className="px-6 pb-6 space-y-3">
                  {/* Thesis / Execution alignment — print version */}
                  {pageLevelDS && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap', marginBottom: '4px' }}>
                      <span style={{ fontSize: '11px', fontWeight: 700, padding: '1px 8px', borderRadius: '9999px', backgroundColor: pageLevelDS.thesisRatingBg, color: pageLevelDS.thesisRatingFg, WebkitPrintColorAdjust: 'exact' }}>
                        Thesis: {pageLevelDS.thesisRating}
                      </span>
                      <span style={{ fontSize: '11px', fontWeight: 700, padding: '1px 8px', borderRadius: '9999px', backgroundColor: pageLevelDS.executionBg, color: pageLevelDS.executionFg, WebkitPrintColorAdjust: 'exact' }}>
                        {pageLevelDS.executionStatus}
                      </span>
                      <span style={{ fontSize: '10px', color: '#64748b' }}>
                        · {pageLevelDS.conviction} conviction · {pageLevelDS.targetPct} target · {pageLevelDS.currentPct} now
                      </span>
                    </div>
                  )}
                  <div className="flex items-center gap-3">
                    <div className={cn("w-3 h-3 rounded-full", sc.dot)} />
                    <span className={cn("text-2xl font-bold", sc.text)}>{headlineStance}</span>
                    <span className="text-slate-400 text-sm">— {Math.round(scorecard.overall_score)}/100</span>
                  </div>
                  <p className="text-xs text-slate-400">{Math.round(scorecard.confidence * 100)}% confidence · {stanceDesc}</p>
                  {verdictText ? (
                    <p className="text-sm text-slate-600 leading-relaxed">{sanitizedVerdictText}</p>
                  ) : (
                    <p className="text-sm text-slate-600 leading-relaxed">{scorecard.bullish_factors[0] || ""}</p>
                  )}
                  {data.key_tension && (
                    <div className="p-3 bg-slate-50 border-l-4 border-slate-300 rounded-r-lg">
                      <p className="text-xs text-slate-500 italic">{data.key_tension}</p>
                    </div>
                  )}
                  {/* Sizing note aligned with execution — print */}
                  {pageLevelDS && (pageLevelDS.executionStatus === 'BUY NOW' || pageLevelDS.executionStatus === 'STAGED BUY' || pageLevelDS.executionStatus === 'WAIT') && (
                    <div style={{ padding: '6px 10px', backgroundColor: '#f8fafc', borderRadius: '6px', borderLeft: `3px solid ${pageLevelDS.executionBg}`, WebkitPrintColorAdjust: 'exact' }}>
                      <p style={{ fontSize: '10px', color: '#1e293b', fontWeight: 500, margin: 0 }}>{pageLevelDS.sizingNote}</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>

          {/* PAGE 8 — Investment Memo (single, final) */}
          <div className="print-page print-page-break">
            <h2 className="text-xl font-bold text-slate-900 mb-6 pb-2 border-b border-slate-200">Investment Memo</h2>
            <div className="space-y-5">
              {topTakeaway && (
                <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Top Takeaway</h3>
                  <p className="text-sm text-slate-700 leading-relaxed">{sanitizedTopTakeaway}</p>
                </div>
              )}
              {investmentThesisBullets.length > 0 && (
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Investment Thesis</h3>
                  <ul className="space-y-1.5">
                    {investmentThesisBullets.map((b, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                        <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0 text-green-500" />
                        {b}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {keyRisksBullets.length > 0 && (
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Key Risks</h3>
                  <ul className="space-y-1.5">
                    {keyRisksBullets.map((b, i) => (
                      <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                        <XCircle className="w-3.5 h-3.5 mt-0.5 shrink-0 text-red-400" />
                        {b}
                      </li>
                    ))}
                  </ul>
                  {scorecard.risk_flags.length > 0 && (
                    <div className="mt-3 p-2.5 bg-amber-50 border border-amber-100 rounded-lg space-y-1">
                      <span className="text-xs font-semibold text-amber-800 flex items-center gap-1"><AlertTriangle className="w-3 h-3" /> Risk Flags</span>
                      {scorecard.risk_flags.map((flag, i) => (
                        <p key={i} className="text-xs text-amber-700">• {flag}</p>
                      ))}
                    </div>
                  )}
                </div>
              )}
              <div className="border-t border-slate-100 pt-4">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Summary Verdict</h3>
                <div className="flex items-center gap-2 mb-1.5">
                  <div className={cn("w-2.5 h-2.5 rounded-full", sc.dot)} />
                  <span className={cn("font-bold", sc.text)}>{headlineStance}</span>
                  <span className="text-slate-400 text-sm">— {Math.round(scorecard.overall_score)}/100 · {Math.round(scorecard.confidence * 100)}% confidence</span>
                </div>
                <p className="text-sm text-slate-600 leading-relaxed">{sanitizedCallout}</p>
                {data.key_tension && (
                  <p className="text-xs text-slate-400 italic mt-1">{data.key_tension}</p>
                )}
                <p className="text-xs text-slate-400 mt-1.5">See Final Verdict page for full analysis and position sizing detail.</p>
              </div>
            </div>
          </div>

        </div>

        {/* ── Section 4: Investment memo ──────────────────────── */}
        <Card className="border-slate-200 shadow-sm animate-fade-in no-print">
          <CardHeader className="pb-2">
            <CardTitle className="text-base font-semibold text-slate-800">Investment Memo</CardTitle>
          </CardHeader>
          <CardContent className="px-6 pb-6 space-y-6">
            {/* Execution context strip — aligns memo to current decision state */}
            {pageLevelDS && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap', padding: '8px 12px', backgroundColor: '#f8fafc', borderRadius: '8px', border: '1px solid #f1f5f9' }}>
                <span style={{ fontSize: '10px', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginRight: '4px' }}>Report Context</span>
                <span style={{ fontSize: '11px', fontWeight: 700, padding: '1px 8px', borderRadius: '9999px', backgroundColor: pageLevelDS.thesisRatingBg, color: pageLevelDS.thesisRatingFg }}>
                  Thesis: {pageLevelDS.thesisRating}
                </span>
                <span style={{ fontSize: '11px', fontWeight: 700, padding: '1px 8px', borderRadius: '9999px', backgroundColor: pageLevelDS.executionBg, color: pageLevelDS.executionFg }}>
                  {pageLevelDS.executionStatus}
                </span>
                <span style={{ fontSize: '11px', fontWeight: 600, padding: '1px 8px', borderRadius: '9999px', backgroundColor: '#f1f5f9', color: '#475569' }}>
                  Conviction: {pageLevelDS.conviction}
                </span>
              </div>
            )}
            {/* Investment Thesis — MemoEngine structured bullets */}
            {investmentThesisBullets.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Investment Thesis</h3>
                <ul className="space-y-2">
                  {investmentThesisBullets.map((b, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                      <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0 text-green-500" />
                      {b}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Key Risks — MemoEngine structured bullets */}
            {keyRisksBullets.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">Key Risks</h3>
                <ul className="space-y-2">
                  {keyRisksBullets.map((b, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-slate-600">
                      <XCircle className="w-3.5 h-3.5 mt-0.5 shrink-0 text-red-400" />
                      {b}
                    </li>
                  ))}
                </ul>
                {scorecard.risk_flags.length > 0 && (
                  <div className="mt-3 p-2.5 bg-amber-50 border border-amber-100 rounded-lg space-y-1">
                    <span className="text-xs font-semibold text-amber-800 flex items-center gap-1">
                      <AlertTriangle className="w-3 h-3" /> Risk Flags
                    </span>
                    {scorecard.risk_flags.map((flag, i) => (
                      <p key={i} className="text-xs text-amber-700">• {flag}</p>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Raw memo toggle */}
            <div className="border-t border-slate-100 pt-4">
              <button
                onClick={() => setShowMemo(!showMemo)}
                className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-600 transition-colors no-print"
              >
                {showMemo ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                {showMemo ? "Hide" : "Show"} full memo
              </button>
              {showMemo && (
                <pre className="mt-3 text-xs text-slate-500 font-mono leading-relaxed whitespace-pre-wrap bg-slate-50 rounded-lg p-4 overflow-auto">
                  {memo}
                </pre>
              )}
            </div>
          </CardContent>
        </Card>

        {/* ── Validation Log ────────────────────────────────── */}
        {data.validation_log && <ValidationLogCard vlog={data.validation_log} />}

        {/* ── Section 5: Verdict + Position sizing ───────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in no-print">
          {/* Verdict */}
          <Card className="border-slate-200 shadow-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-base font-semibold text-slate-800">Final Verdict</CardTitle>
            </CardHeader>
            <CardContent className="px-6 pb-6 space-y-4">
              {/* Thesis / Execution / Conviction alignment strip */}
              {pageLevelDS && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '12px', fontWeight: 700, padding: '2px 10px', borderRadius: '9999px', backgroundColor: pageLevelDS.thesisRatingBg, color: pageLevelDS.thesisRatingFg }}>
                    {pageLevelDS.thesisRating}
                  </span>
                  <ArrowRight className="w-3 h-3 text-slate-300 shrink-0" />
                  <span style={{ fontSize: '12px', fontWeight: 700, padding: '2px 10px', borderRadius: '9999px', backgroundColor: pageLevelDS.executionBg, color: pageLevelDS.executionFg }}>
                    {pageLevelDS.executionStatus}
                  </span>
                  <span style={{ fontSize: '11px', color: '#64748b', marginLeft: '2px' }}>
                    · {pageLevelDS.conviction} conviction · {pageLevelDS.targetPct} target
                  </span>
                </div>
              )}

              <div className="flex items-center gap-3">
                <div className={cn("w-3 h-3 rounded-full", sc.dot)} />
                <span className={cn("text-xl font-bold", sc.text)}>{headlineStance}</span>
                <span className="text-slate-400 text-sm">— {Math.round(scorecard.overall_score)}/100</span>
              </div>

              {verdictText ? (
                <p className="text-sm text-slate-600 leading-relaxed">{sanitizedVerdictText}</p>
              ) : (
                <p className="text-sm text-slate-600 leading-relaxed">
                  {scorecard.bullish_factors[0] || "See full memo for detailed analysis."}
                </p>
              )}

              {data.key_tension && (
                <div className="p-3 bg-slate-50 border-l-4 border-slate-300 rounded-r-lg">
                  <p className="text-xs text-slate-500 italic">{data.key_tension}</p>
                </div>
              )}

              {/* Sizing reminder aligned with execution */}
              {pageLevelDS && (pageLevelDS.executionStatus === 'BUY NOW' || pageLevelDS.executionStatus === 'STAGED BUY' || pageLevelDS.executionStatus === 'WAIT') && (
                <div style={{ padding: '8px 12px', backgroundColor: '#f8fafc', borderRadius: '8px', borderLeft: `3px solid ${pageLevelDS.executionBg}` }}>
                  <p style={{ fontSize: '11px', color: '#1e293b', fontWeight: 500, margin: 0 }}>{pageLevelDS.sizingNote}</p>
                </div>
              )}
            </CardContent>
          </Card>

        </div>

        {/* ── Footer ─────────────────────────────────────────────── */}
        <div className="border-t border-slate-200 pt-6 pb-10 flex flex-col sm:flex-row justify-between gap-2 text-xs text-slate-400">
          <span>Data: Financial Modeling Prep API</span>
          <span>Evaluated: {new Date(data.evaluated_at).toLocaleString()}</span>
        </div>
      </main>
    </div>
  );
}

// ── Helper sub-components ─────────────────────────────────────────────────────

function StatCell({
  label,
  value,
  large,
}: {
  label: string;
  value: string;
  large?: boolean;
}) {
  return (
    <div className="space-y-0.5">
      <p className="text-xs text-slate-400 font-medium">{label}</p>
      <p
        className={cn(
          "font-semibold tabular-nums text-slate-900",
          large ? "text-xl" : "text-sm"
        )}
      >
        {value}
      </p>
    </div>
  );
}

function FactorCard({
  title,
  items,
  variant,
}: {
  title: string;
  items: string[];
  variant: "bullish" | "bearish";
}) {
  const isBullish = variant === "bullish";
  return (
    <Card
      className={cn(
        "shadow-sm border-l-4",
        isBullish ? "border-l-green-400 border-green-100" : "border-l-red-400 border-red-100"
      )}
    >
      <CardHeader className="pb-2">
        <CardTitle
          className={cn(
            "text-sm font-semibold flex items-center gap-2",
            isBullish ? "text-green-700" : "text-red-700"
          )}
        >
          {isBullish ? (
            <TrendingUp className="w-4 h-4" />
          ) : (
            <TrendingDown className="w-4 h-4" />
          )}
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="px-6 pb-5">
        {items.length === 0 ? (
          <p className="text-xs text-slate-400 italic">None identified.</p>
        ) : (
          <ul className="space-y-2">
            {items.map((item, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                {isBullish ? (
                  <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 shrink-0 text-green-500" />
                ) : (
                  <XCircle className="w-3.5 h-3.5 mt-0.5 shrink-0 text-red-400" />
                )}
                {item}
              </li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

function ValuationRow({
  label,
  bear,
  base,
  bull,
}: {
  label: string;
  bear?: number | null;
  base?: number | null;
  bull?: number | null;
}) {
  return (
    <TableRow>
      <TableCell className="pl-6 text-sm text-slate-600">{label}</TableCell>
      <TableCell className="text-center text-sm text-red-600 font-semibold tabular-nums">
        {bear != null ? bear.toFixed(1) + "x" : "—"}
      </TableCell>
      <TableCell className="text-center text-sm text-blue-600 font-semibold tabular-nums">
        {base != null ? base.toFixed(1) + "x" : "—"}
      </TableCell>
      <TableCell className="text-center text-sm text-green-600 font-semibold tabular-nums">
        {bull != null ? bull.toFixed(1) + "x" : "—"}
      </TableCell>
    </TableRow>
  );
}

function formatUpside(target: number, current: number): string {
  const pct = ((target - current) / current) * 100;
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}%`;
}

// ── Return Distribution Chart ─────────────────────────────────────────────────

function ReturnDistributionChart({ mc }: { mc: MCSim }) {
  const SVG_W  = 520;
  const SVG_H  = 82;
  const PAD_L  = 46;  // left padding (avoids label clip for negative P5)
  const PAD_R  = 46;  // right padding
  const TRACK_W = SVG_W - PAD_L - PAD_R;  // 428
  const TRACK_Y = 40;
  const LABEL_Y = 17;   // label names above track
  const VAL_Y   = 60;   // return values below track
  const TICK_H  = 10;   // half-height of normal tick
  const ER_H    = 16;   // half-height of E[R] tick (taller = emphasis)

  const er  = mc.mean_return;
  const p5  = mc.p5_return;
  const p50 = mc.median_return;
  const p95 = mc.p95_return;

  // Axis: 12% padding beyond P5/P95, at least ±15% so the chart always has room
  const axisMin   = Math.min(p5 * 1.12, -0.15);
  const axisMax   = Math.max(p95 * 1.12,  0.15);
  const axisRange = axisMax - axisMin || 0.01;

  const toX = (r: number) =>
    PAD_L + ((r - axisMin) / axisRange) * TRACK_W;

  const p5x   = toX(p5);
  const p50x  = toX(p50);
  const erx   = toX(er);
  const p95x  = toX(p95);
  const zeroX = toX(0);

  // Merge P50 label into E[R] when they're within 34px of each other
  const erP50Close = Math.abs(erx - p50x) < 34;

  // Caption generation
  const spread  = (p95 - p5) * 100;
  const p5Pct   = p5  * 100;
  const p95Pct  = p95 * 100;
  const ud      = p95 > 0 && p5 < 0 ? p95Pct / Math.abs(p5Pct) : null;

  let caption = "";
  if (spread > 80)       caption += `Wide distribution (${spread.toFixed(0)}pp spread). `;
  else if (spread > 40)  caption += `Moderate spread (${spread.toFixed(0)}pp). `;
  else                   caption += `Tight distribution (${spread.toFixed(0)}pp spread). `;

  if      (p5Pct < -30)  caption += `Significant downside tail (P5 ${p5Pct.toFixed(0)}%). `;
  else if (p5Pct < -15)  caption += `Moderate downside risk (P5 ${p5Pct.toFixed(0)}%). `;

  if (ud !== null) {
    if      (ud > 2.0)  caption += `Asymmetric upside: ${ud.toFixed(1)}× upside/downside.`;
    else if (ud > 1.2)  caption += `Slight upside skew: ${ud.toFixed(1)}× upside/downside.`;
    else                caption += `Balanced risk/reward (${ud.toFixed(1)}× upside/downside).`;
  }

  const fmtR = (v: number) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;

  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-1">
        <CardTitle className="text-sm font-semibold text-slate-700">
          Probability Distribution
          <span className="ml-2 text-xs font-normal text-slate-400">
            {mc.n_sims === 0 ? 'driver-based (deterministic)' : `${mc.n_sims.toLocaleString()} simulations · driver-based`}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="px-5 pb-4 pt-2">
        <svg
          viewBox={`0 0 ${SVG_W} ${SVG_H}`}
          className="w-full"
          aria-hidden="true"
          style={{ fontFamily: "ui-sans-serif, system-ui, sans-serif" }}
        >
          {/* Track */}
          <line
            x1={PAD_L} y1={TRACK_Y} x2={PAD_L + TRACK_W} y2={TRACK_Y}
            stroke="#cbd5e1" strokeWidth="1.5"
          />

          {/* Zero baseline (dashed) */}
          {zeroX >= PAD_L && zeroX <= PAD_L + TRACK_W && (
            <>
              <line
                x1={zeroX} y1={TRACK_Y - TICK_H - 2} x2={zeroX} y2={TRACK_Y + TICK_H + 2}
                stroke="#94a3b8" strokeWidth="1" strokeDasharray="3 3"
              />
              <text x={zeroX} y={VAL_Y + 12} textAnchor="middle" fontSize="9" fill="#94a3b8">0%</text>
            </>
          )}

          {/* P5–P95 shaded band */}
          <rect
            x={p5x} y={TRACK_Y - 6} width={p95x - p5x} height={12}
            fill="#6366f1" fillOpacity="0.10" rx="2"
          />

          {/* ── P5 ── */}
          <line
            x1={p5x} y1={TRACK_Y - TICK_H} x2={p5x} y2={TRACK_Y + TICK_H}
            stroke="#ef4444" strokeWidth="1.5"
          />
          <circle cx={p5x} cy={TRACK_Y} r="3.5" fill="#ef4444" />
          <text x={p5x} y={LABEL_Y} textAnchor="middle" fontSize="10" fontWeight="500" fill="#ef4444">P5</text>
          <text x={p5x} y={VAL_Y}   textAnchor="middle" fontSize="10" fill="#dc2626">{fmtR(p5)}</text>

          {/* ── P50 (hidden when too close to E[R]) ── */}
          {!erP50Close && (
            <>
              <line
                x1={p50x} y1={TRACK_Y - TICK_H} x2={p50x} y2={TRACK_Y + TICK_H}
                stroke="#64748b" strokeWidth="1.5"
              />
              <circle cx={p50x} cy={TRACK_Y} r="3.5" fill="#64748b" />
              <text x={p50x} y={LABEL_Y} textAnchor="middle" fontSize="10" fontWeight="500" fill="#64748b">P50</text>
              <text x={p50x} y={VAL_Y}   textAnchor="middle" fontSize="10" fill="#475569">{fmtR(p50)}</text>
            </>
          )}

          {/* ── E[R] (emphasized) ── */}
          <line
            x1={erx} y1={TRACK_Y - ER_H} x2={erx} y2={TRACK_Y + ER_H}
            stroke="#4f46e5" strokeWidth="2"
          />
          <circle cx={erx} cy={TRACK_Y} r="5" fill="#4f46e5" />
          <text x={erx} y={LABEL_Y} textAnchor="middle" fontSize="10" fontWeight="700" fill="#4f46e5">
            {erP50Close ? "E[R]≈P50" : "E[R]"}
          </text>
          <text x={erx} y={VAL_Y} textAnchor="middle" fontSize="10" fontWeight="600" fill="#4f46e5">{fmtR(er)}</text>

          {/* ── P95 ── */}
          <line
            x1={p95x} y1={TRACK_Y - TICK_H} x2={p95x} y2={TRACK_Y + TICK_H}
            stroke="#16a34a" strokeWidth="1.5"
          />
          <circle cx={p95x} cy={TRACK_Y} r="3.5" fill="#16a34a" />
          <text x={p95x} y={LABEL_Y} textAnchor="middle" fontSize="10" fontWeight="500" fill="#16a34a">P95</text>
          <text x={p95x} y={VAL_Y}   textAnchor="middle" fontSize="10" fill="#15803d">{fmtR(p95)}</text>
        </svg>

        {caption && (
          <p className="text-xs text-slate-500 mt-1 leading-relaxed">{caption}</p>
        )}
      </CardContent>
    </Card>
  );
}

// ── Monte Carlo card ──────────────────────────────────────────────────────────

function MCCard({
  mc,
  currentPrice,
}: {
  mc: MCSim;
  currentPrice: number | null;
}) {
  const fmtR = (v: number) =>
    `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
  const fmtPct = (v: number) => `${(v * 100).toFixed(0)}%`;

  // Build price-axis bounds with 3% padding so markers never clip
  const allPrices = [mc.p5_price, mc.p95_price];
  if (currentPrice) allPrices.push(currentPrice);
  const axisMin = Math.min(...allPrices) * 0.97;
  const axisMax = Math.max(...allPrices) * 1.03;
  const axisRange = axisMax - axisMin || 1;
  const barPos = (p: number) => `${((p - axisMin) / axisRange) * 100}%`;
  const barW   = (lo: number, hi: number) =>
    `${((hi - lo) / axisRange) * 100}%`;

  const probRows: [string, number, string, string][] = [
    ["P(gain)",       mc.prob_positive, "bg-green-500",  "text-green-700"],
    ["P(>20% gain)",  mc.prob_20_gain,  "bg-emerald-400","text-emerald-700"],
    ["P(>20% loss)",  mc.prob_loss_20,  "bg-red-400",    "text-red-600"],
  ];

  const retRows: [string, number, string][] = [
    ["P5 (bear)",  mc.p5_return,    "text-red-600"],
    ["P25",        mc.p25_return,   "text-orange-500"],
    ["Median",     mc.median_return,"text-slate-800"],
    ["P75",        mc.p75_return,   "text-blue-600"],
    ["P95 (bull)", mc.p95_return,   "text-green-600"],
  ];

  const udColor =
    mc.upside_downside >= 2.0 ? "text-green-700"
    : mc.upside_downside < 1.0 ? "text-red-600"
    : "text-slate-700";

  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-slate-700 flex items-center justify-between">
          <span>Distribution-Based Valuation</span>
          <span className="text-xs font-normal text-slate-400 tabular-nums">
            {mc.n_sims.toLocaleString()} sims · MC
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="px-6 pb-5 space-y-5">

        {/* ── Hero metrics: E[R] / Skew Ratio / Spread ─────── */}
        <div className="grid grid-cols-3 gap-4 pb-4 border-b border-slate-100">
          <div>
            <p className="text-[10px] text-slate-400 mb-0.5 uppercase tracking-wide font-medium">E[Return]</p>
            <p className={cn("text-lg font-bold tabular-nums", mc.mean_return >= 0 ? "text-green-600" : "text-red-600")}>
              {fmtR(mc.mean_return)}
            </p>
            <p className="text-[10px] text-slate-400">expected return</p>
          </div>
          <div>
            <p className="text-[10px] text-slate-400 mb-0.5 uppercase tracking-wide font-medium">Skew Ratio</p>
            <p className={cn("text-lg font-bold tabular-nums", udColor)}>
              {mc.upside_downside.toFixed(2)}×
            </p>
            <p className="text-[10px] text-slate-400">P95 / |P5|</p>
          </div>
          <div>
            <p className="text-[10px] text-slate-400 mb-0.5 uppercase tracking-wide font-medium">Spread</p>
            <p className="text-lg font-bold tabular-nums text-slate-700">
              {fmtR(mc.p95_return - mc.p5_return)}
            </p>
            <p className="text-[10px] text-slate-400">P95 – P5</p>
          </div>
        </div>

        {/* ── Probability bars ─────────────────────────────── */}
        <div className="grid grid-cols-3 gap-4">
          {probRows.map(([label, val, barCls, txtCls]) => (
            <div key={label} className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="text-slate-500">{label}</span>
                <span className={cn("font-bold tabular-nums", txtCls)}>
                  {fmtPct(val)}
                </span>
              </div>
              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className={cn("h-full rounded-full", barCls)}
                  style={{ width: fmtPct(val) }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* ── Return percentile grid ───────────────────────── */}
        <div className="border-t border-slate-100 pt-4">
          <p className="text-xs text-slate-400 font-medium mb-2.5">
            Return Distribution
          </p>
          <div className="grid grid-cols-5 gap-1 text-center">
            {retRows.map(([label, val, cls]) => (
              <div key={label} className="space-y-0.5">
                <p className="text-[10px] text-slate-400 leading-tight">
                  {label}
                </p>
                <p className={cn("text-sm font-bold tabular-nums", cls)}>
                  {fmtR(val)}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* ── Price distribution bar ───────────────────────── */}
        <div className="border-t border-slate-100 pt-4">
          <p className="text-xs text-slate-400 font-medium mb-4">
            Price Distribution
          </p>
          <div className="relative h-10">
            {/* P5–P95 full range */}
            <div
              className="absolute h-2.5 top-2.5 bg-slate-100 rounded-full"
              style={{ left: barPos(mc.p5_price), width: barW(mc.p5_price, mc.p95_price) }}
            />
            {/* IQR: P25–P75 */}
            <div
              className="absolute h-2.5 top-2.5 bg-blue-200 rounded"
              style={{ left: barPos(mc.p25_price), width: barW(mc.p25_price, mc.p75_price) }}
            />
            {/* Median dot */}
            <div
              className="absolute w-3 h-3 bg-blue-600 rounded-full top-2 -translate-x-1.5"
              style={{ left: barPos(mc.median_price) }}
            />
            {/* Current price marker */}
            {currentPrice && (
              <div
                className="absolute w-px h-6 top-1 bg-slate-500"
                style={{ left: barPos(currentPrice) }}
              >
                <span className="absolute -top-4 left-1 text-[9px] text-slate-500 whitespace-nowrap font-medium">
                  {formatPrice(currentPrice)}
                </span>
              </div>
            )}
            {/* Axis price labels */}
            <span
              className="absolute top-7 text-[9px] text-slate-400 -translate-x-1/2 tabular-nums"
              style={{ left: barPos(mc.p5_price) }}
            >
              {formatPrice(mc.p5_price)}
            </span>
            <span
              className="absolute top-7 text-[9px] text-slate-400 -translate-x-1/2 tabular-nums"
              style={{ left: barPos(mc.p95_price) }}
            >
              {formatPrice(mc.p95_price)}
            </span>
          </div>
        </div>

        {/* ── Footer ──────────────────────────────────────────── */}
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500 border-t border-slate-100 pt-3">
          <span>
            Growth:{" "}
            <span className="font-semibold text-slate-700">
              {(mc.growth_mean * 100).toFixed(1)}%{" "}
              <span className="text-slate-400">
                ±{(mc.growth_std * 100).toFixed(1)}%
              </span>
            </span>
          </span>
          <span>
            Kelly size:{" "}
            <span className="font-semibold text-slate-700">
              {(mc.kelly_fraction * 100).toFixed(1)}%
            </span>
          </span>
          <span>
            P(gain):{" "}
            <span className="font-semibold text-green-700">
              {(mc.prob_positive * 100).toFixed(0)}%
            </span>
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

// ── Peer comparison helpers ───────────────────────────────────────────────────


/** Format a margin/ratio value (0-1) as a percentage string, or "—" */
function fmtPct(v: number | null | undefined, decimals = 1): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(decimals)}%`;
}

/** Format a percentage-already value (e.g. revenue_growth = 15.5) */
function fmtPctRaw(v: number | null | undefined, decimals = 1): string {
  if (v == null) return "—";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(decimals)}%`;
}

/**
 * Sanitize a peer row — convert impossible 0.0 sentinels to null.
 * interest_coverage of 0 means data absent, not a true zero coverage ratio.
 * growth of 0 could be genuine but is more often a missing-data fill; leave growth as-is
 * since 0% growth is valid. Only fix provably-invalid zeros.
 */
function sanitizePeerRow(r: PeerRowType): PeerRowType {
  return {
    ...r,
    interest_coverage: r.interest_coverage === 0 ? null : r.interest_coverage,
  };
}

// Growth outlier caps for median computation (mirrors backend caps)
const _GROWTH_CAP = 300;
const _EPS_GROWTH_CAP_DEC = 1.5; // 150% as decimal

/** Compute positioning label vs. peer median for a set of rows.
 *  Growth fields are outlier-filtered before computing median. */
function peerMedian(rows: PeerRowType[], key: keyof PeerRowType): number | null {
  const isGrowthKey = key === "revenue_growth" || key === "ebitda_growth";
  const isEpsGrowthKey = key === "eps_growth";

  const vals = rows
    .filter((r) => !r.is_target && r[key] != null)
    .map((r) => r[key] as number)
    .filter((v) => {
      if (isGrowthKey) return Math.abs(v) <= _GROWTH_CAP;
      if (isEpsGrowthKey) return Math.abs(v) <= _EPS_GROWTH_CAP_DEC;
      return true;
    });
  if (!vals.length) return null;
  const sorted = [...vals].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

/** Relative positioning badge: premium/discount/in-line */
function RelBadge({
  value,
  median,
  highIsGood,
  isValuation,
}: {
  value: number | null | undefined;
  median: number | null;
  highIsGood?: boolean;
  isValuation?: boolean;
}) {
  if (value == null || median == null || median === 0) return null;
  // Use (value - median) / |median| — works correctly when median is negative.
  // ratio = value / median is wrong for negative medians (e.g. EPS CAGR).
  const absMedian = Math.abs(median);
  const pctDiff = (value - median) / absMedian;
  let label = "";
  let cls = "";

  if (isValuation) {
    // Higher valuation = premium (expensive)
    if (pctDiff > 0.15) { label = "prem"; cls = "bg-amber-100 text-amber-700"; }
    else if (pctDiff < -0.15) { label = "disc"; cls = "bg-green-100 text-green-700"; }
    else { label = "≈med"; cls = "bg-slate-100 text-slate-500"; }
  } else if (highIsGood) {
    if (pctDiff > 0.15) { label = "↑"; cls = "bg-green-100 text-green-700"; }
    else if (pctDiff < -0.15) { label = "↓"; cls = "bg-red-100 text-red-600"; }
    else { label = "≈"; cls = "bg-slate-100 text-slate-500"; }
  } else {
    if (pctDiff > 0.15) { label = "↑"; cls = "bg-red-100 text-red-600"; }
    else if (pctDiff < -0.15) { label = "↓"; cls = "bg-green-100 text-green-700"; }
    else { label = "≈"; cls = "bg-slate-100 text-slate-500"; }
  }

  return (
    <span className={cn("ml-1.5 px-1 py-0.5 rounded text-[10px] font-semibold", cls)}>
      {label}
    </span>
  );
}

interface PeerSectionProps {
  title: string;
  insight: string;
  rows: PeerRowType[];
  columns: {
    header: string;
    key: keyof PeerRowType;
    format: (v: number | null | undefined) => string;
    highIsGood?: boolean;
    isValuation?: boolean;
    align?: "right" | "left";
  }[];
  isPrint?: boolean;
}

function PeerSection({ title, insight, rows, columns, isPrint }: PeerSectionProps) {
  const hasAnyData = rows.some((r) =>
    columns.some((c) => r[c.key] != null)
  );
  if (!hasAnyData) return null;

  const medians = Object.fromEntries(
    columns.map((c) => [c.key, peerMedian(rows, c.key)])
  ) as Record<string, number | null>;

  return (
    <div className={isPrint ? "mb-6" : ""}>
      <Card className="border-slate-200 shadow-sm overflow-hidden">
        <CardHeader className="pb-2 pt-4 px-6">
          <CardTitle className="text-sm font-semibold text-slate-700">{title}</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow className="bg-slate-50">
                <TableHead className="pl-6 font-semibold text-slate-600 w-[160px]">Company</TableHead>
                {columns.map((c) => (
                  <TableHead
                    key={String(c.key)}
                    className={cn(
                      "font-semibold text-slate-600 text-right pr-4",
                      c.align === "left" && "text-left"
                    )}
                  >
                    {c.header}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row, i) => (
                <TableRow
                  key={i}
                  className={cn(
                    row.is_target && "bg-blue-50/70 font-semibold",
                    !row.is_target && "hover:bg-slate-50/70"
                  )}
                >
                  <TableCell className="pl-6 py-2.5">
                    <span className="font-mono font-semibold text-slate-900 text-sm">
                      {row.ticker}
                    </span>
                    {row.company_name && (
                      <span className="text-[11px] text-slate-400 block leading-tight max-w-[150px] truncate">
                        {row.company_name}
                      </span>
                    )}
                  </TableCell>
                  {columns.map((c) => {
                    const val = row[c.key] as number | null | undefined;
                    return (
                      <TableCell
                        key={String(c.key)}
                        className={cn(
                          "text-right tabular-nums pr-4 text-sm",
                          c.align === "left" && "text-left"
                        )}
                      >
                        <span className={cn(val == null && "text-slate-300")}>
                          {c.format(val)}
                        </span>
                        {row.is_target && (
                          <RelBadge
                            value={val}
                            median={medians[String(c.key)]}
                            highIsGood={c.highIsGood}
                            isValuation={c.isValuation}
                          />
                        )}
                      </TableCell>
                    );
                  })}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
      {insight && (
        <p className="text-xs text-slate-500 italic mt-2 px-1">{insight}</p>
      )}
    </div>
  );
}

// ── Trend Summary card ────────────────────────────────────────────────────────

function TrendSummaryCard({ trends }: { trends: TrendResult | null }) {
  if (!trends) return null;
  const metrics = [
    { label: "Revenue Growth", trend: trends.revenue_growth, sig: trends.revenue_growth_sig },
    { label: "Op Margin",      trend: trends.op_margin,      sig: trends.op_margin_sig },
    { label: "Net Margin",     trend: trends.net_margin,     sig: trends.net_margin_sig },
    { label: "ROE",            trend: trends.roe,            sig: trends.roe_sig },
    { label: "ROIC",           trend: trends.roic,           sig: trends.roic_sig },
  ];
  const sigCls = (sig: string) =>
    sig === "↑" ? "text-green-600" :
    sig === "↓" ? "text-red-600" :
    sig === "⚠" ? "text-amber-600" : "text-slate-400";
  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-slate-700">Trend Summary</CardTitle>
      </CardHeader>
      <CardContent className="px-6 pb-5">
        <div className="grid grid-cols-5 gap-3">
          {metrics.map(m => (
            <div key={m.label} className="flex flex-col items-center gap-0.5">
              <span className={cn("text-xl font-bold", sigCls(m.sig))}>{m.sig}</span>
              <span className="text-[11px] font-medium text-slate-600 text-center leading-tight">{m.label}</span>
              <span className="text-[10px] text-slate-400">{m.trend}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ── Valuation Range Bar — four-zone distribution track, PDF-safe ─────────────

function ValuationRangeBar({
  bearPrice, basePrice, bullPrice, currentPrice, mc,
}: {
  bearPrice:    number | null | undefined;
  basePrice:    number | null | undefined;
  bullPrice:    number | null | undefined;
  currentPrice: number | null | undefined;
  mc?:          MCSim | null;
}) {
  if (!currentPrice) return null;

  const useMC = !!(mc && mc.p5_price > 0 && mc.p25_price > 0 &&
                   mc.median_price > 0 && mc.p75_price > 0 && mc.p95_price > 0);

  // Percentile anchors — MC primary, scenario fallback
  const p5  = useMC ? mc!.p5_price     : (bearPrice  != null ? bearPrice  * 0.78 : null);
  const p25 = useMC ? mc!.p25_price    : (bearPrice  ?? null);
  const p50 = useMC ? mc!.median_price : (basePrice  ?? null);
  const p75 = useMC ? mc!.p75_price    : (bullPrice  ?? null);
  const p95 = useMC ? mc!.p95_price    : (bullPrice  != null ? bullPrice  * 1.15 : null);

  if (!p25 || !p50 || !p75) return null;
  if (p75 <= p25) return null;

  const lo = p5  ?? p25 * 0.85;
  const hi = p95 ?? p75 * 1.15;

  // Axis: expand slightly beyond P5/P95, always containing current price
  const axisMin   = Math.min(lo, currentPrice) * 0.97;
  const axisMax   = Math.max(hi, currentPrice) * 1.03;
  const axisRange = axisMax - axisMin;
  if (axisRange <= 0) return null;

  // Percentage position along the axis (0–100)
  const pos = (v: number) =>
    Math.max(0, Math.min(100, ((v - axisMin) / axisRange) * 100));

  const p5pos  = pos(lo);
  const p25pos = pos(p25);
  const p50pos = pos(p50);
  const p75pos = pos(p75);
  const p95pos = pos(hi);
  const nowPos = pos(currentPrice);

  // Flex proportions for four zones (span from P5 to P95)
  const z1 = Math.max(0.5, p25pos - p5pos);    // Red:    Deep Value  (P5–P25)
  const z2 = Math.max(0.5, p50pos - p25pos);   // Yellow: Accumulation (P25–P50)
  const z3 = Math.max(0.5, p75pos - p50pos);   // Gray:   Fair Value  (P50–P75)
  const z4 = Math.max(0.5, p95pos - p75pos);   // Green:  Extended    (P75–P95)

  const fmtDelta = (v: number) => {
    const p = ((v / currentPrice) - 1) * 100;
    return `${p >= 0 ? '+' : ''}${p.toFixed(1)}%`;
  };

  // Current-price zone classification
  const zoneLabel = currentPrice <= p25 ? 'Deep Value'   :
                    currentPrice <= p50 ? 'Accumulation' :
                    currentPrice <= p75 ? 'Fair Value'   : 'Extended';
  const zoneFg    = currentPrice <= p25 ? '#b91c1c' :
                    currentPrice <= p50 ? '#a16207' :
                    currentPrice <= p75 ? '#475569' : '#15803d';
  const zoneBg    = currentPrice <= p25 ? '#fee2e2' :
                    currentPrice <= p50 ? '#fef9c3' :
                    currentPrice <= p75 ? '#f1f5f9' : '#dcfce7';

  const TRACK_H = 12;
  const TRACK_T = 24;
  const TICK_T  = TRACK_T - 5;
  const TICK_H  = TRACK_H + 10;
  const LABEL_T = TRACK_T + TRACK_H + 7;

  const priceGrid = useMC ? [
    { label: 'P25',     price: p25,          fg: '#dc2626', sub: 'Ideal Buy'   },
    { label: 'P50',     price: p50,          fg: '#475569', sub: 'Fair Value'  },
    { label: 'P75',     price: p75,          fg: '#16a34a', sub: 'Trim Zone'   },
    { label: 'Current', price: currentPrice, fg: '#0f172a', sub: 'Market price'},
  ] : [
    { label: 'Bear',    price: p25,          fg: '#dc2626', sub: 'Bear case'   },
    { label: 'Base',    price: p50,          fg: '#475569', sub: 'Base case'   },
    { label: 'Bull',    price: p75,          fg: '#16a34a', sub: 'Bull case'   },
    { label: 'Current', price: currentPrice, fg: '#0f172a', sub: 'Market price'},
  ];

  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-slate-700">
            {useMC ? 'Valuation Distribution' : 'Valuation Range'}
          </CardTitle>
          <span className="text-xs font-semibold px-2 py-0.5 rounded"
            style={{ backgroundColor: zoneBg, color: zoneFg, WebkitPrintColorAdjust: 'exact' }}>
            {zoneLabel}
          </span>
        </div>
      </CardHeader>
      <CardContent className="px-6 pb-5">
        {/* ── Four-zone distribution bar ────────────────────────── */}
        <div style={{ position: 'relative', height: `${LABEL_T + 14}px`, marginBottom: '4px' }}>

          {/* Flex zone bar — spans P5 to P95 along the axis */}
          <div style={{
            position: 'absolute',
            left: `${p5pos}%`, right: `${100 - p95pos}%`,
            top: `${TRACK_T}px`, height: `${TRACK_H}px`,
            borderRadius: '9999px', overflow: 'hidden',
            display: 'flex', WebkitPrintColorAdjust: 'exact',
          }}>
            <div style={{ flex: z1, backgroundColor: '#fca5a5', WebkitPrintColorAdjust: 'exact' }} />
            <div style={{ flex: z2, backgroundColor: '#fde68a', WebkitPrintColorAdjust: 'exact' }} />
            <div style={{ flex: z3, backgroundColor: '#e2e8f0', WebkitPrintColorAdjust: 'exact' }} />
            <div style={{ flex: z4, backgroundColor: '#86efac', WebkitPrintColorAdjust: 'exact' }} />
          </div>

          {/* P25 tick */}
          <div style={{ position: 'absolute', left: `${p25pos}%`,
            top: `${TICK_T}px`, width: '2px', height: `${TICK_H}px`,
            backgroundColor: '#ef4444', transform: 'translateX(-50%)',
            WebkitPrintColorAdjust: 'exact' }} />
          <span style={{ position: 'absolute', left: `${p25pos}%`, top: `${LABEL_T}px`,
            fontSize: '9px', color: '#dc2626', fontWeight: 600,
            transform: 'translateX(-50%)', whiteSpace: 'nowrap' }}>P25</span>

          {/* P50 tick */}
          <div style={{ position: 'absolute', left: `${p50pos}%`,
            top: `${TICK_T}px`, width: '2px', height: `${TICK_H}px`,
            backgroundColor: '#94a3b8', transform: 'translateX(-50%)',
            WebkitPrintColorAdjust: 'exact' }} />
          <span style={{ position: 'absolute', left: `${p50pos}%`, top: `${LABEL_T}px`,
            fontSize: '9px', color: '#475569', fontWeight: 600,
            transform: 'translateX(-50%)', whiteSpace: 'nowrap' }}>P50</span>

          {/* P75 tick */}
          <div style={{ position: 'absolute', left: `${p75pos}%`,
            top: `${TICK_T}px`, width: '2px', height: `${TICK_H}px`,
            backgroundColor: '#16a34a', transform: 'translateX(-50%)',
            WebkitPrintColorAdjust: 'exact' }} />
          <span style={{ position: 'absolute', left: `${p75pos}%`, top: `${LABEL_T}px`,
            fontSize: '9px', color: '#16a34a', fontWeight: 600,
            transform: 'translateX(-50%)', whiteSpace: 'nowrap' }}>P75</span>

          {/* Current price — bold dark bar + "Now" above */}
          <div style={{ position: 'absolute', left: `${nowPos}%`,
            top: `${TRACK_T - 10}px`, width: '3px', height: `${TRACK_H + 20}px`,
            backgroundColor: '#0f172a', transform: 'translateX(-50%)',
            borderRadius: '2px', WebkitPrintColorAdjust: 'exact' }} />
          <span style={{ position: 'absolute', left: `${nowPos}%`,
            top: `${TRACK_T - 19}px`, fontSize: '9px', color: '#0f172a', fontWeight: 700,
            transform: 'translateX(-50%)', whiteSpace: 'nowrap' }}>Now</span>
        </div>

        {/* Zone legend */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 14px', marginBottom: '12px' }}>
          {[
            { color: '#fca5a5', label: 'Deep Value' },
            { color: '#fde68a', label: 'Accumulation' },
            { color: '#e2e8f0', label: 'Fair Value' },
            { color: '#86efac', label: 'Extended' },
          ].map(({ color, label }) => (
            <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
              <div style={{ width: '8px', height: '8px', borderRadius: '2px', backgroundColor: color, flexShrink: 0, WebkitPrintColorAdjust: 'exact' }} />
              <span style={{ fontSize: '10px', color: '#94a3b8' }}>{label}</span>
            </div>
          ))}
        </div>

        {/* ── Price / upside grid ─────────────────────────────────── */}
        <div className="grid grid-cols-4 gap-3 border-t border-slate-100 pt-3">
          {priceGrid.map(({ label, price, fg, sub }) => (
            <div key={label}>
              <p className="text-[10px] text-slate-400 mb-0.5 font-medium">{label}</p>
              <p className="text-sm font-bold tabular-nums" style={{ color: fg }}>
                {formatPrice(price)}
              </p>
              <p className="text-[11px]" style={{ color: label === 'Current' ? '#94a3b8' : fg }}>
                {label === 'Current' ? sub : fmtDelta(price)}
              </p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ── Entry strategy logic engine ───────────────────────────────────────────────

interface EntryLevels {
  // Raw percentile anchors — traceable to MC or scenario mapping
  p5Price:         number;   // P5  = Bear scenario (or MC P5)
  p95Price:        number;   // P95 = Bull scenario (or MC P95)
  strongBuyLow:    number;   // P5
  strongBuyHigh:   number;   // P25 (momentum-adjusted)
  idealBuy:        number;   // P25 or P20 when risk-adjusted (momentum-adjusted)
  starterZoneLow:  number;   // P25 (momentum-adjusted)
  starterZoneHigh: number;   // P40 interpolated (momentum-adjusted)
  fairValue:       number;   // P50
  trimZone:        number;   // P75
  overvalued:      number;   // P95
  momAdjusted:     boolean;
  momDirection:    'down' | 'up' | 'none';
  riskAdjusted:    boolean;
  zone:            'strong_buy' | 'attractive' | 'above_fair' | 'trim';
  interpretation:  string;
  source:          'mc' | 'scenarios';
}

function computeEntryLevels(
  currentPrice:  number,
  bearFallback:  number,
  baseFallback:  number,
  bullFallback:  number,
  momScore:      number,
  riskScore:     number,
  mc?:           MCSim | null,
): EntryLevels {
  const useMC = !!(
    mc &&
    mc.p5_price   > 0 && mc.p25_price > 0 &&
    mc.median_price > 0 && mc.p75_price > 0 && mc.p95_price > 0
  );

  let p5: number, p25: number, p50: number, p75: number, p95: number;
  if (useMC) {
    p5  = mc!.p5_price;
    p25 = mc!.p25_price;
    p50 = mc!.median_price;
    p75 = mc!.p75_price;
    p95 = mc!.p95_price;
  } else {
    // Scenario fallback: Bear = P5, Base = P50, Bull = P95
    // (matches user rule: P5 ≈ Bear, P50 ≈ Base, P95 ≈ Bull)
    //
    // P25 and P75 are linearly interpolated from their percentile positions:
    //   P25 is (25−5)/(50−5) = 20/45 of the way from Bear(P5) to Base(P50)
    //   P75 is (75−50)/(95−50) = 25/45 of the way from Base(P50) to Bull(P95)
    p5  = bearFallback;
    p25 = bearFallback + (baseFallback - bearFallback) * (20 / 45);
    p50 = baseFallback;
    p75 = baseFallback + (bullFallback - baseFallback) * (25 / 45);
    p95 = bullFallback;
  }

  // Interpolated percentiles — weights derived from percentile positions:
  //   P20 is (20−5)/(25−5) = 15/20 = 0.75 of the way from P5 to P25
  //   P40 is (40−25)/(50−25) = 15/25 = 0.60 of the way from P25 to P50
  const p20 = p5  + 0.75 * (p25 - p5);
  const p40 = p25 + 0.60 * (p50 - p25);

  // Momentum and risk flags (informational — do NOT shift levels above currentPrice)
  const momDown      = momScore < 50;
  const momUp        = momScore > 70;
  const momDirection: 'down' | 'up' | 'none' = momDown ? 'down' : momUp ? 'up' : 'none';
  const riskAdj      = riskScore < 60;

  // Zone classification (raw percentiles only)
  const zone: EntryLevels['zone'] =
    currentPrice < p25  ? 'strong_buy' :
    currentPrice <= p50 ? 'attractive'  :
    currentPrice <= p75 ? 'above_fair'  : 'trim';

  // ── Current-price-anchored entry levels ──────────────────────────────────
  // CORE RULE: no "buy" level may exceed currentPrice.
  // Use P20 as the ideal anchor for high-risk names (riskScore < 60).
  const adjP25 = riskAdj ? p20 : p25;

  let strongBuyLow:    number;
  let strongBuyHigh:   number;
  let idealBuy:        number;
  let starterZoneLow:  number;
  let starterZoneHigh: number;

  if (zone === 'strong_buy') {
    // Current IS in the strong buy zone — price is already below P25.
    // Strong Buy Zone: P5 → currentPrice (we're here already)
    // Ideal Buy:       currentPrice — buy immediately, no need to wait
    // Starter Zone:    currentPrice → P40 (acceptable to keep adding up to P40)
    strongBuyLow    = Math.max(0, p5);
    strongBuyHigh   = currentPrice;                          // cap at current
    idealBuy        = currentPrice;                          // buy NOW
    starterZoneLow  = currentPrice;
    starterZoneHigh = momDown
      ? currentPrice                            // weak momentum: no upward starter zone
      : Math.min(p40, currentPrice * 1.05);    // normal: max 5% above current
  } else if (zone === 'attractive') {
    // P25 ≤ current ≤ P50 — good entry but not deep value.
    // Strong Buy Zone: P5 → adjP25 (only on a meaningful pullback)
    // Ideal Buy:       adjP25 (target pullback to P25)
    // Starter Zone:    adjP25 → currentPrice (acceptable to start anywhere here)
    strongBuyLow    = Math.max(0, p5);
    strongBuyHigh   = Math.min(adjP25, currentPrice);      // always ≤ current
    idealBuy        = Math.min(adjP25, currentPrice);      // always ≤ current
    starterZoneLow  = Math.min(adjP25, currentPrice);
    starterZoneHigh = currentPrice;                        // up to but not above current
  } else {
    // current > P50 — above fair value; no strong buy zone.
    // Ideal Buy:    P40–P50 midpoint (requires a real pullback)
    // Starter Zone: P25 → P50 (pullback range), both capped at current
    strongBuyLow    = 0;
    strongBuyHigh   = 0;
    idealBuy        = Math.min((p40 + p50) / 2, currentPrice); // always ≤ current
    starterZoneLow  = Math.min(p25, currentPrice);
    starterZoneHigh = Math.min(p50, currentPrice);             // always ≤ current
  }

  const fairValue  = p50;
  const trimZone   = p75;
  const overvalued = p95;

  // Upside/downside skew label for interpretation
  const udLabel = mc
    ? mc.upside_downside >= 2.5 ? 'asymmetric upside'
    : mc.upside_downside >= 1.5 ? 'favourable skew'
    : mc.upside_downside >= 1.0 ? 'balanced risk/reward'
    : 'downside-skewed'
    : '';

  const pctRaw     = ((fairValue - currentPrice) / currentPrice * 100).toFixed(1);
  const aboveBelow = currentPrice < fairValue
    ? `${pctRaw}% below fair value`
    : `${Math.abs(Number(pctRaw))}% above fair value`;

  let interpretation: string;
  if (zone === 'strong_buy') {
    interpretation = `Trading in the strong buy zone — below the P25 anchor${udLabel ? ` with ${udLabel}` : ''}. Current price is the ideal entry; initiate a position and scale in systematically across tranches.`;
  } else if (zone === 'attractive') {
    interpretation = `Current price is ${aboveBelow}${udLabel ? `, ${udLabel}` : ''}. Attractive entry — a starter position at current is justified; the ideal target is a pullback to P25.`;
  } else if (zone === 'above_fair') {
    interpretation = `Trading above the P50 fair value anchor${udLabel ? ` with ${udLabel}` : ''}. Wait for a pullback toward P40–P50; initiating here requires high conviction in the bull case.`;
  } else {
    interpretation = `Price is above P75 — the distribution suggests limited upside relative to downside. Trim existing positions and avoid new entries.`;
  }

  return {
    p5Price: p5, p95Price: p95,
    strongBuyLow, strongBuyHigh, idealBuy,
    starterZoneLow, starterZoneHigh,
    fairValue, trimZone, overvalued,
    momAdjusted:  momDown || momUp,
    momDirection,
    riskAdjusted: riskAdj,
    zone,
    interpretation,
    source: useMC ? 'mc' : 'scenarios',
  };
}

// ── Decision Summary engine ───────────────────────────────────────────────────

type ValuationStatus = 'Deep Value' | 'Attractive' | 'Fair Value' | 'Extended';
type ExecutionStatus = 'BUY NOW' | 'STAGED BUY' | 'WAIT' | 'HOLD' | 'TRIM' | 'EXIT';

interface DecisionSummary {
  // Thesis (long-term fundamental view)
  thesisRating:         'Buy' | 'Hold' | 'Sell';
  thesisRatingBg:       string;
  thesisRatingFg:       string;
  // Valuation zone
  valuationStatus:      ValuationStatus;
  valuationBg:          string;
  valuationFg:          string;
  // Execution (current action)
  executionStatus:      ExecutionStatus;
  executionBg:          string;
  executionFg:          string;
  // Conviction
  conviction:           'High' | 'Medium' | 'Low';
  // Sizing
  targetPct:            string;
  currentPct:           string;
  starterPct?:          string;
  sizingNote:           string;
  // Triggers
  buyTrigger:           string;
  addTrigger:           string;
  trimTrigger:          string;
  exitTrigger:          string;
}

function computeDecisionSummary(
  lv:           EntryLevels,
  currentPrice: number,
  momScore:     number,
  riskScore:    number,
  mc?:          MCSim | null,
  stance?:      string,    // 'Bullish' | 'Neutral' | 'Bearish'
): DecisionSummary {
  // rrRatio = (P95 - price) / (price - P5)  [Rule: Return = (Target − Price) / Price]
  // Prefer MC percentiles when available; fall back to scenario P5/P95 from EntryLevels
  // so rrRatio is never a silent zero when only scenario prices are present.
  const _p5px  = (mc && mc.p5_price  > 0) ? mc.p5_price  : lv.p5Price;
  const _p95px = (mc && mc.p95_price > 0) ? mc.p95_price : lv.p95Price;
  const rrRatio = (currentPrice > 0 && _p5px > 0 && _p95px > currentPrice)
    ? Math.max(0.001, (_p95px - currentPrice) / currentPrice) /
      Math.max(0.001, (currentPrice - _p5px)  / currentPrice)
    : 0;
  const probPos = mc?.prob_positive ?? 0.5;
  const fp = formatPrice;

  // ── Thesis Rating (long-term fundamental view — from backend stance) ──────
  const thesisRating: 'Buy' | 'Hold' | 'Sell' =
    stance === 'Bullish' ? 'Buy' :
    stance === 'Bearish' ? 'Sell' : 'Hold';

  const _thesisColors: Record<'Buy' | 'Hold' | 'Sell', { bg: string; fg: string }> = {
    Buy:  { bg: '#15803d', fg: '#ffffff' },
    Hold: { bg: '#475569', fg: '#ffffff' },
    Sell: { bg: '#dc2626', fg: '#ffffff' },
  };
  const { bg: thesisRatingBg, fg: thesisRatingFg } = _thesisColors[thesisRating];

  // ── Valuation Status ──────────────────────────────────────────────────────
  const valuationStatus: ValuationStatus =
    lv.zone === 'strong_buy' ? 'Deep Value' :
    lv.zone === 'attractive' ? 'Attractive' :
    lv.zone === 'above_fair' ? 'Fair Value' : 'Extended';

  const _valColors: Record<ValuationStatus, { bg: string; fg: string }> = {
    'Deep Value': { bg: '#dcfce7', fg: '#15803d' },
    'Attractive': { bg: '#dbeafe', fg: '#1d4ed8' },
    'Fair Value': { bg: '#f1f5f9', fg: '#475569' },
    'Extended':   { bg: '#fee2e2', fg: '#b91c1c' },
  };
  const { bg: valuationBg, fg: valuationFg } = _valColors[valuationStatus];

  // ── Execution Status (ordered cascade — sell discipline highest) ──────────
  let executionStatus: ExecutionStatus;
  if      (currentPrice >= lv.overvalued)                                   executionStatus = 'EXIT';
  else if (lv.zone === 'trim')                                               executionStatus = 'TRIM';
  else if (momScore < 35)                                                    executionStatus = 'WAIT';
  else if (lv.zone === 'above_fair')                                        executionStatus = 'HOLD';
  else if (momScore < 50 && lv.zone === 'strong_buy')                       executionStatus = 'STAGED BUY';
  else if (momScore < 50)                                                    executionStatus = 'WAIT';
  else if (lv.zone === 'strong_buy' && riskScore >= 50)                     executionStatus = 'BUY NOW';
  else                                                                       executionStatus = 'STAGED BUY';

  const _execColors: Record<ExecutionStatus, { bg: string; fg: string }> = {
    'BUY NOW':    { bg: '#15803d', fg: '#ffffff' },
    'STAGED BUY': { bg: '#1d4ed8', fg: '#ffffff' },
    'WAIT':       { bg: '#fef9c3', fg: '#92400e' },
    'HOLD':       { bg: '#f1f5f9', fg: '#475569' },
    'TRIM':       { bg: '#fed7aa', fg: '#92400e' },
    'EXIT':       { bg: '#fee2e2', fg: '#b91c1c' },
  };
  const { bg: executionBg, fg: executionFg } = _execColors[executionStatus];

  // ── Conviction ────────────────────────────────────────────────────────────
  const conviction: 'High' | 'Medium' | 'Low' = (() => {
    if (executionStatus === 'EXIT' || executionStatus === 'TRIM' ||
        executionStatus === 'WAIT' || executionStatus === 'HOLD') return 'Low';
    if (riskScore >= 65 && momScore >= 60 && rrRatio >= 2.5 && probPos >= 0.65) return 'High';
    if (riskScore >= 50 && momScore >= 50 && rrRatio >= 1.5)                     return 'Medium';
    return 'Low';
  })();

  // ── Target size (qualitative Kelly proxy) ─────────────────────────────────
  const targetNum = (() => {
    if (executionStatus === 'EXIT' || executionStatus === 'TRIM') return 0;
    if (conviction === 'High')   return rrRatio >= 3.0 ? 4.5 : 3.5;
    if (conviction === 'Medium') return rrRatio >= 2.0 ? 3.0 : 2.0;
    return 1.5;
  })();

  const targetPct = targetNum > 0 ? `${targetNum.toFixed(1)}%` : '—';

  // ── Current recommended (aligned with execution — no contradictions) ──────
  // Rule: Low conviction → cap Recommended Now at 1.0%
  const currentNum = (() => {
    switch (executionStatus) {
      case 'EXIT':       return 0;
      case 'TRIM':       return 0;
      case 'WAIT':       return 0;
      case 'HOLD':       return conviction === 'Low' ? Math.min(targetNum, 1.0) : targetNum;
      case 'BUY NOW':
        if (conviction === 'High')   return targetNum;
        if (conviction === 'Medium') return Math.round(targetNum * 0.6 * 2) / 2;
        // Low: small starter, hard cap 1.0%
        return Math.min(1.0, Math.round(targetNum * 0.35 * 2) / 2);
      case 'STAGED BUY': {
        const phase1 = Math.round((targetNum / 3) * 2) / 2;
        // Low conviction: phase 1 capped at 1.0%
        return conviction === 'Low' ? Math.min(1.0, phase1) : phase1;
      }
    }
  })();

  const currentPct = (() => {
    if (executionStatus === 'TRIM') return 'Reduce 25–50%';
    if (executionStatus === 'EXIT') return 'Exit 75–100%';
    return currentNum > 0 ? `${currentNum.toFixed(1)}%` : '0%';
  })();

  const starterPct = (() => {
    if (executionStatus === 'WAIT') {
      const s = Math.round(targetNum * 0.25 * 2) / 2;
      return s > 0 ? `${s.toFixed(1)}% on confirmation` : undefined;
    }
    if (executionStatus === 'STAGED BUY') {
      return `+${currentNum.toFixed(1)}% / tranche (3 phases)`;
    }
    return undefined;
  })();

  // ── Sizing narrative — language strictly bound to conviction + execution ────
  // Rule: STAGED BUY / WAIT must never use "full", "aggressive", or "high conviction" language
  const sizingNote = (() => {
    switch (executionStatus) {
      case 'BUY NOW':
        if (conviction === 'High')   return 'Full allocation justified. Enter at current price.';
        if (conviction === 'Medium') return 'Initiate partial position now; scale to target over 2 weeks as thesis confirms.';
        return 'Initiate small starter position only. Scale in gradually as momentum and risk improve.';
      case 'STAGED BUY':
        if (conviction === 'Low')
          return `Initiate partial position (${currentNum.toFixed(1)}%) — scale in gradually over 3 tranches, waiting for confirmation at each stage.`;
        return `Phase 1 of 3: ${currentNum.toFixed(1)}% now → add on each 5–10% pullback or after 2-week thesis hold.`;
      case 'WAIT':
        return `Hold at 0%. Wait for confirmation before sizing. Starter ${starterPct ?? 'position on trigger'}.`;
      case 'HOLD':
        return 'Maintain current position. No new buys above fair value (P50).';
      case 'TRIM':
        return 'Reduce exposure now. Risk/reward deteriorates above P75.';
      case 'EXIT':
        return 'Exit majority of position. Distribution fully priced in at P95+.';
    }
  })();

  // ── Triggers ──────────────────────────────────────────────────────────────
  const buyTrigger = (() => {
    if (executionStatus === 'BUY NOW')    return `Enter at current price (${fp(currentPrice)}) — in strong buy zone now.`;
    if (executionStatus === 'STAGED BUY') return `Phase 1: initiate at current price (${fp(currentPrice)}).`;
    if (executionStatus === 'WAIT')       return `Momentum score ≥ 50 OR pullback to ${fp(lv.idealBuy)} (P25).`;
    if (executionStatus === 'HOLD')       return `Pullback to ${fp(lv.idealBuy)} (P25–P40) with fundamentals intact.`;
    return `Pullback to strong buy zone below ${fp(lv.idealBuy)} with improving fundamentals.`;
  })();

  const addTrigger = (() => {
    if (executionStatus === 'BUY NOW')    return `5–10% pullback from entry OR 2-week hold with thesis intact.`;
    if (executionStatus === 'STAGED BUY') return `Phase 2: 5–10% dip from Phase 1. Phase 3: trend confirmation.`;
    if (executionStatus === 'WAIT')       return `After buy trigger fires — add on next 5–10% pullback.`;
    if (executionStatus === 'HOLD')       return `At ${fp(lv.idealBuy)} on pullback; no adds above ${fp(lv.fairValue)} (P50).`;
    return `On pullback to ${fp(lv.idealBuy)} with fundamentals intact.`;
  })();

  const trimTrigger = executionStatus === 'TRIM'
    ? `Now — price already above trim zone (P75 = ${fp(lv.trimZone)}).`
    : `Price reaches ${fp(lv.trimZone)} (P75) — trim 25–50% of position.`;

  const exitTrigger = executionStatus === 'EXIT'
    ? `Now — price at or above P95 (${fp(lv.overvalued)}).`
    : `Price reaches ${fp(lv.overvalued)} (P95) or fundamental thesis break.`;

  return {
    thesisRating, thesisRatingBg, thesisRatingFg,
    valuationStatus, valuationBg, valuationFg,
    executionStatus, executionBg, executionFg,
    conviction,
    targetPct, currentPct, starterPct, sizingNote,
    buyTrigger, addTrigger, trimTrigger, exitTrigger,
  };
}

// ── Entry Strategy card ───────────────────────────────────────────────────────

function EntryStrategyCard({
  vr, currentPrice, momScore, riskScore, mc, stance,
}: {
  vr:           ValuationRange;
  currentPrice: number | null;
  momScore:     number;
  riskScore:    number;
  mc?:          MCSim | null;
  stance?:      string;
}) {
  if (
    !vr.available || !currentPrice ||
    vr.bear_price == null || vr.base_price == null || vr.bull_price == null
  ) return null;

  const lv = computeEntryLevels(
    currentPrice, vr.bear_price, vr.base_price, vr.bull_price, momScore, riskScore, mc,
  );
  const entryDS = computeDecisionSummary(lv, currentPrice, momScore, riskScore, mc, stance);

  const zoneMeta: Record<EntryLevels['zone'], { label: string; bg: string; fg: string }> = {
    strong_buy: { label: 'Strong Buy Zone',  bg: '#dcfce7', fg: '#15803d' },
    attractive: { label: 'Attractive Entry', bg: '#dbeafe', fg: '#1d4ed8' },
    above_fair: { label: 'Above Fair Value', bg: '#fef9c3', fg: '#a16207' },
    trim:       { label: 'Trim Zone',        bg: '#fee2e2', fg: '#b91c1c' },
  };
  const zm = zoneMeta[lv.zone];

  const rows: { label: string; value: string; note: string; bold: boolean; highlight?: boolean; hidden?: boolean }[] = [
    {
      label: 'Current Price',
      value: formatPrice(currentPrice),
      note:  'Market price',
      bold:  false,
    },
    {
      label:     'Strong Buy Zone',
      value:     `${formatPrice(lv.strongBuyLow)} – ${formatPrice(lv.strongBuyHigh)}`,
      note:      'P5 → P25',
      bold:      true,
      highlight: lv.zone === 'strong_buy',
      hidden:    lv.strongBuyLow === 0,
    },
    {
      label: 'Ideal Buy Price',
      value: formatPrice(lv.idealBuy),
      note:  lv.riskAdjusted ? 'P20 (risk-adj.)' : 'P25',
      bold:  true,
    },
    {
      label:     'Starter Buy Zone',
      value:     `${formatPrice(lv.starterZoneLow)} – ${formatPrice(lv.starterZoneHigh)}`,
      note:      lv.zone === 'strong_buy' ? 'Current → P40' : lv.zone === 'attractive' ? 'P25 → Current' : 'Pullback range',
      bold:      false,
      highlight: lv.zone === 'attractive',
      hidden:    momScore < 50 && lv.zone === 'strong_buy',
    },
    {
      label: 'Fair Value',
      value: formatPrice(lv.fairValue),
      note:  'P50 anchor',
      bold:  false,
    },
    {
      label: 'Trim Zone',
      value: `${formatPrice(lv.trimZone)}+`,
      note:  'P75 — reduce',
      bold:  false,
    },
    {
      label: 'Overvalued',
      value: `${formatPrice(lv.overvalued)}+`,
      note:  'P95+',
      bold:  false,
    },
  ];

  // Execution priority — based on current price vs distribution percentiles
  const priority = lv.zone === 'strong_buy' ? 'HIGH' : lv.zone === 'attractive' ? 'MEDIUM' : 'LOW';
  const priorityMeta =
    priority === 'HIGH'   ? { label: 'HIGH PRIORITY',   bg: '#dcfce7', fg: '#15803d' } :
    priority === 'MEDIUM' ? { label: 'MEDIUM PRIORITY', bg: '#dbeafe', fg: '#1d4ed8' } :
                            { label: 'LOW PRIORITY',    bg: '#f1f5f9', fg: '#64748b' };

  return (
    <Card className="border-slate-200 shadow-sm">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-sm font-semibold text-slate-700">
            Entry Strategy &amp; Price Levels
          </CardTitle>
          <div className="flex items-center gap-1.5 flex-wrap">
            <span className="text-xs font-semibold px-2 py-0.5 rounded"
              style={{ backgroundColor: zm.bg, color: zm.fg, WebkitPrintColorAdjust: 'exact' }}>
              {zm.label}
            </span>
            <span className="text-xs font-bold px-2 py-0.5 rounded"
              style={{ backgroundColor: priorityMeta.bg, color: priorityMeta.fg, WebkitPrintColorAdjust: 'exact' }}>
              {priorityMeta.label}
            </span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="px-0 pb-0">
        {/* ── Decision Summary ── */}
        {(() => {
          const ds = computeDecisionSummary(lv, currentPrice, momScore, riskScore, mc, stance);
          const convBadge: Record<DecisionSummary['conviction'], { bg: string; fg: string }> = {
            High:   { bg: '#dcfce7', fg: '#15803d' },
            Medium: { bg: '#dbeafe', fg: '#1d4ed8' },
            Low:    { bg: '#f1f5f9', fg: '#64748b' },
          };
          const cb = convBadge[ds.conviction];
          return (
            <div style={{ padding: '12px 24px 14px', borderBottom: '1px solid #f1f5f9', backgroundColor: '#fafafa' }}>
              {/* Section label */}
              <span style={{ fontSize: '10px', fontWeight: 700, letterSpacing: '0.06em', color: '#94a3b8', textTransform: 'uppercase' as const, display: 'block', marginBottom: '8px' }}>
                Decision Summary
              </span>
              {/* Status badges — Thesis | Valuation | Execution | Conviction */}
              <div style={{ display: 'flex', alignItems: 'flex-end', gap: '10px', flexWrap: 'wrap' as const, marginBottom: '10px' }}>
                <div>
                  <div style={{ fontSize: '9px', color: '#94a3b8', textTransform: 'uppercase' as const, letterSpacing: '0.05em', marginBottom: '3px' }}>Thesis</div>
                  <span style={{ fontSize: '12px', fontWeight: 700, padding: '2px 10px', borderRadius: '9999px', backgroundColor: ds.thesisRatingBg, color: ds.thesisRatingFg, whiteSpace: 'nowrap' as const, WebkitPrintColorAdjust: 'exact' as const }}>
                    {ds.thesisRating}
                  </span>
                </div>
                <div style={{ width: '1px', height: '28px', backgroundColor: '#e2e8f0', alignSelf: 'flex-end', marginBottom: '1px' }} />
                <div>
                  <div style={{ fontSize: '9px', color: '#94a3b8', textTransform: 'uppercase' as const, letterSpacing: '0.05em', marginBottom: '3px' }}>Valuation</div>
                  <span style={{ fontSize: '11px', fontWeight: 600, padding: '2px 9px', borderRadius: '9999px', backgroundColor: ds.valuationBg, color: ds.valuationFg, whiteSpace: 'nowrap' as const, WebkitPrintColorAdjust: 'exact' as const }}>
                    {ds.valuationStatus}
                  </span>
                </div>
                <div>
                  <div style={{ fontSize: '9px', color: '#94a3b8', textTransform: 'uppercase' as const, letterSpacing: '0.05em', marginBottom: '3px' }}>Execution</div>
                  <span style={{ fontSize: '13px', fontWeight: 800, padding: '3px 12px', borderRadius: '9999px', letterSpacing: '0.04em', backgroundColor: ds.executionBg, color: ds.executionFg, whiteSpace: 'nowrap' as const, WebkitPrintColorAdjust: 'exact' as const }}>
                    {ds.executionStatus}
                  </span>
                </div>
                <div>
                  <div style={{ fontSize: '9px', color: '#94a3b8', textTransform: 'uppercase' as const, letterSpacing: '0.05em', marginBottom: '3px' }}>Conviction</div>
                  <span style={{ fontSize: '11px', fontWeight: 600, padding: '2px 9px', borderRadius: '9999px', backgroundColor: cb.bg, color: cb.fg, whiteSpace: 'nowrap' as const, WebkitPrintColorAdjust: 'exact' as const }}>
                    {ds.conviction}
                  </span>
                </div>
              </div>
              {/* Position sizing grid */}
              <div style={{ display: 'grid', gridTemplateColumns: ds.starterPct ? '1fr 1fr 1fr' : '1fr 1fr', gap: '8px', marginBottom: '8px', padding: '8px 10px', backgroundColor: '#ffffff', borderRadius: '8px', border: '1px solid #f1f5f9' }}>
                <div>
                  <div style={{ fontSize: '9px', color: '#94a3b8', textTransform: 'uppercase' as const, letterSpacing: '0.04em', marginBottom: '2px' }}>Target Size</div>
                  <div style={{ fontSize: '16px', fontWeight: 700, color: '#0f172a' }}>{ds.targetPct}</div>
                </div>
                <div>
                  <div style={{ fontSize: '9px', color: '#94a3b8', textTransform: 'uppercase' as const, letterSpacing: '0.04em', marginBottom: '2px' }}>Recommended Now</div>
                  <div style={{ fontSize: '15px', fontWeight: 700, color: (ds.executionStatus === 'WAIT' || ds.executionStatus === 'EXIT') ? '#94a3b8' : '#0f172a' }}>{ds.currentPct}</div>
                </div>
                {ds.starterPct && (
                  <div>
                    <div style={{ fontSize: '9px', color: '#94a3b8', textTransform: 'uppercase' as const, letterSpacing: '0.04em', marginBottom: '2px' }}>
                      {ds.executionStatus === 'STAGED BUY' ? 'Add Structure' : 'On Confirmation'}
                    </div>
                    <div style={{ fontSize: '11px', fontWeight: 600, color: '#475569' }}>{ds.starterPct}</div>
                  </div>
                )}
              </div>
              {/* Sizing note */}
              <p style={{ fontSize: '11px', color: '#1e293b', lineHeight: 1.5, margin: '0 0 10px', fontWeight: 500 }}>
                {ds.sizingNote}
              </p>
              {/* Trigger grid */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 16px' }}>
                {([
                  { label: 'Buy Trigger',  value: ds.buyTrigger,  fg: '#15803d' },
                  { label: 'Add Trigger',  value: ds.addTrigger,  fg: '#1d4ed8' },
                  { label: 'Trim Trigger', value: ds.trimTrigger, fg: '#a16207' },
                  { label: 'Exit Trigger', value: ds.exitTrigger, fg: '#b91c1c' },
                ] as const).map(({ label, value, fg }) => (
                  <div key={label} style={{ paddingBottom: '4px' }}>
                    <span style={{ fontSize: '9px', fontWeight: 700, color: fg, textTransform: 'uppercase' as const, letterSpacing: '0.05em', display: 'block', marginBottom: '2px' }}>
                      {label}
                    </span>
                    <span style={{ fontSize: '10px', color: '#475569', lineHeight: 1.4 }}>
                      {value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          );
        })()}
        {/* ── Entry zone mini-bar ──────────────────────────────────── */}
        {(() => {
          if (!mc || mc.p5_price <= 0 || mc.p95_price <= 0 || lv.source !== 'mc') return null;
          const mxMin = mc.p5_price  * 0.98;
          const mxMax = mc.p95_price * 1.02;
          const mxR   = mxMax - mxMin;
          if (mxR <= 0) return null;
          const tp   = (v: number) => Math.max(0, Math.min(100, ((v - mxMin) / mxR) * 100));
          const z1   = Math.max(0.5, tp(mc.p25_price)    - tp(mc.p5_price));
          const z2   = Math.max(0.5, tp(mc.median_price) - tp(mc.p25_price));
          const z3   = Math.max(0.5, tp(mc.p75_price)    - tp(mc.median_price));
          const z4   = Math.max(0.5, tp(mc.p95_price)    - tp(mc.p75_price));
          const nowP = tp(currentPrice);
          const p5p  = tp(mc.p5_price);
          const p95p = tp(mc.p95_price);
          return (
            <div style={{ padding: '10px 24px 2px' }}>
              <div style={{ position: 'relative', height: '28px' }}>
                <div style={{
                  position: 'absolute', left: `${p5p}%`, right: `${100 - p95p}%`,
                  top: '10px', height: '8px',
                  borderRadius: '9999px', overflow: 'hidden', display: 'flex',
                  WebkitPrintColorAdjust: 'exact',
                }}>
                  <div style={{ flex: z1, backgroundColor: '#fca5a5', WebkitPrintColorAdjust: 'exact' }} />
                  <div style={{ flex: z2, backgroundColor: '#fde68a', WebkitPrintColorAdjust: 'exact' }} />
                  <div style={{ flex: z3, backgroundColor: '#e2e8f0', WebkitPrintColorAdjust: 'exact' }} />
                  <div style={{ flex: z4, backgroundColor: '#86efac', WebkitPrintColorAdjust: 'exact' }} />
                </div>
                <div style={{
                  position: 'absolute', left: `${nowP}%`, top: '4px',
                  width: '3px', height: '20px', backgroundColor: '#0f172a',
                  transform: 'translateX(-50%)', borderRadius: '2px',
                  WebkitPrintColorAdjust: 'exact',
                }} />
                <span style={{
                  position: 'absolute', left: `${nowP}%`, top: '0px',
                  fontSize: '8px', fontWeight: 700, color: '#0f172a',
                  transform: 'translateX(-50%)', whiteSpace: 'nowrap',
                }}>NOW</span>
              </div>
            </div>
          );
        })()}
        <Table>
          <TableBody>
            {rows.filter(r => !r.hidden).map(({ label, value, note, bold, highlight }) => (
              <TableRow key={label} className={highlight ? 'bg-green-50' : bold ? 'bg-slate-50' : ''}>
                <TableCell className="pl-6 text-sm text-slate-600 py-2.5 w-1/3">{label}</TableCell>
                <TableCell className={cn(
                  'text-right tabular-nums text-sm py-2.5',
                  bold ? 'font-bold text-slate-900' : 'text-slate-700',
                )}>
                  {value}
                </TableCell>
                <TableCell className="pr-6 text-right text-[11px] text-slate-400 py-2.5">
                  {note}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {/* Interpretation */}
        <div className="px-6 py-3 border-t border-slate-100">
          <p className="text-xs text-slate-600 leading-relaxed">{sanitizeNarrative(lv.interpretation, entryDS)}</p>
        </div>
        {/* ── Sell Discipline ── */}
        <div className="mx-6 mb-3 rounded-lg border border-slate-200 overflow-hidden">
          <div className="px-3 py-1.5 bg-slate-50 border-b border-slate-200">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
              Sell Discipline
            </span>
          </div>
          <div className="px-3 py-2 grid grid-cols-2 gap-2">
            <div>
              <p className="text-[10px] text-slate-400 mb-0.5">Trim Zone (P75)</p>
              <p className="text-sm font-bold" style={{ color: '#a16207' }}>
                {formatPrice(lv.trimZone)}+
              </p>
              <p className="text-[10px] text-slate-400 mt-0.5">Trim 25–50%</p>
            </div>
            <div>
              <p className="text-[10px] text-slate-400 mb-0.5">Exit Zone (P95)</p>
              <p className="text-sm font-bold" style={{ color: '#b91c1c' }}>
                {formatPrice(lv.overvalued)}+
              </p>
              <p className="text-[10px] text-slate-400 mt-0.5">Exit majority</p>
            </div>
          </div>
          <div className="px-3 pb-2">
            <p className="text-[10px] text-slate-400 leading-relaxed">
              Upside becomes limited beyond P75; risk/reward deteriorates materially above P95.
            </p>
          </div>
        </div>
        {/* Adjustment badges */}
        {(lv.momAdjusted || lv.riskAdjusted) && (
          <div className="px-6 pb-3 flex flex-wrap gap-2">
            {lv.momAdjusted && (
              <span className="text-[11px] px-2 py-0.5 rounded"
                style={{
                  backgroundColor: lv.momDirection === 'down' ? '#fef9c3' : '#dbeafe',
                  color:           lv.momDirection === 'down' ? '#a16207' : '#1d4ed8',
                  WebkitPrintColorAdjust: 'exact',
                }}>
                {lv.momDirection === 'down'
                  ? 'Weak momentum (score < 50) — wait for confirmation before sizing'
                  : 'Strong momentum (score > 70) — trend supports entry'}
              </span>
            )}
            {lv.riskAdjusted && (
              <span className="text-[11px] px-2 py-0.5 rounded"
                style={{ backgroundColor: '#fee2e2', color: '#b91c1c', WebkitPrintColorAdjust: 'exact' }}>
                P20 entry (risk score &lt; 60)
              </span>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Distribution-based position sizing engine ────────────────────────────────

interface DistribSizing {
  rawKelly:       number;
  scaledKelly:    number;
  positionPct:    number;
  tier:           'small' | 'medium' | 'core' | 'high_conviction';
  entryStyle:     'aggressive' | 'staged' | 'wait_for_pullback';
  rrRatio:        number;
  downside:       number;
  upside:         number;
  kellyScale:     'aggressive' | 'standard' | 'conservative';
  reductions:     string[];
  increases:      string[];
  interpretation: string;
}

function computeDistribSizing(
  currentPrice: number,
  mc: MCSim,
  momScore: number,
  riskScore: number,
  beta: number
): DistribSizing {
  const clamp = (lo: number, hi: number, v: number) => Math.max(lo, Math.min(hi, v));

  const p5  = mc.p5_price;
  const p95 = mc.p95_price;
  const p50 = mc.median_price ?? mc.mean_price;

  const downside = Math.max(0.001, (currentPrice - p5) / currentPrice);
  const upside   = Math.max(0.001, (p95 - currentPrice) / currentPrice);
  const rrRatio  = upside / downside;

  const p = clamp(0.01, 0.99, mc.prob_positive ?? 0.55);
  const q = 1 - p;
  const b = Math.max(0.01, rrRatio);
  const rawKelly = Math.max(0, p - q / b);

  // Kelly scale selection
  const spread = upside / downside;   // same as rrRatio — alias for readability
  let kellyScale: 'aggressive' | 'standard' | 'conservative';
  if (rrRatio >= 2.5 && spread >= 2.0 && p >= 0.65 && riskScore >= 65) {
    kellyScale = 'aggressive';
  } else if (rrRatio < 1.2 || spread < 1.0 || p < 0.45) {
    kellyScale = 'conservative';
  } else {
    kellyScale = 'standard';
  }
  const scaleMultiplier = kellyScale === 'aggressive' ? 0.5 : kellyScale === 'standard' ? 0.25 : 0.1;
  let scaled = rawKelly * scaleMultiplier;

  // Reductions
  const reductions: string[] = [];
  const increases:  string[] = [];

  if (beta > 1.5) {
    scaled *= 0.75;
    reductions.push(`High beta (${beta.toFixed(2)}) −25%`);
  }
  if (riskScore < 65) {
    const rfactor = riskScore < 50 ? 0.70 : 0.80;   // −30% / −20%
    scaled *= rfactor;
    reductions.push(`Risk score ${riskScore} ${rfactor === 0.70 ? '−30%' : '−20%'}`);
  }
  if (momScore < 50) {
    const mfactor = momScore < 40 ? 0.60 : 0.75;    // −40% / −25%
    scaled *= mfactor;
    reductions.push(`Weak momentum (${momScore}) ${mfactor === 0.60 ? '−40%' : '−25%'}`);
  }
  // Wide distribution: total P5→P95 span > 120% of current price
  if ((upside + downside) > 1.2) {
    scaled *= 0.80;
    reductions.push('Wide distribution −20%');
  }

  // Boosts
  if (rrRatio >= 2.5 && p >= 0.65) {
    scaled *= 1.15;
    increases.push(`Strong asymmetry (R/R ${rrRatio.toFixed(1)}×) +15%`);
  }
  if (upside / downside < 3.0 && spread < 0.40 && p >= 0.60) {
    scaled *= 1.10;
    increases.push('Tight distribution +10%');
  }

  // Beta hard cap: β > 1.5 → ceiling of 3.0% regardless of Kelly output
  const rawPct = clamp(0.5, 5.0, scaled * 100);
  const positionPct = beta > 1.5 ? Math.min(rawPct, 3.0) : rawPct;
  if (beta > 1.5 && rawPct > 3.0) reductions.push('Beta hard cap 3.0%');

  let tier: DistribSizing['tier'];
  if (positionPct >= 3.5)      tier = 'high_conviction';
  else if (positionPct >= 2.0) tier = 'core';
  else if (positionPct >= 1.0) tier = 'medium';
  else                         tier = 'small';

  let entryStyle: DistribSizing['entryStyle'];
  if (currentPrice > p50) {
    entryStyle = 'wait_for_pullback';
  } else if (currentPrice <= (mc.p25_price ?? p5 * 1.3) && p >= 0.60) {
    entryStyle = 'aggressive';
  } else {
    entryStyle = 'staged';
  }

  // Interpretation
  const tierLabel = tier === 'high_conviction' ? 'high-conviction' : tier;
  const entryLabel =
    entryStyle === 'aggressive' ? 'full allocation' :
    entryStyle === 'staged'     ? 'staged accumulation' :
                                  'wait for pullback';
  const interpretation =
    `Model recommends a ${positionPct.toFixed(1)}% ${tierLabel} position via ${entryLabel}. ` +
    `R/R of ${rrRatio.toFixed(1)}× (upside ${(upside * 100).toFixed(0)}% vs downside ${(downside * 100).toFixed(0)}%) ` +
    `supports ${kellyScale} Kelly sizing at ${(scaleMultiplier * 100).toFixed(0)}× fraction.`;

  return {
    rawKelly, scaledKelly: scaled, positionPct,
    tier, entryStyle,
    rrRatio, downside, upside,
    kellyScale, reductions, increases,
    interpretation,
  };
}

function MCPositionSizingCard({
  mc, currentPrice, momScore, riskScore, beta,
}: {
  mc: MCSim;
  currentPrice: number;
  momScore: number;
  riskScore: number;
  beta: number;
}) {
  if (!mc || mc.p5_price <= 0 || mc.p95_price <= 0 || !mc.prob_positive) return null;

  const sz = computeDistribSizing(currentPrice, mc, momScore, riskScore, beta);

  const tierColors: Record<DistribSizing['tier'], { bg: string; text: string; label: string }> = {
    small:           { bg: '#f1f5f9', text: '#475569', label: 'Small'          },
    medium:          { bg: '#dbeafe', text: '#1d4ed8', label: 'Medium'         },
    core:            { bg: '#dcfce7', text: '#15803d', label: 'Core'           },
    high_conviction: { bg: '#fef9c3', text: '#a16207', label: 'High Conviction'},
  };
  const esColors: Record<DistribSizing['entryStyle'], { bg: string; text: string; label: string }> = {
    aggressive:        { bg: '#dcfce7', text: '#15803d', label: 'Aggressive Entry'   },
    staged:            { bg: '#dbeafe', text: '#1d4ed8', label: 'Staged Accumulation'},
    wait_for_pullback: { bg: '#fef3c7', text: '#92400e', label: 'Wait for Pullback'  },
  };
  const barColor =
    sz.tier === 'high_conviction' ? '#f59e0b' :
    sz.tier === 'core'            ? '#16a34a' :
    sz.tier === 'medium'          ? '#3b82f6' : '#94a3b8';

  const tc = tierColors[sz.tier];
  const ec = esColors[sz.entryStyle];

  const fmt  = (v: number) => v.toFixed(1);
  const fmtp = (v: number) => (v * 100).toFixed(0) + '%';

  // Kelly reference grid values
  const aggrKelly = (sz.rawKelly * 0.50 * 100).toFixed(1);
  const stdKelly  = (sz.rawKelly * 0.25 * 100).toFixed(1);
  const consKelly = (sz.rawKelly * 0.10 * 100).toFixed(1);
  const expReturn = ((mc.prob_positive * sz.upside) - ((1 - mc.prob_positive) * sz.downside)) * 100;

  return (
    <Card className="rounded-xl overflow-hidden" style={{ WebkitPrintColorAdjust: 'exact', printColorAdjust: 'exact' } as React.CSSProperties}>
      <CardHeader className="pb-3">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
          <CardTitle style={{ fontSize: '14px', fontWeight: 600 }}>Position Sizing</CardTitle>
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
            <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '9999px', backgroundColor: tc.bg, color: tc.text, fontWeight: 600, WebkitPrintColorAdjust: 'exact' } as React.CSSProperties}>
              {tc.label}
            </span>
            <span style={{ fontSize: '11px', padding: '2px 8px', borderRadius: '9999px', backgroundColor: ec.bg, color: ec.text, fontWeight: 600, WebkitPrintColorAdjust: 'exact' } as React.CSSProperties}>
              {ec.label}
            </span>
          </div>
        </div>
      </CardHeader>

      <CardContent style={{ paddingTop: 0 }}>
        {/* ── Size display ── */}
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '6px', marginBottom: '12px' }}>
          <span style={{ fontSize: '32px', fontWeight: 700, lineHeight: 1 }}>{fmt(sz.positionPct)}%</span>
          <span style={{ fontSize: '13px', color: '#64748b' }}>of portfolio</span>
        </div>

        {/* ── Allocation bar 0–5% ── */}
        <div style={{ marginBottom: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: '#94a3b8', marginBottom: '4px' }}>
            <span>0%</span><span>1%</span><span>2%</span><span>3%</span><span>4%</span><span>5%</span>
          </div>
          <div style={{ position: 'relative', height: '12px', borderRadius: '9999px', backgroundColor: '#f1f5f9', overflow: 'hidden', WebkitPrintColorAdjust: 'exact' } as React.CSSProperties}>
            <div style={{
              position: 'absolute', top: 0, left: 0,
              height: '100%',
              width: `${Math.min(100, (sz.positionPct / 5) * 100)}%`,
              borderRadius: '9999px',
              backgroundColor: barColor,
              WebkitPrintColorAdjust: 'exact',
            } as React.CSSProperties} />
          </div>
        </div>

        {/* ── 3-metric grid ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px', marginBottom: '12px' }}>
          <div style={{ backgroundColor: '#f8fafc', borderRadius: '8px', padding: '8px 10px' }}>
            <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '2px' }}>R/R Ratio</div>
            <div style={{ fontSize: '16px', fontWeight: 700 }}>{fmt(sz.rrRatio)}×</div>
          </div>
          <div style={{ backgroundColor: '#f8fafc', borderRadius: '8px', padding: '8px 10px' }}>
            <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '2px' }}>½/¼/⅒ Kelly</div>
            <div style={{ fontSize: '12px', fontWeight: 600 }}>
              {aggrKelly}% / {stdKelly}% / {consKelly}%
            </div>
          </div>
          <div style={{ backgroundColor: '#f8fafc', borderRadius: '8px', padding: '8px 10px' }}>
            <div style={{ fontSize: '10px', color: '#64748b', marginBottom: '2px' }}>E[Return]</div>
            <div style={{ fontSize: '16px', fontWeight: 700, color: expReturn >= 0 ? '#15803d' : '#dc2626' }}>
              {expReturn >= 0 ? '+' : ''}{expReturn.toFixed(1)}%
            </div>
          </div>
        </div>

        {/* ── Downside / Upside ── */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginBottom: '12px' }}>
          <div style={{ backgroundColor: '#fef2f2', borderRadius: '8px', padding: '8px 10px', WebkitPrintColorAdjust: 'exact' } as React.CSSProperties}>
            <div style={{ fontSize: '10px', color: '#dc2626', marginBottom: '2px' }}>Downside (P5)</div>
            <div style={{ fontSize: '15px', fontWeight: 700, color: '#dc2626' }}>−{fmtp(sz.downside)}</div>
            <div style={{ fontSize: '10px', color: '#94a3b8' }}>${mc.p5_price.toFixed(2)}</div>
          </div>
          <div style={{ backgroundColor: '#f0fdf4', borderRadius: '8px', padding: '8px 10px', WebkitPrintColorAdjust: 'exact' } as React.CSSProperties}>
            <div style={{ fontSize: '10px', color: '#15803d', marginBottom: '2px' }}>Upside (P95)</div>
            <div style={{ fontSize: '15px', fontWeight: 700, color: '#15803d' }}>+{fmtp(sz.upside)}</div>
            <div style={{ fontSize: '10px', color: '#94a3b8' }}>${mc.p95_price.toFixed(2)}</div>
          </div>
        </div>

        {/* ── Adjustments ── */}
        {(sz.reductions.length > 0 || sz.increases.length > 0) && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '10px' }}>
            {sz.reductions.map((r, i) => (
              <span key={i} style={{ fontSize: '10px', padding: '2px 7px', borderRadius: '9999px', backgroundColor: '#fef2f2', color: '#dc2626', fontWeight: 500, WebkitPrintColorAdjust: 'exact' } as React.CSSProperties}>
                ↓ {r}
              </span>
            ))}
            {sz.increases.map((inc, i) => (
              <span key={i} style={{ fontSize: '10px', padding: '2px 7px', borderRadius: '9999px', backgroundColor: '#f0fdf4', color: '#15803d', fontWeight: 500, WebkitPrintColorAdjust: 'exact' } as React.CSSProperties}>
                ↑ {inc}
              </span>
            ))}
          </div>
        )}

        {/* ── Interpretation ── */}
        <p style={{ fontSize: '11px', color: '#64748b', lineHeight: 1.6, margin: 0 }}>
          {sz.interpretation}
        </p>
      </CardContent>
    </Card>
  );
}

// ── Macro-aware valuation distribution engine ────────────────────────────────

type MacroRegime       = 'expansion' | 'mid_cycle' | 'late_cycle' | 'contraction';
type SectorSensitivity = 'high' | 'medium' | 'low';

interface MacroAdjustment {
  regime:            MacroRegime;
  regimeLabel:       string;
  regimeDescription: string;
  sensitivity:       SectorSensitivity;
  sensitivityLabel:  string;
  sensitivityMult:   number;
  // Multipliers applied to each MC price percentile
  p5Mult:  number;
  p25Mult: number;
  p50Mult: number;
  p75Mult: number;
  p95Mult: number;
  // Original values
  origP5:  number;
  origP25: number;
  origP50: number;
  origP75: number;
  origP95: number;
  // Adjusted values
  adjP5:   number;
  adjP25:  number;
  adjP50:  number;
  adjP75:  number;
  adjP95:  number;
  // Summary metrics
  p50ChangePercent:   number;
  rrRatioOrig:        number;
  rrRatioAdj:         number;
  expectedReturnOrig: number;
  expectedReturnAdj:  number;
  summary:            string;
}

// Base fractional shifts by regime (e.g. −0.08 = −8% to that percentile)
const _REGIME_ADJS: Record<MacroRegime, {
  p5: number; p25: number; p50: number; p75: number; p95: number;
}> = {
  expansion:   { p5: +0.030, p25: +0.030, p50: +0.020, p75: +0.070, p95: +0.100 },
  mid_cycle:   { p5: -0.020, p25: -0.010, p50: -0.030, p75: -0.040, p95: -0.050 },
  late_cycle:  { p5: -0.120, p25: -0.080, p50: -0.060, p75: -0.050, p95: -0.040 },
  contraction: { p5: -0.220, p25: -0.160, p50: -0.120, p75: -0.100, p95: -0.080 },
};

const _REGIME_META: Record<MacroRegime, { label: string; description: string; bg: string; fg: string }> = {
  expansion:   { label: 'Expansion',       description: 'Positive growth momentum',       bg: '#dcfce7', fg: '#15803d' },
  mid_cycle:   { label: 'Mid-Cycle',        description: 'Decelerating but stable',        bg: '#dbeafe', fg: '#1d4ed8' },
  late_cycle:  { label: 'Late Cycle',       description: 'Tightening conditions',          bg: '#fef9c3', fg: '#a16207' },
  contraction: { label: 'Contraction Risk', description: 'Elevated recession probability', bg: '#fee2e2', fg: '#b91c1c' },
};

function classifyMacroRegime(macro: MacroData): MacroRegime {
  const phase  = (macro.cycle_phase          ?? '').toLowerCase();
  const regime = (macro.macro_regime         ?? '').toLowerCase();
  const risk   = (macro.recession_risk_level ?? '').toLowerCase();
  const lei    = (macro.lei_trend            ?? '').toLowerCase();
  const yc     = (macro.yield_spread_trend   ?? '').toLowerCase();
  const score  = macro.macro_score ?? 50;

  if (
    risk.includes('high') ||
    phase.includes('contraction') ||
    regime.includes('recession') ||
    (yc.includes('inverted') && score < 38)
  ) return 'contraction';

  if (
    phase.includes('late') ||
    regime.includes('late cycle') ||
    regime.includes('tightening') ||
    (risk.includes('elevated') && score < 45) ||
    (yc.includes('inverted') && score < 50)
  ) return 'late_cycle';

  if (
    phase.includes('expansion') ||
    phase.includes('early') ||
    regime.includes('expansion') ||
    (score >= 65 && (lei.includes('rising') || lei.includes('improv')))
  ) return 'expansion';

  return 'mid_cycle';
}

function getSectorSensitivity(
  sector: string | null | undefined,
  beta: number,
): { sensitivity: SectorSensitivity; mult: number; label: string } {
  const s = (sector ?? '').toLowerCase();

  // High beta always implies high macro sensitivity
  if (beta >= 1.4) {
    return { sensitivity: 'high', mult: 1.0, label: `High (β=${beta.toFixed(2)})` };
  }

  const LOW_SENS  = ['utilities', 'consumer staples', 'health care', 'healthcare', 'pharmaceutical'];
  const HIGH_SENS = [
    'technology', 'information technology', 'consumer discretionary',
    'real estate', 'materials', 'industrials', 'financials', 'energy',
    'communication services', 'semiconductor',
  ];

  if (LOW_SENS.some(x => s.includes(x)))  return { sensitivity: 'low',    mult: 0.35, label: `Low (${sector ?? 'Defensive'})` };
  if (HIGH_SENS.some(x => s.includes(x))) return { sensitivity: 'high',   mult: 1.0,  label: `High (${sector ?? 'Cyclical'})` };
  return                                         { sensitivity: 'medium', mult: 0.60, label: `Medium (${sector ?? 'Neutral'})` };
}

function computeMacroAdjustment(
  mc:           MCSim,
  macro:        MacroData,
  sector:       string | null | undefined,
  beta:         number,
  currentPrice: number,
): MacroAdjustment {
  const regime  = classifyMacroRegime(macro);
  const regMeta = _REGIME_META[regime];
  const baseAdj = _REGIME_ADJS[regime];
  const { sensitivity, mult: sMult, label: sLabel } = getSectorSensitivity(sector, beta);

  // Effective multiplier = 1 + (base_shift × sensitivity_multiplier)
  const p5Mult  = 1 + baseAdj.p5  * sMult;
  const p25Mult = 1 + baseAdj.p25 * sMult;
  const p50Mult = 1 + baseAdj.p50 * sMult;
  const p75Mult = 1 + baseAdj.p75 * sMult;
  const p95Mult = 1 + baseAdj.p95 * sMult;

  const adjP5  = Math.max(0.01, mc.p5_price     * p5Mult);
  const adjP25 = Math.max(0.01, mc.p25_price    * p25Mult);
  const adjP50 = Math.max(0.01, mc.median_price * p50Mult);
  const adjP75 = Math.max(0.01, mc.p75_price    * p75Mult);
  const adjP95 = Math.max(0.01, mc.p95_price    * p95Mult);

  const p50ChangePercent = (p50Mult - 1) * 100;

  const _up  = (p: number) => Math.max(0.001, (p - currentPrice) / currentPrice);
  const _dn  = (p: number) => Math.max(0.001, (currentPrice - p) / currentPrice);
  const rrRatioOrig = _up(mc.p95_price) / _dn(mc.p5_price);
  const rrRatioAdj  = _up(adjP95)       / _dn(adjP5);

  const expectedReturnOrig = mc.mean_return * 100;
  // Approximate adjusted mean return: shift mean by the same delta as median
  const adjMedianRet   = (adjP50 / currentPrice) - 1;
  const expectedReturnAdj = (mc.mean_return + (adjMedianRet - mc.median_return)) * 100;

  // Summary narrative
  const dirWord  = p50ChangePercent < 0 ? 'lower' : 'higher';
  const absChg   = Math.abs(p50ChangePercent).toFixed(1);
  const sensWord = sLabel.split(' ')[0].toLowerCase();
  const sectNote = sector ? ` (${sector})` : '';
  const summary =
    `${regMeta.description} — ${sensWord} macro sensitivity${sectNote} — ` +
    `shifts the distribution ${dirWord} by ~${absChg}% at fair value. ` +
    (regime === 'contraction'
      ? 'Downside tail expands materially; entry levels and position size reflect elevated risk.'
      : regime === 'late_cycle'
      ? 'Fair value compresses and downside risk increases; conservative entry sizing is warranted.'
      : regime === 'mid_cycle'
      ? 'Modest downward compression applied; overall investment thesis is unchanged.'
      : 'Upside scenarios receive a modest boost reflecting positive macro momentum.');

  return {
    regime, regimeLabel: regMeta.label, regimeDescription: regMeta.description,
    sensitivity, sensitivityLabel: sLabel, sensitivityMult: sMult,
    p5Mult, p25Mult, p50Mult, p75Mult, p95Mult,
    origP5: mc.p5_price, origP25: mc.p25_price, origP50: mc.median_price,
    origP75: mc.p75_price, origP95: mc.p95_price,
    adjP5, adjP25, adjP50, adjP75, adjP95,
    p50ChangePercent, rrRatioOrig, rrRatioAdj,
    expectedReturnOrig, expectedReturnAdj,
    summary,
  };
}

function buildAdjustedMC(mc: MCSim, adj: MacroAdjustment, currentPrice: number): MCSim {
  const toRet = (p: number) => currentPrice > 0 ? (p / currentPrice) - 1 : 0;
  const upAdj = Math.max(0.001, (adj.adjP95 / currentPrice) - 1);
  const dnAdj = Math.max(0.001, 1 - (adj.adjP5 / currentPrice));
  return {
    ...mc,
    p5_price:      adj.adjP5,
    p25_price:     adj.adjP25,
    median_price:  adj.adjP50,
    p75_price:     adj.adjP75,
    p95_price:     adj.adjP95,
    mean_price:    adj.adjP50,
    p5_return:     toRet(adj.adjP5),
    p25_return:    toRet(adj.adjP25),
    median_return: toRet(adj.adjP50),
    p75_return:    toRet(adj.adjP75),
    p95_return:    toRet(adj.adjP95),
    mean_return:   mc.mean_return + (toRet(adj.adjP50) - mc.median_return),
    upside_downside: upAdj / dnAdj,
  };
}

function MacroAdjustedValuationCard({
  adj, currentPrice,
}: {
  adj:          MacroAdjustment;
  currentPrice: number;
}) {
  const regMeta = _REGIME_META[adj.regime];

  const fmtP = (v: number) => `$${v.toFixed(2)}`;
  const fmtR = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`;
  const fmtDelta = (orig: number, adjV: number) => {
    const pct = ((adjV - orig) / Math.abs(orig)) * 100;
    return { text: `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`, pos: pct >= 0 };
  };

  const sensitivityBg =
    adj.sensitivity === 'high'   ? '#fef9c3' :
    adj.sensitivity === 'low'    ? '#f0fdf4' : '#f1f5f9';
  const sensitivityFg =
    adj.sensitivity === 'high'   ? '#a16207' :
    adj.sensitivity === 'low'    ? '#15803d' : '#475569';

  const d5  = fmtDelta(adj.origP5,  adj.adjP5);
  const d50 = fmtDelta(adj.origP50, adj.adjP50);
  const d95 = fmtDelta(adj.origP95, adj.adjP95);
  const rrUp = adj.rrRatioAdj >= adj.rrRatioOrig;
  const erUp = adj.expectedReturnAdj >= adj.expectedReturnOrig;
  const erDeltaPp = adj.expectedReturnAdj - adj.expectedReturnOrig;

  return (
    <Card className="border-slate-200 shadow-sm overflow-hidden"
      style={{ WebkitPrintColorAdjust: 'exact', printColorAdjust: 'exact' } as React.CSSProperties}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <CardTitle className="text-sm font-semibold text-slate-700">Macro-Adjusted Valuation</CardTitle>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-semibold px-2 py-0.5 rounded-full"
              style={{ backgroundColor: regMeta.bg, color: regMeta.fg, WebkitPrintColorAdjust: 'exact' } as React.CSSProperties}>
              {regMeta.label}
            </span>
            <span className="text-xs font-semibold px-2 py-0.5 rounded-full"
              style={{ backgroundColor: sensitivityBg, color: sensitivityFg, WebkitPrintColorAdjust: 'exact' } as React.CSSProperties}>
              {adj.sensitivityLabel} Sensitivity
            </span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="px-0 pb-0">
        <Table>
          <TableHeader>
            <TableRow className="bg-slate-50">
              <TableHead className="pl-6 w-24 text-[10px] font-semibold text-slate-500"></TableHead>
              <TableHead className="text-center text-[10px] font-semibold text-slate-500">P5 Downside</TableHead>
              <TableHead className="text-center text-[10px] font-semibold text-slate-500">Fair Value (P50)</TableHead>
              <TableHead className="text-center text-[10px] font-semibold text-slate-500">P95 Upside</TableHead>
              <TableHead className="text-center text-[10px] font-semibold text-slate-500">R/R</TableHead>
              <TableHead className="pr-6 text-center text-[10px] font-semibold text-slate-500">E[Return]</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {/* Original */}
            <TableRow>
              <TableCell className="pl-6 text-xs text-slate-400 py-2.5">Original</TableCell>
              <TableCell className="text-center text-sm font-semibold text-slate-600 tabular-nums py-2.5">{fmtP(adj.origP5)}</TableCell>
              <TableCell className="text-center text-sm font-semibold text-slate-600 tabular-nums py-2.5">{fmtP(adj.origP50)}</TableCell>
              <TableCell className="text-center text-sm font-semibold text-slate-600 tabular-nums py-2.5">{fmtP(adj.origP95)}</TableCell>
              <TableCell className="text-center text-sm font-semibold text-slate-600 tabular-nums py-2.5">{adj.rrRatioOrig.toFixed(1)}×</TableCell>
              <TableCell className="pr-6 text-center text-sm font-semibold tabular-nums py-2.5"
                style={{ color: adj.expectedReturnOrig >= 0 ? '#15803d' : '#dc2626' }}>
                {fmtR(adj.expectedReturnOrig)}
              </TableCell>
            </TableRow>
            {/* Macro-adjusted */}
            <TableRow style={{ backgroundColor: regMeta.bg, WebkitPrintColorAdjust: 'exact' } as React.CSSProperties}>
              <TableCell className="pl-6 text-xs font-bold py-2.5" style={{ color: regMeta.fg }}>Macro-Adj.</TableCell>
              <TableCell className="text-center text-sm font-bold tabular-nums py-2.5" style={{ color: regMeta.fg }}>{fmtP(adj.adjP5)}</TableCell>
              <TableCell className="text-center text-sm font-bold tabular-nums py-2.5" style={{ color: regMeta.fg }}>{fmtP(adj.adjP50)}</TableCell>
              <TableCell className="text-center text-sm font-bold tabular-nums py-2.5" style={{ color: regMeta.fg }}>{fmtP(adj.adjP95)}</TableCell>
              <TableCell className="text-center text-sm font-bold tabular-nums py-2.5" style={{ color: regMeta.fg }}>{adj.rrRatioAdj.toFixed(1)}×</TableCell>
              <TableCell className="pr-6 text-center text-sm font-bold tabular-nums py-2.5"
                style={{ color: adj.expectedReturnAdj >= 0 ? '#15803d' : '#dc2626' }}>
                {fmtR(adj.expectedReturnAdj)}
              </TableCell>
            </TableRow>
            {/* Delta */}
            <TableRow className="bg-slate-50">
              <TableCell className="pl-6 text-xs text-slate-400 py-2.5">Δ Change</TableCell>
              <TableCell className="text-center text-xs font-bold tabular-nums py-2.5"
                style={{ color: d5.pos ? '#15803d' : '#dc2626' }}>{d5.text}</TableCell>
              <TableCell className="text-center text-xs font-bold tabular-nums py-2.5"
                style={{ color: d50.pos ? '#15803d' : '#dc2626' }}>{d50.text}</TableCell>
              <TableCell className="text-center text-xs font-bold tabular-nums py-2.5"
                style={{ color: d95.pos ? '#15803d' : '#dc2626' }}>{d95.text}</TableCell>
              <TableCell className="text-center text-xs font-bold py-2.5"
                style={{ color: rrUp ? '#15803d' : '#dc2626' }}>
                {rrUp ? '↑' : '↓'} {adj.rrRatioAdj.toFixed(1)}×
              </TableCell>
              <TableCell className="pr-6 text-center text-xs font-bold tabular-nums py-2.5"
                style={{ color: erUp ? '#15803d' : '#dc2626' }}>
                {erUp ? '+' : ''}{erDeltaPp.toFixed(1)}pp
              </TableCell>
            </TableRow>
          </TableBody>
        </Table>
        <div className="px-6 py-3 border-t border-slate-100">
          <p className="text-xs text-slate-600 leading-relaxed">{adj.summary}</p>
        </div>
      </CardContent>
    </Card>
  );
}

// ── Driver-based scenario model card ─────────────────────────────────────────

function DriverModelCard({ vr, currentPrice }: { vr: ValuationRange; currentPrice: number | null }) {
  if (!vr.driver_model_available) return null;
  const fmtP = (v: number | null | undefined) => v != null ? `${(v * 100).toFixed(1)}%` : "—";
  const fmtX = (v: number | null | undefined) => v != null ? `${v.toFixed(1)}×` : "—";
  const fmtFCF = (v: number | null | undefined) => {
    if (v == null) return "—";
    const b = Math.abs(v) >= 1e9 ? v / 1e9 : v / 1e6;
    const unit = Math.abs(v) >= 1e9 ? "B" : "M";
    return `$${b.toFixed(1)}${unit}`;
  };
  return (
    <Card className="border-slate-200 shadow-sm overflow-hidden">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-slate-700">Driver-Based Scenario Model</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <Table>
          <TableHeader>
            <TableRow className="bg-slate-50">
              <TableHead className="pl-6 font-semibold text-slate-600">Driver</TableHead>
              <TableHead className="text-center text-red-600 font-semibold">Bear</TableHead>
              <TableHead className="text-center text-blue-600 font-semibold">Base</TableHead>
              <TableHead className="text-center text-green-600 font-semibold">Bull</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow>
              <TableCell className="pl-6 text-sm text-slate-600">Revenue Growth</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-red-600">{fmtP(vr.scenario_bear_rev_growth)}</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-blue-600">{fmtP(vr.scenario_base_rev_growth)}</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-green-600">{fmtP(vr.scenario_bull_rev_growth)}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell className="pl-6 text-sm text-slate-600">Operating Margin</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-red-600">{fmtP(vr.scenario_bear_op_margin)}</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-blue-600">{fmtP(vr.scenario_base_op_margin)}</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-green-600">{fmtP(vr.scenario_bull_op_margin)}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell className="pl-6 text-sm text-slate-600">FCF Conversion</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-red-600">{fmtP(vr.scenario_bear_fcf_conv)}</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-blue-600">{fmtP(vr.scenario_base_fcf_conv)}</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-green-600">{fmtP(vr.scenario_bull_fcf_conv)}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell className="pl-6 text-sm text-slate-600">Exit Multiple</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-red-600">{fmtX(vr.scenario_bear_exit_mult)}</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-blue-600">{fmtX(vr.scenario_base_exit_mult)}</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-green-600">{fmtX(vr.scenario_bull_exit_mult)}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell className="pl-6 text-sm text-slate-600">Projected FCF</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-red-600">{fmtFCF(vr.scenario_bear_fwd_fcf)}</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-blue-600">{fmtFCF(vr.scenario_base_fwd_fcf)}</TableCell>
              <TableCell className="text-center text-sm tabular-nums text-green-600">{fmtFCF(vr.scenario_bull_fwd_fcf)}</TableCell>
            </TableRow>
            <TableRow className="bg-slate-50 font-semibold">
              <TableCell className="pl-6 font-semibold text-slate-700">Target Price</TableCell>
              <TableCell className="text-center text-red-600 font-bold">{formatPrice(vr.bear_price)}</TableCell>
              <TableCell className="text-center text-blue-600 font-bold">{formatPrice(vr.base_price)}</TableCell>
              <TableCell className="text-center text-green-600 font-bold">{formatPrice(vr.bull_price)}</TableCell>
            </TableRow>
            {currentPrice && (
              <TableRow>
                <TableCell className="pl-6 text-slate-500 text-xs">vs Current ({formatPrice(currentPrice)})</TableCell>
                <TableCell className="text-center text-xs">{vr.bear_price != null ? formatUpside(vr.bear_price, currentPrice) : "—"}</TableCell>
                <TableCell className="text-center text-xs">{vr.base_price != null ? formatUpside(vr.base_price, currentPrice) : "—"}</TableCell>
                <TableCell className="text-center text-xs">{vr.bull_price != null ? formatUpside(vr.bull_price, currentPrice) : "—"}</TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
        {vr.scenario_bear_label && (
          <div className="px-6 py-3 border-t border-slate-100 space-y-1">
            <p className="text-[10px] text-red-500 leading-snug">Bear: {vr.scenario_bear_label}</p>
            <p className="text-[10px] text-blue-500 leading-snug">Base: {vr.scenario_base_label}</p>
            <p className="text-[10px] text-green-500 leading-snug">Bull: {vr.scenario_bull_label}</p>
          </div>
        )}
        {vr.trend_impact_lines && vr.trend_impact_lines.length > 0 && (
          <div className="px-6 py-3 border-t border-slate-100">
            <p className="text-xs font-semibold text-slate-600 mb-1.5">Trend Impact on Valuation</p>
            <ul className="space-y-1">
              {vr.trend_impact_lines.map((line, i) => (
                <li key={i} className="text-xs text-slate-500 flex items-start gap-1.5">
                  <span className="text-slate-300 mt-0.5">•</span>{line}
                </li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ── Peer 5-year historical performance — unified table ────────────────────────
// Single table: rows = Company × Metric, columns = Current / FY-1…FY-5 (always 6).

const _HIST_YEARS = ["Current", "FY-1", "FY-2", "FY-3", "FY-4", "FY-5"] as const;

const _HIST_METRICS: {
  label: string;
  key: keyof HistoricalYear;
  fmt: (v: number | null | undefined) => string;
}[] = [
  { label: "Revenue Growth %", key: "revenue_growth", fmt: v => fmtPctRaw(v as number | null) },
  { label: "Op Margin %",      key: "op_margin",      fmt: v => fmtPct(v as number | null) },
  { label: "Net Margin %",     key: "net_margin",     fmt: v => fmtPct(v as number | null) },
  { label: "ROE %",            key: "roe",            fmt: v => fmtPct(v as number | null) },
  { label: "ROIC %",           key: "roic",           fmt: v => fmtPct(v as number | null) },
];

function PeerHistoricalTable({
  rows,
  peerTrendInsights,
}: {
  rows: PeerRowType[];
  peerTrendInsights?: string[];
}) {
  const rowsWithHistory = rows.filter(r => r.historical && r.historical.length > 0);
  if (!rowsWithHistory.length) return null;

  return (
    <div className="mt-4 border-t border-slate-100 pt-4 space-y-3">
      <h3 className="text-sm font-semibold text-slate-700">5-Year Historical Performance</h3>

      <Card className="border-slate-200 shadow-sm overflow-x-auto print:overflow-visible">
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow className="bg-slate-50">
                <TableHead className="pl-4 print:pl-2 font-semibold text-slate-600 w-[90px] print:w-[64px]">Company</TableHead>
                <TableHead className="font-semibold text-slate-600 w-[120px] print:w-[96px] print:text-[10px]">Metric</TableHead>
                {_HIST_YEARS.map(y => (
                  <TableHead key={y} className="text-right pr-2 print:pr-1 font-semibold text-slate-600 text-xs whitespace-nowrap">{y}</TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {rowsWithHistory.flatMap((row, ri) => {
                const byLabel: Partial<Record<string, HistoricalYear>> = {};
                for (const h of (row.historical ?? [])) byLabel[h.label] = h;
                return _HIST_METRICS.map((m, mi) => (
                  <TableRow
                    key={`${row.ticker}-${String(m.key)}`}
                    className={cn(
                      row.is_target && "bg-blue-50/60",
                      mi === 0 && ri > 0 && "border-t-2 border-slate-200",
                    )}
                  >
                    <TableCell className="pl-4 print:pl-2 py-1 print:py-0.5 align-top">
                      {mi === 0 && (
                        <span className={cn(
                          "font-mono font-semibold text-sm print:text-xs",
                          row.is_target ? "text-blue-700" : "text-slate-900"
                        )}>
                          {row.is_target ? "★ " : ""}{row.ticker}
                        </span>
                      )}
                    </TableCell>
                    <TableCell className="py-1 print:py-0.5 text-xs print:text-[10px] text-slate-500 whitespace-nowrap">{m.label}</TableCell>
                    {_HIST_YEARS.map(yr => {
                      const h = byLabel[yr];
                      const val = h ? (h[m.key] as number | null) : null;
                      return (
                        <TableCell
                          key={yr}
                          className={cn("text-right tabular-nums pr-2 print:pr-1 text-sm print:text-xs py-1 print:py-0.5", val == null && "text-slate-200")}
                        >
                          {m.fmt(val)}
                        </TableCell>
                      );
                    })}
                  </TableRow>
                ));
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {peerTrendInsights && peerTrendInsights.length > 0 && (
        <Card className="border-slate-200 shadow-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold text-slate-700">Peer Trend Insights</CardTitle>
          </CardHeader>
          <CardContent className="px-6 pb-5">
            <ul className="space-y-1.5">
              {peerTrendInsights.map((ins, i) => (
                <li key={i} className="text-sm text-slate-600 flex items-start gap-2">
                  <span className="text-slate-300 mt-1 shrink-0">•</span>{ins}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

