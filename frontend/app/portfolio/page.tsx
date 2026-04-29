"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  ArrowLeft,
  RefreshCw,
  Trash2,
  PieChart,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

// ── Snapshot type — must match what report/[ticker]/page.tsx saves ─────────────

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

// ── Portfolio allocation output types ─────────────────────────────────────────

interface SellSignal {
  action:   'HOLD' | 'TRIM' | 'EXIT';
  trim_pct: number;      // % of position to sell (0 for HOLD)
  reason:   string;
  trigger:  'valuation' | 'momentum' | 'risk' | 'overvalued';
}

interface PositionResult {
  ticker:          string;
  company_name:    string;
  sector:          string;
  current_price:   number;
  weight:          number;     // % of portfolio (stocks only)
  score:           number;     // 0–1 composite
  conviction:      'High' | 'Medium' | 'Low';
  priority:        'HIGH' | 'MEDIUM' | 'LOW';
  entry_style:     string;
  expected_return: number;     // decimal (e.g. 0.23 = 23%)
  rr_ratio:        number;
  beta:            number;
  stance:          string;
  downside_pct:    number;     // (P5–current)/current × 100
  sell_signal:     SellSignal;
}

interface PortfolioResult {
  positions:       PositionResult[];
  cash_pct:        number;
  port_er:         number;     // weighted E[R] in %
  port_beta:       number;
  port_downside:   number;     // weighted avg P5 return %
  diversification: number;    // 0–100
  sector_exposure: Record<string, number>;
}

// ── Sell discipline ───────────────────────────────────────────────────────────

function computeSellSignal(s: PortfolioSnapshotData): SellSignal {
  const { current_price: cp, p50, p75, p95, mom_score, risk_score, stance } = s;
  const fmt = (v: number) => `$${v.toFixed(2)}`;

  if (cp >= p95 * 1.10) {
    return { action: 'EXIT', trim_pct: 100,
      reason: `Price ${fmt(cp)} >10% above P95 (${fmt(p95)}) — fully extended`,
      trigger: 'overvalued' };
  }
  if (cp >= p95) {
    return { action: 'EXIT', trim_pct: 75,
      reason: `Price ${fmt(cp)} at or above P95 (${fmt(p95)}) — exit majority`,
      trigger: 'valuation' };
  }
  if (cp >= p75) {
    const trimPct = cp >= (p75 + p95) / 2 ? 50 : 25;
    return { action: 'TRIM', trim_pct: trimPct,
      reason: `Price ${fmt(cp)} in trim zone (P75 = ${fmt(p75)}, P95 = ${fmt(p95)})`,
      trigger: 'valuation' };
  }
  if (mom_score < 35) {
    return { action: 'TRIM', trim_pct: 25,
      reason: `Momentum score ${mom_score} — sharply negative trend`,
      trigger: 'momentum' };
  }
  if (risk_score < 40) {
    return { action: 'TRIM', trim_pct: 25,
      reason: `Risk score ${risk_score} — elevated balance-sheet / event risk`,
      trigger: 'risk' };
  }
  if (stance === 'Bearish' && cp >= p50) {
    return { action: 'TRIM', trim_pct: 25,
      reason: `Bearish stance with price above fair value (P50 = ${fmt(p50)})`,
      trigger: 'risk' };
  }
  return { action: 'HOLD', trim_pct: 0,
    reason: 'Within buy zone — hold position', trigger: 'valuation' };
}

// ── Portfolio allocation engine ───────────────────────────────────────────────

function computeAllocation(snaps: PortfolioSnapshotData[]): PortfolioResult | null {
  const eligible = snaps.filter(s => s.mc_available && s.current_price > 0 && s.p5 > 0);
  if (eligible.length === 0) return null;

  const MAX_POS    = 5.0;   // % max single position
  const MAX_SECTOR = 25.0;  // % max per sector
  const BETA_CAP   = 3.0;   // % cap for beta > 1.5

  // ── Step 1: composite score ─────────────────────────────────────────────────
  // Formula: 35% E[R] + 25% R/R + 15% Momentum + 15% Risk Score + 10% Quality
  const ers = eligible.map(s => s.expected_return);
  const rrs = eligible.map(s => Math.min(s.rr_ratio, 5));
  const minER = Math.min(...ers),  maxER = Math.max(...ers);
  const minRR = Math.min(...rrs),  maxRR = Math.max(...rrs);
  const erRange = maxER - minER || 0.001;
  const rrRange = maxRR - minRR || 0.001;

  const scored = eligible.map((s, i) => {
    const normER   = (ers[i] - minER) / erRange;
    const normRR   = (rrs[i] - minRR) / rrRange;
    const normMom  = s.mom_score  / 100;
    const normRisk = s.risk_score / 100;
    const quality  = (s.prof_score + s.fin_score) / 200;
    const score = 0.35 * normER + 0.25 * normRR + 0.15 * normMom + 0.15 * normRisk + 0.10 * quality;
    return { ...s, score };
  });

  // ── Step 2: proportional raw weights ──────────────────────────────────────
  const totalScore = scored.reduce((sum, x) => sum + Math.max(0, x.score), 0);
  if (totalScore === 0) return null;
  let weights = scored.map(s => (Math.max(0, s.score) / totalScore) * 100);

  // ── Step 3: iterative constraint clipping + redistribution ────────────────
  for (let iter = 0; iter < 15; iter++) {
    let clipped = false;
    let excess  = 0;

    // Per-stock cap
    for (let i = 0; i < weights.length; i++) {
      const cap = scored[i].beta > 1.5 ? BETA_CAP : MAX_POS;
      if (weights[i] > cap) {
        excess    += weights[i] - cap;
        weights[i] = cap;
        clipped    = true;
      }
    }

    // Sector cap
    const sectorTot = new Map<string, number>();
    for (let i = 0; i < scored.length; i++) {
      const sec = scored[i].sector;
      sectorTot.set(sec, (sectorTot.get(sec) ?? 0) + weights[i]);
    }
    for (const [sec, tot] of sectorTot) {
      if (tot > MAX_SECTOR) {
        const scale = MAX_SECTOR / tot;
        for (let i = 0; i < scored.length; i++) {
          if (scored[i].sector === sec && weights[i] > 0) {
            const old  = weights[i];
            weights[i] = weights[i] * scale;
            excess    += old - weights[i];
            clipped    = true;
          }
        }
      }
    }

    // Redistribute excess to uncapped stocks
    const cappedIdx = new Set<number>();
    for (let i = 0; i < weights.length; i++) {
      const cap = scored[i].beta > 1.5 ? BETA_CAP : MAX_POS;
      if (weights[i] >= cap - 0.001) cappedIdx.add(i);
    }
    const freeTotal = weights.reduce((s, w, i) => (!cappedIdx.has(i) ? s + w : s), 0);
    if (excess > 0.001 && freeTotal > 0) {
      const boostFactor = (freeTotal + excess) / freeTotal;
      for (let i = 0; i < weights.length; i++) {
        if (!cappedIdx.has(i)) {
          const cap  = scored[i].beta > 1.5 ? BETA_CAP : MAX_POS;
          weights[i] = Math.min(cap, weights[i] * boostFactor);
        }
      }
    }

    if (!clipped) break;
  }

  // ── Step 4: cash buffer when opportunity set is weak ──────────────────────
  const avgScore = scored.reduce((s, x) => s + x.score, 0) / scored.length;
  const cashPct  = avgScore < 0.30
    ? Math.min(25, Math.round((0.30 - avgScore) / 0.30 * 30))
    : 0;

  const stockTotal = weights.reduce((s, w) => s + w, 0);
  const targetTotal = 100 - cashPct;
  if (stockTotal > 0 && cashPct > 0) {
    weights = weights.map(w => (w / stockTotal) * targetTotal);
  }

  // ── Step 5: build position results ────────────────────────────────────────
  const positions: PositionResult[] = scored.map((s, i) => {
    const w          = Math.round(weights[i] * 10) / 10;
    const priority: 'HIGH' | 'MEDIUM' | 'LOW' =
      s.current_price <= s.p25 ? 'HIGH' :
      s.current_price <= s.p50 ? 'MEDIUM' : 'LOW';
    const conviction: 'High' | 'Medium' | 'Low' =
      w >= 3.5 ? 'High' : w >= 2.0 ? 'Medium' : 'Low';
    const entry_style =
      s.current_price > s.p50                               ? 'Wait for Pullback' :
      s.current_price <= s.p25 && s.prob_positive >= 0.60   ? 'Aggressive'        : 'Staged';
    const downside_pct =
      s.current_price > 0 ? ((s.p5 - s.current_price) / s.current_price) * 100 : 0;
    return {
      ticker: s.ticker, company_name: s.company_name, sector: s.sector,
      current_price: s.current_price, weight: w, score: s.score,
      conviction, priority, entry_style,
      expected_return: s.expected_return, rr_ratio: s.rr_ratio,
      beta: s.beta, stance: s.stance, downside_pct,
      sell_signal: computeSellSignal(s),
    };
  }).sort((a, b) => b.weight - a.weight);

  // ── Step 6: portfolio-level metrics ───────────────────────────────────────
  const normW      = weights.map(w => w / (weights.reduce((s, x) => s + x, 0) || 1));
  const port_er    = scored.reduce((s, x, i) => s + normW[i] * x.expected_return * 100, 0);
  const port_beta  = scored.reduce((s, x, i) => s + normW[i] * x.beta, 0);
  const port_downside = scored.reduce((s, x, i) => {
    const dn = x.current_price > 0 ? ((x.p5 - x.current_price) / x.current_price) * 100 : 0;
    return s + normW[i] * dn;
  }, 0);

  // HHI-based diversification (higher = more diversified)
  const hhi    = normW.reduce((s, w) => s + w * w, 0);
  const minHHI = normW.length > 1 ? 1 / normW.length : 1;
  const diversification = normW.length > 1
    ? Math.max(0, Math.min(100, Math.round((1 - (hhi - minHHI) / (1 - minHHI)) * 100)))
    : 0;

  const sector_exposure: Record<string, number> = {};
  for (let i = 0; i < scored.length; i++) {
    const sec = scored[i].sector;
    sector_exposure[sec] = (sector_exposure[sec] ?? 0) + weights[i];
  }

  return {
    positions,
    cash_pct:        cashPct,
    port_er:         Math.round(port_er    * 10) / 10,
    port_beta:       Math.round(port_beta  * 100) / 100,
    port_downside:   Math.round(port_downside * 10) / 10,
    diversification,
    sector_exposure,
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const priorityMeta = (p: 'HIGH' | 'MEDIUM' | 'LOW') =>
  p === 'HIGH'   ? { bg: '#dcfce7', fg: '#15803d' } :
  p === 'MEDIUM' ? { bg: '#dbeafe', fg: '#1d4ed8' } :
                   { bg: '#f1f5f9', fg: '#64748b' };

const convictionMeta = (c: 'High' | 'Medium' | 'Low') =>
  c === 'High'   ? { bg: '#dcfce7', fg: '#15803d' } :
  c === 'Medium' ? { bg: '#dbeafe', fg: '#1d4ed8' } :
                   { bg: '#f1f5f9', fg: '#64748b' };

function fmtER(v: number)  { return `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`; }
function fmtP(v: number)   { return `$${v.toFixed(2)}`; }

// ── Main portfolio page ───────────────────────────────────────────────────────

export default function PortfolioPage() {
  const router = useRouter();
  const [snapshots, setSnapshots] = useState<PortfolioSnapshotData[]>([]);
  const [selected,  setSelected]  = useState<Set<string>>(new Set());
  const [result,    setResult]    = useState<PortfolioResult | null>(null);

  useEffect(() => { loadSnapshots(); }, []);

  function loadSnapshots() {
    try {
      const raw = JSON.parse(
        localStorage.getItem(_PORTFOLIO_LS_KEY) || "{}"
      ) as Record<string, PortfolioSnapshotData>;
      const snaps = Object.values(raw);
      setSnapshots(snaps);
      setSelected(new Set(snaps.map(s => s.ticker)));
      setResult(null);
    } catch {
      setSnapshots([]);
    }
  }

  function removeSnapshot(ticker: string) {
    try {
      const raw = JSON.parse(
        localStorage.getItem(_PORTFOLIO_LS_KEY) || "{}"
      ) as Record<string, PortfolioSnapshotData>;
      delete raw[ticker];
      localStorage.setItem(_PORTFOLIO_LS_KEY, JSON.stringify(raw));
      loadSnapshots();
    } catch {}
  }

  function toggleSelected(ticker: string) {
    setSelected(prev => {
      const next = new Set(prev);
      next.has(ticker) ? next.delete(ticker) : next.add(ticker);
      return next;
    });
    setResult(null);
  }

  function buildPortfolio() {
    const chosen = snapshots.filter(s => selected.has(s.ticker));
    setResult(computeAllocation(chosen));
  }

  return (
    <div className="min-h-screen bg-slate-50">

      {/* ── Nav ── */}
      <div className="sticky top-0 z-50 bg-white/95 backdrop-blur border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center gap-4">
          <button
            onClick={() => router.push("/")}
            className="text-slate-400 hover:text-slate-700 transition-colors shrink-0"
          >
            <ArrowLeft className="w-4 h-4" />
          </button>
          <span className="font-semibold text-slate-900">Portfolio Builder</span>
          <span className="text-xs text-slate-400 hidden sm:block">
            Score = 35% E[R] + 25% R/R + 15% Momentum + 15% Risk + 10% Quality
          </span>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={loadSnapshots}
              className="p-1.5 text-slate-400 hover:text-slate-600 transition-colors"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={buildPortfolio}
              disabled={selected.size === 0}
              className="px-4 py-2 bg-slate-900 text-white text-sm font-semibold rounded-lg hover:bg-slate-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Build Portfolio
            </button>
          </div>
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-8 space-y-6">

        {/* ── Empty state ── */}
        {snapshots.length === 0 && (
          <div className="text-center py-24 space-y-4">
            <PieChart className="w-12 h-12 text-slate-200 mx-auto" />
            <p className="text-slate-500 text-sm font-medium">No evaluated stocks yet.</p>
            <p className="text-xs text-slate-400 max-w-xs mx-auto leading-relaxed">
              Evaluate stocks from the home page — each evaluation is automatically saved here.
            </p>
            <button
              onClick={() => router.push("/")}
              className="mt-2 px-5 py-2.5 bg-slate-900 text-white rounded-lg text-sm font-semibold hover:bg-slate-700 transition-colors"
            >
              Evaluate a stock
            </button>
          </div>
        )}

        {/* ── Stock selection table ── */}
        {snapshots.length > 0 && (
          <Card className="border-slate-200 shadow-sm">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between flex-wrap gap-2">
                <CardTitle className="text-sm font-semibold text-slate-700">
                  Evaluated Stocks
                  <span className="ml-2 text-xs font-normal text-slate-400">
                    {selected.size} of {snapshots.length} selected
                  </span>
                </CardTitle>
                <div className="flex gap-3 text-xs">
                  <button
                    onClick={() => setSelected(new Set(snapshots.map(s => s.ticker)))}
                    className="text-blue-600 hover:underline"
                  >
                    Select all
                  </button>
                  <span className="text-slate-300">·</span>
                  <button
                    onClick={() => { setSelected(new Set()); setResult(null); }}
                    className="text-slate-500 hover:underline"
                  >
                    Clear
                  </button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-slate-50 border-b border-slate-200">
                    <tr>
                      <th className="w-10 pl-4 py-2.5" />
                      <th className="text-left pl-4 py-2.5 font-semibold text-slate-600">Ticker</th>
                      <th className="text-left py-2.5 font-semibold text-slate-600 hidden sm:table-cell">Company</th>
                      <th className="text-left py-2.5 font-semibold text-slate-600 hidden lg:table-cell">Sector</th>
                      <th className="text-right py-2.5 font-semibold text-slate-600">Price</th>
                      <th className="text-right py-2.5 pr-3 font-semibold text-slate-600">E[R]</th>
                      <th className="text-right py-2.5 pr-3 font-semibold text-slate-600">R/R</th>
                      <th className="text-center py-2.5 font-semibold text-slate-600">Zone</th>
                      <th className="text-center py-2.5 font-semibold text-slate-600">Score</th>
                      <th className="w-10 pr-4" />
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {snapshots.map(s => {
                      const isSel = selected.has(s.ticker);
                      const zone  =
                        s.current_price <= s.p25 ? 'Strong Buy' :
                        s.current_price <= s.p50 ? 'Attractive' :
                        s.current_price <= s.p75 ? 'Above Fair' : 'Trim';
                      const zoneFg =
                        s.current_price <= s.p25 ? '#15803d' :
                        s.current_price <= s.p50 ? '#1d4ed8' :
                        s.current_price <= s.p75 ? '#a16207' : '#b91c1c';
                      const zoneBg =
                        s.current_price <= s.p25 ? '#dcfce7' :
                        s.current_price <= s.p50 ? '#dbeafe' :
                        s.current_price <= s.p75 ? '#fef9c3' : '#fee2e2';
                      return (
                        <tr
                          key={s.ticker}
                          className={cn(
                            "cursor-pointer hover:bg-slate-50 transition-colors",
                            isSel && "bg-blue-50/40"
                          )}
                          onClick={() => toggleSelected(s.ticker)}
                        >
                          <td className="pl-4 py-3">
                            <input
                              type="checkbox"
                              checked={isSel}
                              readOnly
                              className="rounded accent-blue-600 cursor-pointer"
                            />
                          </td>
                          <td className="pl-4 py-3">
                            <span className="font-mono font-bold text-slate-900">{s.ticker}</span>
                            <span className="block text-[10px] text-slate-400">
                              {new Date(s.evaluated_at).toLocaleDateString()}
                            </span>
                          </td>
                          <td className="py-3 text-slate-600 hidden sm:table-cell max-w-[160px] truncate pr-4">
                            {s.company_name}
                          </td>
                          <td className="py-3 text-slate-500 text-xs hidden lg:table-cell pr-4">
                            {s.sector}
                          </td>
                          <td className="py-3 text-right tabular-nums font-medium text-slate-700 pr-3">
                            {fmtP(s.current_price)}
                          </td>
                          <td className={cn(
                            "py-3 text-right tabular-nums font-semibold pr-3",
                            s.expected_return >= 0 ? "text-green-600" : "text-red-600"
                          )}>
                            {fmtER(s.expected_return * 100)}
                          </td>
                          <td className="py-3 text-right tabular-nums text-slate-700 pr-3">
                            {s.rr_ratio.toFixed(1)}×
                          </td>
                          <td className="py-3 text-center">
                            <span
                              className="text-xs font-semibold px-2 py-0.5 rounded"
                              style={{ backgroundColor: zoneBg, color: zoneFg }}
                            >
                              {zone}
                            </span>
                          </td>
                          <td className="py-3 text-center pr-2">
                            <span className={cn(
                              "font-bold tabular-nums",
                              s.overall_score >= 65 ? "text-green-700" :
                              s.overall_score >= 45 ? "text-amber-600" : "text-red-600"
                            )}>
                              {Math.round(s.overall_score)}
                            </span>
                          </td>
                          <td className="py-3 pr-4 text-center">
                            <button
                              onClick={e => { e.stopPropagation(); removeSnapshot(s.ticker); }}
                              className="text-slate-300 hover:text-red-400 transition-colors"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              {snapshots.some(s => !s.mc_available) && (
                <p className="px-6 py-2.5 text-xs text-amber-600 border-t border-slate-100">
                  Some stocks lack a Monte Carlo distribution and will be excluded from the portfolio calculation.
                </p>
              )}
            </CardContent>
          </Card>
        )}

        {/* ── Portfolio result ── */}
        {result && (
          <>
            {/* Summary metrics row */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {([
                {
                  label: 'Expected Return',
                  value: fmtER(result.port_er),
                  color: result.port_er >= 0 ? '#15803d' : '#dc2626',
                },
                {
                  label: 'Portfolio Beta',
                  value: result.port_beta.toFixed(2),
                  color: result.port_beta > 1.3 ? '#d97706' : '#0f172a',
                },
                {
                  label: 'Weighted Downside (P5)',
                  value: fmtER(result.port_downside),
                  color: '#dc2626',
                },
                {
                  label: 'Diversification',
                  value: `${result.diversification}/100`,
                  color: result.diversification >= 65 ? '#15803d' :
                         result.diversification >= 40 ? '#d97706' : '#dc2626',
                },
              ] as const).map(({ label, value, color }) => (
                <Card key={label} className="border-slate-200 shadow-sm">
                  <CardContent className="p-4">
                    <p className="text-xs text-slate-400 mb-1">{label}</p>
                    <p className="text-2xl font-bold tabular-nums" style={{ color }}>
                      {value}
                    </p>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Allocation table */}
            <Card className="border-slate-200 shadow-sm overflow-hidden">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-semibold text-slate-700">
                  Portfolio Allocation
                  <span className="ml-2 text-xs font-normal text-slate-400">
                    {result.positions.length} positions
                    {result.cash_pct > 0 && ` + ${result.cash_pct}% cash`}
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50 border-b border-slate-200">
                      <tr>
                        <th className="text-left pl-6 py-2.5 font-semibold text-slate-600">Ticker</th>
                        <th className="text-right py-2.5 pr-5 font-semibold text-slate-600">Weight</th>
                        <th className="text-center py-2.5 font-semibold text-slate-600">Conviction</th>
                        <th className="text-center py-2.5 font-semibold text-slate-600">Priority</th>
                        <th className="text-center py-2.5 font-semibold text-slate-600">Entry</th>
                        <th className="text-center py-2.5 font-semibold text-slate-600">Action</th>
                        <th className="text-right py-2.5 pr-4 font-semibold text-slate-600 hidden sm:table-cell">E[R]</th>
                        <th className="text-right py-2.5 pr-4 font-semibold text-slate-600 hidden sm:table-cell">R/R</th>
                        <th className="text-right py-2.5 pr-5 font-semibold text-slate-600 hidden md:table-cell">Downside</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {result.positions.map((pos, i) => {
                        const pm = priorityMeta(pos.priority);
                        const cm = convictionMeta(pos.conviction);
                        const barW = Math.min(100, (pos.weight / 5) * 100);
                        return (
                          <tr
                            key={pos.ticker}
                            className={cn(
                              "hover:bg-slate-50 cursor-pointer transition-colors",
                              i === 0 && "font-medium"
                            )}
                            onClick={() => router.push(`/report/${pos.ticker}`)}
                          >
                            <td className="pl-6 py-3">
                              <div className="flex items-center gap-3">
                                {/* Weight bar */}
                                <div style={{
                                  width: '40px', height: '4px',
                                  backgroundColor: '#f1f5f9',
                                  borderRadius: '9999px', overflow: 'hidden', flexShrink: 0,
                                }}>
                                  <div style={{
                                    width: `${barW}%`, height: '100%',
                                    backgroundColor: '#3b82f6',
                                    borderRadius: '9999px',
                                  }} />
                                </div>
                                <div>
                                  <span className="font-mono font-bold text-slate-900">{pos.ticker}</span>
                                  <span className="text-[10px] text-slate-400 block hidden sm:block">
                                    {pos.sector}
                                  </span>
                                </div>
                              </div>
                            </td>
                            <td className="py-3 text-right tabular-nums pr-5">
                              <span className="font-bold text-slate-900">{pos.weight.toFixed(1)}%</span>
                            </td>
                            <td className="py-3 text-center">
                              <span
                                className="text-xs font-semibold px-2 py-0.5 rounded"
                                style={{ backgroundColor: cm.bg, color: cm.fg }}
                              >
                                {pos.conviction}
                              </span>
                            </td>
                            <td className="py-3 text-center">
                              <span
                                className="text-xs font-bold px-2 py-0.5 rounded"
                                style={{ backgroundColor: pm.bg, color: pm.fg }}
                              >
                                {pos.priority}
                              </span>
                            </td>
                            <td className="py-3 text-center text-xs text-slate-600">
                              {pos.entry_style}
                            </td>
                            <td className="py-3 text-center">
                              {(() => {
                                const sig = pos.sell_signal;
                                const bg =
                                  sig.action === 'EXIT' ? '#fee2e2' :
                                  sig.action === 'TRIM' ? '#fef9c3' : '#f0fdf4';
                                const fg =
                                  sig.action === 'EXIT' ? '#b91c1c' :
                                  sig.action === 'TRIM' ? '#a16207' : '#15803d';
                                const label =
                                  sig.action === 'HOLD' ? 'HOLD' :
                                  sig.action === 'EXIT' ? `EXIT ${sig.trim_pct}%` :
                                  `TRIM ${sig.trim_pct}%`;
                                return (
                                  <span
                                    className="text-xs font-bold px-2 py-0.5 rounded whitespace-nowrap"
                                    style={{ backgroundColor: bg, color: fg }}
                                    title={sig.reason}
                                  >
                                    {label}
                                  </span>
                                );
                              })()}
                            </td>
                            <td className={cn(
                              "py-3 text-right tabular-nums pr-4 hidden sm:table-cell font-semibold",
                              pos.expected_return >= 0 ? "text-green-600" : "text-red-600"
                            )}>
                              {fmtER(pos.expected_return * 100)}
                            </td>
                            <td className="py-3 text-right tabular-nums pr-4 text-slate-700 hidden sm:table-cell">
                              {pos.rr_ratio.toFixed(1)}×
                            </td>
                            <td className="py-3 text-right tabular-nums pr-5 text-red-600 font-semibold hidden md:table-cell">
                              {fmtER(pos.downside_pct)}
                            </td>
                          </tr>
                        );
                      })}

                      {/* Cash row */}
                      {result.cash_pct > 0 && (
                        <tr className="bg-slate-50">
                          <td className="pl-6 py-3">
                            <div className="flex items-center gap-3">
                              <div style={{ width: '40px', height: '4px', backgroundColor: '#f1f5f9', borderRadius: '9999px', overflow: 'hidden', flexShrink: 0 }}>
                                <div style={{ width: `${Math.min(100, (result.cash_pct / 5) * 100)}%`, height: '100%', backgroundColor: '#94a3b8', borderRadius: '9999px' }} />
                              </div>
                              <span className="font-semibold text-slate-400">CASH</span>
                            </div>
                          </td>
                          <td className="py-3 text-right tabular-nums pr-5 font-bold text-slate-500">
                            {result.cash_pct.toFixed(1)}%
                          </td>
                          <td colSpan={7} className="py-3 pl-3 text-xs text-slate-400 italic">
                            Buffer — avg composite score &lt; 30%
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            {/* Sell Discipline Alerts */}
            {result.positions.some(p => p.sell_signal.action !== 'HOLD') && (
              <Card className="border-red-100 shadow-sm" style={{ backgroundColor: '#fff9f9' }}>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-semibold" style={{ color: '#b91c1c' }}>
                    <AlertTriangle className="inline w-4 h-4 mr-1.5" style={{ color: '#ef4444' }} />
                    Sell Discipline Alerts
                  </CardTitle>
                </CardHeader>
                <CardContent className="px-6 pb-5 space-y-2">
                  {result.positions
                    .filter(p => p.sell_signal.action !== 'HOLD')
                    .map(pos => {
                      const sig = pos.sell_signal;
                      const actionBg = sig.action === 'EXIT' ? '#fee2e2' : '#fef9c3';
                      const actionFg = sig.action === 'EXIT' ? '#b91c1c' : '#a16207';
                      const triggerLabel =
                        sig.trigger === 'overvalued' ? 'Overvalued' :
                        sig.trigger === 'valuation'  ? 'Valuation'  :
                        sig.trigger === 'momentum'   ? 'Momentum'   : 'Risk';
                      return (
                        <div
                          key={pos.ticker}
                          className="flex flex-wrap items-start gap-x-3 gap-y-1 py-2.5 border-b border-red-100 last:border-0"
                        >
                          <span className="font-mono font-bold text-slate-900 w-14 shrink-0">
                            {pos.ticker}
                          </span>
                          <span
                            className="text-xs font-bold px-2 py-0.5 rounded shrink-0"
                            style={{ backgroundColor: actionBg, color: actionFg }}
                          >
                            {sig.action === 'HOLD' ? 'HOLD' :
                             sig.action === 'EXIT' ? `EXIT ${sig.trim_pct}%` :
                             `TRIM ${sig.trim_pct}%`}
                          </span>
                          <span
                            className="text-xs px-1.5 py-0.5 rounded shrink-0"
                            style={{ backgroundColor: '#f1f5f9', color: '#64748b' }}
                          >
                            {triggerLabel}
                          </span>
                          <span className="text-xs text-slate-600 flex-1 min-w-[180px]">
                            {sig.reason}
                          </span>
                        </div>
                      );
                    })}
                  <p className="text-[10px] text-slate-400 pt-1">
                    Percentages refer to portion of the position to reduce. Hover the Action badge in the table for full context.
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Sector exposure + Risk summary side by side */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="border-slate-200 shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-semibold text-slate-700">Sector Exposure</CardTitle>
                </CardHeader>
                <CardContent className="px-6 pb-5 space-y-3">
                  {Object.entries(result.sector_exposure)
                    .sort((a, b) => b[1] - a[1])
                    .map(([sector, pct]) => (
                      <div key={sector} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-600">{sector}</span>
                          <span className={cn(
                            "font-semibold tabular-nums",
                            pct > 20 ? "text-amber-600" : "text-slate-700"
                          )}>
                            {pct.toFixed(1)}%
                            {pct > 20 && <span className="ml-1 text-amber-500">⚠</span>}
                          </span>
                        </div>
                        <div style={{
                          height: '6px', borderRadius: '9999px',
                          backgroundColor: '#f1f5f9', overflow: 'hidden',
                        }}>
                          <div style={{
                            height: '100%',
                            width: `${Math.min(100, (pct / 25) * 100)}%`,
                            borderRadius: '9999px',
                            backgroundColor: pct > 20 ? '#f59e0b' : '#3b82f6',
                          }} />
                        </div>
                      </div>
                    ))}
                </CardContent>
              </Card>

              <Card className="border-slate-200 shadow-sm">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-semibold text-slate-700">Risk Summary</CardTitle>
                </CardHeader>
                <CardContent className="px-6 pb-5">
                  <dl className="space-y-2.5 text-sm">
                    {([
                      ['Portfolio Beta',           result.port_beta.toFixed(2),      result.port_beta > 1.3 ? 'text-amber-700' : 'text-slate-800'],
                      ['Weighted E[R]',             fmtER(result.port_er),            result.port_er >= 0 ? 'text-green-700' : 'text-red-600'],
                      ['Weighted Downside (P5)',    fmtER(result.port_downside),      'text-red-600'],
                      ['Diversification Score',     `${result.diversification}/100`,  result.diversification >= 65 ? 'text-green-700' : result.diversification >= 40 ? 'text-amber-600' : 'text-red-600'],
                      ['HIGH priority positions',   String(result.positions.filter(p => p.priority === 'HIGH').length), 'text-green-700'],
                      ['Positions above P50',       String(result.positions.filter(p => p.priority === 'LOW').length),  result.positions.filter(p => p.priority === 'LOW').length > result.positions.length / 2 ? 'text-amber-600' : 'text-slate-700'],
                    ] as const).map(([label, value, cls]) => (
                      <div key={label}>
                        <div className="flex justify-between">
                          <dt className="text-slate-500">{label}</dt>
                          <dd className={cn("font-semibold tabular-nums", cls)}>{value}</dd>
                        </div>
                        <Separator className="mt-2.5" />
                      </div>
                    ))}
                  </dl>

                  {result.port_beta > 1.3 && (
                    <div className="flex items-start gap-2 p-2.5 bg-amber-50 rounded-lg border border-amber-100 mt-4">
                      <AlertTriangle className="w-3.5 h-3.5 text-amber-500 mt-0.5 shrink-0" />
                      <p className="text-xs text-amber-700">
                        Elevated portfolio beta ({result.port_beta.toFixed(2)}). Consider adding defensive names or increasing cash.
                      </p>
                    </div>
                  )}
                  {result.diversification >= 70 && (
                    <div className="flex items-start gap-2 p-2.5 bg-green-50 rounded-lg border border-green-100 mt-3">
                      <CheckCircle2 className="w-3.5 h-3.5 text-green-500 mt-0.5 shrink-0" />
                      <p className="text-xs text-green-700">
                        Well-diversified — no single stock or sector dominates.
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Execution Priority Guide */}
            <Card className="border-slate-200 shadow-sm">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-semibold text-slate-700">
                  <TrendingUp className="inline w-4 h-4 mr-1.5 text-slate-400" />
                  Execution Priority Guide
                </CardTitle>
              </CardHeader>
              <CardContent className="px-6 pb-5">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  {([
                    {
                      priority: 'HIGH' as const,
                      title: 'Buy Now',
                      desc: 'Price ≤ P25 — trading at or below deep value. Full allocation justified. Aggressive or staged entry.',
                    },
                    {
                      priority: 'MEDIUM' as const,
                      title: 'Scale In',
                      desc: 'Price between P25–P50 — attractive but not deep value. Stage entry over 2–4 weeks.',
                    },
                    {
                      priority: 'LOW' as const,
                      title: 'Wait',
                      desc: 'Price above P50 — above fair value. Monitor only. Enter on meaningful pullback toward P40–P50.',
                    },
                  ] as const).map(({ priority, title, desc }) => {
                    const pm = priorityMeta(priority);
                    return (
                      <div key={priority} className="space-y-2">
                        <div className="flex items-center gap-2">
                          <span
                            className="text-xs font-bold px-2 py-0.5 rounded"
                            style={{ backgroundColor: pm.bg, color: pm.fg }}
                          >
                            {priority}
                          </span>
                          <span className="text-sm font-semibold text-slate-700">{title}</span>
                        </div>
                        <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </>
        )}

        {/* Footer */}
        <div className="border-t border-slate-200 pt-4 pb-10 text-xs text-slate-400 text-center">
          Score = 35% E[R] + 25% R/R + 15% Momentum + 15% Risk Score + 10% Quality
          &nbsp;·&nbsp; Max 5%/position &nbsp;·&nbsp; Sector cap 25% &nbsp;·&nbsp; Beta&gt;1.5 cap 3%
        </div>
      </main>
    </div>
  );
}
