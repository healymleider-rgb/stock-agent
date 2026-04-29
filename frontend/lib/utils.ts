import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

// ── Tailwind class helper ─────────────────────────────────────────────────────

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// ── Number formatting ─────────────────────────────────────────────────────────

/**
 * Format a raw number into a compact financial string.
 * e.g. 2_800_000_000_000 → "$2.80T"
 */
export function formatLargeNumber(n: number | null | undefined): string {
  if (n == null || isNaN(n)) return "N/A";
  const abs = Math.abs(n);
  const sign = n < 0 ? "-" : "";
  if (abs >= 1e12) return `${sign}$${(abs / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(2)}`;
}

/**
 * Format a 0-100 score as "72/100".
 */
export function formatPercent(n: number | null | undefined): string {
  if (n == null || isNaN(n)) return "N/A";
  return `${Math.round(n)}/100`;
}

/**
 * Format a price number as currency.
 */
export function formatPrice(n: number | null | undefined): string {
  if (n == null || isNaN(n)) return "N/A";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(n);
}

/**
 * Format a decimal ratio as a percentage string.
 * e.g. 0.15 → "15.0%"
 */
export function formatRatio(n: number | null | undefined): string {
  if (n == null || isNaN(n)) return "N/A";
  return `${(n * 100).toFixed(1)}%`;
}

/**
 * Format a multiplier (P/E, P/S, etc.)
 */
export function formatMultiple(n: number | null | undefined, suffix = "x"): string {
  if (n == null || isNaN(n)) return "N/A";
  return `${n.toFixed(1)}${suffix}`;
}

// ── Color helpers ─────────────────────────────────────────────────────────────

/**
 * Returns Tailwind text + background color classes for stance.
 */
export function stanceColor(stance: string): {
  text: string;
  bg: string;
  border: string;
  dot: string;
} {
  switch (stance) {
    case "Bullish":
      return {
        text: "text-green-700",
        bg: "bg-green-50",
        border: "border-green-200",
        dot: "bg-green-500",
      };
    case "Bearish":
      return {
        text: "text-red-700",
        bg: "bg-red-50",
        border: "border-red-200",
        dot: "bg-red-500",
      };
    default:
      return {
        text: "text-slate-600",
        bg: "bg-slate-50",
        border: "border-slate-200",
        dot: "bg-slate-400",
      };
  }
}

/**
 * Returns a Tailwind color class for a 0-100 score.
 * ≥65 → green, 45-64 → amber, <45 → red
 */
export function scoreColor(score: number): {
  text: string;
  bg: string;
  bar: string;
  ring: string;
} {
  if (score >= 65) {
    return {
      text: "text-green-700",
      bg: "bg-green-50",
      bar: "bg-green-500",
      ring: "#16a34a",
    };
  }
  if (score >= 45) {
    return {
      text: "text-amber-700",
      bg: "bg-amber-50",
      bar: "bg-amber-500",
      ring: "#d97706",
    };
  }
  return {
    text: "text-red-700",
    bg: "bg-red-50",
    bar: "bg-red-500",
    ring: "#dc2626",
  };
}

/**
 * Returns a human-readable label for a 0-100 score.
 */
export function scoreLabel(score: number): string {
  if (score >= 75) return "Strong";
  if (score >= 65) return "Good";
  if (score >= 55) return "Fair";
  if (score >= 45) return "Weak";
  return "Poor";
}

/**
 * Category display name mapping.
 */
export function categoryLabel(key: string): string {
  const map: Record<string, string> = {
    valuation: "Valuation",
    growth: "Growth",
    profitability: "Profitability",
    financial_health: "Financial Health",
    momentum: "Momentum",
    risk: "Risk",
  };
  return map[key] ?? key;
}

/**
 * Macro score color.
 */
export function macroScoreColor(score: number): string {
  if (score >= 65) return "text-green-700";
  if (score >= 45) return "text-amber-700";
  return "text-red-700";
}

/**
 * Recession risk color.
 */
export function recessionRiskColor(level: string): string {
  const l = level.toLowerCase();
  if (l.includes("low")) return "text-green-700 bg-green-50 border-green-200";
  if (l.includes("high")) return "text-red-700 bg-red-50 border-red-200";
  return "text-amber-700 bg-amber-50 border-amber-200";
}

/**
 * Safely round a number to fixed decimals.
 */
export function safeFixed(n: number | null | undefined, decimals = 1): string {
  if (n == null || isNaN(n)) return "N/A";
  return n.toFixed(decimals);
}
