"use client";

import { useState, useEffect, useRef, KeyboardEvent } from "react";
import { useRouter } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  Search,
  TrendingUp,
  BarChart2,
  FileText,
  Clock,
} from "lucide-react";

const MAX_RECENT = 5;
const LS_KEY = "stockeval_recent";

function getRecent(): string[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) || "[]");
  } catch {
    return [];
  }
}

function addRecent(ticker: string) {
  const existing = getRecent().filter((t) => t !== ticker);
  const updated = [ticker, ...existing].slice(0, MAX_RECENT);
  localStorage.setItem(LS_KEY, JSON.stringify(updated));
}

export default function HomePage() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  const [ticker, setTicker] = useState("");
  const [recent, setRecent] = useState<string[]>([]);

  useEffect(() => {
    setRecent(getRecent());
    inputRef.current?.focus();
  }, []);

  const handleSubmit = (tickerVal?: string) => {
    const t = (tickerVal ?? ticker).trim().toUpperCase();
    if (!t) return;
    addRecent(t);
    setRecent(getRecent());
    router.push(`/report/${t}`);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") handleSubmit();
  };

  return (
    <main className="min-h-screen bg-white flex flex-col">
      {/* ── Hero ────────────────────────────────────────────────── */}
      <section className="flex-1 flex flex-col items-center justify-center px-4 pt-20 pb-16">
        <div className="w-full max-w-2xl mx-auto text-center space-y-8 animate-fade-in">
          {/* Wordmark */}
          <div className="space-y-3">
            <div className="inline-flex items-center gap-2 text-xs font-semibold tracking-widest uppercase text-slate-400 mb-2">
              <span className="w-6 h-px bg-slate-300" />
              Institutional Research
              <span className="w-6 h-px bg-slate-300" />
            </div>
            <h1 className="text-5xl sm:text-6xl font-semibold tracking-tight text-slate-900">
              StockEval
            </h1>
            <p className="text-lg text-slate-500 max-w-md mx-auto leading-relaxed">
              Institutional-grade equity research, powered by AI agents.
            </p>
          </div>

          {/* Search bar */}
          <div className="space-y-3">
            <div className="flex gap-2 p-1.5 bg-slate-50 border border-slate-200 rounded-2xl shadow-sm">
              <div className="flex-1 flex items-center gap-3 px-4">
                <Search className="w-4 h-4 text-slate-400 shrink-0" />
                <input
                  ref={inputRef}
                  type="text"
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  onKeyDown={handleKeyDown}
                  placeholder="AAPL"
                  maxLength={10}
                  className="flex-1 bg-transparent text-2xl font-semibold text-slate-900 placeholder:text-slate-300 outline-none tracking-widest"
                  autoComplete="off"
                  spellCheck={false}
                />
              </div>
              <button
                onClick={() => handleSubmit()}
                disabled={!ticker.trim()}
                className={cn(
                  "px-6 py-3 rounded-xl font-semibold text-sm transition-all",
                  "bg-slate-900 text-white hover:bg-slate-700 active:scale-95",
                  "disabled:opacity-40 disabled:cursor-not-allowed disabled:scale-100"
                )}
              >
                Generate Report
              </button>
            </div>

            <p className="text-xs text-slate-400">
              Press{" "}
              <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded text-[10px] font-mono">
                Enter
              </kbd>{" "}
              to submit
            </p>
          </div>

          {/* Recent searches */}
          {recent.length > 0 && (
            <div className="flex items-center gap-2 flex-wrap justify-center">
              <span className="text-xs text-slate-400 flex items-center gap-1">
                <Clock className="w-3 h-3" /> Recent:
              </span>
              {recent.map((t) => (
                <button
                  key={t}
                  onClick={() => handleSubmit(t)}
                  className="text-xs font-mono font-semibold px-3 py-1 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-full transition-colors"
                >
                  {t}
                </button>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* ── How it works ────────────────────────────────────────── */}
      <section className="border-t border-slate-100 py-16 px-4">
        <div className="max-w-4xl mx-auto">
          <p className="text-center text-xs font-semibold tracking-widest uppercase text-slate-400 mb-10">
            How it works
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-8">
            {[
              {
                icon: <Search className="w-5 h-5" />,
                step: "01",
                title: "Enter a ticker",
                desc: "Type any US equity ticker symbol — large cap, mid cap, or small cap.",
              },
              {
                icon: <BarChart2 className="w-5 h-5" />,
                step: "02",
                title: "AI agents analyze",
                desc: "Six specialist agents run fundamental, technical, macro, sentiment, risk, and market analysis in parallel.",
              },
              {
                icon: <FileText className="w-5 h-5" />,
                step: "03",
                title: "Professional report",
                desc: "Receive a full investment memo with scored categories, valuation range, peer comparison, and a clear stance.",
              },
            ].map(({ icon, step, title, desc }) => (
              <div key={step} className="flex flex-col gap-3">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-xl bg-slate-100 flex items-center justify-center text-slate-600">
                    {icon}
                  </div>
                  <span className="text-xs font-mono text-slate-300 font-semibold">{step}</span>
                </div>
                <div>
                  <h3 className="font-semibold text-slate-900 mb-1">{title}</h3>
                  <p className="text-sm text-slate-500 leading-relaxed">{desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────────────────────── */}
      <footer className="border-t border-slate-100 py-6 px-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <p className="text-xs text-slate-400">StockEval — AI Research Platform</p>
          <div className="flex items-center gap-1 text-xs text-slate-400">
            <TrendingUp className="w-3 h-3" />
            Powered by FMP + AI Agents
          </div>
        </div>
      </footer>
    </main>
  );
}
