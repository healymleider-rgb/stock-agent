import type {
  EvaluateResponse,
  HistoryEntry,
  JobStatus,
  StartEvaluationResponse,
} from "./types";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Debug: log base URL on first import (visible in browser console)
if (typeof window !== "undefined") {
  console.log("[StockEval] API base URL:", BASE_URL);
}

// ── Low-level fetchers ────────────────────────────────────────────────────────

export async function startEvaluation(ticker: string): Promise<string> {
  const res = await fetch(`${BASE_URL}/api/evaluate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker: ticker.toUpperCase() }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Failed to start evaluation: ${res.status} ${text}`);
  }
  const data: StartEvaluationResponse = await res.json();
  return data.job_id;
}

export async function pollJob(jobId: string): Promise<JobStatus> {
  const res = await fetch(`${BASE_URL}/api/jobs/${jobId}`);
  if (!res.ok) {
    throw new Error(`Failed to poll job ${jobId}: ${res.status}`);
  }
  return res.json();
}

export async function getHistory(ticker: string): Promise<HistoryEntry[]> {
  const res = await fetch(
    `${BASE_URL}/api/history/${ticker.toUpperCase()}`
  );
  if (!res.ok) {
    return [];
  }
  return res.json();
}

// ── Convenience: start + poll ─────────────────────────────────────────────────

/**
 * Run a full evaluation for a ticker.
 * Polls every 1.5 seconds and calls onProgress(step, progress) on each update.
 * Resolves with the final EvaluateResponse when complete.
 */
export async function evaluate(
  ticker: string,
  onProgress?: (step: string, progress: number) => void
): Promise<EvaluateResponse> {
  const jobId = await startEvaluation(ticker);

  return new Promise<EvaluateResponse>((resolve, reject) => {
    const poll = async () => {
      try {
        const status = await pollJob(jobId);

        onProgress?.(status.step, status.progress);

        if (status.status === "complete" && status.result) {
          resolve(status.result);
          return;
        }

        if (status.status === "error") {
          reject(new Error(status.error || "Evaluation failed."));
          return;
        }

        // Still running — poll again after 1.5s
        setTimeout(poll, 1500);
      } catch (err) {
        reject(err);
      }
    };

    // Start polling immediately
    poll();
  });
}
