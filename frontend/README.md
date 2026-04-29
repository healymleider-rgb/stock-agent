# StockEval Frontend

Institutional-grade equity research platform — full-stack web UI for the StockEval multi-agent system.

## Setup

### 1. Install Python dependencies

```bash
pip install fastapi uvicorn
```

### 2. Start the API server

```bash
cd ~/stock_eval
uvicorn web_api:app --reload --port 8000
```

The API will be available at http://localhost:8000. You can verify it's running at http://localhost:8000/api/health.

### 3. Install frontend dependencies

```bash
cd ~/stock_eval/frontend
npm install
```

### 4. Start the frontend dev server

```bash
npm run dev
```

### 5. Open the app

Navigate to http://localhost:3000

## Usage

1. Enter a ticker symbol (e.g. `AAPL`, `NVDA`, `MSFT`) in the search box on the landing page.
2. Press **Enter** or click **Generate Report**.
3. Wait for the AI agents to complete their analysis (typically 30–90 seconds).
4. Review the full report: score overview, category breakdown, fundamentals, valuation range, peer comparison, and macro analysis.
5. Click **PDF** in the top bar to print or save the report as a PDF.

## Architecture

```
web_api.py              FastAPI server (background job queue)
  └── POST /api/evaluate       → start evaluation, returns job_id
  └── GET  /api/jobs/{job_id}  → poll job status + result
  └── GET  /api/history/{ticker} → past evaluations

frontend/
  app/
    page.tsx            Landing page (ticker input, recent searches)
    report/[ticker]/
      page.tsx          Full report page
  components/ui/        ShadCN UI components
  lib/
    api.ts              API client (startEvaluation, pollJob, evaluate)
    types.ts            TypeScript types
    utils.ts            Formatting + color utilities
```

## Environment Variables

Create a `.env.local` file in the `frontend/` directory to override the API URL:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```
