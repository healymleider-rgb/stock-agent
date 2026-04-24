# stock_agent

StockEval backend. Runs equity evaluation via a multi-agent pipeline,
validates outputs via a 10-block pre-report gate, and exposes results
through a FastAPI backend. Portfolio engine (ticker_analysis_v1.json
producer) is in active development.

## Key documents

- [schemas/README.md](schemas/README.md) — Solo Mode output contract (`ticker_analysis_v1.schema.json`),
  validation rules N1/F1/S1, block coverage by layer
- [FINDINGS.md](FINDINGS.md) — Upstream pipeline bugs surfaced by the validation gate;
  issues that cannot be fixed within `validation_gate.py` and need upstream owners

## Running

```bash
source .venv/bin/activate
uvicorn web_api:app --port 8000
```

## Tests

```bash
python3 -m pytest tests/ -v
```
