"""
validate_example.py — validate ticker_analysis examples against the v1 schema.

Run from stock_agent/:
    python3 schemas/validate_example.py                 # validates all examples
    python3 schemas/validate_example.py msft            # validates msft_ticker_analysis_v1.json
    python3 schemas/validate_example.py nflx msft       # validates specific tickers
"""
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

here = Path(__file__).parent

with open(here / "ticker_analysis_v1.schema.json") as f:
    schema = json.load(f)

try:
    Draft202012Validator.check_schema(schema)
except SchemaError as e:
    print(f"  ✗ Schema is invalid: {e.message}")
    sys.exit(1)

v = Draft202012Validator(schema)

# Resolve target files from args or default to all examples
args = sys.argv[1:]
if args:
    targets = [here / "examples" / f"{ticker.lower()}_ticker_analysis_v1.json"
               for ticker in args]
else:
    targets = sorted((here / "examples").glob("*_ticker_analysis_v1.json"))

if not targets:
    print("  ✗ No example files found")
    sys.exit(1)

failed = 0
for path in targets:
    if not path.exists():
        print(f"  ✗ {path.name}: file not found")
        failed += 1
        continue

    with open(path) as f:
        example = json.load(f)

    errors = sorted(v.iter_errors(example), key=lambda e: list(e.absolute_path))
    if errors:
        print(f"  ✗ {path.name}: {len(errors)} validation error(s):")
        for e in errors:
            loc = " → ".join(str(p) for p in e.absolute_path) or "(root)"
            print(f"      [{loc}] {e.message}")
        failed += 1
    else:
        print(f"✓ {path.name} validates against schema")

if failed:
    sys.exit(1)
