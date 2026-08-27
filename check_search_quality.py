#!/usr/bin/env python
"""Measure search quality and latency against a running server — before/after a model swap.

Switching the embedding model changes the similarity distribution, so the relevance
thresholds that made sense for one model are meaningless for another. Run this against
the current index, save the output, switch the model, re-embed, and run it again.

Usage:
    python check_search_quality.py                       # default queries, localhost:8000
    python check_search_quality.py --url http://host:8000
    python check_search_quality.py --query "和室の背景" --query office
    python check_search_quality.py --json > before.json  # machine-readable, for diffing

Requires only the server to be up; it talks to /api/v1/search over HTTP.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# English queries carried over from the search-cap investigation (see Handover.pyn
# @bug-search-cap) plus the Japanese free-text queries the swap is meant to enable.
DEFAULT_QUERIES = [
    "office",
    "dress",
    "forest",
    "和室の背景",
    "ロングヘアの女性キャラ",
    "中世の鎧を着た戦士",
]

THRESHOLDS = (0.0, 0.5, 0.6, 0.7)


def search(url: str, query: str, min_relevance: float, max_results: int) -> dict:
    payload = json.dumps({
        "query": query,
        "min_relevance": min_relevance,
        "max_results": max_results,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/v1/search",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    elapsed = time.perf_counter() - t0
    body = json.loads(raw)
    return {
        "total": body.get("total", 0),
        "seconds": round(elapsed, 3),
        "payload_kb": round(len(raw) / 1024, 1),
        "top": [r.get("name") for r in body.get("results", [])[:5]],
    }


def main() -> int:
    # Japanese queries and the em dash below are outside cp932, which is still the
    # console codepage on a Japanese Windows install. Degrade unencodable characters
    # rather than let a UnicodeEncodeError abort the run — same guard as main.py.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000", help="Server base URL")
    ap.add_argument("--query", action="append", dest="queries", metavar="TEXT",
                    help="Query to run (repeatable). Overrides the default set.")
    ap.add_argument("--max-results", type=int, default=5000,
                    help="Candidate pool size sent to the server (default: 5000)")
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table")
    args = ap.parse_args()

    queries = args.queries or DEFAULT_QUERIES
    report: dict = {"url": args.url, "queries": {}}

    try:
        with urllib.request.urlopen(f"{args.url.rstrip('/')}/api/v1/settings", timeout=30) as resp:
            report["settings"] = json.loads(resp.read())
    except urllib.error.URLError as exc:
        print(f"ERROR: cannot reach {args.url} — is the server running? ({exc})", file=sys.stderr)
        return 1

    for query in queries:
        per_threshold = {}
        for threshold in THRESHOLDS:
            per_threshold[f"{threshold:.0%}"] = search(
                args.url, query, threshold, args.max_results)
        report["queries"][query] = per_threshold

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    model = (report.get("settings") or {}).get("embedding_model", "?")
    print(f"model (as reported by /settings): {model}")
    print(f"{'query':<24}{'min':>6}{'hits':>8}{'sec':>7}{'KB':>9}  top hit")
    print("-" * 96)
    for query, per_threshold in report["queries"].items():
        for threshold, row in per_threshold.items():
            top = (row["top"][0] or "") if row["top"] else "—"
            print(f"{query:<24}{threshold:>6}{row['total']:>8}{row['seconds']:>7}"
                  f"{row['payload_kb']:>9}  {top[:40]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
