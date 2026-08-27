#!/usr/bin/env python
"""Measure how well Japanese queries reach what their English counterparts find.

The indexed text is English — product names and descriptions from the DAZ store,
path words for filesystem products. A Japanese query therefore has no lexical
overlap to lean on and must be bridged by the embedding model alone. The English
phrasing of the same intent is the ceiling: whatever it returns is what the
Japanese query *should* return.

So the metric is agreement, not a hand-written relevance list — for each pair, how
many of the English query's top 10 the Japanese query also puts in its top 10.
That survives a change of model, which absolute similarity scores do not (see
Handover.pyn @js-switch).

Usage:
    python check_crosslingual.py                       # needs a running server
    python check_crosslingual.py --url http://host:8000
    python check_crosslingual.py --json > e5.json      # for diffing across models

Note: it does not measure whether the catalogue *has* a good answer. For '和室'
it does not — nothing indexed mentions tatami, shoji or ryokan — so the ceiling
there is 'Japanese Restaurant / Onsen / Bath', not an actual Japanese room.
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

PAIRS = [
    ("和室", "Japanese room"),
    ("和室の背景", "Japanese style room interior"),
    ("ロングヘアの女性キャラ", "long hair female character"),
    ("中世の鎧を着た戦士", "medieval armored warrior"),
    ("森の風景", "forest landscape"),
    ("学校の教室", "school classroom"),
]

TOP_N = 10


def search(url: str, query: str, max_results: int) -> list:
    payload = json.dumps({
        "query": query, "min_relevance": 0.0, "max_results": max_results,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/v1/search",
        data=payload, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)["results"]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--max-results", type=int, default=5000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    report: dict = {"url": args.url, "pairs": {}}
    try:
        with urllib.request.urlopen(
                f"{args.url.rstrip('/')}/api/v1/settings", timeout=30) as resp:
            report["settings"] = json.loads(resp.read())
    except urllib.error.URLError as exc:
        print(f"ERROR: cannot reach {args.url} — is the server running? ({exc})",
              file=sys.stderr)
        return 1

    for ja, en in PAIRS:
        ja_hits = search(args.url, ja, args.max_results)
        en_hits = search(args.url, en, args.max_results)
        ja_top = [h["name"] for h in ja_hits[:TOP_N]]
        en_top = [h["name"] for h in en_hits[:TOP_N]]
        report["pairs"][ja] = {
            "english": en,
            "overlap": len(set(ja_top) & set(en_top)),
            "ja_top": ja_top,
            "en_top": en_top,
            "ja_top_score": round(ja_hits[0]["relevance_score"], 4) if ja_hits else None,
        }

    overlaps = [p["overlap"] for p in report["pairs"].values()]
    report["mean_overlap"] = round(sum(overlaps) / len(overlaps), 2)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    model = (report.get("settings") or {}).get("embedding_model", "?")
    print(f"model (as reported by /settings): {model}")
    print(f"{'JA query':<26}{'overlap':>9}{'top score':>11}  JA top hit")
    print("-" * 96)
    for ja, p in report["pairs"].items():
        print(f"{ja:<26}{p['overlap']:>6}/{TOP_N}{p['ja_top_score']:>11}  "
              f"{p['ja_top'][0][:44] if p['ja_top'] else '-'}")
    print(f"\nmean top-{TOP_N} overlap with the English phrasing: "
          f"{report['mean_overlap']}/{TOP_N}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
