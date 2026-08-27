#!/usr/bin/env python
"""Show what Japanese queries are actually translated to, so bad ones can be fixed.

A Japanese search that returns something unrelated almost always fails at the
translation step, not the index (see Handover.pyn @nsfw-gap). This prints the
English each query becomes, and — with --search — what that English then finds,
which is what you need to decide whether a term belongs in the glossary.

Usage:
    python check_translation.py 緊縛 鞭 猿ぐつわ
    python check_translation.py --search 和室          # also show what it returns
    python check_translation.py --file terms.txt      # one query per line
    python check_translation.py --glossary            # check the shipped glossary

When a translation is wrong, add the term to query_glossary.ja-en.json with the
English wording the catalogue actually uses:

    {"terms": {"猿ぐつわ": "gag"}}

Matching is on the whole query, so the entry answers '猿ぐつわ' and leaves longer
phrases containing it to the model.

The translator runs in-process — no server needed unless you pass --search.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def search(url: str, query: str, limit: int = 5) -> list:
    payload = json.dumps({
        "query": query, "min_relevance": 0.0, "max_results": 5000,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{url.rstrip('/')}/api/v1/search",
        data=payload, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return [r["name"] for r in json.load(response)["results"][:limit]]


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queries", nargs="*", help="Japanese queries to translate")
    parser.add_argument("--file", help="Read queries from a file, one per line")
    parser.add_argument("--glossary", action="store_true",
                        help="Translate every key in the shipped glossary")
    parser.add_argument("--search", action="store_true",
                        help="Also show the top hits (needs a running server)")
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    import query_translation as qt

    queries = list(args.queries)
    if args.file:
        queries += [line.strip() for line in
                    Path(args.file).read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    if args.glossary:
        queries += list(qt.load_glossary())
    if not queries:
        parser.error("give at least one query, or --file / --glossary")

    glossary = qt.load_glossary()
    print(f"backend: {qt.active_backend() or '(none — queries pass through)'}")
    print(f"glossary: {len(glossary)} term(s)\n")

    for query in queries:
        started = time.perf_counter()
        english, source = qt.translate_query(query)
        elapsed = (time.perf_counter() - started) * 1000
        how = "glossary" if query.strip() in glossary else (
            "model" if source else "unchanged")
        print(f"{query}")
        print(f"   -> {english!r}   [{how}, {elapsed:.0f} ms]")
        if args.search:
            try:
                for name in search(args.url, query):
                    print(f"      {name}")
            except urllib.error.URLError as exc:
                print(f"      (server unreachable at {args.url}: {exc})")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
