"""Reports which content on disk is missing from the DAZ CMS database.

Walks the DAZ content directories, subtracts everything the CMS already tracks, and
groups the remainder into the logical products the indexer would create. Use it to
sanity-check and tune the grouping heuristics (SCAN_MIN_PRODUCT_DEPTH,
CATEGORY_WORDS) before running `vab load --phase scan`.

Usage:
    python analyze_content_roots.py [DIR1 DIR2 ...] [--show-dirs]

If no directories are given, uses the content roots configured in the DAZ CMS.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from managers import filesystem_scanner  # noqa: E402
from managers.daz_db_analyzer import DazDBAnalyzer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="*",
                        help="Content roots to scan (default: from DAZ CMS database)")
    parser.add_argument("--show-dirs", action="store_true",
                        help="List every constituent directory under each product")
    args = parser.parse_args()

    # Product names routinely contain characters the console codepage cannot encode
    # (en dashes, accents); degrade those rather than aborting the report.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    analyzer = DazDBAnalyzer()
    products = filesystem_scanner.scan(analyzer, roots=args.dirs or None)

    sep = "-" * 44
    total_files = sum(p["file_count"] for p in products)
    print(f"""
{sep}
 Untracked content
{sep}
 Logical products identified: {len(products):>7,}
 Untracked files covered:     {total_files:>7,}
{sep}
""")

    if not products:
        return

    print(f"  {'Files':>5}  {'Vendor / Artist':<28}  Product")
    print(f"  {'-'*5}  {'-'*28}  {'-'*30}")
    for p in products:
        vendor_str = (p["vendor"] or "").ljust(28)
        ctx = f"  [{', '.join(p['context'])}]" if p["context"] else ""
        print(f"  {p['file_count']:>5}  {vendor_str}  {p['name']}{ctx}")
        if args.show_dirs:
            for d in p["all_dirs"]:
                print(f"         {'':28}    {d}")

    analyzer.close()


if __name__ == "__main__":
    main()
