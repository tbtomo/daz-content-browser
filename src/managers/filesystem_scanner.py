"""Discovers content that exists on disk but is not registered in the DAZ CMS.

DAZ Studio's Smart Content pane is backed by the CMS PostgreSQL database; its Content
Library pane is just the folder tree of the configured content directories. Anything
installed by unzipping into a content directory — which is how most non-DAZ-store
content arrives — never gets a CMS row, so the SKU-driven ETL in
``postgres_db_manager`` cannot see it.

This module walks the content roots, subtracts everything the CMS already tracks, and
groups the remainder into plausible products so they can be indexed alongside real
products. Grouping is heuristic: DAZ content has no on-disk product boundary, only
naming conventions.

Two layouts are recognised:

* **Modern** — ``{top}/{figure}/{category}/{vendor}/{product}``, e.g.
  ``People/Genesis 9/Clothing/SomeVendor/Some Outfit``
* **Runtime** — ``Runtime/Libraries/{type}/{vendor}/{product}``, the Poser-era layout

``analyze_content_roots.py`` at the repo root is a CLI wrapper over this module and is
the tool for tuning the heuristics against a real library.
"""

import hashlib
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from utilities import find_thumbnail

logger = logging.getLogger(__name__)


# ─── Vocabulary ────────────────────────────────────────────────────────────────

FIGURE_NAMES = {
    "genesis 9 female", "genesis 9 male", "genesis 9",
    "genesis 8.1 female", "genesis 8.1 male", "genesis 8.1",
    "genesis 8 female", "genesis 8 male", "genesis 8",
    "genesis 3 female", "genesis 3 male", "genesis 3",
    "genesis 2 female", "genesis 2 male", "genesis 2",
    "genesis female", "genesis male", "genesis",
    "victoria 4", "michael 4", "v4", "m4",
    "aiko 6", "victoria 6", "michael 6",
}

RUNTIME_TYPES = {
    "character", "characters", "material", "materials",
    "pose", "poses", "prop", "props", "hair",
    "light", "lights", "camera", "cameras", "scene", "scenes", "hand",
}

CATEGORY_WORDS = {
    "clothing", "characters", "poses", "hair", "props",
    "materials", "anatomy", "environments", "lights", "accessories",
    "shaders", "clothing and accessories",
}

# Maps a directory word to the category vocabulary already used by the CMS-derived
# products (see determine_categories in postgres_db_manager), so filesystem products
# land in the same UI facet rather than creating parallel near-duplicates.
CATEGORY_ALIASES = {
    "clothing": "wardrobe", "clothes": "wardrobe", "wardrobe": "wardrobe",
    "outfits": "wardrobe", "outfit": "wardrobe",
    "characters": "character", "character": "character",
    "poses": "pose", "pose": "pose",
    "props": "prop", "prop": "prop",
    "materials": "materials", "material": "materials",
    "shaders": "shader", "shader": "shader",
    "hair": "hair",
    "lights": "light", "light": "light",
    "cameras": "camera", "camera": "camera",
    "scenes": "scene", "scene": "scene",
    "environments": "scene", "environment": "scene",
    "sets": "set", "set": "set",
    "accessories": "accessory", "accessory": "accessory",
    "animations": "animation", "animation": "animation",
    "morphs": "morph", "morph": "morph",
    "scripts": "script", "script": "script",
    "anatomy": "anatomy",
}

# DAZ Studio user-facing scene/preset files. These live anywhere under a content root.
DAZ_EXTENSIONS = {".duf"}

# Poser-era user-facing files. Only meaningful under Runtime/Libraries — the same
# extensions appear in support folders that the Content Library never shows.
POSER_EXTENSIONS = {
    ".pz2", ".cr2", ".pp2", ".pz3", ".cm2", ".hr2", ".fc2", ".hd2", ".mc6", ".mt5",
}

# Standard DAZ Studio content root folders. Used to recognise (and strip) the wrapper
# directory left behind when an archive is unzipped without merging into the root.
DEFAULT_TOP_LEVEL_DIRS = {
    "people", "props", "poses", "environments", "animals", "vehicles", "scenes",
    "figures", "architecture", "materials", "shader presets", "light presets",
    "render settings", "camera presets", "scripts", "presets", "general",
    "documentation", "runtime", "aniblocks", "data",
}

# Minimum depth (in path components below the content root) a synthesized parent must
# have before sibling directories are merged into it. 5 lets products sit at
# People/{figure}/{category}/{vendor}/{product} while preventing over-merging at the
# vendor or category level.
MIN_PRODUCT_DEPTH = int(os.getenv("SCAN_MIN_PRODUCT_DEPTH", "5"))


# ─── Path classification ───────────────────────────────────────────────────────

def runtime_index(parts):
    """Returns the index of the ``Runtime`` component of a ``Runtime/Libraries`` pair.

    Runtime is not always at the top of the relative path. A product unzipped with
    its archive's own wrapper folder sits at ``{Wrapper}/Runtime/Libraries/...``, and
    a content root registered on a product directory's parent puts every product one
    level down. Anchoring on position 0 would classify those as modern paths and —
    worse — make walk_content_files() reject their Poser files outright.

    Args:
        parts: Relative path components.

    Returns:
        int | None: Index of the ``Runtime`` component, or None if there is no
            ``Runtime/Libraries`` pair.
    """
    for i in range(len(parts) - 1):
        if parts[i].lower() == "runtime" and parts[i + 1].lower() == "libraries":
            return i
    return None


def is_runtime_path(parts) -> bool:
    """True when the relative path passes through Runtime/Libraries."""
    return runtime_index(parts) is not None


def runtime_group_key(parts):
    """Group key for Runtime/Libraries paths: (normalised_brand, leaf_name_lower)."""
    i = runtime_index(parts)
    inner = parts[i + 2:]  # strip everything up to and including Runtime/Libraries
    if inner and inner[0].lower() in RUNTIME_TYPES:
        inner = inner[1:]  # strip type
    if not inner:
        return parts
    brand_norm = inner[0].lower().rstrip("s")  # normalise plural forms
    leaf = inner[-1].lower() if len(inner) > 1 else ""
    return (brand_norm, leaf)


def modern_group_key(parts):
    """Group key for modern paths: normalise figure names so G8/G9 paths merge."""
    return tuple("[FIGURE]" if p.lower() in FIGURE_NAMES else p for p in parts)


def _infer_name_vendor(rel_parts, path_type):
    """Infers (product_name, vendor) from a product root's path components."""
    if path_type == "runtime":
        i = runtime_index(rel_parts)
        inner = rel_parts[i + 2:]  # strip everything up to and including Runtime/Libraries
        if inner and inner[0].lower() in RUNTIME_TYPES:
            inner = inner[1:]  # strip type
        if not inner:
            return ("(unknown)", None)
        if len(inner) == 1:
            return (inner[0], None)
        name = inner[-1]
        vendor = next(
            (p for p in reversed(inner[:-1]) if p.lower() not in CATEGORY_WORDS),
            None
        )
        return (name, vendor)

    # Modern: {top}/{figure}/{category}/{vendor?}/{product}
    fig_idx = next(
        (i for i, p in enumerate(rel_parts) if p.lower() in FIGURE_NAMES), None
    )
    after_fig = rel_parts[fig_idx + 1:] if fig_idx is not None else rel_parts[1:]
    if not after_fig:
        return (rel_parts[-1] if rel_parts else "(unknown)", None)
    if len(after_fig) == 1:
        return (after_fig[0], None)

    # The leaf is the product; the vendor is the nearest preceding component that is
    # not a content category word (Clothing, Poses, ...), so both
    # 'Clothing/Vendor/Product' and 'Vendor/Product' resolve correctly.
    name = after_fig[-1]
    vendor = next(
        (p for p in reversed(after_fig[:-1])
         if p.lower() not in CATEGORY_WORDS and p.lower() not in CATEGORY_ALIASES),
        None,
    )
    return (name, vendor)


def _context_label(rel_parts, path_type):
    """A short label describing the product's context (figure name, or Runtime type)."""
    if path_type == "runtime":
        i = runtime_index(rel_parts)
        return rel_parts[i + 2] if len(rel_parts) > i + 2 else ""
    return next((p for p in rel_parts if p.lower() in FIGURE_NAMES), "")


def infer_category(rel_parts) -> str | None:
    """Maps a recognised category word in the path to the shared category vocabulary.

    In the modern layout the category sits directly after the figure name
    (``People/Genesis 9/Clothing/...``), so anything before the figure — the
    top-level container — is ignored to avoid reading 'People' as a category.
    """
    parts = list(rel_parts)
    fig_idx = next(
        (i for i, p in enumerate(parts) if p.lower() in FIGURE_NAMES), None
    )
    if fig_idx is not None:
        parts = parts[fig_idx + 1:]
    for part in parts:
        alias = CATEGORY_ALIASES.get(part.lower())
        if alias:
            return alias
    return None


def infer_figures(rel_parts) -> list:
    """Returns the canonical figure names appearing in the path, longest match first."""
    found = {p for p in rel_parts if p.lower() in FIGURE_NAMES}
    return sorted(found)


# ─── Content roots ─────────────────────────────────────────────────────────────

def dedupe_roots(roots) -> list:
    """Drops content roots nested inside another root, and roots that do not exist.

    DAZ Studio happily accepts a content directory and one of its own subdirectories
    as separate roots. Scanning both would walk the same files twice and produce
    duplicate products.

    Args:
        roots: Iterable of path strings.

    Returns:
        list[Path]: Existing, non-nested roots, shallowest first.
    """
    existing = []
    for r in roots:
        if not r:
            continue
        p = Path(r)
        if p.is_dir():
            existing.append(p.resolve())
        else:
            logger.warning(f"Content root does not exist, skipping: {r}")

    existing.sort(key=lambda p: len(p.parts))
    kept: list[Path] = []
    for p in existing:
        if any(p == k or str(p).lower().startswith(str(k).lower() + os.sep) for k in kept):
            logger.info(f"Content root {p} is nested inside an existing root — skipping.")
            continue
        kept.append(p)
    return kept


# ─── Filesystem walk ───────────────────────────────────────────────────────────

def walk_content_files(root: Path):
    """Yields all user-facing content files under root.

    Skips the ``data/`` subtree (support geometry/morphs, never shown in the Content
    Library) and only accepts Poser-format files under ``Runtime/Libraries``.

    Args:
        root (Path): Absolute content root.

    Yields:
        Path: Absolute file paths.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        if rel.parts and rel.parts[0].lower() == "data":
            dirnames.clear()
            continue
        in_runtime = is_runtime_path(rel.parts)
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in DAZ_EXTENSIONS or (in_runtime and ext in POSER_EXTENSIONS):
                yield Path(dirpath) / fname


def load_tracked_files(analyzer) -> set:
    """Returns {(normalised_relative_dir, lower_filename)} for every CMS content row."""
    rows = analyzer.get_all_content_paths()
    tracked = set()
    for path, filename in rows:
        norm = (path or "").replace("\\", "/").lower().rstrip("/")
        tracked.add((norm, (filename or "").lower()))
    logger.info(f"Loaded {len(tracked):,} tracked file entries from the CMS.")
    return tracked


def canonical_top_levels(tracked) -> set:
    """Returns the top-level content folder names the CMS actually uses.

    Content is routinely unzipped twice: once merged into the content root
    ('Props/Vendor/Product') and once inside the archive's own wrapper folder
    ('Vendor Product Name/Props/Vendor/Product'). Knowing the canonical top-level
    names lets the grouping step strip that wrapper so both copies fold into one
    product instead of appearing twice.

    Args:
        tracked (set): Output of load_tracked_files().

    Returns:
        set: Lowercased top-level directory names, always including the defaults.
    """
    tops = set(DEFAULT_TOP_LEVEL_DIRS)
    for path, _filename in tracked:
        if path:
            tops.add(path.split("/", 1)[0])
    tops.discard("")
    return tops


def collect_untracked_dirs(roots, tracked, on_root_done=None) -> dict:
    """Walks the roots and returns per-directory tracked/untracked file statistics.

    Args:
        roots (list[Path]): Deduplicated content roots.
        tracked (set): Output of load_tracked_files().
        on_root_done (callable, optional): Called as on_root_done(root) after each
            root is walked, so callers can report progress on a slow scan.

    Returns:
        dict: {Path(dir): {'root': Path, 'tracked': int, 'untracked': int,
                           'files': [Path, ...]}} for directories holding
              at least one untracked file.
    """
    stats: dict = {}
    for root in roots:
        logger.info(f"Scanning {root} …")
        count = 0
        for file in walk_content_files(root):
            count += 1
            rel = file.relative_to(root)
            key = (str(rel.parent).replace("\\", "/").lower(), rel.name.lower())
            entry = stats.get(file.parent)
            if entry is None:
                entry = stats[file.parent] = {
                    "root": root, "tracked": 0, "untracked": 0, "files": []
                }
            if key in tracked:
                entry["tracked"] += 1
            else:
                entry["untracked"] += 1
                entry["files"].append(file)
        logger.info(f"  {count:,} user-facing files")
        if on_root_done:
            on_root_done(root)

    return {d: v for d, v in stats.items() if v["untracked"] > 0 and v["tracked"] == 0}


# ─── Product grouping ──────────────────────────────────────────────────────────

def merge_into_ancestors(dir_stats, content_roots) -> dict:
    """Rolls subdirectories into the shallowest directory that plausibly is a product.

    Builds one prefix tree per content root out of the candidate directories, then
    walks it bottom-up applying two rules:

      A. A directory that itself holds untracked files absorbs everything beneath it.
      B. A directory that is not itself a candidate absorbs its children when two or
         more of them resolved to product roots, it sits at least MIN_PRODUCT_DEPTH
         below the content root, and its own name is not a generic category word
         (so Characters/Effie and Characters/Simona stay separate, while
         G8G9CouplesPoses/Anywhere and .../InPlace merge).

    This is the linear-time equivalent of repeatedly rescanning the whole root set,
    which does not finish on libraries with tens of thousands of untracked directories.

    Args:
        dir_stats (dict): Output of collect_untracked_dirs().
        content_roots (list[Path]): Deduplicated content roots.

    Returns:
        dict: {product_root: {'file_count': int, 'constituent_dirs': [Path, ...]}}
    """
    # node: {"children": {name: node}, "present": bool, "dir": Path|None}
    def new_node():
        return {"children": {}, "present": False, "dir": None}

    trees = {root: new_node() for root in content_roots}

    for directory, data in dir_stats.items():
        root = data["root"]
        tree = trees.get(root)
        if tree is None:
            continue
        node = tree
        for part in directory.relative_to(root).parts:
            node = node["children"].setdefault(part, new_node())
        node["present"] = True
        node["dir"] = directory

    results: dict = {}

    def visit(node, path_parts, root):
        """Returns the list of (product_root, [constituent_dirs]) resolved below node."""
        child_roots = []
        for name, child in node["children"].items():
            child_roots.extend(visit(child, path_parts + (name,), root))

        here = root.joinpath(*path_parts) if path_parts else root

        # Rule A — this directory has its own untracked files: absorb the subtree.
        if node["present"]:
            constituents = [here]
            for _, dirs in child_roots:
                constituents.extend(dirs)
            return [(here, constituents)]

        # Rule B — synthesize this directory as a product root.
        if (
            len(child_roots) >= 2
            and len(path_parts) >= MIN_PRODUCT_DEPTH
            and path_parts
            and path_parts[-1].lower() not in CATEGORY_WORDS
        ):
            constituents = []
            for _, dirs in child_roots:
                constituents.extend(dirs)
            return [(here, constituents)]

        return child_roots

    for root, tree in trees.items():
        for product_root, constituents in visit(tree, (), root):
            results[product_root] = {
                "file_count": sum(
                    dir_stats[d]["untracked"] for d in constituents if d in dir_stats
                ),
                "constituent_dirs": sorted(set(constituents)),
            }

    return results


def strip_wrapper(rel_parts, top_levels) -> tuple:
    """Drops any wrapper directory preceding the first canonical top-level folder.

    'DoctorPervic Anal Hooks/Props/DoctorPervic/Anal Hooks' becomes
    'Props/DoctorPervic/Anal Hooks', so a product unzipped both ways resolves to one
    entry. Paths with no recognisable top-level folder are returned unchanged.
    """
    for i, part in enumerate(rel_parts):
        if part.lower() in top_levels:
            return tuple(rel_parts[i:])
    return tuple(rel_parts)


def group_into_products(merged, content_roots, top_levels=None) -> list:
    """Groups product roots across figures (modern layout) and Runtime types.

    A single product often installs parallel trees for several figures; this folds
    them back into one entry.

    Args:
        merged (dict): Output of merge_into_ancestors().
        content_roots (list[Path]): Deduplicated content roots.
        top_levels (set, optional): Canonical top-level folder names used to strip
            archive wrapper directories. Defaults to DEFAULT_TOP_LEVEL_DIRS.

    Returns:
        list[dict]: Product dicts sorted by vendor then name.
    """
    top_levels = top_levels or DEFAULT_TOP_LEVEL_DIRS
    buckets = defaultdict(list)

    for root, data in merged.items():
        cr = next((c for c in content_roots if str(root).startswith(str(c))), None)
        if cr is None:
            continue
        rel_parts = strip_wrapper(root.relative_to(cr).parts, top_levels)

        if is_runtime_path(rel_parts):
            key = ("runtime", runtime_group_key(rel_parts))
        else:
            key = ("modern", modern_group_key(rel_parts))

        buckets[key].append((root, rel_parts, data))

    products = []
    for (path_type, _group_key), entries in buckets.items():
        total_files = sum(d["file_count"] for _, _, d in entries)
        all_dirs = []
        for _, _, d in entries:
            all_dirs.extend(d["constituent_dirs"])

        rep_root, rel_parts, _ = min(entries, key=lambda e: len(e[0].parts))
        name, vendor = _infer_name_vendor(rel_parts, path_type)

        context_labels = set()
        rel_parts_per_root = []
        figures = set()
        for _, rp, _d in entries:
            rel_parts_per_root.append(rp)
            figures.update(infer_figures(rp))
            lbl = _context_label(rp, path_type)
            if lbl:
                context_labels.add(lbl)

        category = next(
            (c for c in (infer_category(rp) for rp in rel_parts_per_root) if c), None
        )

        products.append({
            "name": name,
            "vendor": vendor,
            "path_type": path_type,
            # Identity of the product, independent of which content root each copy
            # lives in — so the SKU survives deleting one of two installed copies.
            "identity": f"{path_type}|" + "/".join(rel_parts).lower(),
            "file_count": total_files,
            "context": sorted(context_labels),
            "category": category,
            "compatible_figures": sorted(figures),
            "rel_parts": list(rel_parts),
            "paths": sorted(root for root, _rp, _d in entries),
            "all_dirs": sorted(set(all_dirs)),
        })

    return sorted(products, key=lambda p: (p["vendor"] or "", p["name"]))


# ─── Top-level scan ────────────────────────────────────────────────────────────

def scan(analyzer, roots=None, on_progress=None) -> list:
    """Scans the content roots and returns inferred products not tracked by the CMS.

    Args:
        analyzer: DazDBAnalyzer instance (for content roots and tracked files).
        roots (list, optional): Override the content roots to scan.
        on_progress (callable, optional): on_progress(stage, current, total, detail).
            Walking a large library takes a minute with nothing else to report, so
            this fires per phase and per content root to keep callers' UI alive.

    Returns:
        list[dict]: Product dicts as produced by group_into_products().
    """
    content_roots = dedupe_roots(roots if roots is not None else analyzer.get_content_roots())
    if not content_roots:
        logger.warning("No accessible content roots — nothing to scan.")
        return []

    # One step for the CMS index load, one per content root, one for grouping.
    total_steps = len(content_roots) + 2
    step = 0

    def report(detail):
        nonlocal step
        step += 1
        if on_progress:
            on_progress("scan", step, total_steps, detail)

    if on_progress:
        on_progress("scan", 0, total_steps, "reading CMS file index")
    tracked = load_tracked_files(analyzer)
    report("reading CMS file index")

    dir_stats = collect_untracked_dirs(
        content_roots, tracked, on_root_done=lambda root: report(f"scanned {root.name}")
    )
    logger.info(f"{len(dir_stats):,} directories hold untracked content.")

    merged = merge_into_ancestors(dir_stats, content_roots)
    products = group_into_products(merged, content_roots, canonical_top_levels(tracked))
    report("grouping products")
    logger.info(f"Grouped into {len(products):,} filesystem products.")

    # Attach the untracked files themselves so callers can pick a thumbnail without
    # re-walking the tree.
    files_by_dir = {d: v["files"] for d, v in dir_stats.items()}
    for product in products:
        files = []
        for d in product["all_dirs"]:
            files.extend(files_by_dir.get(d, []))
        product["files"] = files

    return products


# ─── Product rows ──────────────────────────────────────────────────────────────

FS_SKU_PREFIX = "fs-"


def make_sku(identity: str) -> str:
    """Returns a stable synthetic SKU for a filesystem product identity."""
    digest = hashlib.sha1(str(identity).lower().encode("utf-8")).hexdigest()
    return f"{FS_SKU_PREFIX}{digest[:16]}"


def is_filesystem_sku(sku: str) -> bool:
    """True for SKUs minted by this module."""
    return bool(sku) and sku.startswith(FS_SKU_PREFIX)


def build_embedding_text(product: dict) -> str:
    """Builds the descriptive paragraph used to embed a filesystem product.

    These products have no store description, so nearly all of the available meaning
    lives in the directory names themselves.
    """
    name = product["name"]
    parts = [f"A 3D asset package titled '{name}'."]
    if product.get("vendor"):
        parts.append(f"Created by the artist or studio: {product['vendor']}.")
    if product.get("category"):
        parts.append(f"It is categorized under: {product['category']}.")
    if product.get("compatible_figures"):
        parts.append(f"It is built for the figures: {', '.join(product['compatible_figures'])}.")

    cat = (product.get("category") or "").lower()
    if cat in ("prop", "set", "scene"):
        parts.append("This is a set of props suitable for decorating digital scenes, environments, and dioramas.")
    elif cat == "character":
        parts.append("This is a character asset for digital art and animation.")
    elif cat == "hair":
        parts.append("This is a hairstyle asset for 3D characters.")
    elif cat == "wardrobe":
        parts.append("It contains clothing or wardrobe items for 3D figures.")
    elif cat == "pose":
        parts.append("It contains poses for 3D figures.")

    # The folder path carries the vendor's own naming — feed it in as free text.
    path_words = " ".join(product.get("rel_parts", []))
    if path_words:
        parts.append(f"It is installed under the content path: {path_words}.")

    parts.append(
        "It was found in the DAZ Studio Content Library and is not registered in Smart Content."
    )
    return " ".join(parts)


def build_product_row(product: dict) -> dict:
    """Converts a scanned product into a row for the SQLite enrichment cache."""
    thumbnail = None
    for file in product.get("files", [])[:200]:
        thumbnail = find_thumbnail(file)
        if thumbnail:
            break

    extensions = sorted({
        os.path.splitext(f.name)[1].lower() for f in product.get("files", [])
    })
    tag_parts = [product.get("category")] + list(product.get("context", []))

    return {
        "sku":                 make_sku(product["identity"]),
        "url":                 None,
        "image_url":           None,
        "store":               None,
        "name":                product["name"],
        "artist":              product.get("vendor") or "",
        "price":               None,
        "description":         None,
        "tags":                ", ".join(t for t in tag_parts if t),
        "formats":             ", ".join(extensions),
        "poly_count":          None,
        "textures_info":       None,
        "required_products":   None,
        "compatible_figures":  ", ".join(product.get("compatible_figures", [])),
        "compatible_software": None,
        "embedding_text":      build_embedding_text(product),
        "last_updated":        _newest_mtime(product),
        "category":            product.get("category"),
        "subcategories":       "",
        "styles":              None,
        "inferred_tags":       None,
        "enriched_at":         datetime.now(timezone.utc).isoformat(),
        "mature":              None,
        "asset_count":         product["file_count"],
        "source":              "filesystem",
        "content_dirs":        json.dumps([str(d) for d in product["all_dirs"]]),
        "thumbnail_path":      thumbnail.as_posix() if thumbnail else None,
    }


def _newest_mtime(product: dict) -> str | None:
    """ISO timestamp of the most recently modified file in the product, if readable."""
    newest = 0.0
    for file in product.get("files", [])[:200]:
        try:
            newest = max(newest, file.stat().st_mtime)
        except OSError:
            continue
    if not newest:
        return None
    return datetime.fromtimestamp(newest, tz=timezone.utc).isoformat()


def list_files_for_dirs(content_dirs) -> list:
    """Lists user-facing content files directly inside the given directories.

    Used by the API to answer 'what files does this filesystem product contain?'
    without a CMS row.

    Args:
        content_dirs (list[str]): Absolute directory paths.

    Returns:
        list[dict]: Dicts with keys path, filename, content_type, resolved_path.
    """
    files = []
    for directory in content_dirs:
        d = Path(directory)
        if not d.is_dir():
            continue
        in_runtime_dir = any(
            part.lower() == "libraries" for part in d.parts
        ) and any(part.lower() == "runtime" for part in d.parts)
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name.lower())
        except OSError as e:
            logger.warning(f"Could not list {d}: {e}")
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            ext = entry.suffix.lower()
            if ext in DAZ_EXTENSIONS or (in_runtime_dir and ext in POSER_EXTENSIONS):
                files.append({
                    "path": str(d),
                    "filename": entry.name,
                    "content_type": ext.lstrip("."),
                    "resolved_path": entry.as_posix(),
                })
    return files
