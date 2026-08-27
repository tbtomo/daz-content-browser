"""Tests for the Content Library filesystem scanner."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from managers import filesystem_scanner as fs  # noqa: E402


class FakeAnalyzer:
    """Stands in for DazDBAnalyzer without touching PostgreSQL."""

    def __init__(self, roots, tracked=()):
        self._roots = [str(r) for r in roots]
        self._tracked = list(tracked)

    def get_content_roots(self):
        return self._roots

    def get_all_content_paths(self):
        return self._tracked


def write(path: Path, content: str = "{}"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def library(tmp_path):
    """A small content root covering both the modern and Runtime layouts."""
    root = tmp_path / "My Library"

    modern = root / "People" / "Genesis 9" / "Clothing" / "AcmeVendor" / "Space Suit"
    write(modern / "Space Suit.duf")
    write(modern / "Space Suit.png", "png")
    write(modern / "Materials" / "Red.duf")

    runtime = root / "Runtime" / "Libraries" / "Poses" / "OtherVendor" / "Action Poses"
    write(runtime / "Jump.pz2")
    write(runtime / "Crouch.pz2")

    # Support data must never be indexed.
    write(root / "data" / "AcmeVendor" / "Space Suit" / "internal.duf")
    # Poser formats outside Runtime/Libraries are not user-facing either.
    write(root / "Props" / "stray.pz2")

    return root


def test_walk_skips_data_and_non_runtime_poser_files(library):
    found = {p.name for p in fs.walk_content_files(library)}
    assert found == {"Space Suit.duf", "Red.duf", "Jump.pz2", "Crouch.pz2"}


def test_scan_groups_products_and_infers_metadata(library):
    products = fs.scan(FakeAnalyzer([library]))
    by_name = {p["name"]: p for p in products}

    assert set(by_name) == {"Space Suit", "Action Poses"}

    suit = by_name["Space Suit"]
    assert suit["vendor"] == "AcmeVendor"
    assert suit["category"] == "wardrobe"
    assert suit["compatible_figures"] == ["Genesis 9"]
    assert suit["file_count"] == 2  # the .duf plus the one in Materials/

    poses = by_name["Action Poses"]
    assert poses["vendor"] == "OtherVendor"
    assert poses["category"] == "pose"
    assert poses["file_count"] == 2


def test_tracked_files_are_excluded(library):
    tracked = [("People/Genesis 9/Clothing/AcmeVendor/Space Suit", "Space Suit.duf")]
    products = fs.scan(FakeAnalyzer([library], tracked))
    names = {p["name"] for p in products}

    # The Space Suit directory is no longer fully untracked, so only its Materials/
    # subdirectory qualifies — and Action Poses is unaffected.
    assert "Action Poses" in names
    assert "Space Suit" not in names


def test_nested_content_roots_are_deduplicated(library):
    nested = library / "People"
    roots = fs.dedupe_roots([str(library), str(nested)])
    assert roots == [library.resolve()]


def test_missing_roots_are_dropped(tmp_path, library):
    roots = fs.dedupe_roots([str(library), str(tmp_path / "does-not-exist")])
    assert roots == [library.resolve()]


def test_build_product_row(library):
    products = fs.scan(FakeAnalyzer([library]))
    suit = next(p for p in products if p["name"] == "Space Suit")
    row = fs.build_product_row(suit)

    assert row["sku"].startswith("fs-")
    assert fs.is_filesystem_sku(row["sku"])
    assert row["source"] == "filesystem"
    assert row["artist"] == "AcmeVendor"
    assert row["compatible_figures"] == "Genesis 9"
    assert row["formats"] == ".duf"
    assert row["asset_count"] == 2
    assert row["thumbnail_path"].endswith("Space Suit.png")
    assert "Space Suit" in row["embedding_text"]
    assert json.loads(row["content_dirs"])


def test_sku_is_stable_across_runs(library):
    first = fs.build_product_row(
        next(p for p in fs.scan(FakeAnalyzer([library])) if p["name"] == "Space Suit")
    )
    second = fs.build_product_row(
        next(p for p in fs.scan(FakeAnalyzer([library])) if p["name"] == "Space Suit")
    )
    assert first["sku"] == second["sku"]


def test_list_files_for_dirs(library):
    modern = library / "People" / "Genesis 9" / "Clothing" / "AcmeVendor" / "Space Suit"
    files = fs.list_files_for_dirs([str(modern)])
    assert [f["filename"] for f in files] == ["Space Suit.duf"]
    assert files[0]["content_type"] == "duf"
    assert Path(files[0]["resolved_path"]).exists()


def test_sibling_directories_merge_into_a_deep_parent(tmp_path):
    """Rule B: two sibling product dirs below MIN_PRODUCT_DEPTH roll into their parent."""
    root = tmp_path / "Lib"
    base = root / "People" / "Genesis 9" / "Poses" / "AcmeVendor" / "CouplePoses"
    write(base / "Anywhere" / "A.duf")
    write(base / "InPlace" / "B.duf")

    products = fs.scan(FakeAnalyzer([root]))
    assert [p["name"] for p in products] == ["CouplePoses"]
    assert products[0]["file_count"] == 2


def test_archive_wrapper_directory_folds_into_the_same_product(tmp_path):
    """The same product unzipped merged and unmerged resolves to one entry, one SKU."""
    root = tmp_path / "Lib"
    merged = root / "Props" / "AcmeVendor" / "Anal Hooks"
    wrapped = root / "AcmeVendor Anal Hooks" / "Props" / "AcmeVendor" / "Anal Hooks"
    write(merged / "Hook.duf")
    write(wrapped / "Hook.duf")

    products = fs.scan(FakeAnalyzer([root]))
    assert [p["name"] for p in products] == ["Anal Hooks"]

    product = products[0]
    assert product["vendor"] == "AcmeVendor"
    assert len(product["all_dirs"]) == 2  # both copies are recorded
    assert product["file_count"] == 2


def test_sku_is_unaffected_by_which_copy_exists(tmp_path):
    """Deleting one of two installed copies must not change the product's SKU."""
    def build(with_wrapper):
        root = tmp_path / ("both" if with_wrapper else "one") / "Lib"
        write(root / "Props" / "AcmeVendor" / "Anal Hooks" / "Hook.duf")
        if with_wrapper:
            write(root / "Acme Bundle" / "Props" / "AcmeVendor" / "Anal Hooks" / "Hook.duf")
        product = fs.scan(FakeAnalyzer([root]))[0]
        return fs.build_product_row(product)["sku"]

    assert build(True) == build(False)


def test_generic_category_directories_do_not_merge(tmp_path):
    """Rule B exclusion: distinct characters under Characters/ stay separate."""
    root = tmp_path / "Lib"
    base = root / "People" / "Genesis 9" / "AcmeVendor" / "Bundle" / "Characters"
    write(base / "Effie" / "Effie.duf")
    write(base / "Simona" / "Simona.duf")

    products = fs.scan(FakeAnalyzer([root]))
    assert sorted(p["name"] for p in products) == ["Effie", "Simona"]


def test_runtime_index_finds_a_nested_runtime_libraries_pair():
    assert fs.runtime_index(("Runtime", "Libraries", "Poses")) == 0
    assert fs.runtime_index(("Wrapper", "Runtime", "Libraries", "Poses")) == 1
    assert fs.runtime_index(("Props", "Runtime")) is None
    assert fs.runtime_index(()) is None


def test_walk_accepts_poser_files_below_a_wrapper_directory(tmp_path):
    """A product unzipped with its archive folder still keeps Runtime/Libraries.

    The CMS also registers a product directory's *parent* as a content root, which
    pushes Runtime one level down the same way. Anchoring on position 0 dropped every
    Poser file in both layouts.
    """
    root = tmp_path / "My Poser Library"
    inner = root / "vendor_product_Poser" / "Runtime" / "Libraries" / "Character" / "Thing"
    write(inner / "Thing.cr2")
    write(root / "vendor_product_Poser" / "stray.pz2")  # outside Runtime/Libraries

    assert {p.name for p in fs.walk_content_files(root)} == {"Thing.cr2"}


def test_product_below_a_wrapper_is_classified_as_runtime(tmp_path):
    root = tmp_path / "My Poser Library"
    inner = root / "vendor_product_Poser" / "Runtime" / "Libraries" / "Character" / "Thing"
    write(inner / "Thing.cr2")

    products = fs.scan(FakeAnalyzer([root]))
    assert len(products) == 1
    assert products[0]["name"] == "Thing"
    assert products[0]["path_type"] == "runtime"
