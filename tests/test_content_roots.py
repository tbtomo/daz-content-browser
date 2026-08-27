"""Tests for how the scan decides which directories to walk.

The CMS and the running DAZ Studio disagree about the content roots, and neither
list contains the other — see resolve_content_roots().
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def resolve():
    from managers.postgres_db_manager import resolve_content_roots
    return resolve_content_roots


def _patched(cms_roots, *, plugin_dirs=None, available=True):
    analyzer = MagicMock()
    analyzer.get_content_roots.return_value = list(cms_roots)
    server = MagicMock()
    server.is_available.return_value = available
    server.get_content_directories.return_value = list(plugin_dirs or [])
    return (patch("managers.postgres_db_manager.daz_pg_analyzer", analyzer),
            patch("managers.postgres_db_manager.daz_script_server", server))


def test_adds_directories_the_cms_does_not_know_about(resolve):
    """A Poser-format library carries no CMS rows, so tblBasePath never lists it."""
    a, s = _patched(["D:/DAZ 3D/Studio/My Library"],
                    plugin_dirs=["D:/DAZ 3D/Studio/My Library", "D:/Poser/My Poser Library"])
    with a, s:
        assert resolve() == ["D:/DAZ 3D/Studio/My Library", "D:/Poser/My Poser Library"]


def test_keeps_cms_roots_the_running_app_does_not_report(resolve):
    a, s = _patched(["D:/DAZ 3D/Studio/My Library/self-made"],
                    plugin_dirs=["D:/DAZ 3D/Studio/My Library"])
    with a, s:
        assert resolve() == ["D:/DAZ 3D/Studio/My Library/self-made",
                             "D:/DAZ 3D/Studio/My Library"]


def test_matches_case_and_separator_insensitively(resolve):
    a, s = _patched(["D:/DAZ 3D/Studio/My Library"],
                    plugin_dirs=[r"d:\daz 3d\studio\my library"])
    with a, s:
        assert resolve() == ["D:/DAZ 3D/Studio/My Library"]


def test_falls_back_to_the_cms_when_daz_studio_is_closed(resolve):
    a, s = _patched(["D:/DAZ 3D/Studio/My Library"],
                    plugin_dirs=["D:/Poser/My Poser Library"], available=False)
    with a, s:
        assert resolve() == ["D:/DAZ 3D/Studio/My Library"]


def test_a_failing_plugin_does_not_abort_the_scan(resolve):
    analyzer = MagicMock()
    analyzer.get_content_roots.return_value = ["D:/DAZ 3D/Studio/My Library"]
    server = MagicMock()
    server.is_available.side_effect = RuntimeError("connection refused")
    with patch("managers.postgres_db_manager.daz_pg_analyzer", analyzer), \
         patch("managers.postgres_db_manager.daz_script_server", server):
        assert resolve() == ["D:/DAZ 3D/Studio/My Library"]
