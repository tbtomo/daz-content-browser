"""Unit tests for DazScriptServerClient — focuses on availability caching."""
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from managers.daz_script_server import DazScriptServerClient


@pytest.fixture
def client():
    return DazScriptServerClient()


def _make_status(detected: bool) -> dict:
    return {
        "plugin_detected": detected,
        "plugin_url": "http://127.0.0.1:18811",
        "version": "1.0" if detected else None,
        "auth_enabled": False if detected else None,
        "active_requests": 0,
    }


class TestIsAvailableCache:
    def test_returns_true_when_plugin_detected(self, client):
        with patch.object(client, "status", return_value=_make_status(True)):
            assert client.is_available() is True

    def test_returns_false_when_plugin_not_detected(self, client):
        with patch.object(client, "status", return_value=_make_status(False)):
            assert client.is_available() is False

    def test_status_called_once_within_ttl(self, client):
        mock_status = MagicMock(return_value=_make_status(True))
        with patch.object(client, "status", mock_status):
            client.is_available()
            client.is_available()
            client.is_available()
        mock_status.assert_called_once()

    def test_status_recalled_after_ttl_expires(self, client):
        mock_status = MagicMock(return_value=_make_status(True))
        with patch.object(client, "status", mock_status):
            with patch("managers.daz_script_server.time") as mock_time:
                mock_time.monotonic.side_effect = [0.0, 11.0]
                client.is_available()
                client.is_available()
        assert mock_status.call_count == 2

    def test_cache_not_used_on_first_call(self, client):
        assert client._availability_cache is None
        mock_status = MagicMock(return_value=_make_status(False))
        with patch.object(client, "status", mock_status):
            client.is_available()
        mock_status.assert_called_once()

    def test_cache_populated_after_first_call(self, client):
        with patch.object(client, "status", return_value=_make_status(True)):
            client.is_available()
        assert client._availability_cache is not None
        result, ts = client._availability_cache
        assert result is True
        assert ts > 0

    def test_cached_value_returned_not_recomputed(self, client):
        with patch.object(client, "status", return_value=_make_status(True)):
            client.is_available()
        # Patch status to return False — cache should still return True
        with patch.object(client, "status", return_value=_make_status(False)):
            assert client.is_available() is True

    def test_ttl_boundary_just_inside(self, client):
        mock_status = MagicMock(return_value=_make_status(True))
        with patch.object(client, "status", mock_status):
            with patch("managers.daz_script_server.time") as mock_time:
                mock_time.monotonic.side_effect = [0.0, 9.99]
                client.is_available()
                client.is_available()
        mock_status.assert_called_once()

    def test_ttl_boundary_just_outside(self, client):
        mock_status = MagicMock(return_value=_make_status(True))
        with patch.object(client, "status", mock_status):
            with patch("managers.daz_script_server.time") as mock_time:
                mock_time.monotonic.side_effect = [0.0, 10.01]
                client.is_available()
                client.is_available()
        assert mock_status.call_count == 2

    def test_each_instance_has_independent_cache(self):
        c1 = DazScriptServerClient()
        c2 = DazScriptServerClient()
        with patch.object(c1, "status", return_value=_make_status(True)):
            c1.is_available()
        assert c2._availability_cache is None


class TestGetContentDirectories:
    """The plugin nests a script's return value under 'result'."""

    def test_reads_paths_out_of_the_result_envelope(self, client):
        response = {
            "success": True,
            "result": {
                "success": True,
                "paths": ["D:/DAZ 3D/Studio/My Library", "D:/Poser/My Poser Library"],
                "native": ["D:/DAZ 3D/Studio/My Library"],
                "poser": ["D:/Poser/My Poser Library"],
                "other": [],
            },
        }
        with patch.object(client, "_ensure_scripts_registered"), \
             patch.object(client, "_execute_registered", return_value=response):
            assert client.get_content_directories() == [
                "D:/DAZ 3D/Studio/My Library",
                "D:/Poser/My Poser Library",
            ]

    def test_missing_envelope_yields_no_directories(self, client):
        with patch.object(client, "_ensure_scripts_registered"), \
             patch.object(client, "_execute_registered", return_value={"success": True}):
            assert client.get_content_directories() == []

    def test_get_content_dirs_script_covers_the_poser_list(self):
        from managers.daz_script_server import _STANDARD_SCRIPTS

        script = _STANDARD_SCRIPTS["get-content-dirs"]["script"]
        # getContentDirectory() returns a DzContentFolder, not a path.
        assert "getContentDirectoryPath" in script
        assert "getNumPoserDirectories" in script
        assert "getPoserDirectoryPath" in script
