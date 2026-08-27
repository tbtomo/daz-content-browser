"""API smoke tests — run in demo mode, no real database required."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from server import app
    return TestClient(app)


def test_status(client):
    r = client.get("/api/v1/status")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "chromadb_connected" in body


def test_products(client):
    r = client.get("/api/v1/products")
    assert r.status_code == 200
    body = r.json()
    assert "products" in body
    assert "total" in body


def test_filters(client):
    r = client.get("/api/v1/filters")
    assert r.status_code == 200
    body = r.json()
    assert "categories" in body
    assert "artists" in body
    assert "compatible_figures" in body


def test_search(client):
    r = client.post("/api/v1/search", json={"query": "fantasy dress", "limit": 5})
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert "total" in body


def test_search_max_results_caps_results(client):
    """max_results caps the number of results returned; client paginates from there."""
    r = client.post("/api/v1/search", json={"query": "fantasy dress", "max_results": 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) <= 2


def test_search_total_reflects_full_result_count(client):
    """total counts all matching results regardless of max_results."""
    r_all = client.post("/api/v1/search", json={"query": "fantasy", "max_results": 500})
    r_cap = client.post("/api/v1/search", json={"query": "fantasy", "max_results": 1})
    assert r_all.status_code == 200
    assert r_cap.status_code == 200
    # A capped response must not report more total hits than the uncapped one
    assert r_cap.json()["total"] <= r_all.json()["total"]


def test_search_returns_unique_skus(client):
    """Each result in a single response must have a distinct SKU."""
    r = client.post("/api/v1/search", json={"query": "outfit", "max_results": 500})
    assert r.status_code == 200
    skus = [p["sku"] for p in r.json()["results"]]
    assert len(skus) == len(set(skus))


def test_settings(client):
    r = client.get("/api/v1/settings")
    assert r.status_code == 200
    body = r.json()
    assert "cms_host" in body
    assert "embedding_model" in body


def test_settings_reports_the_translation_stack(client):
    """A search can now go wrong in the translator, so the page has to name it."""
    body = client.get("/api/v1/settings").json()
    assert "translation_enabled" in body
    if body["translation_enabled"]:
        assert body.get("translation_backend") in ("onnx", "ctranslate2")
        assert body.get("translation_model")
        assert isinstance(body.get("translation_glossary_terms"), int)


def test_the_model_fields_cannot_be_changed(client):
    """Which model answers queries is fixed at load time by EMBEDDING_MODEL_ID.

    The settings form renders these as editable inputs, so without this the page
    would happily store and then display a model that is not the one searching.
    """
    before = client.get("/api/v1/settings").json()

    r = client.put("/api/v1/settings", json={
        "embedding_model": "totally/wrong-model",
        "query_model": "also/wrong",
    })
    assert r.status_code == 200
    assert r.json()["embedding_model"] == before["embedding_model"]
    assert r.json()["query_model"] == before["query_model"]

    after = client.get("/api/v1/settings").json()
    assert after["embedding_model"] == before["embedding_model"]
    assert after["query_model"] == before["query_model"]


def test_other_settings_still_save(client):
    """The read-only guard must not block the fields that are genuinely settable."""
    r = client.put("/api/v1/settings", json={"cms_host": "192.0.2.10"})
    assert r.status_code == 200
    assert r.json()["cms_host"] == "192.0.2.10"
    client.put("/api/v1/settings", json={"cms_host": "127.0.0.1"})


def test_daz_studio_status(client):
    r = client.get("/api/v1/daz-studio/status")
    assert r.status_code == 200
    body = r.json()
    assert "plugin_detected" in body
