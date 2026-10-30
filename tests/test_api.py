"""
Tests for Flask REST API Endpoints and Error Handling.

Covers:
- /api/health
- /api/search
- /api/recommend
- /api/predict-rating
- /api/predict-genre
- /api/trends
- /api/stats
- Web view route responses
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_api_health(client):
    """Assert health check returns status 200 and warehouse table metrics."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert "status" in data
    assert "warehouse_row_counts" in data
    assert data["warehouse_row_counts"]["fact_ratings"] == 100836
    assert data["warehouse_row_counts"]["dim_movie"] == 9742


def test_api_search_success(client):
    """Assert search returns results for common titles."""
    res = client.get("/api/search?q=Toy&limit=5")
    assert res.status_code == 200
    data = res.get_json()
    assert data["count"] > 0
    assert any("Toy Story" in r["title"] for r in data["results"])
    item = data["results"][0]
    assert "movie_id" in item
    assert "avg_rating" in item
    assert "genres" in item


def test_api_search_bad_query(client):
    """Assert search with empty query returns 400."""
    res = client.get("/api/search?q=")
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


def test_api_recommend_valid(client):
    """Assert recommendation returns results with scores and why explanations."""
    res = client.get("/api/recommend?movie_id=1&k=5&method=hybrid")
    assert res.status_code == 200
    data = res.get_json()
    assert data["target_movie"]["movie_id"] == 1
    assert len(data["recommendations"]) <= 5
    if data["recommendations"]:
        first = data["recommendations"][0]
        assert "score" in first
        assert "why" in first
        assert "component_scores" in first


def test_api_recommend_methods(client):
    """Assert content, cf, and rules recommendation methods respond."""
    for m in ["content", "cf", "rules"]:
        res = client.get(f"/api/recommend?movie_id=1&k=3&method={m}")
        assert res.status_code == 200
        data = res.get_json()
        assert data["method"] == m


def test_api_recommend_invalid_movie(client):
    """Assert 404 for nonexistent movie."""
    res = client.get("/api/recommend?movie_id=999999")
    assert res.status_code == 404
    data = res.get_json()
    assert "error" in data


def test_api_predict_rating_success(client):
    """Assert rating prediction returns class and numeric regression."""
    payload = {
        "genres": ["Action", "Sci-Fi"],
        "year": 2021,
        "n_ratings": 120,
        "n_tags": 5,
    }
    res = client.post(
        "/api/predict-rating",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["predicted_class"] in ["Low", "Medium", "High"]
    assert "class_probabilities" in data
    assert 0.5 <= data["predicted_rating_value"] <= 5.0


def test_api_predict_genre_success(client):
    """Assert genre prediction from plot text returns top classes with probabilities."""
    payload = {
        "text": "A secret agent embarks on a dangerous mission to save the world from an evil mastermind."
    }
    res = client.post(
        "/api/predict-genre",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["top_predictions"]) == 3
    assert "probability" in data["top_predictions"][0]


def test_api_predict_genre_missing_text(client):
    """Assert 400 for empty plot text."""
    res = client.post("/api/predict-genre", data=json.dumps({}), content_type="application/json")
    assert res.status_code == 400


def test_api_trends_netflix(client):
    """Assert trends endpoint returns history and 12-month ARIMA forecast."""
    res = client.get("/api/trends?source=netflix")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["history"]) >= 60
    assert len(data["forecast"]) == 12
    assert "model_order" in data


def test_api_trends_movielens(client):
    """Assert MovieLens trends return history and forecast."""
    res = client.get("/api/trends?source=movielens")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["history"]) > 100


def test_api_stats(client):
    """Assert stats endpoint aggregates top genres, histogram, and metrics."""
    res = client.get("/api/stats")
    assert res.status_code == 200
    data = res.get_json()
    assert len(data["top_genres"]) == 10
    assert len(data["rating_histogram"]) > 5
    assert len(data["ratings_per_year"]) > 10


def test_html_views(client):
    """Assert all 5 frontend HTML routes render with status 200."""
    routes = ["/", "/recommend", "/predict", "/trends", "/performance"]
    for route in routes:
        res = client.get(route)
        assert res.status_code == 200, f"Route {route} failed with status {res.status_code}"
        assert b"CineMetrics" in res.data or b"Movie" in res.data
