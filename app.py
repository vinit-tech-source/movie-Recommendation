"""
Flask REST API and Web Application Server.

Course Outcomes: CO1 to CO5.
Integrates the complete data warehouse and machine learning models:
- CO1: Star Schema warehouse queries, OLAP statistics, and metadata retrieval.
- CO2: Apriori Association Rules and Naive Bayes text genre classification.
- CO3: Linear, Ridge, Lasso Regression for numeric rating predictions.
- CO4: Decision Tree and Random Forest rating-class classifiers.
- CO5: ARIMA Time Series forecasts and Hybrid Recommendation Engine.

Endpoints:
- GET  /api/health            Warehouse status and loaded model inventory
- GET  /api/search            Autocomplete movie search (SQL LIKE)
- GET  /api/recommend         Personalized recommendations with 'why' explanations
- POST /api/predict-rating    Movie rating classification and regression prediction
- POST /api/predict-genre     Genre classification from text plot description
- GET  /api/trends            Time series history and 12-month ARIMA forecast
- GET  /api/stats             Aggregated warehouse statistics and model metrics
- Web routes: /, /recommend, /predict, /trends, /performance
"""

import json
import sqlite3
from pathlib import Path
from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np

from mining.apriori_rules import get_rules_for_movie
from mining.arima_model import get_trend_api_data
from mining.classifier import clean_and_stem_text
from mining.recommender import recommend

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
DB_PATH = Path("warehouse.db")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

# In-memory Model and Cache Registry
MODELS = {
    "rules": None,
    "genre_classifier": None,
    "rating_classifier": None,
    "rating_regressor": None,
    "arima": None,
    "recommender": None,
}
METRICS = {}


def get_db_connection() -> sqlite3.Connection:
    """Creates a connection to SQLite with row dictionary factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_all_models() -> None:
    """Loads all persisted machine learning models and metrics at startup."""
    global MODELS, METRICS

    # Load Metrics JSON
    metrics_path = REPORTS_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                METRICS = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metrics.json: {e}")

    # Load Model Artifacts
    files_map = {
        "rules": MODELS_DIR / "rules.joblib",
        "genre_classifier": MODELS_DIR / "genre_classifier.joblib",
        "rating_classifier": MODELS_DIR / "rating_classifier.joblib",
        "rating_regressor": MODELS_DIR / "rating_regressor.joblib",
        "arima": MODELS_DIR / "arima_models.joblib",
        "recommender": MODELS_DIR / "recommender.joblib",
    }
    for key, path in files_map.items():
        if path.exists():
            try:
                MODELS[key] = joblib.load(path)
                print(f"[Loaded] {key} from {path}")
            except Exception as e:
                print(f"Error loading {key}: {e}")
        else:
            print(f"[Missing] Model file {path} not found.")


# Initialize models on load
load_all_models()


# ==============================================================================
# HTML View Routes
# ==============================================================================

@app.route("/")
def index_view():
    return render_template("index.html")


@app.route("/recommend")
def recommend_view():
    return render_template("recommend.html")


@app.route("/predict")
def predict_view():
    return render_template("predict.html")


@app.route("/trends")
def trends_view():
    return render_template("trends.html")


@app.route("/performance")
def performance_view():
    return render_template("performance.html")


# ==============================================================================
# REST API Endpoints
# ==============================================================================

@app.route("/api/health", methods=["GET"])
def api_health():
    """Returns warehouse table row counts and list of loaded models."""
    if not DB_PATH.exists():
        return jsonify({"error": "Warehouse database does not exist"}), 500

    conn = get_db_connection()
    tables = [
        "dim_date", "dim_user", "dim_movie", "dim_genre", "movie_genre",
        "fact_ratings", "fact_tags", "dim_title", "dim_netflix_genre",
        "title_genre", "fact_titles"
    ]
    counts = {}
    for tbl in tables:
        try:
            cnt = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            counts[tbl] = cnt
        except Exception:
            counts[tbl] = -1
    conn.close()

    loaded_status = {k: bool(v is not None) for k, v in MODELS.items()}
    all_ready = all(loaded_status.values())

    return jsonify({
        "status": "healthy" if all_ready else "degraded",
        "database_connected": True,
        "warehouse_row_counts": counts,
        "models_loaded": loaded_status,
        "metrics_loaded": bool(METRICS),
    }), 200


@app.route("/api/search", methods=["GET"])
def api_search():
    """Case-insensitive movie search for autocomplete (returns top matches)."""
    q = request.args.get("q", "").strip()
    if not q or len(q) < 1:
        return jsonify({"error": "Query parameter 'q' must not be empty"}), 400

    try:
        limit = min(int(request.args.get("limit", 10)), 50)
    except ValueError:
        limit = 10

    conn = get_db_connection()
    # Search clean_title or title
    search_sql = """
    SELECT m.movie_id, m.title, m.clean_title, m.year,
           ROUND(AVG(r.rating), 2) as avg_rating,
           COUNT(r.rating_id) as n_ratings,
           (SELECT GROUP_CONCAT(g.genre_name, ', ')
            FROM movie_genre mg
            JOIN dim_genre g ON mg.genre_id = g.genre_id
            WHERE mg.movie_id = m.movie_id) as genres
    FROM dim_movie m
    LEFT JOIN fact_ratings r ON m.movie_id = r.movie_id
    WHERE m.clean_title LIKE ? OR m.title LIKE ?
    GROUP BY m.movie_id
    ORDER BY n_ratings DESC, m.year DESC
    LIMIT ?;
    """
    pattern = f"%{q}%"
    rows = conn.execute(search_sql, (pattern, pattern, limit)).fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "movie_id": r["movie_id"],
            "title": r["title"],
            "clean_title": r["clean_title"],
            "year": r["year"],
            "avg_rating": float(r["avg_rating"]) if r["avg_rating"] is not None else 0.0,
            "n_ratings": int(r["n_ratings"]),
            "genres": [g.strip() for g in r["genres"].split(",")] if r["genres"] else [],
        })
    return jsonify({"query": q, "count": len(results), "results": results}), 200


@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    """Provides recommendations with scores and explanations."""
    movie_id_str = request.args.get("movie_id")
    if not movie_id_str:
        return jsonify({"error": "Missing required 'movie_id' parameter"}), 400

    try:
        movie_id = int(movie_id_str)
    except ValueError:
        return jsonify({"error": "Invalid 'movie_id'; must be an integer"}), 400

    method = request.args.get("method", "hybrid").lower()
    valid_methods = {"content", "cf", "rules", "hybrid"}
    if method not in valid_methods:
        return jsonify({"error": f"Invalid method '{method}'. Must be one of: {list(valid_methods)}"}), 400

    try:
        k = min(int(request.args.get("k", 10)), 30)
    except ValueError:
        k = 10

    # Verify movie exists
    conn = get_db_connection()
    target_row = conn.execute("SELECT movie_id, title, clean_title, year FROM dim_movie WHERE movie_id = ?", (movie_id,)).fetchone()
    conn.close()
    if not target_row:
        return jsonify({"error": f"Movie with id {movie_id} not found in catalog"}), 404

    # Run recommendation
    recs = recommend(
        movie_id=movie_id,
        k=k,
        method=method,
        recommender_payload=MODELS.get("recommender"),
        rules_payload=MODELS.get("rules"),
    )

    # Fetch associated rules for this movie
    rules_used = get_rules_for_movie(movie_id=movie_id, k=5, rules_payload=MODELS.get("rules"))

    return jsonify({
        "target_movie": {
            "movie_id": target_row["movie_id"],
            "title": target_row["title"],
            "clean_title": target_row["clean_title"],
            "year": target_row["year"],
        },
        "method": method,
        "k": k,
        "recommendations": recs,
        "association_rules_used": rules_used,
    }), 200


@app.route("/api/predict-rating", methods=["POST"])
def api_predict_rating():
    """Predicts movie rating class (Low/Med/High) and continuous rating value."""
    data = request.get_json(silent=True) or {}

    genres = data.get("genres", [])
    if not isinstance(genres, list):
        return jsonify({"error": "'genres' must be a list of genre names"}), 400

    try:
        year = int(data.get("year", 2020))
        n_ratings = max(0, int(data.get("n_ratings", 20)))
        n_tags = max(0, int(data.get("n_tags", 2)))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid numerical values for year, n_ratings, or n_tags"}), 400

    clf_payload = MODELS.get("rating_classifier")
    reg_payload = MODELS.get("rating_regressor")

    if not clf_payload or not reg_payload:
        return jsonify({"error": "Rating classification or regression models are not loaded"}), 500

    # 1. Classification
    clf_model = clf_payload["model"]
    feature_cols = clf_payload["feature_cols"]

    row = {
        "year": year,
        "log_n_ratings": np.log1p(n_ratings),
        "n_tags": n_tags,
    }
    for col in feature_cols:
        if col not in row:
            row[col] = 1 if col in genres else 0

    import pandas as pd
    df_feat = pd.DataFrame([row])[feature_cols]

    predicted_class = clf_model.predict(df_feat)[0]
    class_probs = clf_model.predict_proba(df_feat)[0]
    prob_dict = {cls: round(float(p), 4) for cls, p in zip(clf_model.classes_, class_probs)}

    # 2. Continuous Rating Prediction from Regression
    reg_pipe = reg_payload["pipeline"]
    pred_rating = float(reg_pipe.predict(df_feat)[0])
    pred_rating_clipped = round(float(np.clip(pred_rating, 0.5, 5.0)), 2)

    return jsonify({
        "input": {
            "genres": genres,
            "year": year,
            "n_ratings": n_ratings,
            "n_tags": n_tags,
        },
        "predicted_class": predicted_class,
        "class_probabilities": prob_dict,
        "predicted_rating_value": pred_rating_clipped,
    }), 200


@app.route("/api/predict-genre", methods=["POST"])
def api_predict_genre():
    """Predicts top-3 genres from textual plot description using Naive Bayes."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Missing required field 'text' with plot description"}), 400

    genre_payload = MODELS.get("genre_classifier")
    if not genre_payload:
        return jsonify({"error": "Genre classification pipeline is not loaded"}), 500

    pipe = genre_payload["pipeline"]
    cleaned = clean_and_stem_text(text)
    if not cleaned:
        cleaned = text.lower()

    probs = pipe.predict_proba([cleaned])[0]
    classes = pipe.classes_
    top_indices = np.argsort(probs)[::-1][:3]

    predictions = [
        {"genre": classes[i], "probability": round(float(probs[i]), 4)}
        for i in top_indices
    ]
    return jsonify({
        "input_text": text,
        "source_dataset": genre_payload.get("source_used", "unknown"),
        "top_predictions": predictions,
    }), 200


@app.route("/api/trends", methods=["GET"])
def api_trends():
    """Returns historical time series, ADF test, and 12-month ARIMA forecast."""
    source = request.args.get("source", "netflix").lower()
    genre = request.args.get("genre")
    title_type = request.args.get("type")

    if source not in ("netflix", "movielens"):
        return jsonify({"error": "Invalid source; must be 'netflix' or 'movielens'"}), 400

    data = get_trend_api_data(
        source=source,
        genre=genre,
        title_type=title_type,
        db_path=DB_PATH,
        models_dir=MODELS_DIR,
    )
    return jsonify(data), 200


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Returns warehouse summary statistics, distribution histograms, and all model metrics."""
    conn = get_db_connection()

    # 1. Top 10 Genres
    top_genres_rows = conn.execute("""
        SELECT g.genre_name, COUNT(f.rating_id) as count
        FROM fact_ratings f
        JOIN movie_genre mg ON f.movie_id = mg.movie_id
        JOIN dim_genre g ON mg.genre_id = g.genre_id
        GROUP BY g.genre_name
        ORDER BY count DESC
        LIMIT 10;
    """).fetchall()
    top_genres = [{"genre": r["genre_name"], "count": r["count"]} for r in top_genres_rows]

    # 2. Rating Histogram (0.5 buckets)
    hist_rows = conn.execute("""
        SELECT rating, COUNT(*) as count
        FROM fact_ratings
        GROUP BY rating
        ORDER BY rating;
    """).fetchall()
    rating_histogram = [{"rating": float(r["rating"]), "count": r["count"]} for r in hist_rows]

    # 3. Ratings per Year
    r_year_rows = conn.execute("""
        SELECT d.year, COUNT(f.rating_id) as count, ROUND(AVG(f.rating), 2) as avg_rating
        FROM fact_ratings f
        JOIN dim_date d ON f.date_id = d.date_id
        GROUP BY d.year
        ORDER BY d.year;
    """).fetchall()
    ratings_per_year = [
        {"year": r["year"], "count": r["count"], "avg_rating": float(r["avg_rating"] or 0)}
        for r in r_year_rows
    ]

    # 4. Netflix Titles Added per Year by Type
    nf_year_rows = conn.execute("""
        SELECT d.year, t.type, COUNT(ft.show_id) as count
        FROM fact_titles ft
        JOIN dim_title t ON ft.show_id = t.show_id
        JOIN dim_date d ON ft.date_id = d.date_id
        WHERE d.year >= 2015
        GROUP BY d.year, t.type
        ORDER BY d.year, t.type;
    """).fetchall()
    netflix_by_year = [{"year": r["year"], "type": r["type"], "count": r["count"]} for r in nf_year_rows]

    conn.close()

    return jsonify({
        "top_genres": top_genres,
        "rating_histogram": rating_histogram,
        "ratings_per_year": ratings_per_year,
        "netflix_by_year": netflix_by_year,
        "model_metrics": METRICS,
    }), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

