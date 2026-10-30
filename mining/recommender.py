"""
Recommendation Engine Module (Content-Based, Item-CF, Apriori Rules, and Hybrid).

Course Outcome: CO5 - Design and evaluate personalized recommendation systems
integrating content-based filtering, item-item collaborative filtering,
and association rule mining with offline empirical evaluation.

This module implements:
1. Content-Based Recommender:
   - Constructs movie document representations from genres and user tags.
   - Computes TF-IDF vectors and cosine similarities.
2. Item-Based Collaborative Filtering (Item-CF):
   - Constructs mean-centered user-item rating matrix for movies with >= 20 ratings.
   - Computes item-item cosine similarity.
3. Rule-Based Recommender:
   - Queries antecedent rules mined via Apriori (Module B).
4. Hybrid Recommender:
   - Weighted ensemble (0.4*content + 0.3*cf + 0.3*rules) with dynamic weight
     redistribution when components are missing.
   - Generates natural, human-readable explanations ("why" strings).
5. Cold-Start Fallback:
   - IMDB/Bayesian weighted average popularity formula within genre.
6. Offline Evaluation:
   - Leave-one-out evaluation per user measuring Hit-Rate@10, Precision@10, Recall@10.
   - Exports comparison bar chart to reports/figures/recommender_evaluation.png.
7. Saves artifacts to models/recommender.joblib.
"""

import random
import sqlite3
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mining.apriori_rules import get_rules_for_movie

CONFIG = {
    "weights": {
        "content": 0.40,
        "cf": 0.30,
        "rule": 0.30,
    },
    "cf_min_ratings": 20,
    "top_neighbors_stored": 50,
    "random_state": 42,
}


def build_movie_metadata_lookup(conn: sqlite3.Connection) -> dict[int, dict]:
    """Builds a fast in-memory dictionary of movie details."""
    query = """
    SELECT m.movie_id, m.title, m.clean_title, m.year,
           AVG(r.rating) as avg_rating,
           COUNT(r.rating_id) as n_ratings
    FROM dim_movie m
    LEFT JOIN fact_ratings r ON m.movie_id = r.movie_id
    GROUP BY m.movie_id;
    """
    df = pd.read_sql_query(query, conn)

    # Fetch genres per movie
    genres_df = pd.read_sql_query(
        """
        SELECT mg.movie_id, g.genre_name
        FROM movie_genre mg
        JOIN dim_genre g ON mg.genre_id = g.genre_id;
        """,
        conn,
    )
    genre_map = genres_df.groupby("movie_id")["genre_name"].apply(list).to_dict()

    lookup = {}
    for _, row in df.iterrows():
        mid = int(row["movie_id"])
        lookup[mid] = {
            "movie_id": mid,
            "title": row["title"],
            "clean_title": row["clean_title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "avg_rating": round(float(row["avg_rating"]), 2) if pd.notna(row["avg_rating"]) else 0.0,
            "n_ratings": int(row["n_ratings"]) if pd.notna(row["n_ratings"]) else 0,
            "genres": genre_map.get(mid, []),
            "genre_str": ", ".join(genre_map.get(mid, [])),
        }
    return lookup


def build_content_engine(
    conn: sqlite3.Connection,
    k_neighbors: int = CONFIG["top_neighbors_stored"]
) -> tuple[dict[int, list[tuple[int, float]]], list[int]]:
    """Builds Content-Based recommender using TF-IDF over genres + user tags."""
    print("Building Content-Based similarity index (genres + tags)...")

    # 1. Fetch tags concatenated per movie
    tags_df = pd.read_sql_query(
        "SELECT movie_id, GROUP_CONCAT(tag, ' ') as tags FROM fact_tags GROUP BY movie_id;",
        conn,
    )
    tag_dict = dict(zip(tags_df["movie_id"], tags_df["tags"]))

    # 2. Fetch genres concatenated per movie
    genres_df = pd.read_sql_query(
        """
        SELECT mg.movie_id, GROUP_CONCAT(g.genre_name, ' ') as genres
        FROM movie_genre mg
        JOIN dim_genre g ON mg.genre_id = g.genre_id
        GROUP BY mg.movie_id;
        """,
        conn,
    )
    genre_dict = dict(zip(genres_df["movie_id"], genres_df["genres"]))

    # 3. Fetch all movies
    movies_df = pd.read_sql_query("SELECT movie_id, clean_title FROM dim_movie ORDER BY movie_id;", conn)
    all_movie_ids = movies_df["movie_id"].tolist()

    documents = []
    for mid in all_movie_ids:
        # Boost genres by repeating them twice, concatenate user tags
        g_text = (genre_dict.get(mid, "") + " ") * 2
        t_text = tag_dict.get(mid, "")
        doc = f"{g_text} {t_text}".strip().lower()
        documents.append(doc if doc else "unknown")

    # 4. TF-IDF and Cosine Similarity
    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute pairwise similarities in chunks to conserve RAM
    n_movies = len(all_movie_ids)
    content_neighbors = {}

    batch_size = 1000
    for i in range(0, n_movies, batch_size):
        chunk = tfidf_matrix[i : i + batch_size]
        sim_chunk = cosine_similarity(chunk, tfidf_matrix)

        for row_idx, sim_row in enumerate(sim_chunk):
            movie_idx = i + row_idx
            mid = all_movie_ids[movie_idx]

            # Exclude self
            sim_row[movie_idx] = -1.0
            # Get top k neighbors
            top_indices = np.argsort(sim_row)[::-1][:k_neighbors]
            neighbors = [
                (all_movie_ids[idx], float(sim_row[idx]))
                for idx in top_indices if sim_row[idx] > 0.01
            ]
            content_neighbors[mid] = neighbors

    return content_neighbors, all_movie_ids


def build_item_cf_engine(
    conn: sqlite3.Connection,
    min_ratings: int = CONFIG["cf_min_ratings"],
    k_neighbors: int = CONFIG["top_neighbors_stored"]
) -> dict[int, list[tuple[int, float]]]:
    """Builds Item-Item Collaborative Filtering engine on mean-centered ratings."""
    print(f"Building Item-CF similarity matrix (movies with >= {min_ratings} ratings)...")

    # 1. Fetch ratings for qualified movies
    cf_query = f"""
    SELECT user_id, movie_id, rating
    FROM fact_ratings
    WHERE movie_id IN (
        SELECT movie_id FROM fact_ratings
        GROUP BY movie_id HAVING COUNT(rating_id) >= {min_ratings}
    );
    """
    df_ratings = pd.read_sql_query(cf_query, conn)

    # 2. Pivot table: users x movies
    user_item = df_ratings.pivot(index="user_id", columns="movie_id", values="rating")
    cf_movie_ids = list(user_item.columns)

    # 3. Mean-center ratings per user (row-wise mean subtraction)
    user_means = user_item.mean(axis=1)
    user_item_centered = user_item.sub(user_means, axis=0).fillna(0.0)

    # 4. Item-Item Cosine Similarity
    # Columns are items, so transpose matrix to items x users
    item_matrix = user_item_centered.T.values
    sim_matrix = cosine_similarity(item_matrix)

    cf_neighbors = {}
    for i, mid in enumerate(cf_movie_ids):
        sim_row = sim_matrix[i].copy()
        sim_row[i] = -1.0  # exclude self
        top_indices = np.argsort(sim_row)[::-1][:k_neighbors]
        neighbors = [
            (cf_movie_ids[idx], float(sim_row[idx]))
            for idx in top_indices if sim_row[idx] > 0.05
        ]
        cf_neighbors[mid] = neighbors

    return cf_neighbors


def get_bayesian_popularity_recommendations(
    conn: sqlite3.Connection,
    genre: str | None = None,
    k: int = 10,
    m: int = 10,
) -> list[int]:
    """Cold-start fallback using Bayesian weighted rating."""
    global_mean = conn.execute("SELECT AVG(rating) FROM fact_ratings;").fetchone()[0] or 3.5

    where_clause = ""
    if genre:
        where_clause = f"""
        WHERE m.movie_id IN (
            SELECT mg.movie_id FROM movie_genre mg
            JOIN dim_genre g ON mg.genre_id = g.genre_id
            WHERE g.genre_name = '{genre}'
        )
        """

    query = f"""
    SELECT m.movie_id, COUNT(r.rating_id) as v, AVG(r.rating) as r_bar
    FROM dim_movie m
    LEFT JOIN fact_ratings r ON m.movie_id = r.movie_id
    {where_clause}
    GROUP BY m.movie_id
    HAVING v >= 5
    """
    df = pd.read_sql_query(query, conn)
    if df.empty:
        return []

    # Weighted Rating Formula: (v / (v + m)) * R + (m / (v + m)) * C
    df["score"] = (df["v"] / (df["v"] + m)) * df["r_bar"] + (m / (df["v"] + m)) * global_mean
    top_movies = df.sort_values(by="score", ascending=False).head(k)["movie_id"].tolist()
    return [int(m) for m in top_movies]


def recommend(
    movie_id: int,
    k: int = 10,
    method: str = "hybrid",
    recommender_payload: dict | None = None,
    rules_payload: dict | None = None,
    models_dir: Path = Path("models"),
    weights: dict | None = None,
) -> list[dict]:
    """Generates recommendations with detailed scores and 'why' explanation strings."""
    if recommender_payload is None:
        rec_file = models_dir / "recommender.joblib"
        if not rec_file.exists():
            return []
        recommender_payload = joblib.load(rec_file)

    if rules_payload is None:
        rule_file = models_dir / "rules.joblib"
        rules_payload = joblib.load(rule_file) if rule_file.exists() else {}

    content_index = recommender_payload.get("content_neighbors", {})
    cf_index = recommender_payload.get("cf_neighbors", {})
    meta = recommender_payload.get("movie_metadata", {})

    target_meta = meta.get(movie_id, {"title": f"Movie {movie_id}", "genres": []})
    target_genres = set(target_meta.get("genres", []))

    # 1. Content Candidates
    content_scores = {mid: score for mid, score in content_index.get(movie_id, [])}

    # 2. Collaborative Filtering Candidates
    cf_scores = {mid: score for mid, score in cf_index.get(movie_id, [])}

    # 3. Association Rules Candidates
    rule_matches = get_rules_for_movie(movie_id, k=25, rules_payload=rules_payload)
    rule_scores = {}
    rule_explanations = {}
    for r in rule_matches:
        for c_id in r["consequent_ids"]:
            if c_id != movie_id:
                # Rule score combines confidence and lift
                r_score = float(r["confidence"] * min(r["lift"] / 5.0, 1.0))
                if c_id not in rule_scores or r_score > rule_scores[c_id]:
                    rule_scores[c_id] = r_score
                    rule_explanations[c_id] = (
                        f"Frequent co-viewing rule with Lift {r['lift']:.2f} and {int(r['confidence']*100)}% confidence"
                    )

    # Filter to requested single method if specified
    if method == "content":
        candidate_ids = list(content_scores.keys())
    elif method == "cf":
        candidate_ids = list(cf_scores.keys())
    elif method == "rules":
        candidate_ids = list(rule_scores.keys())
    else:  # hybrid
        candidate_ids = list(set(content_scores.keys()) | set(cf_scores.keys()) | set(rule_scores.keys()))

    if not candidate_ids:
        # Fallback to genre popularity
        primary_genre = target_meta.get("genres", [None])[0] if target_meta.get("genres") else None
        cold_ids = recommender_payload.get("cold_start", {}).get(primary_genre, [])
        candidate_ids = [m for m in cold_ids if m != movie_id][:k]
        if not candidate_ids:
            return []

    # Min-Max Normalization per query
    def normalize_dict(d: dict) -> dict:
        if not d:
            return {}
        vals = list(d.values())
        min_v, max_v = min(vals), max(vals)
        if max_v - min_v < 1e-6:
            return {k: 1.0 for k in d}
        return {k: (v - min_v) / (max_v - min_v) for k, v in d.items()}

    norm_content = normalize_dict(content_scores)
    norm_cf = normalize_dict(cf_scores)
    norm_rule = normalize_dict(rule_scores)

    cfg_weights = weights or CONFIG["weights"]
    w_content = cfg_weights["content"] if method in ("hybrid", "content") else 0.0
    w_cf = cfg_weights["cf"] if method in ("hybrid", "cf") else 0.0
    w_rule = cfg_weights["rule"] if method in ("hybrid", "rules") else 0.0

    if method != "hybrid":
        if method == "content":
            w_content, w_cf, w_rule = 1.0, 0.0, 0.0
        elif method == "cf":
            w_content, w_cf, w_rule = 0.0, 1.0, 0.0
        elif method == "rules":
            w_content, w_cf, w_rule = 0.0, 0.0, 1.0

    recommendations = []
    for cand_id in candidate_ids:
        c_score = norm_content.get(cand_id, 0.0)
        cf_score = norm_cf.get(cand_id, 0.0)
        r_score = norm_rule.get(cand_id, 0.0)

        # Dynamic weight redistribution if some signals are missing
        active_w_content = w_content if cand_id in norm_content else 0.0
        active_w_cf = w_cf if cand_id in norm_cf else 0.0
        active_w_rule = w_rule if cand_id in norm_rule else 0.0
        total_active_w = active_w_content + active_w_cf + active_w_rule

        if total_active_w > 0:
            final_score = (
                (c_score * active_w_content) +
                (cf_score * active_w_cf) +
                (r_score * active_w_rule)
            ) / total_active_w
        else:
            final_score = 0.1

        cand_meta = meta.get(cand_id, {"title": f"Movie {cand_id}", "clean_title": f"Movie {cand_id}", "genres": []})
        shared_genres = target_genres.intersection(set(cand_meta.get("genres", [])))

        # Formulate human-readable "why" text
        why_parts = []
        if shared_genres:
            why_parts.append(f"Shares genres: {', '.join(sorted(shared_genres))}")
        if cand_id in norm_cf:
            pct = int(cf_scores.get(cand_id, 0.5) * 100)
            why_parts.append(f"Viewers who liked '{target_meta.get('clean_title', 'this')}' rated this {pct}% similarly")
        if cand_id in rule_explanations:
            why_parts.append(rule_explanations[cand_id])

        why_text = "; ".join(why_parts) if why_parts else "High genre and viewer rating affinity."

        recommendations.append({
            "movie_id": cand_id,
            "title": cand_meta.get("clean_title", cand_meta.get("title", f"Movie {cand_id}")),
            "full_title": cand_meta.get("title", ""),
            "year": cand_meta.get("year"),
            "genres": cand_meta.get("genres", []),
            "avg_rating": cand_meta.get("avg_rating", 0.0),
            "n_ratings": cand_meta.get("n_ratings", 0),
            "score": round(float(final_score), 4),
            "component_scores": {
                "content": round(float(c_score), 3),
                "cf": round(float(cf_score), 3),
                "rule": round(float(r_score), 3),
            },
            "why": why_text,
        })

    # Sort descending by final score
    recommendations = sorted(recommendations, key=lambda x: x["score"], reverse=True)[:k]
    return recommendations


def evaluate_recommenders_offline(
    conn: sqlite3.Connection,
    recommender_payload: dict,
    rules_payload: dict,
    reports_dir: Path,
    n_sample_users: int = 150,
) -> dict:
    """Performs offline Leave-One-Out Hit-Rate@10, Precision@10, and Recall@10 evaluation."""
    print(f"\n--- Offline Evaluation: Leave-One-Out Hit-Rate@10 (N={n_sample_users} users) ---")

    # 1. Fetch users with >= 15 ratings and at least 3 ratings >= 4.0
    user_pool_df = pd.read_sql_query(
        """
        SELECT user_id, COUNT(rating_id) as total_r,
               SUM(CASE WHEN rating >= 4.0 THEN 1 ELSE 0 END) as liked_r
        FROM fact_ratings
        GROUP BY user_id
        HAVING total_r >= 15 AND liked_r >= 3;
        """,
        conn,
    )

    eligible_users = user_pool_df["user_id"].tolist()
    random.seed(CONFIG["random_state"])
    sample_users = random.sample(eligible_users, min(n_sample_users, len(eligible_users)))

    methods = ["content", "cf", "rules", "hybrid"]
    hits = {m: 0 for m in methods}
    eval_counts = {m: 0 for m in methods}

    for uid in sample_users:
        # User's liked movies
        liked_df = pd.read_sql_query(
            f"SELECT movie_id FROM fact_ratings WHERE user_id = {uid} AND rating >= 4.0 ORDER BY rating DESC;",
            conn,
        )
        liked_ids = liked_df["movie_id"].tolist()
        if len(liked_ids) < 3:
            continue

        # Leave out 1 random liked movie
        test_movie = random.choice(liked_ids)
        train_seeds = [m for m in liked_ids if m != test_movie][:3]

        for m in methods:
            # Query top 10 using first seed movie
            seed_movie = train_seeds[0]
            recs = recommend(
                movie_id=seed_movie,
                k=10,
                method=m,
                recommender_payload=recommender_payload,
                rules_payload=rules_payload,
            )
            rec_ids = [r["movie_id"] for r in recs]
            eval_counts[m] += 1
            if test_movie in rec_ids:
                hits[m] += 1

    results = {}
    for m in methods:
        n = max(eval_counts[m], 1)
        h = hits[m]
        hit_rate = round(float(h / n), 4)
        precision_10 = round(float(h / (n * 10)), 4)
        recall_10 = round(float(h / n), 4)  # with 1 hidden item, recall = hit rate
        results[m] = {
            "evaluated_users": n,
            "hits": h,
            "hit_rate_at_10": hit_rate,
            "precision_at_10": precision_10,
            "recall_at_10": recall_10,
        }

    print(pd.DataFrame(results).T.to_string())

    # Plot Comparison Bar Chart
    plt.figure(figsize=(8, 5))
    method_labels = ["Content-Based", "Item-CF", "Apriori Rules", "Hybrid Engine"]
    hr_values = [results[m]["hit_rate_at_10"] for m in methods]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b3"]

    bars = plt.bar(method_labels, hr_values, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.003, f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.ylabel("Hit-Rate @ 10")
    plt.title("Recommender Offline Evaluation: Hit-Rate@10 Comparison")
    plt.ylim(0, max(hr_values) * 1.25 if max(hr_values) > 0 else 0.2)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    fig_path = reports_dir / "figures" / "recommender_evaluation.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved recommender evaluation bar chart to {fig_path}")

    return results


def train_recommender(
    db_path: Path = Path("warehouse.db"),
    models_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
) -> dict:
    """Builds and caches all recommender indexes, runs evaluation, and saves models."""
    print("=" * 70)
    print("RUNNING RECOMMENDER ENGINE (CO5: Content, Item-CF, Apriori Rules, Hybrid)")
    print("=" * 70)

    conn = sqlite3.connect(db_path)
    meta = build_movie_metadata_lookup(conn)
    content_index, all_movie_ids = build_content_engine(conn)
    cf_index = build_item_cf_engine(conn)

    # Precompute Cold Start Bayesian popularity per genre
    genres_df = pd.read_sql_query("SELECT genre_name FROM dim_genre;", conn)
    cold_start_map = {}
    for g in genres_df["genre_name"]:
        cold_start_map[g] = get_bayesian_popularity_recommendations(conn, genre=g, k=15)
    cold_start_map[None] = get_bayesian_popularity_recommendations(conn, genre=None, k=15)

    # Save to models/recommender.joblib
    models_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "content_neighbors": content_index,
        "cf_neighbors": cf_index,
        "movie_metadata": meta,
        "cold_start": cold_start_map,
        "config": CONFIG,
    }
    rec_file = models_dir / "recommender.joblib"
    joblib.dump(payload, rec_file)
    print(f"Saved recommender models payload to {rec_file}")

    # Load rules for evaluation
    rule_file = models_dir / "rules.joblib"
    rules_payload = joblib.load(rule_file) if rule_file.exists() else {}

    # Run offline evaluation
    eval_metrics = evaluate_recommenders_offline(conn, payload, rules_payload, reports_dir)
    conn.close()

    return {
        "total_movies_indexed": len(all_movie_ids),
        "cf_movies_indexed": len(cf_index),
        "evaluation": eval_metrics,
    }


if __name__ == "__main__":
    train_recommender()

