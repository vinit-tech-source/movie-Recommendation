"""
Association Rule Mining Module (Apriori Algorithm).

Course Outcome: CO2 - Apply association rule mining and frequent pattern
discovery on transactional / basket data to identify cross-consumption behavior.

This module:
1. Formulates movie viewership into user baskets (items = movies rated >= 4.0).
2. Filters to the top-300 most frequently rated movies to reduce sparsity.
3. Generates frequent itemsets and association rules using the Apriori algorithm:
   - Strict set (min_support=0.10, max_len=3, min_confidence=0.6, min_lift=1.2) for reporting.
   - Lenient set (min_support=0.05, max_len=2, min_confidence=0.5, min_lift=1.2) for broader app recommendations.
4. Generates visual scatter plot (support vs. confidence colored by lift).
5. Exports models to models/rules.joblib and saves top rules to reports/metrics.json.
6. Provides get_rules_for_movie() for recommender engine integration.
"""

import sqlite3
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Configuration parameters for Apriori mining
CONFIG = {
    "top_n_movies": 300,
    "strict": {
        "min_support": 0.10,
        "max_len": 3,
        "min_confidence": 0.60,
        "min_lift": 1.20,
    },
    "lenient": {
        "min_support": 0.05,
        "max_len": 2,
        "min_confidence": 0.50,
        "min_lift": 1.20,
    },
    "random_state": 42,
}


def build_user_movie_matrix(
    db_path: Path = Path("warehouse.db"),
    top_n: int = CONFIG["top_n_movies"]
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Builds a binary user x movie transaction matrix for ratings >= 4.0."""
    conn = sqlite3.connect(db_path)

    # 1. Fetch top N most rated movies
    top_movies_df = pd.read_sql_query(
        f"""
        SELECT movie_id, COUNT(*) as cnt
        FROM fact_ratings
        GROUP BY movie_id
        ORDER BY cnt DESC
        LIMIT {top_n};
        """,
        conn,
    )
    top_movie_ids = top_movies_df["movie_id"].tolist()
    id_list_str = ",".join(map(str, top_movie_ids))

    # 2. Fetch movie titles map
    movie_info_df = pd.read_sql_query(
        f"SELECT movie_id, clean_title, year FROM dim_movie WHERE movie_id IN ({id_list_str});",
        conn,
    )
    title_map = {
        row["movie_id"]: (
            f"{row['clean_title']} ({row['year']})" if pd.notna(row["year"]) else row["clean_title"]
        )
        for _, row in movie_info_df.iterrows()
    }

    # 3. Fetch baskets: user liked movies (rating >= 4.0)
    baskets_df = pd.read_sql_query(
        f"""
        SELECT user_id, movie_id
        FROM fact_ratings
        WHERE rating >= 4.0 AND movie_id IN ({id_list_str});
        """,
        conn,
    )
    conn.close()

    # 4. Construct boolean pivot matrix
    matrix = (
        baskets_df.groupby(["user_id", "movie_id"])["movie_id"]
        .count()
        .unstack()
        .reset_index()
        .fillna(0)
        .set_index("user_id")
    )
    matrix = matrix > 0
    return matrix, title_map


def mine_rules(
    matrix: pd.DataFrame,
    title_map: dict[int, str],
    min_support: float,
    max_len: int,
    min_confidence: float,
    min_lift: float,
) -> tuple[pd.DataFrame, list[dict]]:
    """Mines frequent itemsets and association rules using mlxtend."""
    frequent_itemsets = apriori(
        matrix,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )

    if frequent_itemsets.empty:
        return pd.DataFrame(), []

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )

    if rules.empty:
        return pd.DataFrame(), []

    # Filter by lift and sort by lift descending, confidence descending
    rules = rules[rules["lift"] > min_lift].copy()
    rules = rules.sort_values(by=["lift", "confidence"], ascending=[False, False]).reset_index(drop=True)

    # Attach human-readable titles and serializable structures
    processed_rules = []
    for _, r in rules.iterrows():
        ant_ids = list(r["antecedents"])
        con_ids = list(r["consequents"])
        ant_titles = [title_map.get(m, f"Movie {m}") for m in ant_ids]
        con_titles = [title_map.get(m, f"Movie {m}") for m in con_ids]

        processed_rules.append({
            "antecedent_ids": ant_ids,
            "consequent_ids": con_ids,
            "antecedent_titles": ant_titles,
            "consequent_titles": con_titles,
            "antecedent_str": ", ".join(ant_titles),
            "consequent_str": ", ".join(con_titles),
            "support": round(float(r["support"]), 4),
            "confidence": round(float(r["confidence"]), 4),
            "lift": round(float(r["lift"]), 4),
            "leverage": round(float(r["leverage"]), 4) if "leverage" in r else 0.0,
        })

    return rules, processed_rules


def plot_rules_scatter(rules_df: pd.DataFrame, output_path: Path) -> None:
    """Generates a Support vs. Confidence scatter plot colored by Lift."""
    plt.figure(figsize=(9, 6))
    if not rules_df.empty:
        scatter = plt.scatter(
            rules_df["support"],
            rules_df["confidence"],
            c=rules_df["lift"],
            cmap="viridis",
            alpha=0.65,
            edgecolors="none",
            s=45,
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label("Lift", fontsize=11)
    plt.title("Association Rules: Support vs. Confidence (Colored by Lift)", fontsize=13, pad=12)
    plt.xlabel("Support", fontsize=11)
    plt.ylabel("Confidence", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def train_apriori(
    db_path: Path = Path("warehouse.db"),
    models_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
) -> dict:
    """Orchestrates Apriori mining, model persistence, and returns metrics."""
    print("=" * 70)
    print("RUNNING APRIORI ASSOCIATION MINING (CO2)")
    print("=" * 70)

    matrix, title_map = build_user_movie_matrix(db_path, CONFIG["top_n_movies"])
    print(f"Constructed User x Movie Matrix: {matrix.shape[0]} users x {matrix.shape[1]} movies")

    # 1. Mine Strict Rules (for Reporting & Primary Table)
    strict_cfg = CONFIG["strict"]
    print(f"Mining strict rules (min_support={strict_cfg['min_support']}, max_len={strict_cfg['max_len']})...")
    strict_df, strict_rules = mine_rules(
        matrix, title_map,
        min_support=strict_cfg["min_support"],
        max_len=strict_cfg["max_len"],
        min_confidence=strict_cfg["min_confidence"],
        min_lift=strict_cfg["min_lift"],
    )
    print(f"Strict rules mined: {len(strict_rules)} rules (lift > {strict_cfg['min_lift']})")

    # 2. Mine Lenient Rules (for High Coverage Recommender Fallback)
    lenient_cfg = CONFIG["lenient"]
    print(f"Mining lenient rules (min_support={lenient_cfg['min_support']}, max_len={lenient_cfg['max_len']})...")
    lenient_df, lenient_rules = mine_rules(
        matrix, title_map,
        min_support=lenient_cfg["min_support"],
        max_len=lenient_cfg["max_len"],
        min_confidence=lenient_cfg["min_confidence"],
        min_lift=lenient_cfg["min_lift"],
    )
    print(f"Lenient rules mined: {len(lenient_rules)} rules")

    # 3. Plot Scatter Chart
    fig_path = reports_dir / "figures" / "top_rules.png"
    plot_rules_scatter(strict_df, fig_path)
    print(f"Saved scatter chart to {fig_path}")

    # 4. Save Models
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = models_dir / "rules.joblib"
    rules_payload = {
        "strict_rules": strict_rules,
        "lenient_rules": lenient_rules,
        "title_map": title_map,
        "config": CONFIG,
    }
    joblib.dump(rules_payload, joblib_path)
    print(f"Saved association rules model to {joblib_path}")

    # 5. Extract Top 20 Rules for viva and report
    top_20 = strict_rules[:20] if len(strict_rules) >= 20 else strict_rules
    print("\n--- Top 5 Association Rules Sample ---")
    for i, r in enumerate(top_20[:5], 1):
        print(f"Rule #{i}: {r['antecedent_str']}  -->  {r['consequent_str']}")
        print(f"         Support: {r['support']:.3f} | Confidence: {r['confidence']:.3f} | Lift: {r['lift']:.3f}")

    metrics = {
        "user_count": int(matrix.shape[0]),
        "movie_count": int(matrix.shape[1]),
        "strict_rule_count": len(strict_rules),
        "lenient_rule_count": len(lenient_rules),
        "top_rules": top_20,
    }
    return metrics


def get_rules_for_movie(
    movie_id: int,
    k: int = 5,
    rules_payload: dict | None = None,
    models_dir: Path = Path("models"),
) -> list[dict]:
    """Retrieves top k association rules where movie_id is in the antecedent.

    Falls back to lenient rules if strict rules yield no match.
    """
    if rules_payload is None:
        model_file = models_dir / "rules.joblib"
        if not model_file.exists():
            return []
        rules_payload = joblib.load(model_file)

    strict_matches = [
        r for r in rules_payload.get("strict_rules", [])
        if movie_id in r["antecedent_ids"]
    ]

    if len(strict_matches) >= k:
        return strict_matches[:k]

    # Fill deficit from lenient rules
    lenient_matches = [
        r for r in rules_payload.get("lenient_rules", [])
        if movie_id in r["antecedent_ids"]
    ]
    seen_consequents = {tuple(r["consequent_ids"]) for r in strict_matches}
    combined = list(strict_matches)
    for r in lenient_matches:
        tup = tuple(r["consequent_ids"])
        if tup not in seen_consequents:
            combined.append(r)
            seen_consequents.add(tup)
            if len(combined) >= k:
                break
    return combined[:k]


if __name__ == "__main__":
    train_apriori()

