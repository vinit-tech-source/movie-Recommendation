"""
Classification Mining Module (Text & Tabular Classifiers).

Course Outcomes:
- CO2: Apply text mining and Naive Bayes probabilistic classification
  on unstructured textual movie descriptions.
- CO4: Apply Decision Trees and Random Forests (ensemble learning)
  on structured multidimensional movie metadata to classify ratings.

This module:
1. Implements Genre Classification from Plot Descriptions:
   - Uses TMDB 5000 if available, else falls back to Netflix descriptions.
   - Cleans text: lowercase, punctuation removal, NLTK stopwords, Porter stemming.
   - Fits TF-IDF (1-2 grams) + MultinomialNB (tuned via GridSearchCV).
   - Trains LogisticRegression as a baseline comparison.
   - Exports confusion matrix plot and models/genre_classifier.joblib.
2. Implements Movie Rating-Class Prediction:
   - Filters movies with >= 5 ratings into Low (<3.0), Medium (3.0-3.99), High (>=4.0).
   - Engineered features: 19 genre flags, release year, log1p(n_ratings), n_tags.
     Zero data leakage (no rating averages used).
   - Stratified train/test split (80/20) + 5-fold cross-validation.
   - Trains DecisionTreeClassifier and RandomForestClassifier with balanced class weights.
   - Exports feature importance plot, shallow tree plot, confusion matrix, and models/rating_classifier.joblib.
"""

import json
import re
import sqlite3
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Ensure NLTK stopwords are downloaded safely
try:
    from nltk.corpus import stopwords
    _ = stopwords.words("english")
except Exception:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_and_stem_text(text: str) -> str:
    """Preprocesses text: lowercases, removes punctuation, removes stopwords, applies Porter stemming."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    filtered = [STEMMER.stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(filtered)


# ==============================================================================
# C1: Genre Classification from Plot Text (Naive Bayes - CO2)
# ==============================================================================

def load_genre_text_data(
    db_path: Path = Path("warehouse.db"),
    data_dir: Path = Path("data")
) -> tuple[pd.DataFrame, str]:
    """Loads plot text and labels from TMDB 5000 if available, else falls back to Netflix."""
    tmdb_candidates = [p for p in data_dir.rglob("tmdb_5000_movies.csv") if p.is_file()]
    if tmdb_candidates:
        tmdb_path = tmdb_candidates[0]
        print(f"Loading TMDB 5000 dataset from {tmdb_path}...")
        df_tmdb = pd.read_csv(tmdb_path)
        # Parse overview and primary genre from genres JSON
        records = []
        for _, row in df_tmdb.iterrows():
            overview = str(row.get("overview", "")).strip()
            if not overview or overview.lower() == "nan":
                continue
            genres_raw = str(row.get("genres", "[]"))
            try:
                g_list = json.loads(genres_raw)
                if g_list:
                    primary_genre = g_list[0]["name"]
                    records.append({"text": overview, "label": primary_genre})
            except Exception:
                continue
        df = pd.DataFrame(records)
        return df, "tmdb_5000_movies"

    # Fallback to Netflix
    print("TMDB 5000 movies file not present. Falling back to Netflix descriptions (data/netflix_titles.csv)...")
    conn = sqlite3.connect(db_path)
    df_netflix = pd.read_sql_query(
        """
        SELECT t.show_id, t.description as text,
               (SELECT g.name FROM title_genre tg
                JOIN dim_netflix_genre g ON tg.ngenre_id = g.ngenre_id
                WHERE tg.show_id = t.show_id LIMIT 1) as label
        FROM dim_title t
        WHERE t.description IS NOT NULL AND t.description != '';
        """,
        conn,
    )
    conn.close()
    df_netflix = df_netflix.dropna(subset=["text", "label"])
    return df_netflix[["text", "label"]], "netflix_descriptions"


def train_genre_classifier(
    db_path: Path = Path("warehouse.db"),
    data_dir: Path = Path("data"),
    models_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
) -> dict:
    """Trains Naive Bayes text classifier and comparison LogisticRegression."""
    print("\n" + "-" * 60)
    print("MODULE C1: GENRE CLASSIFICATION FROM TEXT (CO2)")
    print("-" * 60)

    raw_df, source_used = load_genre_text_data(db_path, data_dir)
    print(f"Dataset source used: {source_used} ({len(raw_df)} records)")

    # Select top 8 most frequent classes
    top_labels = raw_df["label"].value_counts().head(8).index.tolist()
    df_filtered = raw_df[raw_df["label"].isin(top_labels)].copy().reset_index(drop=True)
    print(f"Filtered to top {len(top_labels)} classes: {top_labels}")
    print(f"Total training samples: {len(df_filtered)}")

    # Clean text with Porter stemmer
    print("Preprocessing text (stemming, stopwords removal)...")
    df_filtered["clean_text"] = df_filtered["text"].apply(clean_and_stem_text)

    X = df_filtered["clean_text"]
    y = df_filtered["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: TF-IDF + MultinomialNB
    nb_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
        ("nb", MultinomialNB()),
    ])

    param_grid = {"nb__alpha": [0.01, 0.1, 0.5, 1.0, 2.0]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(nb_pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_nb = grid.best_estimator_
    best_alpha = grid.best_params_["nb__alpha"]
    print(f"Best MultinomialNB alpha: {best_alpha}")

    # Evaluate Naive Bayes on Test Set
    y_pred_nb = best_nb.predict(X_test)
    acc_nb = float(accuracy_score(y_test, y_pred_nb))
    f1_macro_nb = float(f1_score(y_test, y_pred_nb, average="macro"))
    f1_weighted_nb = float(f1_score(y_test, y_pred_nb, average="weighted"))
    prec_macro_nb = float(precision_score(y_test, y_pred_nb, average="macro", zero_division=0))
    rec_macro_nb = float(recall_score(y_test, y_pred_nb, average="macro", zero_division=0))

    report_nb = classification_report(y_test, y_pred_nb, output_dict=True, zero_division=0)
    print(f"MultinomialNB Results -> Accuracy: {acc_nb:.4f} | Macro F1: {f1_macro_nb:.4f}")

    # Comparison Model: Logistic Regression
    lr_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
        ("lr", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced", solver="lbfgs")),
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred_lr = lr_pipe.predict(X_test)
    acc_lr = float(accuracy_score(y_test, y_pred_lr))
    f1_macro_lr = float(f1_score(y_test, y_pred_lr, average="macro"))

    print(f"Comparison Logistic Regression -> Accuracy: {acc_lr:.4f} | Macro F1: {f1_macro_lr:.4f}")

    # Plot Confusion Matrix for Naive Bayes
    classes = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_pred_nb, labels=classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
    )
    plt.title(f"Naive Bayes Confusion Matrix (Genre from Text)\nAccuracy: {acc_nb:.2%}, Macro F1: {f1_macro_nb:.3f}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = reports_dir / "figures" / "cm_nb_genre.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # Save Fitted Model
    model_payload = {
        "pipeline": best_nb,
        "classes": list(best_nb.classes_),
        "source_used": source_used,
        "best_alpha": best_alpha,
    }
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = models_dir / "genre_classifier.joblib"
    joblib.dump(model_payload, joblib_path)
    print(f"Saved genre classifier pipeline to {joblib_path}")

    return {
        "source_used": source_used,
        "classes": classes,
        "sample_count": len(df_filtered),
        "naive_bayes": {
            "best_alpha": best_alpha,
            "accuracy": round(acc_nb, 4),
            "precision_macro": round(prec_macro_nb, 4),
            "recall_macro": round(rec_macro_nb, 4),
            "f1_macro": round(f1_macro_nb, 4),
            "f1_weighted": round(f1_weighted_nb, 4),
            "classification_report": report_nb,
        },
        "logistic_regression": {
            "accuracy": round(acc_lr, 4),
            "f1_macro": round(f1_macro_lr, 4),
        },
    }


# ==============================================================================
# C2: Movie Rating-Class Prediction (Decision Tree & Random Forest - CO4)
# ==============================================================================

def build_rating_feature_matrix(db_path: Path = Path("warehouse.db")) -> tuple[pd.DataFrame, list[str], float]:
    """Builds tabular feature matrix for movies with >= 5 ratings without data leakage."""
    conn = sqlite3.connect(db_path)

    # 1. Fetch movies with rating counts >= 5 and bucket into Low, Medium, High
    movies_query = """
    SELECT m.movie_id, m.year,
           AVG(r.rating) as avg_rating,
           COUNT(r.rating_id) as n_ratings,
           (SELECT COUNT(*) FROM fact_tags t WHERE t.movie_id = m.movie_id) as n_tags
    FROM dim_movie m
    JOIN fact_ratings r ON m.movie_id = r.movie_id
    GROUP BY m.movie_id
    HAVING COUNT(r.rating_id) >= 5;
    """
    df_movies = pd.read_sql_query(movies_query, conn)

    # Impute missing years with median year
    median_year = float(df_movies["year"].dropna().median())
    df_movies["year"] = df_movies["year"].fillna(median_year).astype(int)

    # Create target: Low (<3.0), Medium (3.0 <= r < 4.0), High (>=4.0)
    def assign_class(r: float) -> str:
        if r < 3.0:
            return "Low"
        elif r < 4.0:
            return "Medium"
        return "High"

    df_movies["rating_class"] = df_movies["avg_rating"].apply(assign_class)

    # Log-transform rating counts for skewed distribution
    df_movies["log_n_ratings"] = np.log1p(df_movies["n_ratings"])

    # 2. Fetch genre binary flags (19 genres)
    genres_df = pd.read_sql_query("SELECT genre_id, genre_name FROM dim_genre ORDER BY genre_id;", conn)
    all_genres = genres_df["genre_name"].tolist()

    movie_genres = pd.read_sql_query(
        """
        SELECT mg.movie_id, g.genre_name
        FROM movie_genre mg
        JOIN dim_genre g ON mg.genre_id = g.genre_id;
        """,
        conn,
    )
    conn.close()

    # Pivot genres into one-hot columns
    genre_dummies = (
        movie_genres.groupby(["movie_id", "genre_name"])["genre_name"]
        .count()
        .unstack(fill_value=0)
    )
    # Ensure all 19 genres are represented
    for g in all_genres:
        if g not in genre_dummies.columns:
            genre_dummies[g] = 0
    genre_dummies = (genre_dummies > 0).astype(int)

    # Merge features
    feature_df = df_movies.merge(genre_dummies, on="movie_id", how="left")
    for g in all_genres:
        feature_df[g] = feature_df[g].fillna(0).astype(int)

    feature_cols = ["year", "log_n_ratings", "n_tags"] + all_genres
    return feature_df, feature_cols, median_year


def train_rating_classifier(
    db_path: Path = Path("warehouse.db"),
    models_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
) -> dict:
    """Trains Decision Tree and Random Forest rating-class classifiers."""
    print("\n" + "-" * 60)
    print("MODULE C2: RATING-CLASS CLASSIFIER (CO4)")
    print("-" * 60)

    df, feature_cols, median_year = build_rating_feature_matrix(db_path)
    print(f"Constructed feature matrix: {len(df)} movies x {len(feature_cols)} features")
    print(f"Class distribution:\n{df['rating_class'].value_counts().to_string()}")

    X = df[feature_cols]
    y = df["rating_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Decision Tree Tuning
    print("Tuning Decision Tree Classifier...")
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        {"max_depth": [3, 5, 7, 10, None]},
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_
    y_pred_dt = best_dt.predict(X_test)

    acc_dt = float(accuracy_score(y_test, y_pred_dt))
    f1_macro_dt = float(f1_score(y_test, y_pred_dt, average="macro"))
    print(f"Decision Tree (max_depth={best_dt.max_depth}) -> Accuracy: {acc_dt:.4f} | Macro F1: {f1_macro_dt:.4f}")

    # 2. Random Forest Tuning
    print("Tuning Random Forest Classifier...")
    rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42),
        {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 15, None]},
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    y_pred_rf = best_rf.predict(X_test)

    acc_rf = float(accuracy_score(y_test, y_pred_rf))
    f1_macro_rf = float(f1_score(y_test, y_pred_rf, average="macro"))
    f1_weighted_rf = float(f1_score(y_test, y_pred_rf, average="weighted"))
    prec_macro_rf = float(precision_score(y_test, y_pred_rf, average="macro", zero_division=0))
    rec_macro_rf = float(recall_score(y_test, y_pred_rf, average="macro", zero_division=0))
    report_rf = classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)

    print(f"Random Forest (n_est={best_rf.n_estimators}, depth={best_rf.max_depth}) -> Accuracy: {acc_rf:.4f} | Macro F1: {f1_macro_rf:.4f}")

    # 3. Plot Confusion Matrix for Random Forest
    labels = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_test, y_pred_rf, labels=labels)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=labels, yticklabels=labels)
    plt.title(f"Random Forest Confusion Matrix (Rating Class)\nAccuracy: {acc_rf:.2%}, Macro F1: {f1_macro_rf:.3f}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    cm_path = reports_dir / "figures" / "cm_rf_rating.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved rating confusion matrix to {cm_path}")

    # 4. Feature Importance Bar Chart
    importances = best_rf.feature_importances_
    feat_series = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(12)
    plt.figure(figsize=(9, 5))
    feat_series.sort_values().plot(kind="barh", color="#4c72b0")
    plt.title("Top Feature Importances (Random Forest Rating-Class Prediction)")
    plt.xlabel("Gini Importance")
    plt.tight_layout()
    fi_path = reports_dir / "figures" / "feature_importance_rf.png"
    plt.savefig(fi_path, dpi=200)
    plt.close()
    print(f"Saved feature importance chart to {fi_path}")

    # 5. Shallow Tree (depth 3) for viva explanation
    shallow_tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
    shallow_tree.fit(X_train, y_train)
    plt.figure(figsize=(18, 9))
    plot_tree(
        shallow_tree,
        feature_names=feature_cols,
        class_names=shallow_tree.classes_,
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.title("Shallow Decision Tree (Depth 3) - Viva Architecture Demonstration")
    plt.tight_layout()
    tree_path = reports_dir / "figures" / "shallow_tree.png"
    plt.savefig(tree_path, dpi=200)
    plt.close()
    print(f"Saved shallow decision tree to {tree_path}")

    # 6. Save Best Model
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = models_dir / "rating_classifier.joblib"
    payload = {
        "model": best_rf,
        "feature_cols": feature_cols,
        "classes": list(best_rf.classes_),
        "median_year": median_year,
    }
    joblib.dump(payload, joblib_path)
    print(f"Saved rating classifier model to {joblib_path}")

    return {
        "sample_count": len(df),
        "feature_count": len(feature_cols),
        "decision_tree": {
            "max_depth": best_dt.max_depth,
            "accuracy": round(acc_dt, 4),
            "f1_macro": round(f1_macro_dt, 4),
        },
        "random_forest": {
            "n_estimators": best_rf.n_estimators,
            "max_depth": best_rf.max_depth,
            "accuracy": round(acc_rf, 4),
            "precision_macro": round(prec_macro_rf, 4),
            "recall_macro": round(rec_macro_rf, 4),
            "f1_macro": round(f1_macro_rf, 4),
            "f1_weighted": round(f1_weighted_rf, 4),
            "classification_report": report_rf,
            "top_features": feat_series.to_dict(),
        },
    }


def train_classifiers() -> dict:
    """Executes both C1 and C2 training workflows."""
    c1_metrics = train_genre_classifier()
    c2_metrics = train_rating_classifier()
    return {"genre_classification": c1_metrics, "rating_classification": c2_metrics}


def predict_genre_from_text(
    text: str,
    k: int = 3,
    models_dir: Path = Path("models")
) -> list[dict]:
    """Predicts top-k genres with probabilities given plot text."""
    model_path = models_dir / "genre_classifier.joblib"
    if not model_path.exists():
        return []
    payload = joblib.load(model_path)
    pipe = payload["pipeline"]
    cleaned = clean_and_stem_text(text)
    if not cleaned:
        cleaned = text.lower()

    probs = pipe.predict_proba([cleaned])[0]
    classes = pipe.classes_
    top_indices = np.argsort(probs)[::-1][:k]

    return [
        {"genre": classes[i], "probability": round(float(probs[i]), 4)}
        for i in top_indices
    ]


def predict_movie_rating_class(
    genres: list[str],
    year: int,
    n_ratings: int,
    n_tags: int,
    models_dir: Path = Path("models")
) -> dict:
    """Predicts rating class (Low, Medium, High) and class probabilities."""
    model_path = models_dir / "rating_classifier.joblib"
    if not model_path.exists():
        return {"error": "Model not found"}
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_cols = payload["feature_cols"]

    # Construct input row
    row = {
        "year": year,
        "log_n_ratings": np.log1p(n_ratings),
        "n_tags": n_tags,
    }
    for col in feature_cols:
        if col not in row:
            row[col] = 1 if col in genres else 0

    df_input = pd.DataFrame([row])[feature_cols]
    pred_class = model.predict(df_input)[0]
    probs = model.predict_proba(df_input)[0]

    prob_dict = {cls: round(float(p), 4) for cls, p in zip(model.classes_, probs)}
    return {
        "predicted_class": pred_class,
        "probabilities": prob_dict,
    }


if __name__ == "__main__":
    train_classifiers()
