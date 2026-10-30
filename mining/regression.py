"""
Movie Rating Regression Module.

Course Outcomes:
- CO3: Apply linear regression and regularized regression (Ridge, Lasso)
  on multidimensional numerical and categorical features.
- CO4: Apply non-linear decision tree regression to capture feature interactions
  and compare performance against linear baselines.

This module:
1. Targets continuous movie average rating for movies with >= 5 ratings (3,650 samples).
2. Features: 19 genre indicators, release year, log1p(n_ratings), and n_tags.
3. Fits and evaluates:
   - LinearRegression (OLS)
   - Ridge Regression (L2 regularization, tuned via RidgeCV)
   - Lasso Regression (L1 regularization, feature selection via LassoCV)
   - DecisionTreeRegressor (non-linear comparison)
4. Evaluates MSE, RMSE, MAE, R² for both Train and Test sets.
5. Identifies Lasso's active non-zero feature coefficients.
6. Exports Actual vs. Predicted and Residual plots to reports/figures/.
7. Saves the best regressor pipeline to models/rating_regressor.joblib.
"""

import sqlite3
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def load_regression_data(db_path: Path = Path("warehouse.db")) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Loads feature matrix and continuous target (avg_rating) without leakage."""
    conn = sqlite3.connect(db_path)

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

    # Impute missing release year with median
    median_year = float(df_movies["year"].dropna().median())
    df_movies["year"] = df_movies["year"].fillna(median_year).astype(int)
    df_movies["log_n_ratings"] = np.log1p(df_movies["n_ratings"])

    # Fetch 19 genres
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

    # One-hot genre matrix
    genre_dummies = (
        movie_genres.groupby(["movie_id", "genre_name"])["genre_name"]
        .count()
        .unstack(fill_value=0)
    )
    for g in all_genres:
        if g not in genre_dummies.columns:
            genre_dummies[g] = 0
    genre_dummies = (genre_dummies > 0).astype(int)

    feature_df = df_movies.merge(genre_dummies, on="movie_id", how="left")
    for g in all_genres:
        feature_df[g] = feature_df[g].fillna(0).astype(int)

    numeric_cols = ["year", "log_n_ratings", "n_tags"]
    return feature_df, numeric_cols, all_genres


def train_regressors(
    db_path: Path = Path("warehouse.db"),
    models_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
) -> dict:
    """Trains and compares Linear, Ridge, Lasso, and Decision Tree regressors."""
    print("=" * 70)
    print("RUNNING REGRESSION MODULE (CO3: Linear/Ridge/Lasso & CO4: Decision Tree)")
    print("=" * 70)

    df, numeric_cols, genre_cols = load_regression_data(db_path)
    feature_cols = numeric_cols + genre_cols

    X = df[feature_cols]
    y = df["avg_rating"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Dataset split: {len(X_train)} train samples, {len(X_test)} test samples")

    # Column transformer: Standardize numeric columns; pass binary genre flags through
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", "passthrough", genre_cols),
        ]
    )

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5),
        "Lasso": LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, random_state=42, max_iter=5000),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42),
    }

    comparison_results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model),
        ])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Predictions
        y_train_pred = pipe.predict(X_train)
        y_test_pred = pipe.predict(X_test)

        train_mse = float(mean_squared_error(y_train, y_train_pred))
        test_mse = float(mean_squared_error(y_test, y_test_pred))
        train_rmse = float(np.sqrt(train_mse))
        test_rmse = float(np.sqrt(test_mse))
        train_mae = float(mean_absolute_error(y_train, y_train_pred))
        test_mae = float(mean_absolute_error(y_test, y_test_pred))
        train_r2 = float(r2_score(y_train, y_train_pred))
        test_r2 = float(r2_score(y_test, y_test_pred))

        comparison_results[name] = {
            "train_mse": round(train_mse, 4),
            "test_mse": round(test_mse, 4),
            "train_rmse": round(train_rmse, 4),
            "test_rmse": round(test_rmse, 4),
            "train_mae": round(train_mae, 4),
            "test_mae": round(test_mae, 4),
            "train_r2": round(train_r2, 4),
            "test_r2": round(test_r2, 4),
        }

    # Summary table
    comp_df = pd.DataFrame(comparison_results).T
    print("\n--- Regression Model Comparison Table ---")
    print(comp_df[["test_rmse", "test_mae", "test_r2", "train_r2"]].to_string())

    # Lasso Non-Zero Feature Coefficients Analysis
    lasso_model = fitted_pipelines["Lasso"].named_steps["regressor"]
    lasso_coefs = lasso_model.coef_
    lasso_features = {}
    for col, coef in zip(feature_cols, lasso_coefs):
        if abs(coef) > 1e-5:
            lasso_features[col] = round(float(coef), 4)

    print(f"\nLasso Optimal Alpha: {lasso_model.alpha_:.5f}")
    print("Lasso Non-Zero Coefficients (Top Feature Impact):")
    for feat, val in sorted(lasso_features.items(), key=lambda x: abs(x[1]), reverse=True)[:8]:
        print(f"  {feat:20s}: {val:+.4f}")

    # Best Model Selection based on Test RMSE
    best_name = min(comparison_results.keys(), key=lambda k: comparison_results[k]["test_rmse"])
    best_pipe = fitted_pipelines[best_name]
    print(f"\nSelected Best Regressor: {best_name} (Test RMSE: {comparison_results[best_name]['test_rmse']:.4f})")

    # Generate Plots: Actual vs. Predicted & Residuals for the best model
    reports_dir.mkdir(parents=True, exist_ok=True)
    y_test_pred_best = best_pipe.predict(X_test)
    residuals = y_test - y_test_pred_best

    # 1. Actual vs Predicted
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_test_pred_best, alpha=0.4, color="#1f77b4", edgecolors="none")
    plt.plot([0.5, 5.0], [0.5, 5.0], color="red", linestyle="--", linewidth=1.5, label="Perfect Fit (y=x)")
    plt.xlabel("Actual Average Rating")
    plt.ylabel("Predicted Average Rating")
    plt.title(f"{best_name} Regression: Actual vs. Predicted\n(Test R² = {comparison_results[best_name]['test_r2']:.3f}, RMSE = {comparison_results[best_name]['test_rmse']:.3f})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path1 = reports_dir / "figures" / "regression_actual_vs_pred.png"
    plt.savefig(plot_path1, dpi=200)
    plt.close()
    print(f"Saved actual vs. predicted plot to {plot_path1}")

    # 2. Residual Distribution Plot
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=35, color="#2ca02c", edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"{best_name} Residual Distribution (Mean: {residuals.mean():.3f}, Std: {residuals.std():.3f})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plot_path2 = reports_dir / "figures" / "regression_residuals.png"
    plt.savefig(plot_path2, dpi=200)
    plt.close()
    print(f"Saved residual plot to {plot_path2}")

    # Save Pipeline
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = models_dir / "rating_regressor.joblib"
    joblib_payload = {
        "pipeline": best_pipe,
        "best_model_name": best_name,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "genre_cols": genre_cols,
        "median_year": float(df["year"].median()),
    }
    joblib.dump(joblib_payload, joblib_path)
    print(f"Saved regression model to {joblib_path}")

    metrics = {
        "models": comparison_results,
        "best_model": best_name,
        "lasso_alpha": float(lasso_model.alpha_),
        "lasso_non_zero_coefficients": lasso_features,
    }
    return metrics


def predict_movie_rating(
    genres: list[str],
    year: int,
    n_ratings: int,
    n_tags: int,
    models_dir: Path = Path("models"),
) -> float:
    """Predicts expected average movie rating given features."""
    model_path = models_dir / "rating_regressor.joblib"
    if not model_path.exists():
        return 3.5  # reasonable fallback
    payload = joblib.load(model_path)
    pipe = payload["pipeline"]
    feature_cols = payload["feature_cols"]

    row = {
        "year": year,
        "log_n_ratings": np.log1p(n_ratings),
        "n_tags": n_tags,
    }
    for col in feature_cols:
        if col not in row:
            row[col] = 1 if col in genres else 0

    df_in = pd.DataFrame([row])[feature_cols]
    pred = pipe.predict(df_in)[0]
    return round(float(np.clip(pred, 0.5, 5.0)), 2)


if __name__ == "__main__":
    train_regressors()

