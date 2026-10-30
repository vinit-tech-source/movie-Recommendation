"""
Master Orchestration Script for Data Mining & Analysis Mini Project.

Course Outcomes: CO1 to CO5.

Executes all analytical and mining pipelines end-to-end:
1. Module B (CO2): Apriori Association Rules Mining
2. Module C (CO2, CO4): Genre Text Classification & Rating-Class Prediction
3. Module D (CO3, CO4): Linear, Ridge, Lasso & Decision Tree Regression
4. Module E (CO5): ARIMA Time Series Modeling & 12-Month Forecasting
5. Module F (CO5): Content, Item-CF, Apriori & Hybrid Recommender Engine

Generates and persists:
- models/*.joblib
- reports/figures/*.png
- reports/metrics.json (all empirical metrics consumed by the Flask web application)
"""

import json
import time
from pathlib import Path

from mining.apriori_rules import train_apriori
from mining.arima_model import train_arima
from mining.classifier import train_classifiers
from mining.recommender import train_recommender
from mining.regression import train_regressors


def run_pipeline() -> dict:
    """Executes all mining modules, persists models, figures, and metrics.json."""
    start_time = time.time()
    print("=" * 80)
    print("STARTING END-TO-END DATA MINING & MODEL TRAINING PIPELINE")
    print("=" * 80)

    db_path = Path("warehouse.db")
    if not db_path.exists():
        raise FileNotFoundError("warehouse.db not found! Please run python etl.py first.")

    models_dir = Path("models")
    reports_dir = Path("reports")
    figures_dir = reports_dir / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "pipeline_version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 1. Module B: Apriori Association Rules
    print("\n>>> [1/5] Executing Module B: Apriori Association Rules (CO2)...")
    apriori_metrics = train_apriori(db_path, models_dir, reports_dir)
    metrics["apriori"] = apriori_metrics

    # 2. Module C: Classifiers
    print("\n>>> [2/5] Executing Module C: Classification (CO2, CO4)...")
    clf_metrics = train_classifiers()
    metrics["classification"] = clf_metrics

    # 3. Module D: Regression
    print("\n>>> [3/5] Executing Module D: Regression Models (CO3, CO4)...")
    reg_metrics = train_regressors(db_path, models_dir, reports_dir)
    metrics["regression"] = reg_metrics

    # 4. Module E: Time Series / ARIMA
    print("\n>>> [4/5] Executing Module E: Time Series / ARIMA (CO5)...")
    arima_metrics = train_arima(db_path, models_dir, reports_dir)
    metrics["time_series"] = arima_metrics

    # 5. Module F: Recommender Engine
    print("\n>>> [5/5] Executing Module F: Recommendation Engine & Evaluation (CO5)...")
    rec_metrics = train_recommender(db_path, models_dir, reports_dir)
    metrics["recommender"] = rec_metrics

    # Save to reports/metrics.json
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved consolidated metrics to {metrics_path}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE IN {elapsed:.2f} SECONDS")
    print("=" * 80)

    # Print Final Summary Table
    print("\n" + "#" * 80)
    print("FINAL MODEL PERFORMANCE SUMMARY TABLE (Real Metrics from Run)")
    print("#" * 80)
    print(f"1. APRIORI (CO2):")
    print(f"   Strict Rules (supp>=0.10, conf>=0.60, lift>1.2): {apriori_metrics['strict_rule_count']}")
    print(f"   Lenient Rules (supp>=0.05, conf>=0.50):           {apriori_metrics['lenient_rule_count']}")

    nb = clf_metrics["genre_classification"]["naive_bayes"]
    rf = clf_metrics["rating_classification"]["random_forest"]
    print(f"\n2. CLASSIFICATION (CO2 & CO4):")
    print(f"   Genre Naive Bayes (CO2)     -> Acc: {nb['accuracy']:.4f} | Macro F1: {nb['f1_macro']:.4f} (Source: {clf_metrics['genre_classification']['source_used']})")
    print(f"   Rating Random Forest (CO4)  -> Acc: {rf['accuracy']:.4f} | Macro F1: {rf['f1_macro']:.4f} | Top: {list(rf['top_features'].keys())[:3]}")

    best_reg = reg_metrics["best_model"]
    reg_val = reg_metrics["models"][best_reg]
    print(f"\n3. REGRESSION (CO3 & CO4):")
    print(f"   Best Regressor ({best_reg})  -> Test RMSE: {reg_val['test_rmse']:.4f} | Test MAE: {reg_val['test_mae']:.4f} | Test R²: {reg_val['test_r2']:.4f}")

    nf_ts = arima_metrics["netflix"]
    print(f"\n4. TIME SERIES / ARIMA (CO5):")
    print(f"   Netflix Titles ARIMA{tuple(nf_ts['order'])} -> Test RMSE: {nf_ts['holdout_evaluation']['arima_rmse']:.2f} (vs Naive {nf_ts['holdout_evaluation']['naive_rmse']:.2f})")

    rec_eval = rec_metrics["evaluation"]
    print(f"\n5. RECOMMENDER ENGINE (CO5):")
    print(f"   Offline Leave-One-Out Hit-Rate@10:")
    print(f"   - Content-Based:   {rec_eval['content']['hit_rate_at_10']:.4f}")
    print(f"   - Item-CF:         {rec_eval['cf']['hit_rate_at_10']:.4f}")
    print(f"   - Apriori Rules:   {rec_eval['rules']['hit_rate_at_10']:.4f}")
    print(f"   - Hybrid Engine:   {rec_eval['hybrid']['hit_rate_at_10']:.4f}")
    print("#" * 80 + "\n")

    return metrics


if __name__ == "__main__":
    run_pipeline()

