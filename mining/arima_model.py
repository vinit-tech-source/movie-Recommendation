"""
Time Series Analysis and ARIMA Forecasting Module.

Course Outcome: CO5 - Apply advanced data mining, temporal pattern discovery,
and time series forecasting techniques (ARIMA, Differencing, Stationarity Analysis).

This module:
1. Extracts temporal series from the Fact Constellation warehouse:
   - Primary: Netflix monthly titles added (2016-01 to 2021-08, 68 months).
   - Secondary: MovieLens monthly ratings (2000-01 to 2018-08, log1p transformed).
2. Performs Augmented Dickey-Fuller (ADF) tests before and after differencing (d=1).
3. Produces ACF, PACF, and residual diagnostics plots.
4. Executes grid search over (p, d=1, q) orders minimizing AIC.
5. Evaluates out-of-sample test RMSE over a 12-month holdout against:
   - Naive Baseline (persistence: repeat last known value)
   - Seasonal Naive Baseline (repeat value from 12 months prior)
6. Refits on full historical series and forecasts 12 months ahead with 95% confidence intervals.
7. Saves models and metadata to models/arima_models.joblib and figures to reports/figures/.
8. Supports interactive filtered forecasts with minimum sample threshold safeguards (>= 36 points).
"""

import sqlite3
import warnings
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Suppress Convergence / Invertibility convergence warnings during parameter grid search
warnings.filterwarnings("ignore")


def get_time_series(
    source: str = "netflix",
    genre: str | None = None,
    title_type: str | None = None,
    db_path: Path = Path("warehouse.db"),
) -> pd.Series:
    """Extracts monthly aggregated time series from the warehouse."""
    conn = sqlite3.connect(db_path)

    if source.lower() == "netflix":
        where_clauses = ["d.full_date >= '2016-01-01'", "d.full_date <= '2021-08-31'"]
        joins = [
            "JOIN dim_date d ON ft.date_id = d.date_id",
            "JOIN dim_title t ON ft.show_id = t.show_id",
        ]

        if title_type:
            where_clauses.append(f"t.type = '{title_type}'")
        if genre:
            joins.append("JOIN title_genre tg ON t.show_id = tg.show_id")
            joins.append("JOIN dim_netflix_genre g ON tg.ngenre_id = g.ngenre_id")
            where_clauses.append(f"g.name = '{genre}'")

        query = f"""
        SELECT d.full_date, COUNT(DISTINCT ft.show_id) as count
        FROM fact_titles ft
        {' '.join(joins)}
        WHERE {' AND '.join(where_clauses)}
        GROUP BY d.full_date;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return pd.Series(dtype=float)

        df["dt"] = pd.to_datetime(df["full_date"])
        # Resample monthly start
        ts = df.set_index("dt").resample("MS")["count"].sum()
        # Fill any missing months with 0
        idx = pd.date_range("2016-01-01", "2021-08-01", freq="MS")
        ts = ts.reindex(idx, fill_value=0)
        ts.name = "netflix_titles"
        return ts

    elif source.lower() == "movielens":
        where_clauses = ["d.full_date >= '2000-01-01'", "d.full_date <= '2018-08-31'"]
        joins = ["JOIN dim_date d ON f.date_id = d.date_id"]

        if genre:
            joins.append("JOIN movie_genre mg ON f.movie_id = mg.movie_id")
            joins.append("JOIN dim_genre g ON mg.genre_id = g.genre_id")
            where_clauses.append(f"g.genre_name = '{genre}'")

        query = f"""
        SELECT d.full_date, COUNT(f.rating_id) as count
        FROM fact_ratings f
        {' '.join(joins)}
        WHERE {' AND '.join(where_clauses)}
        GROUP BY d.full_date;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return pd.Series(dtype=float)

        df["dt"] = pd.to_datetime(df["full_date"])
        ts = df.set_index("dt").resample("MS")["count"].sum()
        idx = pd.date_range("2000-01-01", "2018-08-01", freq="MS")
        ts = ts.reindex(idx, fill_value=0)
        ts.name = "movielens_ratings"
        return ts

    else:
        conn.close()
        raise ValueError(f"Unknown source: {source}")


def check_stationarity(series: pd.Series) -> dict:
    """Runs Augmented Dickey-Fuller test on raw and 1st differenced series."""
    # Raw series ADF
    res_raw = adfuller(series.dropna())
    # 1st Difference ADF
    diff_series = series.diff().dropna()
    res_diff = adfuller(diff_series)

    return {
        "raw": {
            "adf_statistic": round(float(res_raw[0]), 4),
            "p_value": float(f"{res_raw[1]:.5e}"),
            "critical_values": {k: round(v, 3) for k, v in res_raw[4].items()},
            "is_stationary": bool(res_raw[1] < 0.05),
        },
        "differenced_d1": {
            "adf_statistic": round(float(res_diff[0]), 4),
            "p_value": float(f"{res_diff[1]:.5e}"),
            "critical_values": {k: round(v, 3) for k, v in res_diff[4].items()},
            "is_stationary": bool(res_diff[1] < 0.05),
        },
    }


def grid_search_arima(
    train_series: pd.Series,
    p_values: range = range(0, 4),
    d: int = 1,
    q_values: range = range(0, 4),
) -> tuple[tuple[int, int, int], float]:
    """Finds optimal ARIMA(p,d,q) order by minimizing AIC."""
    best_aic = float("inf")
    best_order = (1, d, 1)

    for p in p_values:
        for q in q_values:
            try:
                model = ARIMA(train_series, order=(p, d, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, d, q)
            except Exception:
                continue

    return best_order, round(float(best_aic), 2)


def evaluate_holdout(
    series: pd.Series,
    order: tuple[int, int, int],
    holdout_months: int = 12,
    log_transformed: bool = False,
) -> dict:
    """Fits ARIMA on train split, compares 12-month test RMSE against naive baselines."""
    train = series.iloc[:-holdout_months]
    test = series.iloc[-holdout_months:]

    # Fit ARIMA on training split
    model = ARIMA(train, order=order)
    res = model.fit()
    forecast = res.forecast(steps=holdout_months)

    if log_transformed:
        # Invert log1p for true count comparison
        test_actual = np.expm1(test)
        forecast_actual = np.expm1(forecast)
        train_actual = np.expm1(train)
    else:
        test_actual = test
        forecast_actual = forecast
        train_actual = train

    # Model RMSE
    model_rmse = float(np.sqrt(np.mean((test_actual.values - forecast_actual.values) ** 2)))

    # Baseline 1: Naive (repeat last training value)
    last_val = train_actual.iloc[-1]
    naive_pred = np.full(shape=len(test_actual), fill_value=last_val)
    naive_rmse = float(np.sqrt(np.mean((test_actual.values - naive_pred) ** 2)))

    # Baseline 2: Seasonal Naive (repeat value from 12 months ago)
    if len(train_actual) >= 12:
        snaive_pred = train_actual.iloc[-12:].values
        snaive_rmse = float(np.sqrt(np.mean((test_actual.values - snaive_pred) ** 2)))
    else:
        snaive_rmse = naive_rmse

    return {
        "holdout_months": holdout_months,
        "arima_rmse": round(model_rmse, 3),
        "naive_rmse": round(naive_rmse, 3),
        "seasonal_naive_rmse": round(snaive_rmse, 3),
        "improvement_over_naive_pct": round(float((naive_rmse - model_rmse) / naive_rmse * 100), 2),
    }


def fit_forecast_full(
    series: pd.Series,
    order: tuple[int, int, int],
    forecast_steps: int = 12,
    log_transformed: bool = False,
) -> tuple[ARIMA, pd.DataFrame, dict]:
    """Refits ARIMA on entire series and forecasts next 12 months with 95% CI."""
    model = ARIMA(series, order=order)
    results = model.fit()

    # Forecast with confidence intervals
    fc_res = results.get_forecast(steps=forecast_steps)
    fc_mean = fc_res.predicted_mean
    fc_ci = fc_res.conf_int(alpha=0.05)

    # Next 12 month indices
    last_date = series.index[-1]
    fc_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=forecast_steps, freq="MS")

    if log_transformed:
        pred_val = np.expm1(fc_mean.values)
        lower_ci = np.clip(np.expm1(fc_ci.iloc[:, 0].values), 0, None)
        upper_ci = np.expm1(fc_ci.iloc[:, 1].values)
    else:
        pred_val = np.clip(fc_mean.values, 0, None)
        lower_ci = np.clip(fc_ci.iloc[:, 0].values, 0, None)
        upper_ci = np.clip(fc_ci.iloc[:, 1].values, 0, None)

    df_fc = pd.DataFrame({
        "forecast": np.round(pred_val, 1),
        "lower_ci": np.round(lower_ci, 1),
        "upper_ci": np.round(upper_ci, 1),
    }, index=fc_dates)

    # Residual diagnostics
    resid = results.resid
    lb_test = acorr_ljungbox(resid.dropna(), lags=[10], return_df=True)
    lb_pvalue = float(lb_test["lb_pvalue"].iloc[0])

    diagnostics = {
        "aic": round(float(results.aic), 2),
        "bic": round(float(results.bic), 2),
        "ljung_box_p_value": float(f"{lb_pvalue:.4f}"),
        "residuals_white_noise": bool(lb_pvalue > 0.05),
    }

    return results, df_fc, diagnostics


def generate_time_series_plots(
    ts_netflix: pd.Series,
    netflix_order: tuple[int, int, int],
    df_fc_netflix: pd.DataFrame,
    ts_ml: pd.Series,
    ml_order: tuple[int, int, int],
    df_fc_ml: pd.DataFrame,
    reports_dir: Path,
) -> None:
    """Generates ACF/PACF, forecast, and residual diagnostic charts."""
    fig_dir = reports_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Netflix ACF & PACF plot (d=1 difference)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    diff_nf = ts_netflix.diff().dropna()
    plot_acf(diff_nf, ax=axes[0], lags=20, title="Netflix Differenced (d=1) ACF")
    plot_pacf(diff_nf, ax=axes[1], lags=20, title="Netflix Differenced (d=1) PACF")
    plt.tight_layout()
    plt.savefig(fig_dir / "arima_acf_pacf_netflix.png", dpi=200)
    plt.close()

    # 2. Netflix 12-Month Forecast Plot with 95% Confidence Band
    plt.figure(figsize=(10, 5))
    plt.plot(ts_netflix.index, ts_netflix.values, label="Historical Titles Added", color="#1f77b4", linewidth=2)
    plt.plot(df_fc_netflix.index, df_fc_netflix["forecast"].values, label=f"ARIMA{netflix_order} Forecast", color="#d62728", linestyle="--", linewidth=2)
    plt.fill_between(
        df_fc_netflix.index,
        df_fc_netflix["lower_ci"].values,
        df_fc_netflix["upper_ci"].values,
        color="#d62728", alpha=0.18, label="95% Confidence Interval"
    )
    plt.title(f"Netflix Monthly Titles Added: Historical Trend & 12-Month Forecast (ARIMA{netflix_order})")
    plt.xlabel("Date")
    plt.ylabel("Titles Added / Month")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "arima_forecast_netflix.png", dpi=200)
    plt.close()

    # 3. Netflix Residual Diagnostics Plot
    model = ARIMA(ts_netflix, order=netflix_order)
    resid = model.fit().resid
    plt.figure(figsize=(9, 4))
    plt.plot(resid.index, resid.values, color="#2ca02c", linewidth=1.2)
    plt.axhline(0, color="red", linestyle="--")
    plt.title(f"Netflix ARIMA{netflix_order} Residuals Over Time")
    plt.xlabel("Date")
    plt.ylabel("Residual Value")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "arima_residuals_netflix.png", dpi=200)
    plt.close()

    # 4. MovieLens Forecast Plot
    plt.figure(figsize=(10, 5))
    plt.plot(ts_ml.index, ts_ml.values, label="Historical Ratings Count", color="#ff7f0e", linewidth=1.5)
    plt.plot(df_fc_ml.index, df_fc_ml["forecast"].values, label=f"ARIMA{ml_order} Forecast", color="#9467bd", linestyle="--", linewidth=2)
    plt.fill_between(
        df_fc_ml.index,
        df_fc_ml["lower_ci"].values,
        df_fc_ml["upper_ci"].values,
        color="#9467bd", alpha=0.18, label="95% Confidence Interval"
    )
    plt.title(f"MovieLens Monthly Ratings Activity: Historical Trend & 12-Month Forecast (ARIMA{ml_order})")
    plt.xlabel("Date")
    plt.ylabel("Ratings / Month")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "arima_forecast_movielens.png", dpi=200)
    plt.close()


def train_arima(
    db_path: Path = Path("warehouse.db"),
    models_dir: Path = Path("models"),
    reports_dir: Path = Path("reports"),
) -> dict:
    """Executes time series and ARIMA pipeline for Netflix and MovieLens."""
    print("=" * 70)
    print("RUNNING TIME SERIES / ARIMA FORECASTING (CO5)")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Primary Series: Netflix Monthly Titles (2016-01 to 2021-08)
    # ---------------------------------------------------------
    print("\n--- 1. Analyzing Netflix Series (2016-01 to 2021-08) ---")
    ts_netflix = get_time_series(source="netflix", db_path=db_path)
    print(f"Loaded Netflix Series: {len(ts_netflix)} months (range: {ts_netflix.index[0].strftime('%Y-%m')} to {ts_netflix.index[-1].strftime('%Y-%m')})")

    # ADF Stationarity Test
    nf_adf = check_stationarity(ts_netflix)
    print(f"Raw Series ADF:  stat = {nf_adf['raw']['adf_statistic']:.4f}, p = {nf_adf['raw']['p_value']} (Stationary: {nf_adf['raw']['is_stationary']})")
    print(f"Diff (d=1) ADF:  stat = {nf_adf['differenced_d1']['adf_statistic']:.4f}, p = {nf_adf['differenced_d1']['p_value']} (Stationary: {nf_adf['differenced_d1']['is_stationary']})")

    # Order search with d=1
    print("Executing Grid Search over (p in 0..3, d=1, q in 0..3)...")
    nf_order, nf_aic = grid_search_arima(ts_netflix, p_values=range(0, 4), d=1, q_values=range(0, 4))
    print(f"Optimal Netflix Order: ARIMA{nf_order} with AIC: {nf_aic}")

    # Holdout Evaluation (last 12 months)
    nf_eval = evaluate_holdout(ts_netflix, order=nf_order, holdout_months=12, log_transformed=False)
    print("Netflix 12-Month Holdout Comparison:")
    print(f"  ARIMA{nf_order} RMSE:        {nf_eval['arima_rmse']}")
    print(f"  Naive Baseline RMSE:        {nf_eval['naive_rmse']}")
    print(f"  Seasonal Naive RMSE:        {nf_eval['seasonal_naive_rmse']}")
    print(f"  Improvement over Naive:     {nf_eval['improvement_over_naive_pct']}%")

    # Refit and forecast
    _, df_fc_netflix, nf_diag = fit_forecast_full(ts_netflix, order=nf_order, forecast_steps=12, log_transformed=False)
    print(f"Netflix 12-Month Forecast Mean: {df_fc_netflix['forecast'].mean():.1f} titles/mo")

    # ---------------------------------------------------------
    # 2. Secondary Series: MovieLens Ratings (2000-01 to 2018-08)
    # ---------------------------------------------------------
    print("\n--- 2. Analyzing MovieLens Series (2000-01 to 2018-08) ---")
    ts_ml_raw = get_time_series(source="movielens", db_path=db_path)
    print(f"Loaded MovieLens Series: {len(ts_ml_raw)} months")
    # Log-transform due to user burstiness
    ts_ml_log = np.log1p(ts_ml_raw)

    ml_adf = check_stationarity(ts_ml_log)
    print(f"MovieLens (log) Diff ADF: stat = {ml_adf['differenced_d1']['adf_statistic']:.4f}, p = {ml_adf['differenced_d1']['p_value']}")

    ml_order, ml_aic = grid_search_arima(ts_ml_log, p_values=range(0, 4), d=1, q_values=range(0, 4))
    print(f"Optimal MovieLens Order: ARIMA{ml_order} with AIC: {ml_aic}")

    ml_eval = evaluate_holdout(ts_ml_log, order=ml_order, holdout_months=12, log_transformed=True)
    print(f"MovieLens 12-Month Holdout RMSE -> ARIMA: {ml_eval['arima_rmse']} vs Naive: {ml_eval['naive_rmse']}")

    _, df_fc_ml, ml_diag = fit_forecast_full(ts_ml_log, order=ml_order, forecast_steps=12, log_transformed=True)

    # Generate charts
    generate_time_series_plots(
        ts_netflix, nf_order, df_fc_netflix,
        ts_ml_raw, ml_order, df_fc_ml,
        reports_dir
    )
    print("Saved all ARIMA time series figures to reports/figures/")

    # Save to models/
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = models_dir / "arima_models.joblib"
    payload = {
        "netflix": {
            "order": nf_order,
            "aic": nf_aic,
            "series": ts_netflix,
            "forecast_df": df_fc_netflix,
            "metrics": nf_eval,
            "diagnostics": nf_diag,
            "stationarity": nf_adf,
        },
        "movielens": {
            "order": ml_order,
            "aic": ml_aic,
            "series": ts_ml_raw,
            "forecast_df": df_fc_ml,
            "metrics": ml_eval,
            "diagnostics": ml_diag,
            "stationarity": ml_adf,
        },
    }
    joblib.dump(payload, joblib_path)
    print(f"Saved ARIMA models payload to {joblib_path}")

    # Helper JSON-friendly dictionary for metrics.json
    metrics_summary = {
        "netflix": {
            "order": list(nf_order),
            "aic": nf_aic,
            "data_points": len(ts_netflix),
            "date_range": [ts_netflix.index[0].strftime("%Y-%m"), ts_netflix.index[-1].strftime("%Y-%m")],
            "stationarity_adf": nf_adf,
            "holdout_evaluation": nf_eval,
            "diagnostics": nf_diag,
            "forecast_next_12_months": [
                {"date": d.strftime("%Y-%m"), "forecast": row["forecast"], "lower_ci": row["lower_ci"], "upper_ci": row["upper_ci"]}
                for d, row in df_fc_netflix.iterrows()
            ],
        },
        "movielens": {
            "order": list(ml_order),
            "aic": ml_aic,
            "data_points": len(ts_ml_raw),
            "date_range": [ts_ml_raw.index[0].strftime("%Y-%m"), ts_ml_raw.index[-1].strftime("%Y-%m")],
            "holdout_evaluation": ml_eval,
            "diagnostics": ml_diag,
        },
    }
    return metrics_summary


def get_trend_api_data(
    source: str = "netflix",
    genre: str | None = None,
    title_type: str | None = None,
    db_path: Path = Path("warehouse.db"),
    models_dir: Path = Path("models"),
) -> dict:
    """Provides time series history and forecast for the Flask /api/trends endpoint."""
    # Check if this is the default unfiltered call, return cached model
    if not genre and not title_type:
        payload_file = models_dir / "arima_models.joblib"
        if payload_file.exists():
            data = joblib.load(payload_file)
            src_key = "netflix" if source.lower() == "netflix" else "movielens"
            entry = data[src_key]
            series = entry["series"]
            fc_df = entry["forecast_df"]
            return {
                "source": source,
                "history": [{"date": d.strftime("%Y-%m"), "value": float(v)} for d, v in series.items()],
                "forecast": [
                    {"date": d.strftime("%Y-%m"), "forecast": float(row["forecast"]), "lower_ci": float(row["lower_ci"]), "upper_ci": float(row["upper_ci"])}
                    for d, row in fc_df.iterrows()
                ],
                "model_order": list(entry["order"]),
                "test_rmse": entry["metrics"]["arima_rmse"],
                "naive_rmse": entry["metrics"]["naive_rmse"],
                "adf_pvalue": entry["stationarity"]["differenced_d1"]["p_value"],
                "message": "Full historical series with 12-month ARIMA forecast.",
            }

    # Filtered dynamic calculation
    ts = get_time_series(source=source, genre=genre, title_type=title_type, db_path=db_path)
    if ts.empty or len(ts) < 36 or ts.sum() < 50:
        return {
            "source": source,
            "history": [{"date": d.strftime("%Y-%m"), "value": float(v)} for d, v in ts.items()],
            "forecast": None,
            "model_order": None,
            "message": "Filtered series contains fewer than 36 monthly points or insufficient volume; forecast omitted to prevent unreliable estimates.",
        }

    # Fit quick ARIMA(1,1,1) on filtered series
    try:
        model = ARIMA(ts, order=(1, 1, 1))
        res = model.fit()
        fc_res = res.get_forecast(steps=12)
        fc_mean = fc_res.predicted_mean
        fc_ci = fc_res.conf_int()
        last_d = ts.index[-1]
        fc_dates = pd.date_range(last_d + pd.DateOffset(months=1), periods=12, freq="MS")

        forecast_list = []
        for i, d in enumerate(fc_dates):
            forecast_list.append({
                "date": d.strftime("%Y-%m"),
                "forecast": round(float(np.clip(fc_mean.iloc[i], 0, None)), 1),
                "lower_ci": round(float(np.clip(fc_ci.iloc[i, 0], 0, None)), 1),
                "upper_ci": round(float(np.clip(fc_ci.iloc[i, 1], 0, None)), 1),
            })

        return {
            "source": source,
            "history": [{"date": d.strftime("%Y-%m"), "value": float(v)} for d, v in ts.items()],
            "forecast": forecast_list,
            "model_order": [1, 1, 1],
            "message": "Filtered series with dynamically fitted ARIMA(1,1,1) forecast.",
        }
    except Exception as e:
        return {
            "source": source,
            "history": [{"date": d.strftime("%Y-%m"), "value": float(v)} for d, v in ts.items()],
            "forecast": None,
            "message": f"Could not fit ARIMA model: {str(e)}",
        }


if __name__ == "__main__":
    train_arima()

