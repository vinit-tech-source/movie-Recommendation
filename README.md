# 🎬 CineMetrics: Movie / OTT Recommendation & Trend Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/Flask-3.1-black?style=for-the-badge&logo=flask&logoColor=white" alt="Flask 3.1" />
  <img src="https://img.shields.io/badge/Scikit--Learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/SQLite-Fact%20Constellation-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite" />
  <img src="https://img.shields.io/badge/Pytest-19%20Passing-success?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest" />
  <img src="https://img.shields.io/badge/Academic-CO1%20to%20CO5-blue?style=for-the-badge" alt="Curriculum CO1-CO5" />
</p>

> **An End-to-End Analytical Data Mining & Machine Learning Architecture** uniting theatrical viewer preferences from the **MovieLens 100k** dataset with commercial streaming catalog distribution patterns from the **Netflix Titles** dataset. Features an automated ETL pipeline loading into an ACID-compliant **Fact Constellation Data Warehouse**, five machine learning & time-series mining modules, and an ultra-modern light/white SaaS web application.

---

## 📑 Student & Submission Metadata
* **Submitted by:** **Vinit Shinde**
* **Roll No:** 37
* **PRN:** RBT24CS178
* **Branch:** B.Tech – Computer Engineering
* **Subject:** Data Mining and Analytics (Course Code: `CS3203T-A`)
* **Academic Year:** 2026–2027 | **Semester:** 1
* **Faculty Guide:** Prof. Jit Shendekar
* **Institution:** Rajarshi Shahu College of Engineering (Autonomous), Tathawade, Pune
* **Comprehensive Project Report:** [reports/PROJECT_REPORT.pdf](reports/PROJECT_REPORT.pdf)

---

## 🏛️ System Architecture

```
                  ┌───────────────────────────────┐
                  │    Raw Benchmark Datasets     │
                  │  • MovieLens 100k (Ratings)   │
                  │  • Netflix Titles (Catalog)   │
                  └───────────────┬───────────────┘
                                  │
                                  ▼ [Automated ETL Pipeline: etl.py]
                  ┌───────────────────────────────┐
                  │ Fact Constellation Warehouse  │
                  │  SQLite Database: warehouse.db│
                  │  5,340 Conformed Calendar Dates│
                  │  100,836 Ratings • 9,742 Movies│
                  │  8,807 Titles • 0 Orphan Keys │
                  └───────────────┬───────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Frequent Pattern│    │  Classification  │    │  Trend & Series  │
│  Mining (CO2/CO4)│    │ & Text NLP (CO2) │    │ Modeling (CO3/5) │
│  • Apriori Rules │    │ • Multinomial NB │    │ • Lasso (L1)     │
│  • 935 Co-View   │    │ • Random Forest  │    │ • ARIMA(3,1,3)   │
│    Rules (Lift   │    │   (63.01% Acc)   │    │   (+22.5% Gain)  │
│    up to 6.15x)  │    │ • Plot Synopses  │    │ • ADF Tests      │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │   Hybrid Recommender Engine   │
                  │   0.4 Content + 0.3 CF +      │
                  │   0.3 Apriori Association     │
                  │   Leave-One-Out Hit-Rate: 4.7%│
                  └───────────────┬───────────────┘
                                  │
                                 ▼
                  ┌───────────────────────────────┐
                  │   Flask Web Application & API │
                  │   Ultra-Clean White SaaS UI   │
                  │   Real-time Predictive Forms  │
                  │   Chart.js Visual Analytics   │
                  └───────────────────────────────┘
```

---

## 📊 Empirical Evaluation & Benchmark Results

All metrics reported below were produced on strict holdout test splits (`train_all.py`):

| Pipeline Module | Course Outcome | Model Algorithm | Primary Evaluation Metric | Secondary Metric | Optimal Hyperparameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Market-Basket Mining** | **CO2** | Apriori Algorithm | **935 Strict Rules Mined** | Top Lift: **6.15x** | $\text{Support} \ge 0.10, \text{Confidence} \ge 0.60$ |
| **Genre Text Mining** | **CO2** | Multinomial Naive Bayes | **Accuracy: 50.49%** | Macro F1: 0.4757 | $\alpha = 0.1$, TF-IDF (1–2 grams) |
| **Rating Classification** | **CO4** | **Random Forest Classifier** | **Accuracy: 63.01%** | Macro F1: 0.5576 | $n_{\text{est}}=200, \text{depth}=10, \text{balanced}$ |
| **Rating Classification** | **CO4** | Decision Tree Classifier | Accuracy: 50.68% | Macro F1: 0.4766 | $\text{max\_depth} = 10$ |
| **Rating Regression** | **CO3** | **Lasso Regression (L1)** | **Test RMSE: 0.4683** | **Test $R^2$: 0.3105** | $\alpha = 0.00041$ (Sparse Features) |
| **Rating Regression** | **CO3** | Ridge Regression (L2) | Test RMSE: 0.4684 | Test $R^2$: 0.3103 | $\alpha = 10.0$ |
| **Time Series Forecast** | **CO5** | **$\text{ARIMA}(3,1,3)$ on Netflix** | **Test RMSE: 41.42** | **+22.47% vs. Naive** | $d=1$ (ADF $p = 2.82 \times 10^{-8}$) |
| **Hybrid Recommender** | **CO5** | **Hybrid (Content + CF + Rules)**| **Hit-Rate@10: 0.0467** | Precision@10: 0.0047 | Weights: $(0.4, 0.3, 0.3)$ |

---

## 🎯 Course Outcomes (CO1 to CO5) Mapping

* **CO1 — Data Preprocessing, Transformation & Warehousing:**
  * Implemented in `etl.py`: Automated ingestion creating a conformed Fact Constellation schema in SQLite (`warehouse.db`) with `dim_date`, `dim_genre`, `dim_movie`, `dim_user`, `dim_title`, `fact_ratings`, and `fact_titles`. Verified 0 orphan keys and 11 relational indexes.
* **CO2 — Association Rule Mining & Text Classification:**
  * Implemented in `mining/apriori_rules.py` & `mining/classifier.py`: Binary co-viewing matrix ($599 \text{ users} \times 300 \text{ movies}$) mined via Apriori. Plot synopsis text processed using NLTK Porter Stemming and TF-IDF to train Multinomial Naive Bayes.
* **CO3 — Linear & Regularized Regression Analysis:**
  * Implemented in `mining/regression.py`: Standardized pipeline evaluating OLS, Ridge, and Lasso. Lasso identified active weights (Documentary $+0.599$, Animation $+0.450$, Children $-0.318$, Horror $-0.196$).
* **CO4 — Decision Trees & Ensemble Methods:**
  * Implemented in `mining/classifier.py`: Decision Tree and balanced Random Forest ensemble predicting movie rating tiers (Low, Medium, High).
* **CO5 — Temporal Forecasting & Recommendation Systems:**
  * Implemented in `mining/arima_model.py` & `mining/recommender.py`: Stationarity testing (ADF) with first-order differencing ($d=1$) and $\text{ARIMA}(3,1,3)$ grid search. Three-way hybrid recommender with Leave-One-Out Hit-Rate@10 offline evaluation.

---

## 🚀 Quickstart Guide (Run in 3 Minutes)

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/vinit-tech-source/movie-Recommendation.git
cd movie-Recommendation

# Create and activate virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Build Warehouse & Train Machine Learning Models
```bash
# 1. Run automated ETL pipeline (creates warehouse.db)
python etl.py

# 2. Train all 6 machine learning & time-series models
python train_all.py
```

### 3. Launch Web Application
```bash
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

### 4. Run Automated Test Suite
```bash
python -m pytest tests/
```
*(All 19/19 tests pass verifying schema referential integrity, orphan keys, and API contracts).*

---

## 🌐 Web Interface Walkthrough

* **Dashboard (`/`):** Real-time warehouse stats, interactive autocomplete search with title preview, and catalog KPI cards.
* **Recommendations (`/recommend`):** Multi-strategy recommendation engine (Hybrid, Content-Based, Item-Item CF, Apriori Rules) with transparent "Why Recommended" explanations.
* **Predictor (`/predict`):** Interactive metadata form predicting rating tiers (Random Forest) + continuous rating (Lasso), plus an NLP plot classifier predicting genres from story synopses.
* **Trends & Forecast (`/trends`):** Interactive Chart.js time-series dashboard showing historical Netflix additions and 12-month ARIMA forecasts with 95% confidence bands.
* **Model Benchmarks (`/performance`):** Comprehensive academic benchmark tables, out-of-sample error comparisons, and Recommender Hit-Rate@10 metrics.

---

## 🔌 REST API Endpoints

The Flask application exposes a complete REST API:

| Endpoint | Method | Description | Example Query / Payload |
| :--- | :--- | :--- | :--- |
| `/api/health` | `GET` | Health check, warehouse row counts & loaded model status | `GET /api/health` |
| `/api/search` | `GET` | Autocomplete movie search with rating aggregates | `GET /api/search?q=matrix&limit=5` |
| `/api/recommend` | `GET` | Returns Top-K recommendations with component scores | `GET /api/recommend?movie_id=1&method=hybrid&k=10` |
| `/api/predict-rating` | `POST` | Predicts rating tier (Random Forest) & rating value (Lasso) | `POST {"genres": ["Action", "Sci-Fi"], "year": 2022, "n_ratings": 75}` |
| `/api/predict-genre` | `POST` | Classifies plot synopsis text into genres | `POST {"text": "A detective chases a mastermind criminal..."}` |
| `/api/trends` | `GET` | Historical monthly additions & 12-month ARIMA forecast | `GET /api/trends?source=netflix` |
| `/api/stats` | `GET` | Warehouse aggregates (ratings histogram, top genres) | `GET /api/stats` |

---

## 📂 Project Directory Structure

```
movie-Recommendation/
├── data/                                 # Benchmark raw CSV datasets
│   ├── ml-latest-small/                  # MovieLens small dataset (ratings, movies, tags, links)
│   ├── netflix_titles.csv/               # Netflix streaming catalog
│   └── tmdb_5000_credits.csv             # TMDB credits catalog
├── etl.py                                # Automated ETL & Fact Constellation Warehouse Builder
├── train_all.py                          # Master training script executing Modules B through F
├── app.py                                # Flask REST API and web application server
├── requirements.txt                      # Project dependencies
├── warehouse.db                          # Analytical SQLite Data Warehouse
├── .gitignore                            # Standard git ignore definitions
├── mining/                               # Core Data Mining Modules
│   ├── __init__.py
│   ├── apriori_rules.py                  # Module B (CO2): Apriori Association Rule Mining
│   ├── classifier.py                     # Module C (CO2, CO4): Naive Bayes & Random Forest
│   ├── regression.py                     # Module D (CO3, CO4): Linear, Ridge, Lasso & Tree Regressor
│   ├── arima_model.py                    # Module E (CO5): ADF Tests & ARIMA Time Series Forecasting
│   └── recommender.py                    # Module F (CO5): Hybrid Recommender Engine
├── models/                               # Serialized ML Model Payloads (.joblib)
│   ├── rules.joblib                      # 935 strict Apriori association rules
│   ├── genre_classifier.joblib           # TF-IDF + MultinomialNB genre classifier
│   ├── rating_classifier.joblib          # Random Forest rating-class model
│   ├── rating_regressor.joblib           # Lasso continuous rating regressor
│   ├── arima_models.joblib               # Fitted ARIMA models and future forecasts
│   └── recommender.joblib                # Precomputed similarity matrices & metadata
├── reports/                              # Academic Reports & Evaluation Figures
│   ├── PROJECT_REPORT.pdf                # Formal 5-Page Mini-Project Report (PDF)
│   ├── PROJECT_REPORT.html               # Printable HTML Report Template
│   ├── PROJECT_REPORT.md                 # Markdown Report Documentation
│   ├── metrics.json                      # Automated training evaluation metrics audit
│   ├── figures/                          # 12 evaluation plots & confusion matrix heatmaps
│   └── screenshots/                      # High-resolution UI screenshots
├── templates/                            # Jinja2 HTML5 Templates (White SaaS UI)
│   ├── base.html                         # Base layout with navigation and Chart.js CDN
│   ├── index.html                        # Dashboard & instant search autocomplete
│   ├── recommend.html                    # Recommendation engine & association rules table
│   ├── predict.html                      # Interactive rating & genre prediction forms
│   ├── trends.html                       # Time series charts with 12-month forecast & filters
│   └── performance.html                  # Empirical metrics dashboard
├── static/
│   ├── css/style.css                     # Modern white/light SaaS design system
│   └── js/                               # Frontend scripts (main.js, recommend.js, predict.js, trends.js)
├── tests/
│   ├── test_etl.py                       # Warehouse schema integrity & referential tests
│   └── test_api.py                       # REST API endpoint smoke & contract tests
└── README.md                             # Comprehensive project documentation
```

---

## 📜 License & Acknowledgments
* **Academic Submission:** Mini Project for Data Mining and Analytics (`CS3203T-A`), B.Tech Computer Engineering, Rajarshi Shahu College of Engineering, Pune.
* **Datasets:** [GroupLens MovieLens 100k](https://grouplens.org/datasets/movielens/latest/) and [Kaggle Netflix Titles](https://www.kaggle.com/datasets/shivamb/netflix-shows).
* **Author:** Vinit Shinde (GitHub: [@vinit-tech-source](https://github.com/vinit-tech-source))
