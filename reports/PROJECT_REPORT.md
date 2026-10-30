# RAJARSHI SHAHU COLLEGE OF ENGINEERING
### Tathawade, Pune – 411033
### Department of Computer Engineering

---

# MINI PROJECT REPORT
## MOVIE / OTT RECOMMENDATION AND TREND ANALYSIS
### An Integrated Approach Using Star Schema ETL, Apriori Association Rules, Multi-Model Classification, Time Series ARIMA, and Hybrid Collaborative Filtering

**Subject:** Data Mining and Analytics  
**Subject Code:** CS3203T-A  
**Academic Year:** 2026–2027  
**Semester:** 1  

**Submitted by:**  
Vinit Shinde  
Roll No: 37  
PRN: RBT24CS178  
B.Tech – Computer Engineering  

**Faculty Guide:**  
Prof. Jit Shendekar  

---

## CONTENTS

| Section | Title |
|---|---|
| 1 | Abstract |
| 2 | Introduction |
| 3 | Problem Statement |
| 4 | Aim and Objectives |
| 5 | Scope of the Project |
| 6 | Syllabus Mapping |
| 7 | Dataset Description |
| 8 | Methodology and Algorithm |
| 9 | System Architecture |
| 10 | Software and Tools |
| 11 | Experimental Output and Evaluation |
| 12 | Limitations and Future Scope |
| 13 | Conclusion |
| 14 | References |

---

## 1. ABSTRACT

With the exponential growth of online streaming services and digital media catalogs, platforms face significant hurdles in helping users discover relevant content, understanding audience preferences, and projecting temporal content expansion. This mini-project presents an end-to-end analytical data mining system that integrates community-driven theatrical viewer ratings from the **MovieLens 100k** dataset (100,836 ratings, 9,742 movies) with streaming platform catalog additions from the **Netflix Titles** dataset (8,807 titles).

The system executes an automated ETL pipeline that structures the data into a conformed **Fact Constellation Data Warehouse** in SQLite across 5,340 conformed calendar dates. On top of this warehouse, five core data mining and analytics modules are developed:
1. **Apriori Association Rule Mining:** Analyzes user viewing baskets (ratings $\ge 4.0$) on the top 300 movies, discovering **935 high-lift co-viewing rules** (with Lift reaching 6.15 for complementary titles like *Kill Bill: Vol. 1* $\rightarrow$ *Vol. 2*).
2. **Text Mining & Naive Bayes Classification:** Cleans unstructured plot descriptions using NLTK Porter Stemming and unigram/bigram TF-IDF vectorization to categorize movie plot synopses into genres with **50.49% accuracy** (0.4757 macro-F1).
3. **Supervised Rating-Class Prediction:** Extracts 22 leak-free features (19 genre flags, release year, $\log(1+n_{\text{ratings}})$, and tag counts) to train Decision Tree and balanced Random Forest classifiers, achieving **63.01% accuracy** (0.5576 macro-F1) across Low, Medium, and High rating tiers.
4. **Regularized Regression Analysis:** Applies Linear (OLS), Ridge (L2), and Lasso (L1) regression to predict continuous ratings. Lasso achieves a **Test RMSE of 0.4683** ($R^2 = 0.3105$) and isolates active genre weights (Documentary $+0.599$, Animation $+0.450$, Children $-0.318$).
5. **Temporal Trend Analysis & ARIMA Forecasting:** Evaluates Augmented Dickey-Fuller (ADF) stationarity ($p = 2.82 \times 10^{-8}$ after $d=1$ differencing) and optimizes an **ARIMA(3,1,3)** model on monthly Netflix additions, achieving a **Test RMSE of 41.42** (a **22.47% improvement** over baseline persistence) and projecting 12-month future platform growth with 95% confidence intervals.
6. **Hybrid Recommendation Engine:** Fuses Content-based TF-IDF, Item-Item Collaborative Filtering (mean-centered cosine similarity), and Apriori association rules with dynamic weight redistribution and explainable *"why"* reasoning. Offline Leave-One-Out evaluation across 150 users achieves a **Hit-Rate@10 of 0.0467**.

All capabilities are served through a lightweight Flask REST API and an interactive, responsive analytical web dashboard.

---

## 2. INTRODUCTION

Modern digital entertainment platforms rely heavily on data mining techniques to extract actionable patterns from massive interaction logs. Streaming catalogs contain thousands of titles spanning multiple genres, directors, countries, and release periods. Users generate millions of rating events, search queries, and session interactions. Examining these transactions manually is impossible due to extreme matrix sparsity and combinatorial item interactions.

Data mining automatically uncovers recurring consumption behaviors, identifies cross-genre affinities, predicts content reception, and projects platform catalog trajectories. For example, if viewers who enjoy dystopian science fiction consistently rate neo-noir thrillers highly, association rule mining detects this co-consumption pattern. Similarly, statistical time series models reveal seasonal patterns in platform content acquisition, allowing stakeholders to plan licensing schedules.

This project implements a complete, runnable college mini-project demonstrating the end-to-end data mining lifecycle—from raw CSV ingestion and multidimensional schema construction to descriptive association mining, predictive supervised learning, time series forecasting, and personalized recommendation.

---

## 3. PROBLEM STATEMENT

To ingest, cleanse, and model heterogeneous movie rating transactions and streaming catalog records into a multidimensional Fact Constellation warehouse, and to apply association rule mining, text classification, predictive regression, and ARIMA temporal modeling so that co-viewing behaviors, rating tiers, and platform catalog trends can be reliably discovered, evaluated, and served through an explainable recommendation interface.

---

## 4. AIM AND OBJECTIVES

**Aim:** To design, implement, and empirically evaluate an integrated Movie and OTT Recommendation and Trend Analysis system using multidimensional warehousing and core data mining algorithms.

**Objectives:**
- Load, validate, and inspect the MovieLens 100k and Netflix Titles benchmark datasets.
- Cleanse and transform raw records: extract release years via regular expressions, handle missing attributes, parse duration units, and construct an analytical SQLite Fact Constellation schema with conformed date dimensions.
- Formulate user ratings into transactional baskets and apply the Apriori algorithm to discover frequent itemsets and co-viewing association rules using support, confidence, and lift.
- Implement an NLTK-based text mining pipeline with Porter Stemming and unigram/bigram TF-IDF vectorization to classify unstructured plot synopses into genres using Multinomial Naive Bayes.
- Engineer leak-free metadata features and train Decision Tree and Random Forest classifiers to predict movie rating tiers (Low, Medium, High) with balanced class weights.
- Train and compare Linear, Ridge, and Lasso regression models to predict continuous movie ratings and analyze active L1 feature selection coefficients.
- Perform Augmented Dickey-Fuller stationarity tests and fit ARIMA models on monthly catalog additions to evaluate 12-month holdout accuracy and forecast platform growth with 95% confidence intervals.
- Construct an explainable Hybrid Recommender (Content-based + Item-CF + Apriori Rules) and perform offline Leave-One-Out Hit-Rate@10 evaluation.
- Deploy an interactive, responsive Flask web application and Chart.js dashboard for real-time inference and visualization.

---

## 5. SCOPE OF THE PROJECT

The project encompasses the offline analytical processing, model training, empirical evaluation, and interactive web serving of entertainment datasets. The system supports analytical OLAP queries, real-time title autocomplete search, multi-strategy recommendations with explainable text, rating and genre inference forms, and interactive time series visualizations.

The project operates on benchmark historical catalogs (MovieLens small and Netflix Titles); it does not provide live streaming CDN media hosting, establish medical or causal behavioral claims, or automatically alter production licensing contracts.

---

## 6. SYLLABUS MAPPING

| Syllabus Unit / Topic | Use in this Project | Course Outcome |
|---|---|---|
| **Unit 1: Introduction to Data Mining & Warehousing** | Purpose and application of data mining to entertainment datasets; dimensional modeling; Fact Constellation schema; OLAP aggregation queries. | **CO1** |
| **Unit 2: Data Preparation & Preprocessing** | Cleansing missing years/dates; handling categorical unknowns; regex feature extraction; $\log(1+x)$ variance stabilization; Porter Stemming; TF-IDF document vectorization. | **CO1, CO2** |
| **Unit 3: Regression Analysis** | Ordinary Least Squares (OLS), Ridge (L2 penalty), Lasso (L1 feature selection), and DecisionTreeRegressor for continuous rating prediction. | **CO3, CO4** |
| **Unit 4: Association Rules & Classification** | Market-basket Apriori frequent itemset mining (Support, Confidence, Lift); Multinomial Naive Bayes text classification; Decision Tree and balanced Random Forest classifiers. | **CO2, CO4** |
| **Unit 5: Advanced Data Mining & Recommenders** | Augmented Dickey-Fuller stationarity testing ($d=1$); ARIMA(3,1,3) temporal forecasting; Content-based TF-IDF; Item-Item Collaborative Filtering; Hybrid ensemble; Leave-One-Out Hit-Rate@10 evaluation. | **CO5** |

---

## 7. DATASET DESCRIPTION

The system ingests two established benchmark entertainment datasets:

### 1. MovieLens 100k Small Dataset (`ml-latest-small`)
- **`ratings.csv` (100,836 rows):** User ratings on a 0.5 to 5.0 scale across 610 users and 9,724 rated movies, recorded between March 1996 and September 2018. Includes `userId`, `movieId`, `rating`, `timestamp`.
- **`movies.csv` (9,742 rows):** Titles and pipe-delimited genre strings. 13 titles lack a release year in parentheses and were cleanly assigned `year = NULL`; 34 movies have `(no genres listed)` and were preserved without bridge rows.
- **`tags.csv` (3,683 rows):** User-applied freeform tags (`tag`, `timestamp`), lowercased and joined per movie for TF-IDF document representation.
- **`links.csv` (9,742 rows):** External identifiers (`imdbId`, `tmdbId`) cast to nullable integers and linked to dimension tables.

### 2. Netflix Titles Dataset (`netflix_titles.csv`)
- **8,807 rows:** Complete catalog metadata for movies and TV shows added to Netflix between 2008 and September 2021.
- **Attributes:** `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description`.
- **Data Cleansing:** Leading whitespace was stripped from `date_added`; 10 null dates were safely preserved in `dim_title` with `date_id = NULL`. Missing directors (2,634), cast (825), and countries (831) were imputed with `"Unknown"`. Durations were parsed into numerical values and unit strings (e.g., 90 min $\rightarrow$ 90, "min"; 2 Seasons $\rightarrow$ 2, "Seasons").

---

## 8. METHODOLOGY AND ALGORITHM

The system executes an 8-stage data mining pipeline:

1. **Step 1 – Ingestion & Path Discovery:** Automatically locates data files regardless of directory nesting using `pathlib.Path.rglob`.
2. **Step 2 – Data Preprocessing & Fact Constellation Schema Loading:** Idempotently drops and rebuilds `warehouse.db`. Normalizes entities into conformed `dim_date` (5,340 dates), `dim_user` (610), `dim_movie` (9,742), `dim_genre` (19), `dim_title` (8,807), and `dim_netflix_genre` (42). Fact tables `fact_ratings` (100,836 rows) and `fact_titles` (8,797 rows) share `dim_date`, forming a **Fact Constellation Schema** with zero orphan foreign keys.
3. **Step 3 – Basket Construction & Apriori Mining:** Baskets represent individual users. Items are movies rated $\ge 4.0$. Filtered to the top 300 most-rated movies to construct a dense $599 \times 300$ boolean matrix. Executed `mlxtend.frequent_patterns.apriori` with $\text{min\_support} = 0.10$ and generated association rules filtered by $\text{confidence} \ge 0.60$ and $\text{lift} > 1.2$. A secondary lenient set ($\text{support} \ge 0.05, \text{confidence} \ge 0.50$) was generated for high recommender coverage.
4. **Step 4 – Plot Text Preprocessing & Naive Bayes Classification:** Filtered Netflix descriptions into top 8 categories ($N=6,664$). Removed punctuation and NLTK stop words, applied Porter Stemming, fitted unigram/bigram `TfidfVectorizer(max_features=20000)`, and tuned `MultinomialNB` hyperparameter $\alpha$ via 5-fold cross-validation.
5. **Step 5 – Leak-Free Feature Engineering & Rating Classification:** Selected movies with $\ge 5$ ratings ($N=3,650$). Target was bucketed into Low ($<3.0$), Medium ($3.0 \le r < 4.0$), and High ($\ge 4.0$). Engineered 22 leak-free features (19 genre flags, release year, $\log(1+n_{\text{ratings}})$, and tag count). Average rating was strictly excluded to prevent data leakage. Stratified 80/20 train/test split; trained Decision Tree and balanced Random Forest ($n=200$ trees, depth 10).
6. **Step 6 – Regularized Regression Modeling:** Standardized numerical columns using `ColumnTransformer` inside a `Pipeline`. Evaluated Ordinary Least Squares (OLS), Ridge (L2 penalty), Lasso (L1 penalty), and DecisionTreeRegressor. Evaluated MSE, RMSE, MAE, and $R^2$ on train and test sets.
7. **Step 7 – Stationarity Testing & ARIMA Forecasting:** Extracted monthly Netflix additions (2016-01 to 2021-08, 68 months). Executed Augmented Dickey-Fuller test. Applied first differencing ($d=1$). Optimized $(p, 1, q)$ grid minimizing AIC. Evaluated out-of-sample 12-month holdout RMSE against Naive and Seasonal Naive baselines. Validated white noise residuals via Ljung-Box test and projected 12-month future additions with 95% confidence intervals.
8. **Step 8 – Hybrid Recommendation & Offline Evaluation:** Combined Content-based TF-IDF, mean-centered Item-Item Collaborative Filtering, and Apriori association rules with dynamic weight redistribution ($0.4 \times \text{Content} + 0.3 \times \text{CF} + 0.3 \times \text{Rules}$). Performed Leave-One-Out Hit-Rate@10 offline evaluation across 150 test users.

### Key Mathematical Formulations

| Measure / Technique | Mathematical Formula | Analytical Meaning |
|---|---|---|
| **Support** | $\text{Support}(A \rightarrow B) = P(A \cup B) = \frac{\text{Count}(A \cup B)}{N}$ | Fraction of all user baskets containing both movies $A$ and $B$. |
| **Confidence** | $\text{Confidence}(A \rightarrow B) = \frac{P(A \cup B)}{P(A)} = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$ | Likelihood that a user likes movie $B$ given that they liked movie $A$. |
| **Lift** | $\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)} = \frac{P(A \cup B)}{P(A) \cdot P(B)}$ | Ratio of observed co-occurrence to expected occurrence under statistical independence. Lift $> 1$ indicates true complementary association. |
| **Naive Bayes** | $P(c \mid d) \propto P(c) \prod_{i=1}^{n} P(w_i \mid c)$ | Maximum A Posteriori (MAP) probability of genre $c$ given plot word tokens $w_i$. |
| **Lasso Regression** | $\min_{\beta} \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda \sum_{j=1}^{p} |\beta_j|$ | Penalizes OLS residual sum of squares by L1 absolute sum, shrinking non-informative genre coefficients to exactly zero. |
| **ARIMA($p,d,q$)** | $\left(1 - \sum_{i=1}^p \phi_i B^i\right) (1 - B)^d y_t = c + \left(1 + \sum_{j=1}^q \theta_j B^j\right) \epsilon_t$ | Combines $p$ autoregressive lags, $d$ non-seasonal differences, and $q$ moving average error terms. |
| **Item-Item Cosine** | $\text{sim}(i, j) = \frac{\sum_u (r_{u,i} - \bar{r}_u)(r_{u,j} - \bar{r}_u)}{\sqrt{\sum_u (r_{u,i} - \bar{r}_u)^2} \sqrt{\sum_u (r_{u,j} - \bar{r}_u)^2}}$ | Cosine similarity across mean-centered user rating vectors, eliminating individual user rating scale bias. |
| **Hybrid Score** | $S_{\text{hybrid}} = \frac{w_c S_{\text{content}} + w_{cf} S_{\text{cf}} + w_r S_{\text{rule}}}{w_c + w_{cf} + w_r}$ | Normalized multi-strategy score with dynamic redistribution when individual signals are missing. |

---

## 9. SYSTEM ARCHITECTURE

```
+-------------------------------------------------------------------------------+
|                             RAW DATA SOURCES                                  |
|     MovieLens 100k CSVs (100,836 ratings)   |   Netflix Titles CSV (8,807)     |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                      ETL & WAREHOUSE ENGINE (etl.py - CO1)                    |
|  - Regex release year extraction, timestamp formatting, null imputation        |
|  - Fact Constellation Schema: dim_date (5,340), dim_movie (9,742),            |
|    dim_title (8,807), fact_ratings (100,836), fact_titles (8,797)             |
|  - Relational indexes on join keys; 0 orphan foreign keys verified            |
+---------------------------------------+---------------------------------------+
                                        |
       +--------------------------------+-------------------------------+
       |                                |                               |
       v                                v                               v
+---------------+             +-------------------+             +---------------+
| APRIORI RULES |             |  CLASSIFICATION   |             |  REGRESSION   |
| (apriori.py)  |             |  (classifier.py)  |             | (regression.py|
| 599x300 matrix|             | - Naive Bayes text|             | - OLS Linear  |
| 935 rules     |             |   genre (50.5%)   |             | - Ridge (L2)  |
| Lift up to    |             | - Random Forest   |             | - Lasso (L1)  |
| 6.15x (CO2)   |             |   rating (63.0%)  |             |   RMSE: 0.468 |
+-------+-------+             +---------+---------+             +-------+-------+
        |                               |                               |
        +-------------------------------+-------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                    TEMPORAL & RECOMMENDATION MINING                           |
|  - arima_model.py (CO5): ADF Test (p=2.82e-8), ARIMA(3,1,3) 12-mo forecast    |
|  - recommender.py (CO5): Hybrid Engine (Content + CF + Rules) + Hit-Rate@10   |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       SERVING & PRESENTATION LAYER                            |
|  - Flask REST API (app.py): /api/recommend, /api/predict-*, /api/trends       |
|  - Web Dashboard: Jinja2, Chart.js (Explore, Recommend, Predict, Trends)     |
+-------------------------------------------------------------------------------+
```

---

## 10. SOFTWARE AND TOOLS

- **Programming Language:** Python 3.12 (standard modern scientific computing stack).
- **Data Engineering & ETL:** Pandas 2.3.3, NumPy 2.5.2, Pathlib.
- **Data Warehouse Engine:** SQLite 3 (embedded relational database with foreign key support).
- **Machine Learning Libraries:** Scikit-Learn 1.6.1 (`TfidfVectorizer`, `MultinomialNB`, `RandomForestClassifier`, `LassoCV`, `RidgeCV`, `Pipeline`).
- **Association Rule Mining:** mlxtend 0.25.0 (`apriori`, `association_rules`).
- **Time Series & Statistics:** statsmodels 0.15.0 (`adfuller`, `ARIMA`, `acorr_ljungbox`).
- **Natural Language Processing:** NLTK 3.10.4 (`stopwords`, `PorterStemmer`).
- **Web Serving & Visualization:** Flask 3.1.3, Jinja2, Chart.js via CDN, Matplotlib 3.11, Seaborn 0.13.
- **Verification & Testing:** Pytest 9.1.1 (19 automated unit/integration tests), Playwright 1.64.0 (automated browser verification).

---

## 11. EXPERIMENTAL OUTPUT AND EVALUATION

All metrics were computed by running `python train_all.py` on the real datasets:

### 1. Apriori Association Rules Mining (CO2)
- **Frequent Itemsets Discovered:** 891 itemsets ($\text{support} \ge 0.10$).
- **Strict Rules Generated:** 935 rules ($\text{confidence} \ge 0.60, \text{lift} > 1.2$).
- **Lenient Rules Generated:** 2,206 rules ($\text{support} \ge 0.05, \text{confidence} \ge 0.50$).

#### Top Discovered Association Rules (Real Values)

| # | Antecedent (If Liked) | Consequent (Then Liked) | Support | Confidence | Lift |
|---|---|---|---|---|---|
| 1 | Kill Bill: Vol. 1 (2003) | Kill Bill: Vol. 2 (2004) | 0.1085 | 0.7386 | **6.145** |
| 2 | Kill Bill: Vol. 2 (2004) | Kill Bill: Vol. 1 (2003) | 0.1085 | 0.9028 | **6.145** |
| 3 | The Godfather (1972), Goodfellas (1990) | The Godfather: Part II (1974) | 0.1018 | 0.7922 | **4.435** |
| 4 | The Matrix (1999), Terminator 2 (1991) | The Terminator (1984) | 0.1035 | 0.6739 | **4.294** |
| 5 | The Terminator (1984) | The Matrix (1999), Terminator 2 (1991) | 0.1035 | 0.6596 | **4.294** |

### 2. Multi-Model Classification Benchmarks (CO2, CO4)

| Task | Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|---|
| **Genre from Text (CO2)** | **Multinomial Naive Bayes** | **50.49%** | **0.5056** | **0.4697** | **0.4757** | **0.5034** |
| Genre from Text (CO2) | Logistic Regression | 51.54% | 0.5284 | 0.5212 | 0.5215 | 0.5186 |
| **Rating Tier (CO4)** | Decision Tree | 50.68% | 0.4746 | 0.4812 | 0.4766 | 0.5283 |
| **Rating Tier (CO4)** | **Random Forest (Best)** | **63.01%** | **0.5694** | **0.5630** | **0.5576** | **0.6385** |

*Note on Genre Classification:* A ~50% top-1 accuracy is normal and expected given that Netflix synopses are concise single sentences (~25 words) with substantial cross-genre overlap (e.g., Action vs. Thriller).

### 3. Continuous Rating Regression Models (CO3, CO4)

| Model | Test RMSE | Test MAE | Test R² | Train R² | Key Observation |
|---|---|---|---|---|---|
| **Lasso Regression (L1)** | **0.4683** | **0.3707** | **0.3105** | 0.3032 | **Best Generalization** ($\alpha = 0.00041$). Active sparse feature selection. |
| Ridge Regression (L2) | 0.4684 | 0.3705 | 0.3103 | 0.3034 | Shrinks collinear genre indicators smoothly. |
| Linear Regression (OLS) | 0.4684 | 0.3705 | 0.3103 | 0.3034 | Unregularized baseline. |
| Decision Tree Regressor | 0.4810 | 0.3756 | 0.2727 | 0.3553 | Captures non-linear thresholds ($\text{depth} = 5$). |

*Lasso Active Non-Zero Coefficients:*
- **Positive Rating Weights:** Documentary ($+0.5992$), Animation ($+0.4497$), Drama ($+0.2904$), Film-Noir ($+0.2236$), $\log(1+n_{\text{ratings}})$ ($+0.1596$).
- **Negative Rating Weights:** Children ($-0.3179$), Horror ($-0.1962$).

### 4. Time Series Modeling & ARIMA Forecasting (CO5)
- **Netflix Monthly Additions (2016-01 to 2021-08, 68 months):**
  - Raw Series ADF test: statistic $= -2.9141$, $p = 0.0437$ (borderline non-stationary).
  - Differenced ($d=1$) ADF test: statistic $= -6.3367$, $p = 2.819 \times 10^{-8}$ (**strongly stationary**).
  - Selected Order: **ARIMA(3, 1, 3)** (AIC $= 660.23$).
  - 12-Month Holdout Test RMSE: **41.42** vs. Naive Baseline: **53.42** (**22.47% improvement**).
  - Ljung-Box diagnostic: $p = 0.812 > 0.05$ (residuals behave as Gaussian white noise).
- **MovieLens Monthly Rating Volume (2000-01 to 2018-08, 224 months, $\log(1+x)$):**
  - Differenced ADF test: statistic $= -8.3707$, $p = 2.67 \times 10^{-13}$.
  - Selected Order: **ARIMA(1, 1, 2)** (AIC $= 604.72$). Holdout RMSE $= 460.52$ vs. Naive $= 506.84$.

### 5. Recommender Offline Leave-One-Out Evaluation ($N=150$ users, CO5)

| Recommendation Strategy | Hit-Rate @ 10 | Precision @ 10 | Recall @ 10 | Coverage & Strengths |
|---|---|---|---|---|
| Content-Based (TF-IDF) | 0.0067 | 0.0007 | 0.0067 | 100% catalog coverage; effective for niche titles with unique tags. |
| **Item-Item CF** | **0.0467** | **0.0047** | **0.0467** | High accuracy on dense items with $\ge 20$ community ratings. |
| Apriori Rules | 0.0267 | 0.0027 | 0.0267 | High confidence when rules fire; limited on tail catalog. |
| **Hybrid Ensemble Engine** | **0.0467** | **0.0047** | **0.0467** | **Best balance**: matches peak CF precision while eliminating cold-start failures. |

---

## 12. LIMITATIONS AND FUTURE SCOPE

### Academic Limitations
1. **Catalog Domain Asymmetry:** MovieLens ratings span 1996–2018 and reflect theatrical consumer behavior. Netflix additions span 2008–2021 and reflect licensing deals. While conformed on calendar dates in the warehouse, they represent different consumer populations.
2. **Short Text Lexical Sparsity:** Single-sentence Netflix plot descriptions (~25 words) contain inherent ambiguity across overlapping genres (e.g., Action vs. Thriller), placing an empirical ceiling on unigram/bigram classification accuracy.
3. **Rating Skewness:** In consumer rating portals, users rarely review titles they dislike. Consequently, Medium and High ratings dominate, requiring balanced loss weights during training.

### Future Scope
1. **Matrix Factorization & Deep Learning:** Incorporating Singular Value Decomposition (SVD) and Neural Collaborative Filtering (NCF) to capture latent embeddings.
2. **Context-Aware Streaming Logs:** Integrating device telemetry, viewing duration, and pause events to provide real-time session-based recommendations.
3. **Cloud Native Deployment:** Containerizing the application with Docker and Kubernetes for continuous retraining pipelines.

---

## 13. CONCLUSION

This mini-project successfully designed, implemented, and empirically verified an integrated Data Mining and Analytics system spanning all course outcomes (**CO1 through CO5**). An automated ETL pipeline loaded 100,836 rating events and 8,807 streaming entries into an analytical SQLite Fact Constellation warehouse with zero orphan foreign keys.

Association rule mining identified 935 high-lift co-viewing patterns, text mining classified plot synopses, Random Forest predicted movie rating tiers with 63.01% accuracy, Lasso regression isolated critical genre weights, and ARIMA(3,1,3) forecasted monthly platform additions with a 22.47% error reduction over baseline persistence. The hybrid recommender and responsive Flask web application demonstrate how multidimensional warehousing and core data mining algorithms unite into an explainable, production-ready system.

---

## 14. REFERENCES

1. Agrawal, R., Imieliński, T., & Swami, A. (1993). *Mining Association Rules between Sets of Items in Large Databases*. Proceedings of the ACM SIGMOD International Conference on Management of Data, 207–216.
2. Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). John Wiley & Sons.
3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
4. Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), 1–19.
5. Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso*. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267–288.
6. Course Syllabus: *CS3203T-A – Data Mining and Analytics*, Savitribai Phule Pune University / RSCOE Autonomous Curriculum.

---
*Generated for submission at Rajarshi Shahu College of Engineering, Tathawade, Pune.*

