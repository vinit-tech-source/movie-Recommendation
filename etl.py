"""
ETL Pipeline and Star Schema / Fact Constellation Warehouse Builder.

Course Outcome: CO1 - Understand, evaluate and apply data preprocessing,
data transformation, and multidimensional data modeling techniques (ETL,
Star Schema, Fact Constellation).

This module:
1. Auto-discovers input CSV datasets (MovieLens and Netflix).
2. Cleanses and transforms data (handling nulls, dates, regex year extraction,
   genre splitting, duration parsing, rating categorization).
3. Constructs an analytical SQLite Data Warehouse (warehouse.db) using a
   conformed Fact Constellation schema with composite keys, foreign keys,
   and performance indexes.
4. Produces a comprehensive Data Quality and Audit Report.
5. Executes and displays 5 sanity OLAP queries.
"""

import re
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd


def locate_data_files(base_data_dir: Path) -> dict[str, Path]:
    """Auto-detects data file paths handling nested zip/folder extraction structures."""
    files = {}

    # Find ratings.csv and locate its parent folder for MovieLens files
    ratings_candidates = [
        p for p in base_data_dir.rglob("ratings.csv") if p.is_file()
    ]
    if not ratings_candidates:
        raise FileNotFoundError(f"Could not find ratings.csv in {base_data_dir}")
    ml_dir = ratings_candidates[0].parent
    files["ratings"] = ml_dir / "ratings.csv"
    files["movies"] = ml_dir / "movies.csv"
    files["tags"] = ml_dir / "tags.csv"
    files["links"] = ml_dir / "links.csv"

    # Find netflix_titles.csv
    netflix_candidates = [
        p for p in base_data_dir.rglob("netflix_titles.csv") if p.is_file()
    ]
    if not netflix_candidates:
        raise FileNotFoundError(f"Could not find netflix_titles.csv in {base_data_dir}")
    files["netflix"] = netflix_candidates[0]

    # Optional TMDB 5000 movies file
    tmdb_candidates = [
        p for p in base_data_dir.rglob("tmdb_5000_movies.csv") if p.is_file()
    ]
    files["tmdb"] = tmdb_candidates[0] if tmdb_candidates else None

    return files


def extract_year_and_clean_title(title: str) -> tuple[str, int | None]:
    """Extract release year with regex and return (clean_title, year).

    13 MovieLens titles have no (year) -> returns (clean_title, None).
    """
    if not isinstance(title, str):
        return str(title), None
    match = re.search(r"\((\d{4})\)\s*$", title.strip())
    if match:
        year = int(match.group(1))
        clean_title = title[: match.start()].strip()
        return clean_title, year
    return title.strip(), None


def parse_duration(val: str | None) -> tuple[int | None, str]:
    """Parse Netflix duration like '90 min' or '2 Seasons' into (value, unit)."""
    if not isinstance(val, str) or not val.strip():
        return None, "Unknown"
    val = val.strip()
    match = re.match(r"^(\d+)\s*(.*)$", val)
    if match:
        return int(match.group(1)), match.group(2).strip() or "Unknown"
    return None, "Unknown"


def create_schema(conn: sqlite3.Connection) -> None:
    """Creates the conformed Fact Constellation schema in SQLite with FKs and indexes."""
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = OFF;")

    # Drop existing tables for complete idempotence
    tables = [
        "fact_titles",
        "title_genre",
        "dim_netflix_genre",
        "dim_title",
        "fact_tags",
        "fact_ratings",
        "movie_genre",
        "dim_genre",
        "dim_movie",
        "dim_user",
        "dim_date",
    ]
    for tbl in tables:
        cur.execute(f"DROP TABLE IF EXISTS {tbl};")

    schema_sql = """
    -- Conformed Dimension: Date
    CREATE TABLE dim_date (
        date_id INTEGER PRIMARY KEY,
        full_date TEXT NOT NULL,
        day INTEGER NOT NULL,
        month INTEGER NOT NULL,
        month_name TEXT NOT NULL,
        quarter INTEGER NOT NULL,
        year INTEGER NOT NULL
    );

    -- MovieLens Dimension: User
    CREATE TABLE dim_user (
        user_id INTEGER PRIMARY KEY
    );

    -- MovieLens Dimension: Movie
    CREATE TABLE dim_movie (
        movie_id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        clean_title TEXT NOT NULL,
        year INTEGER,
        imdb_id INTEGER,
        tmdb_id INTEGER
    );

    -- MovieLens Dimension: Genre & Bridge
    CREATE TABLE dim_genre (
        genre_id INTEGER PRIMARY KEY AUTOINCREMENT,
        genre_name TEXT UNIQUE NOT NULL
    );

    CREATE TABLE movie_genre (
        movie_id INTEGER NOT NULL,
        genre_id INTEGER NOT NULL,
        PRIMARY KEY (movie_id, genre_id),
        FOREIGN KEY (movie_id) REFERENCES dim_movie(movie_id) ON DELETE CASCADE,
        FOREIGN KEY (genre_id) REFERENCES dim_genre(genre_id) ON DELETE CASCADE
    );

    -- MovieLens Fact: Ratings
    CREATE TABLE fact_ratings (
        rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        movie_id INTEGER NOT NULL,
        date_id INTEGER NOT NULL,
        rating REAL NOT NULL,
        rating_class TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES dim_user(user_id),
        FOREIGN KEY (movie_id) REFERENCES dim_movie(movie_id),
        FOREIGN KEY (date_id) REFERENCES dim_date(date_id)
    );

    -- MovieLens Fact: Tags (used in content-based recommendations)
    CREATE TABLE fact_tags (
        user_id INTEGER NOT NULL,
        movie_id INTEGER NOT NULL,
        date_id INTEGER NOT NULL,
        tag TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES dim_user(user_id),
        FOREIGN KEY (movie_id) REFERENCES dim_movie(movie_id),
        FOREIGN KEY (date_id) REFERENCES dim_date(date_id)
    );

    -- Netflix Dimension: Title
    CREATE TABLE dim_title (
        show_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        type TEXT NOT NULL,
        rating TEXT NOT NULL,
        duration_value INTEGER,
        duration_unit TEXT NOT NULL,
        country_primary TEXT NOT NULL,
        description TEXT NOT NULL,
        release_year INTEGER NOT NULL
    );

    -- Netflix Dimension: Genre & Bridge
    CREATE TABLE dim_netflix_genre (
        ngenre_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL
    );

    CREATE TABLE title_genre (
        show_id TEXT NOT NULL,
        ngenre_id INTEGER NOT NULL,
        PRIMARY KEY (show_id, ngenre_id),
        FOREIGN KEY (show_id) REFERENCES dim_title(show_id) ON DELETE CASCADE,
        FOREIGN KEY (ngenre_id) REFERENCES dim_netflix_genre(ngenre_id) ON DELETE CASCADE
    );

    -- Netflix Fact: Titles (conformed on date_id)
    CREATE TABLE fact_titles (
        show_id TEXT NOT NULL,
        date_id INTEGER NOT NULL,
        release_year INTEGER NOT NULL,
        duration_value INTEGER,
        FOREIGN KEY (show_id) REFERENCES dim_title(show_id),
        FOREIGN KEY (date_id) REFERENCES dim_date(date_id)
    );

    -- Performance Indexes on all join keys
    CREATE INDEX idx_ratings_user ON fact_ratings(user_id);
    CREATE INDEX idx_ratings_movie ON fact_ratings(movie_id);
    CREATE INDEX idx_ratings_date ON fact_ratings(date_id);
    CREATE INDEX idx_ratings_rating ON fact_ratings(rating);
    CREATE INDEX idx_movie_genre_genre ON movie_genre(genre_id);
    CREATE INDEX idx_tags_movie ON fact_tags(movie_id);
    CREATE INDEX idx_tags_user ON fact_tags(user_id);
    CREATE INDEX idx_fact_titles_date ON fact_titles(date_id);
    CREATE INDEX idx_fact_titles_show ON fact_titles(show_id);
    CREATE INDEX idx_title_genre_genre ON title_genre(ngenre_id);
    CREATE INDEX idx_dim_movie_title ON dim_movie(clean_title);
    """
    cur.executescript(schema_sql)
    conn.commit()
    cur.execute("PRAGMA foreign_keys = ON;")


def run_etl(db_path: Path = Path("warehouse.db"), data_dir: Path = Path("data")) -> dict:
    """Executes the complete ETL pipeline and returns metrics."""
    print("=" * 70)
    print("STARTING ETL PIPELINE (CO1: Fact Constellation Warehouse Builder)")
    print("=" * 70)

    files = locate_data_files(data_dir)
    print(f"MovieLens source: {files['ratings'].parent}")
    print(f"Netflix source:   {files['netflix']}")
    if files["tmdb"]:
        print(f"TMDB source:      {files['tmdb']}")
    else:
        print("TMDB source:      Not present (fallback will be used for Module C)")

    # 1. Read Raw CSVs
    df_ratings_raw = pd.read_csv(files["ratings"])
    df_movies_raw = pd.read_csv(files["movies"])
    df_tags_raw = pd.read_csv(files["tags"])
    df_links_raw = pd.read_csv(files["links"])
    df_netflix_raw = pd.read_csv(files["netflix"])

    quality_report = {
        "ratings_raw_rows": len(df_ratings_raw),
        "movies_raw_rows": len(df_movies_raw),
        "tags_raw_rows": len(df_tags_raw),
        "links_raw_rows": len(df_links_raw),
        "netflix_raw_rows": len(df_netflix_raw),
        "ratings_dropped": 0,
        "movies_dropped": 0,
        "netflix_dropped": 0,
    }

    # 2. MovieLens Transformation
    # Clean movies & extract years
    parsed_titles = [extract_year_and_clean_title(t) for t in df_movies_raw["title"]]
    df_movies = df_movies_raw.copy()
    df_movies["clean_title"] = [p[0] for p in parsed_titles]
    df_movies["year"] = [p[1] for p in parsed_titles]

    # Join links
    df_links = df_links_raw.copy()
    # tmdbId may have NaN, handle safely
    df_links["imdbId"] = df_links["imdbId"].astype("Int64")
    df_links["tmdbId"] = df_links["tmdbId"].astype("Int64")
    df_movies = df_movies.merge(df_links, on="movieId", how="left")

    # Ratings clean
    df_ratings = df_ratings_raw.dropna(subset=["userId", "movieId", "rating", "timestamp"]).copy()
    df_ratings = df_ratings.drop_duplicates(subset=["userId", "movieId"])
    quality_report["ratings_dropped"] = len(df_ratings_raw) - len(df_ratings)

    # Date transformation for ratings
    dt_ratings = pd.to_datetime(df_ratings["timestamp"], unit="s", utc=True)
    df_ratings["full_date"] = dt_ratings.dt.strftime("%Y-%m-%d")
    df_ratings["date_id"] = dt_ratings.dt.strftime("%Y%m%d").astype(int)

    # Rating categorization
    def categorize_rating(r: float) -> str:
        if r < 3.0:
            return "Low"
        elif r < 4.0:
            return "Medium"
        return "High"

    df_ratings["rating_class"] = df_ratings["rating"].apply(categorize_rating)

    # Tags clean
    df_tags = df_tags_raw.dropna(subset=["userId", "movieId", "tag"]).copy()
    dt_tags = pd.to_datetime(df_tags["timestamp"], unit="s", utc=True)
    df_tags["full_date"] = dt_tags.dt.strftime("%Y-%m-%d")
    df_tags["date_id"] = dt_tags.dt.strftime("%Y%m%d").astype(int)
    df_tags["tag"] = df_tags["tag"].astype(str).str.strip().str.lower()

    # 3. Netflix Transformation
    df_netflix = df_netflix_raw.copy()
    df_netflix["director"] = df_netflix["director"].fillna("Unknown").str.strip()
    df_netflix["cast"] = df_netflix["cast"].fillna("Unknown").str.strip()
    df_netflix["country"] = df_netflix["country"].fillna("Unknown").str.strip()
    df_netflix["country_primary"] = df_netflix["country"].apply(
        lambda c: c.split(",")[0].strip() if c != "Unknown" else "Unknown"
    )
    df_netflix["rating"] = df_netflix["rating"].fillna("Unknown").str.strip()
    df_netflix["description"] = df_netflix["description"].fillna("").str.strip()
    df_netflix["title"] = df_netflix["title"].fillna("").str.strip()
    df_netflix["type"] = df_netflix["type"].fillna("Movie").str.strip()

    # Duration parsing
    parsed_dur = [parse_duration(d) for d in df_netflix["duration"]]
    df_netflix["duration_value"] = [p[0] for p in parsed_dur]
    df_netflix["duration_unit"] = [p[1] for p in parsed_dur]

    # Netflix date_added parsing (strip leading whitespace, coerce nulls)
    df_netflix["date_added_clean"] = df_netflix["date_added"].astype(str).str.strip()
    df_netflix["parsed_date"] = pd.to_datetime(df_netflix["date_added_clean"], errors="coerce")
    df_netflix["date_id"] = df_netflix["parsed_date"].dt.strftime("%Y%m%d")
    # Where parsed_date is NaT, date_id is NaN

    # 4. Build Conformed dim_date
    # Collect all unique dates across ratings, tags, and netflix
    dates_from_ratings = pd.DataFrame({"full_date": df_ratings["full_date"].unique()})
    dates_from_tags = pd.DataFrame({"full_date": df_tags["full_date"].unique()})
    valid_netflix_dates = df_netflix["parsed_date"].dropna()
    dates_from_netflix = pd.DataFrame({
        "full_date": valid_netflix_dates.dt.strftime("%Y-%m-%d").unique()
    })

    all_dates = pd.concat([dates_from_ratings, dates_from_tags, dates_from_netflix], ignore_index=True)
    all_dates = all_dates.drop_duplicates().dropna()
    all_dates["dt"] = pd.to_datetime(all_dates["full_date"])
    all_dates = all_dates.sort_values("dt").reset_index(drop=True)

    df_dim_date = pd.DataFrame({
        "date_id": all_dates["dt"].dt.strftime("%Y%m%d").astype(int),
        "full_date": all_dates["dt"].dt.strftime("%Y-%m-%d"),
        "day": all_dates["dt"].dt.day,
        "month": all_dates["dt"].dt.month,
        "month_name": all_dates["dt"].dt.strftime("%B"),
        "quarter": all_dates["dt"].dt.quarter,
        "year": all_dates["dt"].dt.year,
    })

    # 5. Connect to SQLite and Load Tables
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    create_schema(conn)

    print("\n--- Loading Conformed Dimensions & Fact Tables ---")

    # Load dim_date
    df_dim_date.to_sql("dim_date", conn, if_exists="append", index=False)
    print(f"Loaded dim_date:            {len(df_dim_date):,d} dates")

    # Load dim_user
    users = pd.DataFrame({"user_id": sorted(df_ratings["userId"].unique())})
    users.to_sql("dim_user", conn, if_exists="append", index=False)
    print(f"Loaded dim_user:            {len(users):,d} users")

    # Load dim_movie
    movie_cols = ["movieId", "title", "clean_title", "year", "imdbId", "tmdbId"]
    dim_movie = df_movies[movie_cols].rename(
        columns={"movieId": "movie_id", "imdbId": "imdb_id", "tmdbId": "tmdb_id"}
    )
    dim_movie.to_sql("dim_movie", conn, if_exists="append", index=False)
    print(f"Loaded dim_movie:           {len(dim_movie):,d} movies")

    # Load dim_genre and movie_genre bridge
    genre_set = set()
    movie_genre_pairs = []
    for _, row in df_movies.iterrows():
        mid = row["movieId"]
        genre_str = str(row["genres"])
        if genre_str == "(no genres listed)":
            continue
        for g in genre_str.split("|"):
            g_clean = g.strip()
            if g_clean:
                genre_set.add(g_clean)
                movie_genre_pairs.append((mid, g_clean))

    genre_map = {name: i + 1 for i, name in enumerate(sorted(genre_set))}
    dim_genre = pd.DataFrame([{"genre_id": gid, "genre_name": gname} for gname, gid in genre_map.items()])
    dim_genre.to_sql("dim_genre", conn, if_exists="append", index=False)
    print(f"Loaded dim_genre:           {len(dim_genre):,d} unique genres")

    movie_genre_df = pd.DataFrame([
        {"movie_id": mid, "genre_id": genre_map[gname]}
        for mid, gname in movie_genre_pairs
    ]).drop_duplicates()
    movie_genre_df.to_sql("movie_genre", conn, if_exists="append", index=False)
    print(f"Loaded movie_genre bridge:  {len(movie_genre_df):,d} mappings")

    # Load fact_ratings
    ratings_to_load = df_ratings[["userId", "movieId", "date_id", "rating", "rating_class"]].rename(
        columns={"userId": "user_id", "movieId": "movie_id"}
    )
    ratings_to_load.to_sql("fact_ratings", conn, if_exists="append", index=False)
    print(f"Loaded fact_ratings:        {len(ratings_to_load):,d} ratings")

    # Load fact_tags
    tags_to_load = df_tags[["userId", "movieId", "date_id", "tag"]].rename(
        columns={"userId": "user_id", "movieId": "movie_id"}
    )
    tags_to_load.to_sql("fact_tags", conn, if_exists="append", index=False)
    print(f"Loaded fact_tags:           {len(tags_to_load):,d} tags")

    # Load dim_title (Netflix)
    title_cols = [
        "show_id", "title", "type", "rating", "duration_value",
        "duration_unit", "country_primary", "description", "release_year"
    ]
    dim_title = df_netflix[title_cols]
    dim_title.to_sql("dim_title", conn, if_exists="append", index=False)
    print(f"Loaded dim_title (Netflix): {len(dim_title):,d} titles")

    # Load dim_netflix_genre and title_genre bridge
    ngenre_set = set()
    title_genre_pairs = []
    for _, row in df_netflix.iterrows():
        sid = row["show_id"]
        cats = str(row["listed_in"]) if pd.notna(row["listed_in"]) else ""
        for c in cats.split(","):
            c_clean = c.strip()
            if c_clean:
                ngenre_set.add(c_clean)
                title_genre_pairs.append((sid, c_clean))

    ngenre_map = {name: i + 1 for i, name in enumerate(sorted(ngenre_set))}
    dim_ngenre = pd.DataFrame([{"ngenre_id": gid, "name": gname} for gname, gid in ngenre_map.items()])
    dim_ngenre.to_sql("dim_netflix_genre", conn, if_exists="append", index=False)
    print(f"Loaded dim_netflix_genre:   {len(dim_ngenre):,d} categories")

    title_genre_df = pd.DataFrame([
        {"show_id": sid, "ngenre_id": ngenre_map[gname]}
        for sid, gname in title_genre_pairs
    ]).drop_duplicates()
    title_genre_df.to_sql("title_genre", conn, if_exists="append", index=False)
    print(f"Loaded title_genre bridge:  {len(title_genre_df):,d} mappings")

    # Load fact_titles (Only titles with valid date_added)
    valid_netflix_facts = df_netflix.dropna(subset=["date_id"]).copy()
    valid_netflix_facts["date_id"] = valid_netflix_facts["date_id"].astype(int)
    fact_titles = valid_netflix_facts[["show_id", "date_id", "release_year", "duration_value"]]
    fact_titles.to_sql("fact_titles", conn, if_exists="append", index=False)
    print(f"Loaded fact_titles:         {len(fact_titles):,d} titles (with valid dates)")
    print(f"Netflix titles without date:{len(df_netflix) - len(fact_titles)} titles preserved in dim_title with date=NULL")

    # 6. Sanity OLAP Queries
    print("\n" + "=" * 70)
    print("SANITY OLAP QUERIES (Verifying Data Constellation Schema)")
    print("=" * 70)

    # OLAP Query 1: Ratings count per year
    print("\n[OLAP 1] Ratings Per Year (Sample 5 Recent Years):")
    q1 = """
    SELECT d.year, COUNT(f.rating_id) AS total_ratings, ROUND(AVG(f.rating), 3) AS avg_rating
    FROM fact_ratings f
    JOIN dim_date d ON f.date_id = d.date_id
    GROUP BY d.year
    ORDER BY d.year DESC
    LIMIT 5;
    """
    print(pd.read_sql_query(q1, conn).to_string(index=False))

    # OLAP Query 2: Top 10 Genres by Rating Count
    print("\n[OLAP 2] Top 10 Genres by Total Rating Count:")
    q2 = """
    SELECT g.genre_name, COUNT(f.rating_id) AS rating_count
    FROM fact_ratings f
    JOIN movie_genre mg ON f.movie_id = mg.movie_id
    JOIN dim_genre g ON mg.genre_id = g.genre_id
    GROUP BY g.genre_name
    ORDER BY rating_count DESC
    LIMIT 10;
    """
    print(pd.read_sql_query(q2, conn).to_string(index=False))

    # OLAP Query 3: Average Rating Per Genre
    print("\n[OLAP 3] Average Rating Per Genre (Top 8):")
    q3 = """
    SELECT g.genre_name, ROUND(AVG(f.rating), 3) AS avg_rating, COUNT(f.rating_id) AS n_ratings
    FROM fact_ratings f
    JOIN movie_genre mg ON f.movie_id = mg.movie_id
    JOIN dim_genre g ON mg.genre_id = g.genre_id
    GROUP BY g.genre_name
    HAVING COUNT(f.rating_id) >= 1000
    ORDER BY avg_rating DESC
    LIMIT 8;
    """
    print(pd.read_sql_query(q3, conn).to_string(index=False))

    # OLAP Query 4: Netflix Titles Added Per Year by Type
    print("\n[OLAP 4] Netflix Titles Added Per Year by Type (Recent 6 Years):")
    q4 = """
    SELECT d.year, t.type, COUNT(ft.show_id) AS titles_added
    FROM fact_titles ft
    JOIN dim_title t ON ft.show_id = t.show_id
    JOIN dim_date d ON ft.date_id = d.date_id
    WHERE d.year >= 2016
    GROUP BY d.year, t.type
    ORDER BY d.year DESC, t.type;
    """
    print(pd.read_sql_query(q4, conn).to_string(index=False))

    # OLAP Query 5: Summary Rows Per Fact Table & Dimensions
    print("\n[OLAP 5] Warehouse Row Count Audit:")
    q5_tables = [
        "dim_date", "dim_user", "dim_movie", "dim_genre", "movie_genre",
        "fact_ratings", "fact_tags", "dim_title", "dim_netflix_genre",
        "title_genre", "fact_titles"
    ]
    counts = []
    for tbl in q5_tables:
        c = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        counts.append({"table_name": tbl, "row_count": c})
    print(pd.DataFrame(counts).to_string(index=False))

    conn.close()
    print("\nETL Pipeline completed successfully! warehouse.db is ready.")
    print("=" * 70)

    quality_report.update({
        "dim_date": len(df_dim_date),
        "dim_user": len(users),
        "dim_movie": len(dim_movie),
        "fact_ratings": len(ratings_to_load),
        "fact_tags": len(tags_to_load),
        "dim_title": len(dim_title),
        "fact_titles": len(fact_titles),
    })
    return quality_report


if __name__ == "__main__":
    run_etl()

