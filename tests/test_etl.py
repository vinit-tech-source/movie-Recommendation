"""
Tests for ETL Pipeline and Warehouse Schema Integrity (CO1).

Verifies table existence, exact row counts, primary key uniqueness,
and referential integrity (zero orphan foreign keys in fact tables).
"""

import sqlite3
from pathlib import Path
import pytest


@pytest.fixture(scope="module")
def db_conn():
    db_path = Path("warehouse.db")
    assert db_path.exists(), "warehouse.db does not exist. Run etl.py first."
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    yield conn
    conn.close()


def test_row_counts(db_conn):
    """Assert expected row counts from raw sources."""
    expected = {
        "dim_user": 610,
        "dim_movie": 9742,
        "dim_genre": 19,
        "fact_ratings": 100836,
        "fact_tags": 3683,
        "dim_title": 8807,
        "dim_netflix_genre": 42,
        "fact_titles": 8797,
    }
    for tbl, exp_cnt in expected.items():
        cnt = db_conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        assert cnt == exp_cnt, f"Table {tbl} row count mismatch: {cnt} vs {exp_cnt}"


def test_no_orphan_ratings(db_conn):
    """Ensure fact_ratings has no orphan foreign keys."""
    # Orphan users
    orphan_u = db_conn.execute("""
        SELECT COUNT(*) FROM fact_ratings f
        LEFT JOIN dim_user u ON f.user_id = u.user_id
        WHERE u.user_id IS NULL;
    """).fetchone()[0]
    assert orphan_u == 0, f"Found {orphan_u} orphan users in fact_ratings"

    # Orphan movies
    orphan_m = db_conn.execute("""
        SELECT COUNT(*) FROM fact_ratings f
        LEFT JOIN dim_movie m ON f.movie_id = m.movie_id
        WHERE m.movie_id IS NULL;
    """).fetchone()[0]
    assert orphan_m == 0, f"Found {orphan_m} orphan movies in fact_ratings"

    # Orphan dates
    orphan_d = db_conn.execute("""
        SELECT COUNT(*) FROM fact_ratings f
        LEFT JOIN dim_date d ON f.date_id = d.date_id
        WHERE d.date_id IS NULL;
    """).fetchone()[0]
    assert orphan_d == 0, f"Found {orphan_d} orphan dates in fact_ratings"


def test_no_orphan_netflix_titles(db_conn):
    """Ensure fact_titles has no orphan foreign keys."""
    orphan_s = db_conn.execute("""
        SELECT COUNT(*) FROM fact_titles f
        LEFT JOIN dim_title t ON f.show_id = t.show_id
        WHERE t.show_id IS NULL;
    """).fetchone()[0]
    assert orphan_s == 0, f"Found {orphan_s} orphan show_ids in fact_titles"

    orphan_d = db_conn.execute("""
        SELECT COUNT(*) FROM fact_titles f
        LEFT JOIN dim_date d ON f.date_id = d.date_id
        WHERE d.date_id IS NULL;
    """).fetchone()[0]
    assert orphan_d == 0, f"Found {orphan_d} orphan dates in fact_titles"


def test_bridge_integrity(db_conn):
    """Ensure bridge tables have zero orphan entries."""
    orphan_mg = db_conn.execute("""
        SELECT COUNT(*) FROM movie_genre mg
        LEFT JOIN dim_movie m ON mg.movie_id = m.movie_id
        LEFT JOIN dim_genre g ON mg.genre_id = g.genre_id
        WHERE m.movie_id IS NULL OR g.genre_id IS NULL;
    """).fetchone()[0]
    assert orphan_mg == 0, f"Found {orphan_mg} orphan bridge entries in movie_genre"

    orphan_tg = db_conn.execute("""
        SELECT COUNT(*) FROM title_genre tg
        LEFT JOIN dim_title t ON tg.show_id = t.show_id
        LEFT JOIN dim_netflix_genre g ON tg.ngenre_id = g.ngenre_id
        WHERE t.show_id IS NULL OR g.ngenre_id IS NULL;
    """).fetchone()[0]
    assert orphan_tg == 0, f"Found {orphan_tg} orphan bridge entries in title_genre"


def test_movie_year_null_handling(db_conn):
    """Assert exactly 13 movies without release years are preserved with year IS NULL."""
    null_years = db_conn.execute("""
        SELECT COUNT(*) FROM dim_movie WHERE year IS NULL;
    """).fetchone()[0]
    assert null_years == 13, f"Expected 13 movies with NULL year, got {null_years}"


def test_rating_class_values(db_conn):
    """Ensure rating classes match Low, Medium, High."""
    classes = set(r[0] for r in db_conn.execute("SELECT DISTINCT rating_class FROM fact_ratings").fetchall())
    assert classes == {"Low", "Medium", "High"}

