from __future__ import annotations

import pandas as pd
import numpy as np

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

def test_compute_quality_flags_constant_column():
    n_rows = 10
    data = {
        'ConstantCol': [10] * n_rows,
        'B': list(range(n_rows))
    }
    df = pd.DataFrame(data)

    summary = summarize_dataset(df)
    missing = missing_table(df)

    flags = compute_quality_flags(summary, missing)

    assert flags['has_constant_columns'] is True

    expected_score = 1.0 - 0.2 - 0.15
    assert np.isclose(flags['quality_score'], expected_score)


def test_compute_quality_flags_high_cardinality():
    n_unique_values = 6 
    data = {
        'CardID': [f'ID_{i}' for i in range(1, n_unique_values + 1)],
        'Group': ['A'] * n_unique_values
    }
    df = pd.DataFrame(data)

    summary = summarize_dataset(df)
    missing = missing_table(df)

    flags = compute_quality_flags(summary, missing)

    assert flags['has_high_cardinality_categoricals'] is True
    assert flags['has_constant_columns'] is True

    expected_score = 1.0 - 0.2 - 0.15 - 0.05
    assert np.isclose(flags['quality_score'], expected_score)