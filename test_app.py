"""
Unit tests for app.py

Run with:  pytest test_app.py -v
"""
import numpy as np
import pandas as pd
import pytest

import app
from conftest import MOCK_DF, N, EMBED_DIM


# ── CONSTANTS ────────────────────────────────────────────────────

RESULT_COLS = ["Rank", "id", "title", "main_category", "score", "arXiv Link"]
ALL_METHODS = ["TF-IDF", "BM25", "SBERT + FAISS", "Compare All Three"]


# ── HELPERS ──────────────────────────────────────────────────────

def assert_valid_result(df: pd.DataFrame, expected_rows: int = app.TOP_K) -> None:
    """Assert a result DataFrame has the correct structure and row count."""
    assert list(df.columns) == RESULT_COLS
    assert len(df) == expected_rows
    assert (df["Rank"] == range(1, expected_rows + 1)).all()
    assert df["arXiv Link"].str.startswith("https://arxiv.org/abs/").all()
    assert df["score"].between(-1.1, 1.1).all()  # cosine / L2 similarity range


def assert_empty_result(df: pd.DataFrame) -> None:
    """Assert a DataFrame is the standard empty result (correct columns, zero rows)."""
    assert list(df.columns) == RESULT_COLS
    assert len(df) == 0


def assert_error_result(df: pd.DataFrame) -> None:
    """Assert a DataFrame is an error result."""
    assert "Error" in df.columns
    assert len(df) == 1


# ── _validate_query ───────────────────────────────────────────────

class TestValidateQuery:
    def test_empty_string(self):
        assert app._validate_query("") is not None

    def test_whitespace_only(self):
        assert app._validate_query("   ") is not None

    def test_none_like_empty(self):
        assert app._validate_query("") is not None

    def test_valid_query(self):
        assert app._validate_query("deep learning") is None

    def test_single_word(self):
        assert app._validate_query("transformer") is None

    def test_query_at_max_length(self):
        assert app._validate_query("a" * app.MAX_QUERY_LENGTH) is None

    def test_query_exceeds_max_length(self):
        result = app._validate_query("a" * (app.MAX_QUERY_LENGTH + 1))
        assert result is not None
        assert str(app.MAX_QUERY_LENGTH) in result

    def test_returns_string_on_invalid(self):
        assert isinstance(app._validate_query(""), str)

    def test_returns_none_on_valid(self):
        assert app._validate_query("neural networks") is None


# ── _empty_result ─────────────────────────────────────────────────

class TestEmptyResult:
    def test_columns(self):
        result = app._empty_result()
        assert list(result.columns) == RESULT_COLS

    def test_no_rows(self):
        assert len(app._empty_result()) == 0

    def test_returns_dataframe(self):
        assert isinstance(app._empty_result(), pd.DataFrame)


# ── _error_result ─────────────────────────────────────────────────

class TestErrorResult:
    def test_has_error_column(self):
        result = app._error_result("something went wrong")
        assert "Error" in result.columns

    def test_message_preserved(self):
        msg = "test error message"
        result = app._error_result(msg)
        assert result["Error"].iloc[0] == msg

    def test_single_row(self):
        assert len(app._error_result("oops")) == 1


# ── _build_result_df ──────────────────────────────────────────────

class TestBuildResultDf:
    def test_output_columns(self):
        result = app._build_result_df([0, 1, 2], np.array([0.9, 0.8, 0.7]))
        assert list(result.columns) == RESULT_COLS

    def test_rank_starts_at_one(self):
        result = app._build_result_df([0, 1, 2], np.array([0.9, 0.8, 0.7]))
        assert result["Rank"].tolist() == [1, 2, 3]

    def test_scores_rounded_to_4dp(self):
        result = app._build_result_df([0], np.array([0.123456789]))
        assert result["score"].iloc[0] == 0.1235

    def test_arxiv_link_format(self):
        result = app._build_result_df([0], np.array([0.9]))
        assert result["arXiv Link"].iloc[0] == "https://arxiv.org/abs/2101.00001"

    def test_correct_paper_selected(self):
        result = app._build_result_df([2], np.array([0.5]))
        assert result["title"].iloc[0] == MOCK_DF.iloc[2]["title"]

    def test_index_reset(self):
        result = app._build_result_df([3, 4, 5], np.array([0.7, 0.6, 0.5]))
        assert result.index.tolist() == [0, 1, 2]

    def test_row_count_matches_indices(self):
        indices = [0, 1, 2, 3]
        result = app._build_result_df(indices, np.array([0.9, 0.8, 0.7, 0.6]))
        assert len(result) == len(indices)


# ── tfidf_search ──────────────────────────────────────────────────

class TestTfidfSearch:
    def test_returns_dataframe(self):
        assert isinstance(app.tfidf_search("deep learning"), pd.DataFrame)

    def test_result_structure(self):
        result = app.tfidf_search("deep learning")
        assert list(result.columns) == RESULT_COLS

    def test_default_top_k(self):
        result = app.tfidf_search("deep learning")
        assert len(result) == app.TOP_K

    def test_custom_top_k(self):
        result = app.tfidf_search("deep learning", top_k=3)
        assert len(result) == 3

    def test_query_lowercased(self):
        app.tfidf.transform.reset_mock()
        app.tfidf_search("Deep Learning")
        call_args = app.tfidf.transform.call_args[0][0]
        assert call_args == ["deep learning"]

    def test_ranks_are_sequential(self):
        result = app.tfidf_search("neural networks")
        assert result["Rank"].tolist() == list(range(1, app.TOP_K + 1))


# ── bm25_search ───────────────────────────────────────────────────

class TestBm25Search:
    def test_returns_dataframe(self):
        assert isinstance(app.bm25_search("information retrieval"), pd.DataFrame)

    def test_result_structure(self):
        result = app.bm25_search("information retrieval")
        assert list(result.columns) == RESULT_COLS

    def test_default_top_k(self):
        result = app.bm25_search("information retrieval")
        assert len(result) == app.TOP_K

    def test_custom_top_k(self):
        result = app.bm25_search("text search", top_k=2)
        assert len(result) == 2

    def test_query_tokenized_and_lowercased(self):
        app.bm25.get_scores.reset_mock()
        app.bm25_search("Deep Neural Network")
        tokens = app.bm25.get_scores.call_args[0][0]
        assert tokens == ["deep", "neural", "network"]

    def test_results_ordered_by_score_descending(self):
        result = app.bm25_search("machine learning")
        scores = result["score"].tolist()
        assert scores == sorted(scores, reverse=True)


# ── faiss_search ──────────────────────────────────────────────────

class TestFaissSearch:
    def test_returns_dataframe(self):
        assert isinstance(app.faiss_search("semantic similarity"), pd.DataFrame)

    def test_result_structure(self):
        result = app.faiss_search("semantic similarity")
        assert list(result.columns) == RESULT_COLS

    def test_default_top_k(self):
        result = app.faiss_search("semantic similarity")
        assert len(result) == app.TOP_K

    def test_custom_top_k(self):
        result = app.faiss_search("transformers", top_k=3)
        assert len(result) == 3

    def test_query_lowercased_before_encoding(self):
        app.sbert.encode.reset_mock()
        app.faiss_search("BERT Language Model")
        call_args = app.sbert.encode.call_args[0][0]
        assert call_args == ["bert language model"]

    def test_fetches_top_k_plus_one_from_index(self):
        """FAISS fetches top_k+1 to allow dropping the source paper in title search."""
        app.index.search.reset_mock()
        app.faiss_search("deep learning", top_k=5)
        _, k = app.index.search.call_args[0][1], app.index.search.call_args[0][1]
        assert k == app.TOP_K + 1


# ── run_search ────────────────────────────────────────────────────

class TestRunSearch:
    def test_empty_query_returns_three_empty_dfs(self):
        r1, r2, r3 = app.run_search("", "TF-IDF")
        assert_empty_result(r1)
        assert_empty_result(r2)
        assert_empty_result(r3)

    def test_whitespace_query_returns_three_empty_dfs(self):
        r1, r2, r3 = app.run_search("   ", "BM25")
        assert_empty_result(r1)
        assert_empty_result(r2)
        assert_empty_result(r3)

    def test_tfidf_method_populates_only_first_output(self):
        r1, r2, r3 = app.run_search("deep learning", "TF-IDF")
        assert_valid_result(r1)
        assert_empty_result(r2)
        assert_empty_result(r3)

    def test_bm25_method_populates_only_second_output(self):
        r1, r2, r3 = app.run_search("deep learning", "BM25")
        assert_empty_result(r1)
        assert_valid_result(r2)
        assert_empty_result(r3)

    def test_faiss_method_populates_only_third_output(self):
        r1, r2, r3 = app.run_search("deep learning", "SBERT + FAISS")
        assert_empty_result(r1)
        assert_empty_result(r2)
        assert_valid_result(r3)

    def test_compare_all_populates_all_three_outputs(self):
        r1, r2, r3 = app.run_search("deep learning", "Compare All Three")
        assert_valid_result(r1)
        assert_valid_result(r2)
        assert_valid_result(r3)

    def test_returns_tuple_of_three(self):
        result = app.run_search("neural networks", "TF-IDF")
        assert len(result) == 3

    def test_all_outputs_are_dataframes(self):
        for method in ALL_METHODS:
            outputs = app.run_search("machine learning", method)
            assert all(isinstance(o, pd.DataFrame) for o in outputs)

    def test_exception_returns_error_in_all_three(self, monkeypatch):
        monkeypatch.setattr(app, "tfidf_search", lambda *_: (_ for _ in ()).throw(RuntimeError("boom")))
        r1, r2, r3 = app.run_search("deep learning", "TF-IDF")
        assert_error_result(r1)
        assert_error_result(r2)
        assert_error_result(r3)

    def test_query_at_max_length_is_accepted(self):
        r1, r2, r3 = app.run_search("a" * app.MAX_QUERY_LENGTH, "TF-IDF")
        assert_valid_result(r1)

    def test_query_over_max_length_returns_empty(self):
        r1, r2, r3 = app.run_search("a" * (app.MAX_QUERY_LENGTH + 1), "TF-IDF")
        assert_empty_result(r1)
        assert_empty_result(r2)
        assert_empty_result(r3)


# ── search_by_title ───────────────────────────────────────────────

class TestSearchByTitle:
    def test_none_title_returns_three_empty_dfs(self):
        r1, r2, r3 = app.search_by_title(None, "TF-IDF")
        assert_empty_result(r1)
        assert_empty_result(r2)
        assert_empty_result(r3)

    def test_empty_string_title_returns_three_empty_dfs(self):
        r1, r2, r3 = app.search_by_title("", "BM25")
        assert_empty_result(r1)
        assert_empty_result(r2)
        assert_empty_result(r3)

    def test_unknown_title_returns_error_in_all_three(self):
        r1, r2, r3 = app.search_by_title("Nonexistent Paper Title XYZ", "TF-IDF")
        assert_error_result(r1)
        assert_error_result(r2)
        assert_error_result(r3)

    def test_valid_title_returns_results(self):
        title = MOCK_DF["title"].iloc[0]
        r1, r2, r3 = app.search_by_title(title, "Compare All Three")
        assert_valid_result(r1)
        assert_valid_result(r2)
        assert_valid_result(r3)

    def test_valid_title_all_methods(self):
        title = MOCK_DF["title"].iloc[1]
        for method in ALL_METHODS:
            results = app.search_by_title(title, method)
            assert len(results) == 3
            assert all(isinstance(r, pd.DataFrame) for r in results)

    def test_error_message_contains_title(self):
        bad_title = "Not A Real Paper"
        r1, _, _ = app.search_by_title(bad_title, "TF-IDF")
        assert bad_title in r1["Error"].iloc[0]
