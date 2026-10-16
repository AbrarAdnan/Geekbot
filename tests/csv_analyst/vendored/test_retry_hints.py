"""PLAN.md Phase 6A.5: each hint's regex must match the real error string it
was derived from (evaluation/reports/ failures), and an unrecognized error
must return no hint rather than a false match."""

from __future__ import annotations

from csv_analyst.validation.retry_hints import retry_hint


def test_matches_groupby_self_aggregate_collision():
    err = "ValueError: cannot insert allergies, already exists"
    assert retry_hint(err) is not None
    assert "size()" in retry_hint(err)


def test_matches_dt_accessor_on_string_column():
    err = "AttributeError: Can only use .dt accessor with datetimelike values"
    assert retry_hint(err) is not None
    assert "to_datetime" in retry_hint(err)


def test_matches_line_continuation_syntax_error():
    err = "syntax error: unexpected character after line continuation character (code.py, line 2)"
    assert retry_hint(err) is not None


def test_matches_key_error():
    err = "KeyError: 'salary'"
    hint = retry_hint(err)
    assert hint is not None
    assert "column name" in hint


def test_matches_tuple_key_error_for_comma_indexing():
    err = "KeyError: ('patient_id', 'first_name')"
    hint = retry_hint(err)
    assert hint is not None
    assert "double brackets" in hint


def test_matches_numpy_ndarray_has_no_attribute():
    err = "AttributeError: 'numpy.ndarray' object has no attribute 'sort_values'"
    hint = retry_hint(err)
    assert hint is not None
    # the old advice (`pd.Series(...).sort_values()`) led the model straight
    # into `.reset_index(name=...)`, which adds a stray `index` column
    assert "to_frame()" in hint
    assert "reset_index(name=" in hint


def test_matches_merge_rejected_by_ast_gate():
    hint = retry_hint("disallowed attribute: merge")
    assert hint is not None
    assert "unstack(fill_value=0)" in hint


def test_matches_groupby_object_has_no_attribute():
    err = "AttributeError: 'SeriesGroupBy' object has no attribute 'sub'"
    hint = retry_hint(err)
    assert hint is not None
    assert "assign(" in hint
    assert ".dt.days" in hint


def test_key_error_for_a_real_column_says_it_was_dropped_earlier():
    hint = retry_hint("KeyError: 'region'", columns=["region", "sales"])
    assert "dropped" in hint


def test_key_error_for_a_data_value_mentions_pivot():
    # e.g. a gender value 'F' that was filtered out before unstacking
    hint = retry_hint("KeyError: 'F'", columns=["gender", "province_name"])
    assert "pivot/unstack" in hint


def test_key_error_for_a_compound_guess_keeps_last_word_advice():
    hint = retry_hint("KeyError: 'product_category'", columns=["category", "sales"])
    assert "last word" in hint


def test_unrecognized_error_returns_none():
    assert retry_hint("TimeoutError: connection reset") is None
