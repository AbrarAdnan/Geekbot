"""Sandbox execution tests: real subprocess round-trip, all result kinds,
timeout, and hostile snippets that must fail cleanly instead of crashing
the parent (PLAN.md Phase 3)."""

from __future__ import annotations

import pandas as pd

from csv_analyst.contracts import ResultKind
from csv_analyst.execution.runner import execute_code


def _df():
    return pd.DataFrame({
        "region": ["North", "South", "North", "South"],
        "revenue": [100.0, 200.0, 150.0, 50.0],
    })


def test_execute_scalar_result():
    result = execute_code(_df(), "result = df['revenue'].mean()")
    assert result.ok
    assert result.kind == ResultKind.SCALAR
    assert result.value == 125.0


def test_execute_series_result():
    result = execute_code(_df(), "result = df.groupby('region')['revenue'].mean()")
    assert result.ok
    assert result.kind == ResultKind.SERIES
    assert set(result.value["region"]) == {"North", "South"}


def test_execute_dataframe_result():
    result = execute_code(_df(), "result = df.groupby('region')['revenue'].mean().reset_index()")
    assert result.ok
    assert result.kind == ResultKind.DATAFRAME
    assert list(result.value.columns) == ["region", "revenue"]


def test_execute_dict_result():
    result = execute_code(_df(), "result = {'total': df['revenue'].sum(), 'count': len(df)}")
    assert result.ok
    assert result.kind == ResultKind.DICT
    assert result.value == {"total": 500.0, "count": 4}


def test_execute_nan_scalar_becomes_none():
    result = execute_code(_df(), "result = df[df['region'] == 'nowhere']['revenue'].mean()")
    assert result.ok
    assert result.kind == ResultKind.SCALAR
    assert result.value is None


def test_execute_missing_result_variable_fails():
    result = execute_code(_df(), "x = df['revenue'].mean()")
    assert not result.ok
    assert "result" in result.error


def test_execute_runtime_error_fails_cleanly():
    result = execute_code(_df(), "result = df['does_not_exist'].mean()")
    assert not result.ok
    assert result.error


def test_execute_timeout():
    code = "result = sum(i for i in range(10**9))"
    result = execute_code(_df(), code, timeout_s=1)
    assert not result.ok
    assert "timeout" in result.error


def test_execute_unsupported_result_type_fails():
    result = execute_code(_df(), "result = [1, 2, 3]")
    assert not result.ok
    assert "unsupported result type" in result.error


def _dated_df():
    return pd.DataFrame({
        "day": pd.to_datetime(["2018-09-08", "2018-09-09", "2018-09-09", "2018-09-10"]),
        "at": pd.to_datetime(["2018-09-09 10:30", "2018-09-10 00:00", "2018-09-11 00:00", "2018-09-12 00:00"]),
    })


def test_execute_python_date_scalar_is_iso_string():
    # the real hospital/h04 shape: idxmax over a series grouped by `.dt.date`
    # returns a datetime.date, which used to fail as "unsupported result type"
    code = "result = df.groupby(df['day'].dt.date)['day'].count().diff().idxmax()"
    result = execute_code(_dated_df(), code)
    assert result.ok
    assert result.kind == ResultKind.SCALAR
    assert result.value == "2018-09-09"


def test_execute_timestamp_scalar_drops_midnight_time():
    result = execute_code(_dated_df(), "result = df['day'].max()")
    assert result.ok
    assert result.value == "2018-09-10"


def test_execute_timestamp_scalar_keeps_real_time():
    result = execute_code(_dated_df(), "result = df['at'].min()")
    assert result.ok
    assert result.value == "2018-09-09T10:30:00"


def test_execute_period_and_timedelta_scalars():
    period = execute_code(_dated_df(), "result = df['day'].dt.to_period('M').iloc[0]")
    assert period.ok and period.value == "2018-09"
    delta = execute_code(_dated_df(), "result = (df['at'] - df['day']).max()")
    assert delta.ok and delta.value == "2 days 00:00:00"


def test_execute_nat_scalar_becomes_none():
    result = execute_code(_dated_df(), "result = df[df['day'] > '2030-01-01']['day'].max()")
    assert result.ok
    assert result.value is None


def test_execute_empty_result_is_legitimate():
    result = execute_code(_df(), "result = df[df['region'] == 'nowhere']")
    assert result.ok
    assert result.kind == ResultKind.DATAFRAME
    assert result.value.empty


def test_execute_parquet_serialization_failure_returns_typed_result(monkeypatch):
    def _raise(*args, **kwargs):
        raise ImportError("no pyarrow")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _raise)
    result = execute_code(_df(), "result = df['revenue'].sum()")
    assert not result.ok
    assert "serializ" in result.error.lower()


def test_execute_dataframe_keeps_named_index_labels():
    # `unstack()` leaves the group keys in a named index; dropping it would
    # show the counts with no idea which group they belong to
    frame = pd.DataFrame({
        "province": ["A", "A", "A", "B", "B"],
        "gender": ["M", "M", "F", "F", "M"],
    })
    code = (
        "t = df.groupby(['province', 'gender']).size().unstack(fill_value=0)\n"
        "result = t[t['M'] > t['F']]"
    )
    result = execute_code(frame, code)
    assert result.ok
    assert result.kind == ResultKind.DATAFRAME
    assert list(result.value["province"]) == ["A"]


def test_execute_dataframe_drops_unnamed_filter_index():
    result = execute_code(_df(), "result = df[df['region'] == 'South'][['region', 'revenue']]")
    assert result.ok
    assert list(result.value.columns) == ["region", "revenue"]


def test_execute_dataframe_named_index_already_a_column_still_serializes():
    result = execute_code(_df(), "result = df.set_index('region', drop=False)")
    assert result.ok
    assert list(result.value.columns) == ["region", "revenue"]


def test_execute_series_with_plain_range_index_has_no_index_column():
    code = "result = df['revenue'].drop_duplicates().sort_values().reset_index(drop=True)"
    result = execute_code(_df(), code)
    assert result.ok
    assert list(result.value.columns) == ["revenue"]
    assert list(result.value["revenue"]) == [50.0, 100.0, 150.0, 200.0]


def test_execute_series_with_label_index_keeps_labels():
    # null count per column: the unnamed index holds the column names
    result = execute_code(_df(), "result = df.isna().sum()")
    assert result.ok
    assert set(result.value["index"]) == {"region", "revenue"}


def test_execute_mixed_row_keeps_numbers_as_numbers_not_dates():
    # A row picked with .loc mixes a date with numbers in one object column.
    # pyarrow used to type that column as date and read 2 / 100.0 as days
    # since 1970 ('1970-01-03', '1970-04-11').
    df = pd.DataFrame({"d": ["2018-09-15", "2018-09-16", "2018-09-16"], "x": [1, 2, 3]})
    code = (
        "t = df.assign(date=pd.to_datetime(df['d']).dt.date).groupby('date')['x'].count().reset_index()\n"
        "t['pct'] = t['x'].diff() / t['x'].shift(1) * 100\n"
        "result = t.loc[1]"
    )
    result = execute_code(df, code)
    assert result.ok
    values = dict(zip(result.value["index"], result.value.iloc[:, 1]))
    assert values["date"] == "2018-09-16"
    assert values["x"] == "2"
    assert values["pct"] == "100.0"
    assert not any(str(v).startswith("1970") for v in values.values())


def test_execute_mixed_int_float_object_column_stays_numeric():
    result = execute_code(_df(), "result = pd.Series([1, 2.5, 3], index=['a', 'b', 'c'], dtype='object')")
    assert result.ok
    assert list(result.value.iloc[:, 1]) == [1.0, 2.5, 3.0]


def test_execute_single_type_date_column_is_unchanged():
    df = pd.DataFrame({"d": ["2018-09-15", "2018-09-16"]})
    result = execute_code(df, "result = pd.to_datetime(df['d']).dt.date.to_frame('day')")
    assert result.ok
    assert [str(v) for v in result.value["day"]] == ["2018-09-15", "2018-09-16"]
