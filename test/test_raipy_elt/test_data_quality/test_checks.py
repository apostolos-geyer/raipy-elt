import pytest
from pytest import fixture
from pytest import mark
import pandas as pd
from raipy_elt.data_quality.checks import check_column_value_sets
from raipy_elt.data_quality.errors import ColCheckOnEmptyDF, ColCheckOnMissingCol
from raipy_elt.data_quality.models import (
    ColumnDomainResult,
    LEFT_NOT_RIGHT,
    RIGHT_NOT_LEFT,
    BOTH,
    EQUAL,
    DF_EMPTY,
    COL_MISSING,
)


# Helper DataFrames
@fixture
def df_123_abc_short():
    return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})


@fixture
def df_1234_abc():
    return pd.DataFrame({"col1": [1, 2, 3, 4], "col2": ["a", "a", "b", "c"]})


@fixture
def df_123_abc_long():
    return pd.DataFrame(
        {"col1": [1, 2, 3, 2, 2, 1, 3], "col2": ["a", "b", "c", "b", "b", "c", "a"]}
    )


@fixture
def df_235_abd():
    return pd.DataFrame({"col1": [2, 3, 5], "col2": ["a", "b", "d"]})


@fixture
def df_empty():
    return pd.DataFrame()


@mark.parametrize(
    "l,r",
    [
        ("df_123_abc_short", "df_123_abc_short"),
        ("df_123_abc_long", "df_123_abc_short"),
        ("df_123_abc_short", "df_abc_123_long"),
        ("df_123_abc_long", "df_123_abc_long"),
    ],
)
def test_equal_columns(l, r, request):
    ldf = request.getfixturevalue(l)
    rdf = request.getfixturevalue(r)

    result = check_column_value_sets("col1", ldf, rdf)
    assert isinstance(result, ColumnDomainResult)
    assert result.rel == EQUAL
    assert result.lnotr is None
    assert result.rnotl is None

    multi_results = check_column_value_sets(["col1", "col2"], ldf, rdf)
    col1res = multi_results["col1"]
    col2res = multi_results["col2"]
    assert col1res == col2res == result
    # should be the same since we know column values are
    # from the same set, therefore we should get identical result
    # objects


### Tests for LEFT_NOT_RIGHT Relationship
@mark.parametrize(
    "l,r", [("df_1234_abc", "df_123_abc_short"), ("df_1234_abc", "df_123_abc_long")]
)
def test_left_not_right(l, r, request):
    ldf = request.getfixturevalue(l)
    rdf = request.getfixturevalue(r)
    result = check_column_value_sets("col1", ldf, rdf)
    assert isinstance(result, ColumnDomainResult)
    assert result.rel == LEFT_NOT_RIGHT
    assert result.lnotr == pd.Series([4])  # Value 1 is in left but not in right


### Tests for RIGHT_NOT_LEFT Relationship
@mark.parametrize(
    "l,r", [("df_123_abc_short", "df_1234_abc"), ("df_123_abc_long", "df_1234_abc")]
)
def test_right_not_left(l, r, request):
    ldf = request.getfixturevalue(l)
    rdf = request.getfixturevalue(r)
    result = check_column_value_sets("col1", ldf, rdf)
    assert isinstance(result, ColumnDomainResult)
    assert result.rel == RIGHT_NOT_LEFT
    assert result.rnotl == pd.Series([4])  # Value 4 is in right but not in left


### Tests for BOTH Relationship
def test_both_columns(df_1234_abc, df_235_abd):
    ldf, rdf = df_1234_abc, df_235_abd
    result = check_column_value_sets(["col1", "col2"], ldf, rdf)
    c1res = result["col1"]
    c2res = result["col2"]

    assert c1res.rel == BOTH
    assert c2res.rel == BOTH
    assert c1res.lnotr == pd.Series([1, 4])
    assert c1res.rnotl == pd.Series([5])
    assert c2res.lnotr == pd.Series(["c"])
    assert c2res.rnotl == pd.Series(["d"])


### Tests for Empty DataFrames
def test_both_empty(df_empty):
    with pytest.raises(ColCheckOnEmptyDF) as exc:
        check_column_value_sets("col1", df_empty, df_empty)
    assert exc.value.result.rel == DF_EMPTY | BOTH


def test_left_empty(df_empty, df_235_abd):
    with pytest.raises(ColCheckOnEmptyDF) as exc:
        check_column_value_sets("col1", df_empty, df_235_abd)
    assert exc.value.result.rel == DF_EMPTY | RIGHT_NOT_LEFT


def test_right_empty(df_235_abd, df_empty):
    with pytest.raises(ColCheckOnEmptyDF) as exc:
        check_column_value_sets("col1", df_235_abd, df_empty)
    assert exc.value.result.rel == DF_EMPTY | LEFT_NOT_RIGHT


### Tests for Missing Columns
def test_missing_columns(df_235_abd):
    l, r = df_235_abd.copy(), df_235_abd.copy()  # noqa: E741

    with pytest.raises(ColCheckOnMissingCol) as exc:
        check_column_value_sets("col3", l, r)
    assert exc.value.result.rel == COL_MISSING | BOTH

    l["col3"] = l["col2"]
    with pytest.raises(ColCheckOnMissingCol) as exc:
        check_column_value_sets("col3", l, r)
    assert exc.value.result.rel == COL_MISSING | LEFT_NOT_RIGHT

    with pytest.raises(ColCheckOnMissingCol) as exc:
        check_column_value_sets("col3", r, l)
    assert exc.value.result.rel == COL_MISSING | RIGHT_NOT_LEFT
