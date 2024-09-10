import pytest
from raipy_elt.data_quality.errors import ColCheckOnEmptyDF, ColCheckOnMissingCol
from raipy_elt.data_quality.models import (
    ColumnDomainResult,
    DF_EMPTY,
    COL_MISSING,
    LEFT_NOT_RIGHT,
    RIGHT_NOT_LEFT,
    BOTH,
    EQUAL,
)


def test_empty_dataframe_error_message():
    result = ColumnDomainResult(DF_EMPTY | RIGHT_NOT_LEFT, None, None)
    with pytest.raises(ColCheckOnEmptyDF) as exc:
        raise ColCheckOnEmptyDF(result)
    assert "Left dataframe is empty" in str(exc.value)


def test_missing_column_error_message():
    result = ColumnDomainResult(COL_MISSING | LEFT_NOT_RIGHT, ["missing_col"], None)
    with pytest.raises(ColCheckOnMissingCol) as exc:
        raise ColCheckOnMissingCol(result)
    assert "Right dataframe is missing column(s) ['missing_col']" in str(exc.value)


@pytest.mark.parametrize(
    "rel, expected_error_message",
    [
        (EQUAL, "DF_EMPTY flag should be in column domain result"),
        (LEFT_NOT_RIGHT, "DF_EMPTY flag should be in column domain result"),
        (RIGHT_NOT_LEFT, "DF_EMPTY flag should be in column domain result"),
        (BOTH, "DF_EMPTY flag should be in column domain result"),
    ],
)
def test_invalid_empty_df_exception_construction(rel, expected_error_message):
    """
    Test that constructing `ColCheckOnEmptyDF` without the `DF_EMPTY` flag raises ValueError.
    """
    invalid_result = ColumnDomainResult(rel, None, None)

    with pytest.raises(ValueError) as exc:
        raise ColCheckOnEmptyDF(invalid_result)

    assert expected_error_message in str(exc.value)


@pytest.mark.parametrize(
    "rel, expected_error_message",
    [
        (EQUAL, "COL_MISSING flag should be in column domain result"),
        (LEFT_NOT_RIGHT, "COL_MISSING flag should be in column domain result"),
        (RIGHT_NOT_LEFT, "COL_MISSING flag should be in column domain result"),
        (BOTH, "COL_MISSING flag should be in column domain result"),
    ],
)
def test_invalid_missing_col_exception_construction(rel, expected_error_message):
    """
    Test that constructing `ColCheckOnMissingCol` without the `COL_MISSING` flag raises ValueError.
    """
    invalid_result = ColumnDomainResult(rel, None, None)

    with pytest.raises(ValueError) as exc:
        raise ColCheckOnMissingCol(invalid_result)

    assert expected_error_message in str(exc.value)


@pytest.mark.parametrize(
    "rel, expected_error_message",
    [
        (DF_EMPTY, "should have flag indicating which is empty"),
        (DF_EMPTY | COL_MISSING, "should have flag indicating which is empty"),
    ],
)
def test_invalid_empty_df_exception_construction_without_side_flag(
    rel, expected_error_message
):
    """
    Test that constructing `ColCheckOnEmptyDF` without LEFT_NOT_RIGHT or RIGHT_NOT_LEFT raises ValueError.
    """
    invalid_result = ColumnDomainResult(rel, None, None)

    with pytest.raises(ValueError) as exc:
        raise ColCheckOnEmptyDF(invalid_result)

    assert expected_error_message in str(exc.value)


@pytest.mark.parametrize(
    "rel, expected_error_message",
    [
        (COL_MISSING, "should have flag indicating which is missing"),
        (COL_MISSING | DF_EMPTY, "should have flag indicating which is missing"),
    ],
)
def test_invalid_missing_col_exception_construction_without_side_flag(
    rel, expected_error_message
):
    """
    Test that constructing `ColCheckOnMissingCol` without LEFT_NOT_RIGHT or RIGHT_NOT_LEFT raises ValueError.
    """
    invalid_result = ColumnDomainResult(rel, None, None)

    with pytest.raises(ValueError) as exc:
        raise ColCheckOnMissingCol(invalid_result)

    assert expected_error_message in str(exc.value)
