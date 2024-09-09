from typing import overload

import pandas as pd

from raipy_elt.data_quality import errors
from raipy_elt.data_quality.models import (
    COL_MISSING,
    DF_EMPTY,
    ColumnDomainResult,
)


@overload
def check_column_value_sets(
    cols: str, left: pd.DataFrame, right: pd.DataFrame
) -> ColumnDomainResult:
    """
    Check that the column `col` of `left` and `right` have the same set of values

    :param col: the column(s) to check
    :param left: the left DataFrame
    :param right: the right DataFrame
    :returns: the result of the check
    """
    ...


@overload
def check_column_value_sets(
    cols: list[str] | tuple[str], left: pd.DataFrame, right: pd.DataFrame
) -> dict[str, ColumnDomainResult]:
    """
    Check that the columns `cols` of `left` and `right` have the same set of values

    :param cols: the columns to check
    :param left: the left DataFrame
    :param right: the right DataFrame
    :returns: the result of the check
    """
    ...


def check_column_value_sets(
    cols: str | list[str] | tuple[str], left: pd.DataFrame, right: pd.DataFrame
) -> ColumnDomainResult | dict[str, ColumnDomainResult]:
    """
    Check that the column or columns `cols` of `left` and `right` have the same set of values

    :param cols: the column(s) to check
    :param left: the left DataFrame
    :param right: the right DataFrame
    :returns: the result of the check
    :raises: EmptyDataframe if either is empty, MissingColumn if either is missing any cols specified
    """

    lempty, rempty = left.empty, right.empty
    if lempty or rempty:
        res: ColumnDomainResult
        match (lempty, rempty):
            case (True, True):
                res = ColumnDomainResult.both(flag=DF_EMPTY)
            case (True, False):
                res = ColumnDomainResult.right_not_left(
                    flag=DF_EMPTY
                )  # right has values and left doesnt
            case (False, True):
                res = ColumnDomainResult.left_not_right(flag=DF_EMPTY)  # vice versa
        raise errors.EmptyDataframe(result=res)  # type: ignore (one of the above cases is true by lempty or rempty)

    if isinstance(cols, str):
        missing_l, missing_r = cols not in left.columns, cols not in right.columns
        if missing_l or missing_r:
            res: ColumnDomainResult
            match (missing_l, missing_r):
                case (True, True):
                    res = ColumnDomainResult.both(cols, cols, COL_MISSING)
                case (True, False):
                    res = ColumnDomainResult.right_not_left(cols, COL_MISSING)
                case (False, True):
                    res = ColumnDomainResult.left_not_right(cols, COL_MISSING)
            raise errors.MissingColumn(result=res)  # type: ignore (one of the above cases is true by missing_l or missing_r)
        return _check_column_value_sets(cols, left, right)
    elif isinstance(cols, tuple | list):
        missing_from_l, missing_from_r = (
            [col for col in cols if col not in df] for df in (left, right)
        )
        if missing_from_l or missing_from_r:
            match (missing_from_l, missing_from_r):
                case [[], mr]:
                    res = ColumnDomainResult.left_not_right(mr, COL_MISSING)
                case [ml, []]:
                    res = ColumnDomainResult.right_not_left(ml, COL_MISSING)
                case [ml, mr]:
                    res = ColumnDomainResult.both(mr, ml, COL_MISSING)
            raise errors.MissingColumn(result=res)
        return {col: _check_column_value_sets(col, left, right) for col in cols}


def _check_column_value_sets(
    col: str, left: pd.DataFrame, right: pd.DataFrame
) -> ColumnDomainResult:
    """
    Check that the column `col` of `left` and `right` have the same domain
    should only be called if verified that col is in both dataframes and that both dataframes
    arent empty

    :param col: the column to check
    :param left: the left DataFrame
    :param right: the right DataFrame
    :returns: the result of the check
    """

    lnotr, rnotl = (
        _coldiff(col, _l, _r) for (_l, _r) in ((left, right), (right, left))
    )
    lcol: pd.Series | pd.DataFrame = lnotr[col]
    rcol: pd.Series | pd.DataFrame = rnotl[col]

    assert isinstance(lcol, pd.Series)
    assert isinstance(rcol, pd.Series)

    match (lcol.empty, rcol.empty):
        case (True, True):
            return ColumnDomainResult.equal()
        case (False, True):
            return ColumnDomainResult.left_not_right(values=lcol)
        case (True, False):
            return ColumnDomainResult.right_not_left(values=rcol)
        case (False, False):
            return ColumnDomainResult.both(lnotr=lcol, rnotl=rcol)


def _coldiff(col: str, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """makes mypy / pyright shut up"""
    diff = left[~left[col].isin(right[col])]
    assert isinstance(diff, pd.DataFrame)
    return diff
