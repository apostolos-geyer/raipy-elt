from enum import Enum, auto
from typing import NamedTuple, Sequence, overload

import pandas as pd

class ColumnDomainRelationship(Enum):
    """
    Enumeration used when checking that (a) column(s) of two dataframes
    have values from the same set.
    """

    EQUAL = auto() 
    """
    Let `D_1` and `D_2` be `DataFrame`s with shared columns `C_0`, ... `C_n`.  

    We say `D_1` and `D_2` have `ColumnDomainRelationship.EQUAL` on `C_i` ∈ [0, n] iff

    `D_1[C_i].unique()` ⊆ `D_2[C_i].unique()` AND `D_2[C_i].unique()` ⊆ `D_1[C_i].unique()`

    i.e the set of the values present in `D_1[C_i]` is the same as those in `D_2[C_i]`
    """

    LEFT_NOT_RIGHT = auto()
    """
    Let `D_1` and `D_2` be DataFrames with shared columns `C_0`, ... `C_n`.

    We say `D_1` and `D_2` have `ColumnDomainRelationship.LEFT_NOT_RIGHT` on `C_i` ∈ [0, n] iff

    `D_1[C_i].unique()` ⊆ `D_2[C_i].unique()` ^ `D_2[C_i].unique()` ⊈ `D_1[C_i].unique()`

    i.e `D_2[C_i]` is covered by `D_1[C_i]` but `D_1[C_i]` is not covered by `D_2[C_i]`
    """


    RIGHT_NOT_LEFT = auto()
    """
    Let `D_1` and `D_2` be DataFrames with shared columns `C_0`, ... `C_n`.

    We say `D_1` and `D_2` have `ColumnDomainRelationship.RIGHT_NOT_LEFT` on `C_i` ∈ [0, n] iff

    `D_2[C_i].unique()` ⊆ `D_1[C_i].unique()` AND `D_1[C_i].unique()` ⊈ `D_2[C_i].unique()`

    i.e `D_1[C_i]` is covered by `D_2[C_i]` but `D_2[C_i]` is not covered by `D_1[C_i]`
    """


    BOTH = auto()
    """
    Let `D_1` and `D_2` be DataFrames with shared columns `C_0`, ... `C_n`.

    We say `D_1` and `D_2` have `ColumnDomainRelationship.BOTH` on `C_i` ∈ [0, n] iff

    `D_1[C_i].unique()` ⊈ `D_2[C_i].unique()` AND `D_2[C_i].unique()` ⊈ `D_1[C_i].unique()`

    i.e ∃ v ∈ `D_1[C_i]` s.t v ∉ `D_2[C_i]` and vice versa
    """


class ColumnDomainResult(NamedTuple):
    """
    NamedTuple for the result of the column domain check
    """

    relationship: ColumnDomainRelationship = ColumnDomainRelationship.EQUAL
    """
    The outcome of the column domain check
    """

    left_not_right: pd.Series | None = None
    """
    The values in the left DataFrame that are not in the right DataFrame
    """


    right_not_left: pd.Series | None = None
    """
    The values in the right DataFrame that are not in the left DataFrame
    """


def _check_column_value_sets(col: str, left: pd.DataFrame, right: pd.DataFrame) -> ColumnDomainResult:
    """
    Check that the column `col` of `left` and `right` have the same domain

    :param col: the column to check
    :param left: the left DataFrame
    :param right: the right DataFrame
    :returns: the result of the check
    """

    lnotr = left[~left[col].isin(right[col])][col]
    rnotl = right[~right[col].isin(left[col])][col]
    match (lnotr.empty, rnotl.empty):
        case (True, True):
            return ColumnDomainResult()
        case (False, True):
            return ColumnDomainResult(ColumnDomainRelationship.LEFT_NOT_RIGHT, lnotr)
        case (True, False):
            return ColumnDomainResult(ColumnDomainRelationship.RIGHT_NOT_LEFT, right_not_left=rnotl)
        case (False, False):
            return ColumnDomainResult(ColumnDomainRelationship.BOTH, lnotr, rnotl)
    
@overload
def check_column_value_sets(col: str, left: pd.DataFrame, right: pd.DataFrame) -> ColumnDomainResult:
    """
    Check that the column `col` of `left` and `right` have the same set of values

    :param col: the column(s) to check
    :param left: the left DataFrame
    :param right: the right DataFrame
    :returns: the result of the check
    """
    ...

@overload
def check_column_value_sets(cols: Sequence[str], left: pd.DataFrame, right: pd.DataFrame) -> dict[str, ColumnDomainResult]:
    """
    Check that the columns `cols` of `left` and `right` have the same set of values

    :param cols: the columns to check
    :param left: the left DataFrame
    :param right: the right DataFrame
    :returns: the result of the check
    """
    ...

def check_column_value_sets(cols: str | Sequence[str], left: pd.DataFrame, right: pd.DataFrame) -> ColumnDomainResult | dict[str, ColumnDomainResult]:
    """
    Check that the column or columns `cols` of `left` and `right` have the same set of values

    :param cols: the column(s) to check
    :param left: the left DataFrame
    :param right: the right DataFrame
    :returns: the result of the check
    """

    if isinstance(cols, str):
        return _check_column_value_sets(cols, left, right)
    elif isinstance(cols, Sequence):
        return {col: _check_column_value_sets(col, left, right) for col in cols}
    else:
        raise TypeError("cols must be a str or a Sequence[str]")
