import sys
from enum import Flag, auto
from typing import Optional, Self, TypeAlias

import pandas as pd
from attrs import frozen


class ColumnDomainRelationship(Flag):
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

    DF_EMPTY = auto()
    """
    Flag to add when relationship is indicating left or right is empty
    """

    COL_MISSING = auto()
    """
    Flag to add when relationship is indicating left or right is missing column
    """


EQUAL, LEFT_NOT_RIGHT, RIGHT_NOT_LEFT, BOTH, DF_EMPTY, COL_MISSING = (
    ColumnDomainRelationship
)


ValueDiff: TypeAlias = Optional[pd.Series | str | list[str]]
ValueDiff.__doc__ = """
value difference representation. use a pd.Series when representing
difference between values in columns, a str or list[str] when representing column(s) not present,
and None when rel is EQUAL or representing emptiness of dataframes
"""


@frozen
class ColumnDomainResult:
    """
    NamedTuple for the result of the column domain check
    """

    rel: ColumnDomainRelationship
    """
    The outcome of the column domain check
    """

    lnotr: ValueDiff
    """
    The values in the left DataFrame that are not in the right DataFrame
    """

    rnotl: ValueDiff
    """
    The values in the right DataFrame that are not in the left DataFrame
    """

    @classmethod
    def equal(cls) -> Self:
        return cls(EQUAL, None, None)

    @classmethod
    def left_not_right(
        cls, values: ValueDiff = None, flag: Optional[ColumnDomainRelationship] = None
    ) -> Self:
        rel = LEFT_NOT_RIGHT if flag is None else LEFT_NOT_RIGHT | flag
        return cls(rel, values, None)

    @classmethod
    def right_not_left(
        cls, values: ValueDiff = None, flag: Optional[ColumnDomainRelationship] = None
    ) -> Self:
        rel = RIGHT_NOT_LEFT if flag is None else RIGHT_NOT_LEFT | flag
        return cls(rel, None, values)

    @classmethod
    def both(
        cls,
        lnotr: ValueDiff = None,
        rnotl: ValueDiff = None,
        flag: Optional[ColumnDomainRelationship] = None,
    ) -> Self:
        rel = BOTH if flag is None else BOTH | flag
        return cls(rel, lnotr, rnotl)
