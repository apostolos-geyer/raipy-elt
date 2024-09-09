from typing import Self
from raipy_elt.data_quality.models import (
    ColumnDomainResult,
    LEFT_NOT_RIGHT,
    RIGHT_NOT_LEFT,
    BOTH,
    DF_EMPTY,
    COL_MISSING,
)


class ColCheckOnEmptyDF(ValueError):
    result: ColumnDomainResult

    def __init__(self: Self, result: ColumnDomainResult) -> None:
        if DF_EMPTY not in result.rel:
            raise ValueError(
                "invalid construction of ColCheckOnEmptyDF exception. DF_EMPTY flag should be in column domain result"
            )
        elif not any(
            cdr in result.rel for cdr in (LEFT_NOT_RIGHT, RIGHT_NOT_LEFT, BOTH)
        ):
            raise ValueError(
                "invalid construction of ColCheckOnEmptyDF exception. should have flag indicating which is empty (LEFT_NOT_RIGHT (right empty), RIGHT_NOT_LEFT (left empty), or BOTH)"
            )

        self.result = result
        super().__init__(self._msg())

    def _msg(self: Self) -> str:
        tmpl = "Column value check is invalid. {why}"
        why = ""
        if LEFT_NOT_RIGHT in self.result.rel:
            why = "Right dataframe is empty"
        elif RIGHT_NOT_LEFT in self.result.rel:
            why = "Left dataframe is empty"
        elif BOTH in self.result.rel:
            why = "Both dataframes are empty"
        return tmpl.format(why=why)


class ColCheckOnMissingCol(KeyError):
    result: ColumnDomainResult

    def __init__(self: Self, result: ColumnDomainResult) -> None:
        if COL_MISSING not in result.rel:
            raise ValueError(
                "invalid construction of ColCheckOnMissingCol exception. COL_MISSING flag should be in column domain result"
            )
        elif not any(
            cdr in result.rel for cdr in (LEFT_NOT_RIGHT, RIGHT_NOT_LEFT, BOTH)
        ):
            raise ValueError(
                "invalid construction of ColCheckOnMissingCol exception. should have flag indicating which is missing (LEFT_NOT_RIGHT (missing in right), RIGHT_NOT_LEFT (missing in left), or BOTH)"
            )

        self.result = result
        super().__init__(self._msg())

    def _msg(self: Self) -> str:
        tmpl = "Column value check is invalid. {why}"
        why = ""
        if LEFT_NOT_RIGHT in self.result.rel:
            why = f"Right dataframe is missing column(s) {self.result.lnotr}"
        if RIGHT_NOT_LEFT in self.result.rel:
            why = f"Left dataframe is missing column(s) {self.result.rnotl}"
        if BOTH in self.result.rel:
            why = f"Right dataframe is missing column(s) {self.result.lnotr}, Left dataframe is missing column(s) {self.result.rnotl}"
        return tmpl.format(why=why)


class UncoveredAssessmentError(Exception):
    def __init__(
        self, file_with: str, file_without: str, missing_ids: list[int], *args
    ):
        super().__init__(args)
        self.file_with = file_with
        self.file_without = file_without
        self.missing_ids = missing_ids
