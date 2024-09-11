from functools import cache
from pathlib import Path
from typing import Callable

import pandas as pd


def enforce_datetime64us(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces that all datetime columns are datetime64[us] type

    :param df: the dataframe to conform
    :returns: the passed dataframe with the type inforced
    """
    dtcols = df.select_dtypes("datetime")

    df[dtcols.columns] = dtcols.astype("datetime64[us]")

    return df


def unique_assessment_id(data: pd.DataFrame) -> pd.Series:
    """
    A helper function to create a unique assessment ID

    :param data: the data to create the unique assessment ID from
    :returns: the unique assessment ID column
    """
    return (
        data["INSTANCE"].astype(str)
        + "|"
        + data["PROVINCE"].astype(str)
        + "|"
        + data["Facility ID"].astype(str)
        + "|"
        + data["Resident ID"].astype(str)
        + "|"
        + data["Assessment ID"].astype(str)
        + "|"
        + data["FILE_DATE"].dt.strftime("%Y-%m-%d")
    )


@cache
def format_enforcer(dtype, date_format) -> Callable[[pd.DataFrame], pd.DataFrame]:
    raise NotImplementedError("this is fucked")

    """
    A factory function to create a function that enforces data types and date formats

    :param dtype: the data types to enforce
    :param date_format: the date formats to enforce
    :returns: a function that enforces the data types and date formats
    """

    def enforce_format(df: pd.DataFrame) -> pd.DataFrame:
        if dtype:
            df = df.astype(dtype)
        if date_format:
            for col_name in date_format.keys():
                df[col_name] = pd.to_datetime(
                    df[col_name], format=date_format[col_name]
                )

        return df

    return enforce_format


def _clean_parse_line(line: str, sep="|") -> list[str]:
    return [substr.strip('"') for substr in line.strip().split(sep)]


def manual_parse(bad_file: Path, sep="|") -> pd.DataFrame:
    """
    Opens a file and reads line by line to parse it into a pandas DataFrame
    """

    with bad_file.open("r") as f:
        header_line = f.readline()
        columns = _clean_parse_line(header_line, sep=sep)

        df = pd.DataFrame(
            [dict(zip(columns, _clean_parse_line(line, sep=sep))) for line in f]
        )

    return df


def list_col_nonempty(df: pd.DataFrame, list_col: str) -> pd.Series:
    """series of bool indicating if a column composed of lists is non empty"""
    return df[list_col].apply(lambda el: False if not el else True)
