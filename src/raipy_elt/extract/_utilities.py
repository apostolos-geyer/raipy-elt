from typing import Literal, Sequence
from functools import reduce
from operator import or_

import pandas as pd
from attrs import asdict, define

def flatten_dicts(dicts: Sequence[dict], initial: dict | None = None) -> dict:
    """
    Flatten multiple dicts into one using a left fold (reduce) operation

    :param dicts: dictionaries to fold over
    :param initial: starting dictionary, optional

    :returns: the flattened (joined) dicts
    """
    initial = initial or {}
    return reduce(
        or_,
        dicts,
        initial,
    )


def enforce_datetime64us(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces that all datetime columns are datetime64[us] type

    :param df: the dataframe to conform
    :returns: the passed dataframe with the type inforced
    """
    dtcols = df.select_dtypes("datetime")

    df[dtcols.columns] = dtcols.astype("datetime64[us]")

    return df


def handle_bad_lines(bad_line: list[str]) -> list[str] | None:
    """
    Called by read_csv with python engine as a fallback in case of a
    read error.

    :param bad_line: the bad line split on the sep character
    :returns: a list of column values
    """
    return [x.strip('"') for x in bad_line]


@define
class MetadataParser:
    """
    Utility class for parsing file metadata
    """

    file_name: str
    file_date: pd.Timestamp
    instance: Literal["EXT", "REV"]
    province: str

    @classmethod
    def parse(cls, fname: str):
        meta = fname[:-4].split("_")  # file name excluding .csv suffix split on _
        instance = meta[0]  # instance is the first item
        province = meta[
            2
        ]  # province is the third item e.g EXT_MDSAssessmentExtract_(Ontario)
        file_date = pd.to_datetime(meta[-1], cache=True)  # file date is the last item

        return cls(
            file_name=fname,
            instance=instance,
            province=province,
            file_date=file_date,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def get_mapping(self) -> dict:
        return {k.upper(): v for (k, v) in self.to_dict().items()}
