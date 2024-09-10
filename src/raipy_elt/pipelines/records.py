# define our metadata structure
from typing import Literal
from pathlib import Path

import pandas as pd
from attrs import asdict, define, field

INSTANCE_DTYPE = pd.CategoricalDtype(categories=["EXT", "REV"])
PROVINCE_DTYPE = pd.CategoricalDtype(categories=["Ontario", "Alberta"])
EXTRACT_TYPE_DTYPE = pd.CategoricalDtype(categories=["Scoring", "Questions"])
FILE_METADATA_COLS = ["FILE_NAME", "FILE_DATE", "INSTANCE", "PROVINCE", "EXTRACT_TYPE"]


@define
class FileMetadata:
    """
    Utility class for parsing file metadata
    """

    file_name: str
    file_date: pd.Timestamp
    instance: pd.Categorical
    province: str
    extract_type: Literal["Scoring", "Questions"]

    @classmethod
    def parse(cls, fname: str):
        meta = fname[:-4].split("_")  # file name excluding .csv suffix split on _
        instance = meta[0]  # instance is the first item
        province = meta[
            2
        ]  # province is the third item e.g EXT_MDSAssessmentExtract_(Ontario)
        file_date = pd.to_datetime(meta[-1], cache=True)  # file date is the last item
        extract_type = "Scoring" if "Scoring" in fname else "Questions"
        return cls(
            file_name=fname,
            instance=instance,
            province=province,
            file_date=file_date,
            extract_type=extract_type,
        )

    def get_mapping(self) -> dict:
        return {k.upper(): v for k, v in asdict(self).items()}

    @staticmethod
    def dataframe(records: list["FileMetadata"]) -> pd.DataFrame:
        return pd.DataFrame([record.get_mapping() for record in records])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> list["FileMetadata"]:
        return [
            cls(
                file_name=row.FILE_NAME,
                instance=row.INSTANCE,
                province=row.PROVINCE,
                file_date=row.FILE_DATE,
                extract_type=row.EXTRACT_TYPE,
            )
            for row in df.itertuples(index=False)
        ]


# a data structure and helper for read metadata construction

READ_METADATA_COLS = [
    "N_ROWS",
    "N_RESIDENTS",
    "N_FACILITIES",
    "N_UNIQUE_ASSESSMENTS",
    "ERRORS",
]


@define
class ReadMetadata:
    n_rows: int
    n_residents: int
    n_facilities: int
    n_unique_assessments: int
    read_errors: list[str] | None
    read_success: bool

    @classmethod
    def from_data(
        cls, data: pd.DataFrame | None = None, errors: list[str] | None = None
    ):
        return (
            cls(
                n_rows=data.shape[0],
                n_residents=data["Resident ID"].nunique(),
                n_unique_assessments=data.UNIQUE_ASSESSMENT_ID.nunique(),
                n_facilities=data["Facility ID"].nunique(),
                read_errors=errors or [],
                read_success=True,
            )
            if data is not None
            else cls(
                n_rows=-1,
                n_residents=-1,
                n_unique_assessments=-1,
                n_facilities=-1,
                read_errors=errors or [],
                read_success=False,
            )
        )

    def get_mapping(self) -> dict:
        return {k.upper(): v for (k, v) in asdict(self).items()}

    @staticmethod
    def dataframe(records: list["ReadMetadata"]) -> pd.DataFrame:
        return pd.DataFrame([record.get_mapping() for record in records]).astype(
            dtype={
                "N_ROWS": pd.Int64Dtype(),
                "N_RESIDENTS": pd.Int64Dtype(),
                "N_FACILITIES": pd.Int64Dtype(),
                "N_UNIQUE_ASSESSMENTS": pd.Int64Dtype(),
                "SUCCESS": bool,
            }
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> list["ReadMetadata"]:
        return [
            cls(
                n_rows=row.N_ROWS,
                n_residents=row.N_RESIDENTS,
                n_facilities=row.N_FACILITIES,
                n_unique_assessments=row.N_UNIQUE_ASSESSMENTS,
                errors=row.ERRORS,
                success=row.SUCCESS,
            )
            for row in df.itertuples(index=False)
        ]


@define
class IngestionRecord:
    file_metadata: FileMetadata
    read_metadata: ReadMetadata
    ingest_errors: list[str]
    ingest_success: bool
    processed_timestamp: pd.Timestamp = field(factory=pd.Timestamp.now)

    def get_mapping(self) -> dict:
        return {
            **self.file_metadata.get_mapping(),
            **self.read_metadata.get_mapping(),
            "INGEST_ERRORS": self.ingest_errors,
            "INGEST_SUCCESS": self.ingest_success,
            "PROCESSED_TIMESTAMP": self.processed_timestamp,
        }

    @staticmethod
    def dataframe(records: list["IngestionRecord"]) -> pd.DataFrame:
        return pd.DataFrame([record.get_mapping() for record in records]).astype(
            dtype={
                "N_ROWS": pd.Int64Dtype(),
                "N_RESIDENTS": pd.Int64Dtype(),
                "N_FACILITIES": pd.Int64Dtype(),
                "N_UNIQUE_ASSESSMENTS": pd.Int64Dtype(),
                "INSTANCE": INSTANCE_DTYPE,
                "PROVINCE": PROVINCE_DTYPE,
                "EXTRACT_TYPE": EXTRACT_TYPE_DTYPE,
                "SUCCESS": bool,
            }
        )
