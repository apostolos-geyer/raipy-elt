import logging
from functools import cache
from pathlib import Path
from typing import Callable, Literal, NamedTuple, TypeAlias

import pandas as pd

from raipy_elt.pipelines.pipeline import (
    ExitPipeline,
    ParamMapping,
    Pipeline,
    ResultMapping,
)
from raipy_elt.pipelines.records import FileMetadata, IngestionRecord, ReadMetadata
from raipy_elt.utilities.dataframe import (
    enforce_datetime64us,
    manual_parse,
    unique_assessment_id,
)
from raipy_elt.utilities.delta import (
    DeltaTable,
    write_deltalake,
)
from raipy_elt.utilities.misc import flatten_dicts

YAML_CONF = """\
---
Questions:
    columns:
        - Facility ID
        - Resident ID
        - Unit
        - Assessment ID
        - Assessment Type
        - question_key
        - item_value
        - Answer
    dates:
        - ARD: '%Y-%m-%d %H:%M:%S'
        - Answer_Date: '%Y-%m-%d %H:%M:%S'

Scoring:
    columns:
      - Facility ID
      - Resident ID
      - Unit
      - Assessment ID
      - RAI-MDS Status
      - Rug Group
      - CMI: Float32
      - CPS: Float32
      - DRS: Float32
      - COM: Float32
      - PAIN: Float32
      - CHESS: Float32
      - ISE: Float32
      - ADL_SHORT: Float32
      - ADL_LONG: Float32
      - ADL_HIER: Float32
      - ABS: Float32
      - PSI: Float32
      - PURS: Float32
      - FRS: Float32
    dates:
      - ARD: '%Y-%m-%d %H:%M:%S'
      - Assessment Locked Date: '%Y-%m-%d %H:%M:%S'
      - Date Assessment Created: '%Y-%m-%d %H:%M:%S'
      - Date Assessment Revised: '%Y-%m-%d %H:%M:%S'
"""

GLOB_CSV = "*.csv"

LOGGER = logging.getLogger(__name__)

FileFilter: TypeAlias = Callable[[FileMetadata], bool]


class _TryReadResult(NamedTuple):
    """
    A named tuple to hold the result of an attempt to read a file

    NamedTuples do not have behaviour associated with them and serve more as a data structure
    for making code more self documenting
    """

    data: pd.DataFrame
    read_metadata: ReadMetadata


class _IngestionResult(NamedTuple):
    """
    A named tuple to hold the result of an ingestion, i.e
    - the data
    - the record of the ingestion (read metadata, file metadata, processed timestamp)

    NamedTuples do not have behaviour associated with them and serve more as a data structure
    for making code more self documenting
    """

    data: pd.DataFrame
    record: IngestionRecord


class _ReadPairResult(NamedTuple):
    """
    A pair of ingestion results, one for questions and one for scoring

    NamedTuples do not have behaviour associated with them and serve more as a data structure
    for making code more self documenting
    """

    questions: _IngestionResult
    scoring: _IngestionResult


class IncomingPairs(NamedTuple):
    """
    A named tuple to hold the incoming questions and scoring files

    NamedTuples do not have behaviour associated with them and serve more as a data structure
    for making code more self documenting
    """

    questions: list[FileMetadata]
    scoring: list[FileMetadata]


@cache
def _get_conf() -> dict:
    import yaml

    return yaml.safe_load(YAML_CONF)


@cache
def _get_pandas_kwargs(
    extract_type: Literal["Questions", "Scoring"], conf: dict | None = None
) -> dict:
    """
    Retrieve the pandas kwargs for reading a file based on the extract type

    :param extract_type: the type of extract
    :param conf: the configuration to use, optional

    :returns: the pandas kwargs
    """
    return {
        "sep": "|",
        "engine": "c",
        "date_format": flatten_dicts(
            (config := (conf or _get_conf()[extract_type]))["dates"]
        ),
        "cache_dates": True,
        "dtype": flatten_dicts(
            [
                {col: "object"} if isinstance(col, str) else col
                for col in config["columns"]
            ],
        ),
        "low_memory": False,
    }


def _try_read(file_metadata: FileMetadata, raw_dir: Path) -> _TryReadResult:
    """
    Attempts to read a file and returns the data (a DataFrame) and the read metadata as a NamedTuple
    """
    LOGGER.info(f"Reading file: {file_metadata.file_name}")

    errors = []
    data = None
    data_augmented = None

    pandas_kwds = _get_pandas_kwargs(file_metadata.extract_type)
    try:
        data = pd.read_csv(
            raw_dir / file_metadata.file_name,
            **pandas_kwds,
        )

    except pd.errors.ParserError as pe:
        LOGGER.error(f"Parser error occurred: {pe}")
        errors.append(pe)
        try:
            data = manual_parse(
                raw_dir / file_metadata.file_name,
                sep="|",
            )
            LOGGER.info("Manual parse successful")

        except Exception as e:
            LOGGER.error(f"An error occurred: {e}")
            errors.append(e)

    except Exception as e:
        LOGGER.error(f"An error occurred: {e}")
        errors.append(e)

    if data is not None:
        LOGGER.info(f"Successfully read {file_metadata.file_name}")
        data_augmented = (
            data.assign(
                INGESTION_TIMESTAMP=pd.Timestamp.now(),
                **file_metadata.get_mapping(),
                UNIQUE_ASSESSMENT_ID=unique_assessment_id,
            )
            .pipe(enforce_datetime64us)
            .astype(pandas_kwds["dtype"])
        )

        for col_name in pandas_kwds["date_format"].keys():
            data_augmented[col_name] = pd.to_datetime(
                data_augmented[col_name], format=pandas_kwds["date_format"][col_name]
            )

    return _TryReadResult(
        data=data_augmented,
        read_metadata=ReadMetadata.from_data(data_augmented, errors),
    )


def _read_pair(
    questions_file_metadata: FileMetadata,
    scoring_file_metadata: FileMetadata,
    raw_dir: Path,
) -> _ReadPairResult:
    """
    Read a pair of files (a questions file and scoring file) and return the results as a named tuple
    """
    q_df, q_read_metadata = _try_read(questions_file_metadata, raw_dir)
    s_df, s_read_metadata = _try_read(scoring_file_metadata, raw_dir)

    q_ingestion_record = IngestionRecord(
        file_metadata=questions_file_metadata,
        read_metadata=q_read_metadata,
        processed_timestamp=pd.Timestamp.now(),
    )
    s_ingestion_record = IngestionRecord(
        file_metadata=scoring_file_metadata,
        read_metadata=s_read_metadata,
        processed_timestamp=pd.Timestamp.now(),
    )

    return _ReadPairResult(
        questions=_IngestionResult(data=q_df, record=q_ingestion_record),
        scoring=_IngestionResult(data=s_df, record=s_ingestion_record),
    )


def _write_to_delta_if_successful(
    ingested: _IngestionResult,
    bronze_dir: Path,
    mode: Literal["append", "overwrite"] = "append",
) -> DeltaTable:
    if not ingested.record.read_metadata.success:
        LOGGER.error(
            f"{ingested.record.file_metadata.file_name} not successfully read, skipping"
        )
        return

    data = ingested.data
    data.FILE_DATE = pd.to_datetime(ingested.data.FILE_DATE, format="%Y-%m-%d")

    write_deltalake(
        bronze_dir / ingested.record.file_metadata.extract_type,
        data=data,
        mode=mode,
        partition_by="FILE_DATE",
    )

    deltatable = DeltaTable(bronze_dir / ingested.record.file_metadata.extract_type)
    return deltatable


_RawToBronze = Pipeline.define(
    name="raw_to_bronze",
    parameters={
        "raw_dir",
        "bronze_dir",
        "file_glob",
        "file_metadata_filters",
        "mode",
    },
    file_glob=GLOB_CSV,
    file_metadata_filters=lambda x: True,
    mode="append",
)


@_RawToBronze.stage(
    "init",
    use_params=[
        ParamMapping(from_param="raw_dir", as_arg="raw_dir"),
    ],
)
def init(raw_dir: Path) -> None:
    """
    Initialize the raw to bronze pipeline
    """
    _RawToBronze.logger.info("Initializing raw to bronze pipeline")

    _RawToBronze.variables[
        "RECORD_NAME"
    ] = f"INGESTION_AT_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')}"

    _RawToBronze.logger.info("Attatching file handler to logger")

    # create a file handler
    handler = logging.FileHandler(
        raw_dir / f'{_RawToBronze.variables["RECORD_NAME"]}.log'
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
    )

    # create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
    )

    # attach a file handler to the logger,
    _RawToBronze.logger.addHandler(handler)
    _RawToBronze.logger.addHandler(console_handler)

    # replace the module logger with the pipeline logger
    global LOGGER
    LOGGER = _RawToBronze.logger


@_RawToBronze.stage(
    "detect_incoming",
    use_params=[
        ParamMapping(from_param="raw_dir", as_arg="raw_dir"),
        ParamMapping(from_param="file_glob", as_arg="glob"),
        ParamMapping(from_param="file_metadata_filters", as_arg="filters"),
    ],
)
def detect_incoming(
    raw_dir: Path,
    glob: str = GLOB_CSV,
    filters: list[FileFilter] | FileFilter | None = None,
) -> list[FileMetadata]:
    """
    Detect incoming files in the given raw directory

    :param raw_dir: the directory with the raw files
    """

    all_incoming: list[FileMetadata] = [
        FileMetadata.parse(file.name) for file in raw_dir.glob(glob)
    ]

    if filters is None:
        return all_incoming

    if callable(filters):
        return [file for file in all_incoming if filters(file)]

    return [file for file in all_incoming if all(f(file) for f in filters)]


@_RawToBronze.stage(
    use_outputs=[
        ResultMapping(
            from_stage="detect_incoming",
            as_arg="incoming",
        )
    ]
)
def divide_extract_types(incoming: list[FileMetadata]) -> IncomingPairs:
    """
    Divide the incoming extracts into questions and scoring files
    """

    def order(df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(
            by=["FILE_DATE", "INSTANCE", "PROVINCE", "FILE_NAME"]
        ).reset_index(drop=True)

    asdf = FileMetadata.dataframe(incoming)
    questions: list[FileMetadata] = (
        asdf.query("EXTRACT_TYPE == 'Questions'")
        .pipe(order)
        .pipe(FileMetadata.from_dataframe)
    )

    scoring: list[FileMetadata] = (
        asdf.query("EXTRACT_TYPE == 'Scoring'")
        .pipe(order)
        .pipe(FileMetadata.from_dataframe)
    )

    return IncomingPairs(questions=questions, scoring=scoring)


@_RawToBronze.stage(
    use_outputs=[
        ResultMapping(
            from_stage="divide_extract_types",
            as_arg="incoming_pairs",
        )
    ]
)
def validate_incoming(incoming_pairs: IncomingPairs) -> bool:
    """
    Validate the incoming pairs of questions and scoring files
    """
    questions, scoring = incoming_pairs
    assert len(questions) == len(
        scoring
    ), "Questions and scoring files are not equal in number"

    try:
        for qfile, sfile in zip(questions, scoring):
            assert qfile.file_name == sfile.file_name.replace("Scoring", "")
    except AssertionError as e:
        raise ExitPipeline(
            "Questions and scoring files are not aligned, there is a bug, not user error.",
            error=True,
        ) from e

    return True


@_RawToBronze.stage(
    use_params=[
        ParamMapping(
            from_param="raw_dir",
            as_arg="raw_dir",
        ),
        ParamMapping(
            from_param="bronze_dir",
            as_arg="bronze_dir",
        ),
        ParamMapping(
            from_param="mode",
            as_arg="mode",
        ),
    ],
    use_outputs=[
        ResultMapping(
            from_stage="divide_extract_types",
            as_arg="incoming_pairs",
        )
    ],
)
def ingest(
    incoming_pairs: IncomingPairs,
    raw_dir: Path,
    bronze_dir: Path,
    mode: Literal["append", "overwrite"],
) -> pd.DataFrame:
    """
    Ingest the incoming pairs of questions and scoring files

    :param incoming_pairs: the incoming pairs of questions and scoring files
    :param raw_dir: the directory with the raw files
    :param bronze_dir: the directory for the bronze tables

    :returns: a DataFrame with the ingestion records
    """

    ingestion_records: list[IngestionRecord] = []

    questions, scoring = incoming_pairs

    pair_no = 0
    for questions_file_metadata, scoring_file_metadata in zip(questions, scoring):
        LOGGER.info(
            f"Processing files: {questions_file_metadata.file_name} and {scoring_file_metadata.file_name}"
        )
        read_pair_result = _read_pair(
            questions_file_metadata, scoring_file_metadata, raw_dir
        )
        LOGGER.info(
            f"Read Questions: {read_pair_result.questions.record.read_metadata}"
        )
        LOGGER.info(f"Read Scoring: {read_pair_result.scoring.record.read_metadata}")

        ingestion_records.extend(
            [read_pair_result.questions.record, read_pair_result.scoring.record]
        )

        if mode == "overwrite" and pair_no == 0:
            LOGGER.info(
                "Overwriting delta table for first pair of files. This will not delete the pre-existing data but mark it as 'deleted' in the transaction log."
            )

            questions_ingested = _write_to_delta_if_successful(
                read_pair_result.questions, bronze_dir, mode="overwrite"
            )
            scoring_ingested = _write_to_delta_if_successful(
                read_pair_result.scoring, bronze_dir, mode="overwrite"
            )
        else:
            questions_ingested = _write_to_delta_if_successful(
                read_pair_result.questions, bronze_dir
            )
            scoring_ingested = _write_to_delta_if_successful(
                read_pair_result.scoring, bronze_dir
            )

        if questions_ingested and scoring_ingested:
            LOGGER.info(
                f"Successfully ingested {questions_file_metadata.file_name} and {scoring_file_metadata.file_name}"
            )
        else:
            LOGGER.error(
                f"Failed to ingest {questions_file_metadata.file_name} and {scoring_file_metadata.file_name}"
            )

    return pd.DataFrame([record.get_mapping() for record in ingestion_records])


@_RawToBronze.stage(
    use_params=[
        ParamMapping(
            from_param="raw_dir",
            as_arg="raw_dir",
        ),
    ],
    use_outputs=[
        ResultMapping(
            from_stage="ingest",
            as_arg="ingestion_records",
        )
    ],
)
def save_ingestion_records(ingestion_records: pd.DataFrame, raw_dir: Path) -> None:
    """
    Save the ingestion records to the bronze directory

    :param ingestion_records: the DataFrame with the ingestion records
    :param bronze_dir: the directory for the bronze tables
    """

    ingestion_records.to_csv(
        raw_dir / f'{_RawToBronze.variables["RECORD_NAME"]}.csv',
        index=False,
    )


@_RawToBronze.stage("cleanup")
def cleanup() -> None:
    """
    Remove the file handler from the logger
    """
    LOGGER.info("Cleaning up raw to bronze pipeline")

    # remove the file handler
    for handler in _RawToBronze.logger.handlers:
        if isinstance(handler, logging.FileHandler | logging.StreamHandler):
            _RawToBronze.logger.removeHandler(handler)


RawToBronze = _RawToBronze  # re-export after defining the pipeline
