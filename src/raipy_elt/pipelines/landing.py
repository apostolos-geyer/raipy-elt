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
from raipy_elt.data_quality import (
    checks as dq_checks,
    errors as dq_errors,
    models as dq_models,
)
from raipy_elt.utilities.dataframe import (
    enforce_datetime64us,
    manual_parse,
    unique_assessment_id,
    list_col_nonempty,
)
from raipy_elt.utilities.delta import (
    DeltaTable,
    write_deltalake,
)
from raipy_elt.utilities.archiving import CompressAlg, move_files, tarball_files
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

GLOB_ASSESSMENT = "*MDSAssessment*.csv"

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

        expect_object = [c for c, t in pandas_kwds["dtype"].items() if t == "object"]
        n = len(data_augmented)
        for col in expect_object:
            nulls = int(data_augmented[col].isna().sum())
            if nulls != 0:
                msg = f"{col=} has {nulls=} out of {n=} entries missing. They will be filled with an empty string"
                if nulls == n:
                    LOGGER.warning(msg)
                else:
                    LOGGER.info(msg)

            data_augmented[col] = data_augmented[col].fillna("")

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
    qfname, sfname = questions_file_metadata.file_name, scoring_file_metadata.file_name
    left, right = (
        f"left dataframe (questions file {qfname})",
        f"right dataframe (scoring file {sfname})",
    )
    q_df, q_read_metadata = _try_read(questions_file_metadata, raw_dir)
    s_df, s_read_metadata = _try_read(scoring_file_metadata, raw_dir)

    errs: list[str] = []
    succ: bool = True

    try:
        check = dq_checks.check_column_value_sets(  # might raise a dq_errors error
            "UNIQUE_ASSESSMENT_ID", q_df, s_df
        )
        if check.rel is not dq_models.ColumnDomainRelationship.EQUAL:
            msg = f"Unique Assessment ID sets not equal between {left} and {right}. Relationship: {check.rel}."
            errs.append(msg)
            LOGGER.error(msg)
    except Exception as err:
        succ = False
        msg = ""
        match err:  # handle aforementioned dq_errors err here
            case dq_errors.ColCheckOnMissingCol() as ccm:
                res = ccm.result
                rel = res.rel
                if dq_models.LEFT_NOT_RIGHT in rel:
                    msg = f"UNIQUE_ASSESSMENT_ID missing from {right}"
                elif dq_models.RIGHT_NOT_LEFT in rel:
                    msg = f"UNIQUE_ASSESSMENT_ID missing from {left}"
                elif dq_models.BOTH in rel:
                    msg = f"UNIQUE_ASSESSMENT_ID is missing {left} and {right}"
            case dq_errors.ColCheckOnEmptyDF() as cce:
                res = cce.result
                rel = res.rel
                if dq_models.LEFT_NOT_RIGHT in rel:
                    msg = f"{left} has values and {right} doesnt"
                elif dq_models.RIGHT_NOT_LEFT in rel:
                    msg = f"{right} has valeus and {left} doesnt"
                elif dq_models.BOTH in rel:
                    msg = f"{left} and {right} are empty"
            case _:
                msg = (
                    f"An unexpected exception occurred while validating {left} and {right}",
                )
        errs.append(f"{err} [{msg}]")

        LOGGER.exception(f"{msg}. Files will be skipped.")

    q_ingestion_record = IngestionRecord(
        file_metadata=questions_file_metadata,
        read_metadata=q_read_metadata,
        ingest_errors=errs,
        ingest_success=succ,
        processed_timestamp=pd.Timestamp.now(),
    )
    s_ingestion_record = IngestionRecord(
        file_metadata=scoring_file_metadata,
        read_metadata=s_read_metadata,
        ingest_errors=errs,
        ingest_success=succ,
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
) -> DeltaTable | None:
    if not (
        ingested.record.read_metadata.read_success and ingested.record.ingest_success
    ):
        LOGGER.error(
            f"{ingested.record.file_metadata.file_name} not successfully read, skipping"
        )
        return None

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
        "archive_dir",
        "archive_behaviour",
        "archive_err_behaviour",
        "archive_compress_alg",
    },
    file_glob=GLOB_ASSESSMENT,
    file_metadata_filters=lambda x: True,
    mode="append",
    archive_behaviour=("tarball", "ingested"),
    archive_err_behaviour=("move", "ingested-review"),
    archive_compress_alg=CompressAlg.GZIP,
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
    _RawToBronze.variables["TIMESTAMP"] = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")

    _RawToBronze.variables["RECORD_NAME"] = (
        f"INGESTION_AT_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')}"
    )

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
    glob: str = GLOB_ASSESSMENT,
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

    for pair_no, (questions_file_metadata, scoring_file_metadata) in enumerate(
        zip(questions, scoring)
    ):
        LOGGER.info(
            f"Processing files: {questions_file_metadata.file_name} and {scoring_file_metadata.file_name}"
        )
        read_pair_result = _read_pair(
            questions_file_metadata, scoring_file_metadata, raw_dir
        )
        LOGGER.info(f"Read Questions: {read_pair_result.questions.record}")
        LOGGER.info(f"Read Scoring: {read_pair_result.scoring.record}")

        qrecord = read_pair_result.questions.record
        srecord = read_pair_result.scoring.record

        # should_write: bool = False
        # try:
        #     check_res = (
        #         dq_checks.check_column_value_sets(  # might raise a dq_errors error
        #             "UNIQUE_ASSESSMENT_ID",
        #             read_pair_result.scoring.data,
        #             read_pair_result.questions.data,
        #         )
        #     )

        #     if check_res.rel is not dq_models.ColumnDomainRelationship.EQUAL:
        #         LOGGER.error(
        #             f"Unique Assessment ID sets are not equal between questions and scoring. Relationship: {check_res}. Will be skipped"
        #         )
        #     else:
        #         should_write = True
        # except Exception as err:
        #     qfn = questions_file_metadata.file_name
        #     sfn = scoring_file_metadata.file_name
        #     msg = ""
        #     match err:  # handle aforementioned dq_errors err here
        #         case dq_errors.ColCheckOnMissingCol() as ccm:
        #             res = ccm.result
        #             rel = res.rel
        #             if dq_models.LEFT_NOT_RIGHT in rel:
        #                 msg = "UNIQUE_ASSESSMENT_ID missing from questions"
        #             elif dq_models.RIGHT_NOT_LEFT in rel:
        #                 msg = "UNIQUE_ASSESSMENT_ID missing from scoring"
        #             elif dq_models.BOTH in rel:
        #                 msg = "UNIQUE_ASSESSMENT_ID is missing"
        #         case dq_errors.ColCheckOnEmptyDF() as cce:
        #             res = cce.result
        #             rel = res.rel
        #             if dq_models.LEFT_NOT_RIGHT in rel:
        #                 msg = (
        #                     f"Questions file {qfn} is empty. Scoring file {sfn} is not."
        #                 )
        #             elif dq_models.RIGHT_NOT_LEFT in rel:
        #                 msg = (
        #                     f"Scoring file {sfn} is empty. Questions file {qfn} is not."
        #                 )
        #             elif dq_models.BOTH in rel:
        #                 msg = f"Both questions ({qfn}) and scoring ({sfn}) are empty."
        #         case _:
        #             msg = (
        #                 f"An unexpected exception occurred while validating files {qfn} and {sfn}",
        #             )

        #     LOGGER.exception(f"{msg}. Files will be skipped.")

        if mode == "overwrite" and pair_no == 0:
            LOGGER.info(
                "Overwriting delta table for first pair of files. This will not delete the pre-existing data but mark it as 'deleted' in the transaction log."
            )

            questions_ingested_to = _write_to_delta_if_successful(
                read_pair_result.questions, bronze_dir, mode="overwrite"
            )
            scoring_ingested_to = (
                None
                if not questions_ingested_to
                else _write_to_delta_if_successful(
                    read_pair_result.scoring, bronze_dir, mode="overwrite"
                )
            )
        else:
            questions_ingested_to = _write_to_delta_if_successful(
                read_pair_result.questions, bronze_dir
            )
            scoring_ingested_to = (
                None
                if not questions_ingested_to
                else _write_to_delta_if_successful(read_pair_result.scoring, bronze_dir)
            )

        qrecord.ingest_success = questions_ingested_to is not None
        srecord.ingest_success = scoring_ingested_to is not None

        if questions_ingested_to and scoring_ingested_to:
            LOGGER.info(
                f"Successfully ingested {questions_file_metadata.file_name} and {scoring_file_metadata.file_name}"
            )
        else:
            LOGGER.error(
                f"Failed to ingest {questions_file_metadata.file_name} and {scoring_file_metadata.file_name}"
            )

        ingestion_records.extend((qrecord, srecord))

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


@_RawToBronze.stage(
    use_params=[
        ParamMapping(from_param="archive_behaviour", as_arg="behaviour"),
        ParamMapping(from_param="archive_err_behaviour", as_arg="err_behaviour"),
        ParamMapping(from_param="raw_dir", as_arg="raw_dir"),
        ParamMapping(from_param="archive_dir", as_arg="archive_dir"),
        ParamMapping(from_param="archive_compress_alg", as_arg="alg"),
    ],
    use_outputs=[
        ResultMapping(
            from_stage="ingest",
            as_arg="ingestion_records",
        )
    ],
)
def archive_raws(
    behaviour: None | tuple[Literal["move"], str] | tuple[Literal["tarball"], str],
    err_behaviour: (
        None
        | Literal["include"]
        | tuple[Literal["move"], str]
        | tuple[Literal["tarball"], str]
    ),
    raw_dir: Path,
    archive_dir: Path,
    alg: CompressAlg,
    ingestion_records: pd.DataFrame,
) -> None:
    """
    archive raws cleans up the ingested raw data according to the behaviours provided. files that failed to ingest (were not written to the delta)
    will not be moved.

    :param behaviour: how files that ingested cleanly (no read errors or ingest errors) should be handled.

        - if None, no files will be moved.
        - if a tuple ("move", str), files will be moved to a directory under the archive_dir named by the second element of the tuple, concatenated
            with the current date and time.
        - if a tuple ("tarball", str), files will be moved to a tarball under the archive_dir named by the second element of the tuple concatenated
            with the current date and time, and suffixed with .tar.{suffix according to compression algorithm, either xz, gz, or bz2}

    :param err_behaviour: how files that ingested with errors (i.e were still written to delta, but had issues) should be handled.

        - if None, no files will be moved.
        - if "include", they will be included with the files without errors
        - if a tuple ("move", str), handled the same as above
        - if a tuple ("tarball", str), handled the same as above

    :param raw_dir: the path to the raw directory (to prepend to the file names from the ingestion_records)
    :param archive_dir: the path in which the archives should be placed
    :param alg: the compression algorithm
    :param ingestion_records: dataframe of ingestion records indicating errors and if the file was saved to delta, must have columns READ_ERRORS, INGEST_ERRORS, INGEST_SUCCESS
    """
    if behaviour is None:
        return

    successes = ingestion_records[ingestion_records["INGEST_SUCCESS"]]
    reidxs, ieidxs = (
        list_col_nonempty(successes, ecol) for ecol in ("READ_ERRORS", "INGEST_ERRORS")
    )
    erridxs = reidxs | ieidxs

    include_errs = isinstance(err_behaviour, str) and (err_behaviour == "include")
    if include_errs:
        fs = [raw_dir / fname for fname in successes["FILE_NAME"]]
        match behaviour:
            case ("move", str() as to_dir):
                move_files(
                    fs,
                    dest_dir=archive_dir
                    / f'{to_dir}{_RawToBronze.variables["TIMESTAMP"]}',
                    logger=LOGGER,
                )
            case ("tarball", str() as to_arc):
                tarball_files(
                    fs,
                    dest_dir=archive_dir,
                    dest_fname=f'{to_arc}{_RawToBronze.variables["TIMESTAMP"]}',
                    cmprsn=alg,
                    logger=LOGGER,
                )
    else:
        noerr_fs = [raw_dir / fname for fname in successes[~erridxs]["FILE_NAME"]]
        err_fs = [
            raw_dir / fname for fname in successes[erridxs]["FILE_NAME"]
        ]  # dont include
        for behave, fs in ((behaviour, noerr_fs), (err_behaviour, err_fs)):
            match behave:
                case ("move", str() as to_dir):
                    move_files(
                        fs,
                        dest_dir=archive_dir
                        / f'{to_dir}{_RawToBronze.variables["TIMESTAMP"]}',
                        logger=LOGGER,
                    )
                case ("tarball", str() as to_arc):
                    tarball_files(
                        fs,
                        dest_dir=archive_dir,
                        dest_fname=f'{to_arc}{_RawToBronze.variables["TIMESTAMP"]}',
                        cmprsn=alg,
                        logger=LOGGER,
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
