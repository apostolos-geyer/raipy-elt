import logging
from pprint import pformat
from collections import defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path
import tarfile
from typing import Callable, Literal, TypedDict

import pandas as pd
import pyarrow
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import TableNotFoundError

from raipy_elt.extract._utilities import (
    MetadataParser,
    enforce_datetime64us,
    flatten_dicts,
    handle_bad_lines,
)
from raipy_elt.extract.configs import BRONZE, DIRNAMES, RAW, RawConfig, read_config

SEP = "|"
DEFAULT_READER_ENGINE = "pyarrow"
FALLBACK_READER_ENGINE = "python"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)


class TryFallback(Exception):
    pass


class InconsistentSchema(Exception):
    def __init__(self, *args, df: pd.DataFrame):
        super().__init__(*args)
        self.df = df


def _make_reader(raw_config: RawConfig) -> Callable[[Path, bool], pd.DataFrame]:
    """
    Uses a structured configuration dict to return a reader function that will appropriately read
    the csv file and raise appropriate warnings to monitor data ingestion and ensure observability.

    :param raw_config: the configuration for the raw data
    :returns: a that takes a path and a boolean fallback=False or True. If there is an issue reading the data using the
        DEFAULT_READER_ENGINE then the function will raise TryFallback. This event should be logged. If the column set
        does not match the expected then it will raise InconsistentSchema. The read dataframe can still be accessed via
        the `df` field of the InconsistentSchema exception. This should also be logged.
    """

    logging.debug(f"[INIT] Setting up reader with configuration: \n{pformat(raw_config, indent=4)}")

    columns_conf = raw_config.get("columns")
    untyped: list[str] = [col for col in columns_conf if isinstance(col, str)]
    typed: list[dict[str, str]] = [col for col in columns_conf if isinstance(col, dict)]
    dtype = flatten_dicts(typed, initial={col: "string" for col in untyped})

    dates_conf = flatten_dicts(raw_config.get("dates"))
    parse_dates = list(dates_conf.keys())
    date_format = dates_conf

    reader_base = partial(
        pd.read_csv, parse_dates=parse_dates, date_format=date_format, sep=SEP
    )

    def reader(path: Path, fallback: bool = False) -> pd.DataFrame:
        stage_tag = "[READ_DFS]"
        logging.info(f"{stage_tag} Attempting to read file: {path}")
        try:
            df = reader_base(path, dtype=dtype, engine=DEFAULT_READER_ENGINE)
        except Exception as e:
            logging.warning(
                f"{stage_tag} Default engine failed for file {path}, triggering fallback.",
                exc_info=True,
            )
            if not fallback:
                raise TryFallback from e
            df = reader_base(
                path,
                dtype=defaultdict(lambda: "object", dtype),
                engine=FALLBACK_READER_ENGINE,
                on_bad_lines=handle_bad_lines,
            )

        want = set(dtype.keys()) | set(parse_dates)
        have = set(df.columns)

        unexpected_columns = have - want
        if len(unexpected_columns) > 0:
            logging.error(f"{stage_tag} Inconsistent schema detected for file {path}.")
            raise InconsistentSchema(
                f"Want columns: {want}. Have columns: {have}. Difference: {unexpected_columns}",
                df=df,
            )

        logging.info(
            f"{stage_tag} Successfully read file with consistent schema: {path}"
        )
        return df

    return reader


def _get_or_create_delta(
    path: Path, schema: pyarrow.Schema, name: str | None = None
) -> DeltaTable:
    """
    Retrieves or creates a DeltaTable at the given path with the given schema

    :param path: the path to the DeltaTable
    :param schema: the schema of the DeltaTable
    :param name: the name of the DeltaTable
    :returns: the DeltaTable
    """
    logging.info(f"[WRITE_DELTA] Accessing or creating DeltaTable at {path}.")
    try:
        return DeltaTable(path)
    except TableNotFoundError:
        logging.info(f"[WRITE_DELTA] DeltaTable not found, creating new table at {path}.")
        table = DeltaTable.create(table_uri=path, schema=schema, name=name)
        return table


class RawToBronzeStages(TypedDict):
    SETUP: Callable[[], None]
    LIST_INCOMING: Callable[[], list[Path]]
    READ_DFS: Callable[[list[Path]], pd.DataFrame]
    WRITE_DELTA: Callable[[pd.DataFrame], DeltaTable]
    ARCHIVE_PROCESSED: Callable[[DeltaTable], None]
    CLEANUP: Callable[[], None]


def gen_stages(table: Literal['questions', 'scoring'], data_dir: Path, record_name: str | None = None) -> RawToBronzeStages:
    """
    Generates the stages for ingesting raw data into a bronze DeltaTable

    :param table: the table to ingest
    :param data_dir: the directory containing the raw data
    :param record_name: the name of the record for the ingestion

    :returns: a dictionary of functions representing the stages
    """

    # constants wrt the stages
    RAW_DIR = data_dir / DIRNAMES[RAW]
    PROCESSED_DIR = RAW_DIR / "processed"
    RAW_CONF = read_config(RAW, table)
    BRONZE_DIR = data_dir / DIRNAMES[BRONZE]
    READER = _make_reader(RAW_CONF)
    RECORD_NAME = record_name or f'{table}_ingested_{datetime.today().strftime("%Y-%m-%d")}'

    logging.info(f"[INIT] Setting up stages for table: {table}.\n")
    logging.info(f"RAW_DIR: {RAW_DIR}\n")
    logging.info(f"RAW_CONF: {RAW_CONF}\n")
    logging.info(f"BRONZE_DIR: {BRONZE_DIR}\n")
    
    logging.info(f"[INIT] Records of ingestion and archived files will be under the name {RECORD_NAME}...\n")
    logging.info(f'Logs will be found under {str(RAW_DIR)}/processed/{RECORD_NAME}.log\n')
    logging.info(f'Source data will be found under {str(RAW_DIR)}/processed/{RECORD_NAME}.tar.gz\n')

    # Add file handler to the root logger
    def setup(*args, **kwargs) -> None:
        nonlocal PROCESSED_DIR, RECORD_NAME
        logging.info(f"[SETUP] Setting up ingestion stages for table: {table}.")
        if not PROCESSED_DIR.exists():
            logging.info(f"[INIT] Creating processed directory: {PROCESSED_DIR}.")
            PROCESSED_DIR.mkdir()
        file_handler = logging.FileHandler(PROCESSED_DIR / f"{RECORD_NAME}.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - [%(filename)s:%(lineno)d] - %(message)s')
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)

    def list_incoming(*args, **kwargs) -> list[Path]:
        """
        list_incoming lists the incoming files in the raw directory that match the glob pattern
        specified in the configuration file.

        :returns: a list of Path objects representing the incoming files
        """
        nonlocal RAW_DIR, RAW_CONF
        logging.info(f"[LIST_INCOMING] Listing incoming {table} files in directory: {RAW_DIR}.")
        globpat: str = RAW_CONF.get("glob")
        incoming: list[Path] = list(RAW_DIR.glob(globpat))
        logging.info(f"[LIST_INCOMING] Found {len(incoming)} files matching pattern {globpat}.")
        return incoming

    def read_dfs(incoming: list[Path], *args, **kwargs) -> pd.DataFrame:
        """
        read_dfs reads the incoming files and applies metadata to the dataframe

        :param incoming: a list of Path objects representing the incoming files
        :returns: a dataframe with the metadata applied
        """
        nonlocal READER

        logging.info("[READ_DFS] Reading dataframes from incoming files.")
        dfs = []
        for file in incoming:
            metadata = MetadataParser.parse(file.name).get_mapping()
            try:
                df = READER(file, False)
            except TryFallback:
                logging.warning(f"[READ_DFS] Fallback reader activated for file: {file}.")
                df = READER(file, True)
            df = df.assign(**metadata, INGESTION_TIMESTAMP=pd.to_datetime(datetime.now()))
            df = enforce_datetime64us(df)
            dfs.append(df)
        joined = pd.concat(dfs, ignore_index=True)
        return joined

    def write_delta(df: pd.DataFrame, *args, **kwargs) -> DeltaTable:
        """
        write_delta writes the dataframe to the bronze directory as a DeltaTable

        :param df: the dataframe to write
        :returns: the DeltaTable
        """
        nonlocal table, BRONZE_DIR
        logging.info(f"[WRITE_DELTA] Writing dataframe to DeltaTable at {BRONZE_DIR / table}.")
        schema = pyarrow.Schema.from_pandas(df)
        deltatable = _get_or_create_delta(path=BRONZE_DIR / table, schema=schema, name=table)
        write_deltalake(deltatable, data=df, schema=schema, mode="append")
        return deltatable


    def archive_processed(delta: DeltaTable) -> list[Path]:
        nonlocal RAW_DIR, PROCESSED_DIR, RECORD_NAME
        logging.info(f"[ARCHIVE_PROCESSED] Archiving processed files to {PROCESSED_DIR / RECORD_NAME}.tar.gz.")
        output_path = PROCESSED_DIR / f"{RECORD_NAME}.tar.gz"

        file_paths = delta.to_pandas(columns=['FILE_NAME']).FILE_NAME.unique()
        file_paths = [(RAW_DIR / file) for file in file_paths]

        with tarfile.open(output_path, "w:gz") as tar:
            for file in file_paths:
                tar.add(file, arcname=file.name)
        
        initial_size = sum(file.stat().st_size for file in file_paths)
        final_size = output_path.stat().st_size
        logging.info(f"[ARCHIVE_PROCESSED] Archived {len(file_paths)} files to {output_path}.")
        logging.info(f"[ARCHIVE_PROCESSED] Initial size: {initial_size} bytes. Final size: {final_size} bytes.")

        return file_paths
    
    def cleanup(files: list[Path], *args, **kwargs) -> None:
        logging.info(f"[CLEANUP] Cleaning up ingestion stages for table: {table}.")

        for file in files:
            logging.info(f"[CLEANUP] Removing file: {file}.")
            file.unlink()

        logging.info("[CLEANUP] Removing file handler from root logger.")
        logging.getLogger().removeHandler(logging.getLogger().handlers[-1])

    return {
        "SETUP": setup,
        "LIST_INCOMING": list_incoming,
        "READ_DFS": read_dfs,
        "WRITE_DELTA": write_delta,
        "ARCHIVE_PROCESSED": archive_processed,
        "CLEANUP": cleanup,
    }

def run_stages(stages: RawToBronzeStages) -> None:
    """
    Runs the stages for ingesting raw data into a bronze DeltaTable

    :param stages: the stages to run
    """
    logging.info("[RUN_STAGES] Running stages.")
    from functools import reduce
    reduce(lambda f, g: g(f), stages.values(), None)
    logging.info("[RUN_STAGES] Stages complete.")