from functools import cache
from pathlib import Path
from typing import overload

import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake  # noqa: F401
from deltalake.exceptions import TableNotFoundError


def get_arrow_schema(df: pd.DataFrame) -> pa.Schema:
    """
    Get the Arrow schema from a pandas DataFrame

    :param df: the DataFrame to get the schema from
    :returns: the Arrow schema
    """
    return pa.Schema.from_pandas(df)


def get_change_data(
    delta: DeltaTable,
    from_version: int = 0,
    to_version: int | None = None,
) -> pd.DataFrame:
    """
    Get the change data from a DeltaTable. Convenience wrapper that uses the dataframe interchange format as
    sometimes the recommended API can have issues converting from arrow format to pandas

    :param delta: the DeltaTable to get the change data from
    :param from_version: the starting version
    :param to_version: the ending version
    :param columns: the columns to include in the change data
    :returns: the change data
    """
    cdf: pa.Table = delta.load_cdf(
        starting_version=from_version, ending_version=to_version
    ).read_all()
    cdf: pd.api.interchange.DataFrame = cdf.__dataframe__()
    cdf: pd.DataFrame = pd.api.interchange.from_dataframe(cdf)
    return cdf


@overload
def get_or_create_delta(
    path: Path,
    for_data: pd.DataFrame,
    name: str | None = None,
    partition_cols: list[str] | None = None,
) -> DeltaTable:
    """
    Retrieves or creates a DeltaTable at the given path with a schema inferred from the given data

    :param path: the path to the DeltaTable
    :param for_data: the data to infer the schema from
    :param name: the name of the DeltaTable
    :returns: the DeltaTable
    """
    ...


@overload
def get_or_create_delta(
    path: Path,
    schema: pa.Schema,
    name: str | None = None,
    partition_cols: list[str] | None = None,
) -> DeltaTable:
    """
    Retrieves or creates a DeltaTable at the given path with the given schema

    :param path: the path to the DeltaTable
    :param schema: the schema of the DeltaTable
    :param name: the name of the DeltaTable
    :returns: the DeltaTable
    """
    ...


@cache
def get_or_create_delta(
    path: Path,
    for_data: pd.DataFrame | None = None,
    schema: pa.Schema | None = None,
    name: str | None = None,
    partition_by: list[str] | None = None,
) -> DeltaTable:
    """
    Retrieves or creates a DeltaTable at the given path with the given schema or inferred from the given data

    If both `for_data` and `schema` are provided, `schema` will be used

    :param path: the path to the DeltaTable
    :param schema: the schema of the DeltaTable
    :param name: the name of the DeltaTable
    :param partition_by: the columns to partition by
    :returns: the DeltaTable
    """

    # TODO: incorporate logging

    create_kwds = dict(table_uri=path, name=name, partition_by=partition_by)

    match (schema, for_data):
        case (None, None):
            raise ValueError("Either `for_data` or `schema` must be provided")
        case (None, for_data):
            create_kwds["schema"] = get_arrow_schema(for_data)
        case (schema, _):
            create_kwds["schema"] = schema

    try:
        table = DeltaTable(path)
    except TableNotFoundError:
        table = DeltaTable.create(**create_kwds)

    return table
