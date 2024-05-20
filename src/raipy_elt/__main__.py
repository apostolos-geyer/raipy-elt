import os
from pathlib import Path

import click


def raw_to_bronze():
    import os
    from pathlib import Path

    if not (data_dir := os.getenv("DATA_DIR", None)):
        print(
            "DATA_DIR environment variable not set. Please set it to the path of the data directory"
            "\n(The directory that contains the subdirectory 0-raw, etc...)"
        )

        print("i.e, call the script like this: DATA_DIR=/path/to/data raw2bronze")
        exit(1)
    
    from raipy_elt.pipelines.landing import RawToBronze

    RawToBronze.set_parameters(
        raw_dir = Path(data_dir) / "0-raw",
        bronze_dir = Path(data_dir) / "1-bronze"
    )

    RawToBronze.run()


@click.group()
def raipy_elt():
    pass


@raipy_elt.command()
@click.option(
    "--data-dir",
    "-d",
    type=str,
    help="The directory containing the raw data",
    default=lambda: os.environ.get("DATA_DIR", None),
)
def ingest_raw(data_dir: os.PathLike | None):
    """
    Ingest raw data into the bronze layer
    """

    if data_dir is None:
        data_dir = click.prompt(
            "Please enter the path to the data directory.."
            "\nhint: you can set the DATA_DIR environment variable or pass -d /path/to/data_dir to avoid the prompt."
        )

    from raipy_elt.pipelines.landing import RawToBronze

    RawToBronze.set_parameters(
        raw_dir = Path(data_dir) / "0-raw",
        bronze_dir = Path(data_dir) / "1-bronze"
    )

    RawToBronze.run()


def main():
    raipy_elt()
