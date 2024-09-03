from pathlib import Path
import logging
import tarfile
from typing import Optional, Sequence
from enum import StrEnum

from raipy_elt.utilities.misc import get_or_mkdir


class CompressAlg(StrEnum):
    """
    the compression algorithms natively supported by Python
    """

    GZIP = "gz"
    BZIP2 = "bz2"
    LZMA = "xz"


GZIP, BZIP2, LZMA = CompressAlg


def tarball_files(
    src_files: Sequence[Path],
    dest_dir: Path,
    dest_fname: str,
    cmprsn: Optional[CompressAlg] = GZIP,
    retain_structure: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    receives a list (sequence) of files from which to create a tarball for cold storage and creates a
    tarfile `{dest_dir}/{dest_file_name}.tar.{alg}`

    :param src_files: list of files to include in the archive.
    :param dest_dir: directory in which the tarball should be placed
    :param dest_fname: file name for the tarball (excluding the parent directories), without suffix.
    :param alg: compression algorithm to use (defaults to gzip)
    :param retain_structure: retain the directory structure above the compressed files (false by default)
    :logger: the logger used for logging messages in the function
    """
    logger = logger or logging.getLogger(__file__)
    debug, info, warning = logger.debug, logger.info, logger.warning

    debug(
        f"tarball_files requested with parameters {src_files=}, {dest_dir=}, {dest_fname=}, {cmprsn=}, {retain_structure=}"
    )

    for file in src_files:
        info(f"checking {file=} exists and is a file")
        if not file.exists():
            warning("received a file that does not exist.")
            raise FileNotFoundError(
                f"attempt to include {file=} in tarball, but file does not exist."
            )

        if not file.is_file():
            warning("received a path to something other than a file.")
            raise ValueError(
                f"attempt to include {file=} in tarball, but path does not point to a file."
            )
        info(f"verified {file=} ok.")

    dest_dir = get_or_mkdir(dest_dir)

    dest_pth = dest_dir / f"{dest_fname}.tar{"" if not cmprsn else cmprsn}"
    md = f'w:{"" if not cmprsn else cmprsn}'

    debug(f"attempting to open file {dest_pth=} for writing in mode {md=}")
    with tarfile.open(dest_pth, mode=md) as tar:
        info(f"tar file {dest_pth} opened")
        for file in src_files:
            try:
                tar.add(file, recursive=retain_structure)
                debug(f"added {file=}.")
            except:
                warning(f"an error occurred adding {file=}", exc_info=True)
                raise
        info("all files added to tarball.")

    info("returning path to new tarball")
    return dest_pth
