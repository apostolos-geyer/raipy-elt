import os
from typing import Sequence
from pathlib import Path
from operator import or_ 
from functools import reduce


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


def get_or_mkdir(path: Path | os.PathLike | str) -> Path:
    """
    Get a directory path and create it if it doesn't exist

    :param path: the path to the directory
    :returns: the directory path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path