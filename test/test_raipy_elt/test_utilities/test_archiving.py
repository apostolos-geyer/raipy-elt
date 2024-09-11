from raipy_elt.utilities.archiving import tarball_files, CompressAlg
import tarfile
import pytest
import logging
from pathlib import Path


@pytest.fixture
def fxtr_files_dumb(tmp_path):
    """
    Fixture to create a temporary directory with some files to compress.
    """
    # Create a directory and some files to be added to the tarball
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    file_paths = []
    for i in range(3):
        file_path = test_dir / f"file{i}.txt"
        file_path.write_text(f"This is test file {i}")
        file_paths.append(file_path)

    return file_paths, tmp_path


@pytest.mark.parametrize(
    "compression_alg, retain_structure, expected_extension",
    [
        (CompressAlg.GZIP, False, ".tar.gz"),
        (CompressAlg.BZIP2, False, ".tar.bz2"),
        (CompressAlg.LZMA, False, ".tar.xz"),
        (CompressAlg.GZIP, True, ".tar.gz"),
        (CompressAlg.BZIP2, True, ".tar.bz2"),
        (CompressAlg.LZMA, True, ".tar.xz"),
        (None, False, ".tar"),  # No compression algorithm case
        (None, True, ".tar"),  # No compression with retain_structure=True
    ],
)
def test_tarball_files(
    fxtr_files_dumb, compression_alg, retain_structure, expected_extension
):
    """
    Parameterized test for the tarball_files function, including cases without compression.
    """
    src_files, tmp_dir = fxtr_files_dumb
    dest_fname = "test_archive"
    dest_dir = tmp_dir / "output_dir"
    dest_dir.mkdir()

    logger = logging.getLogger("test_logger")

    # Run the tarball_files function
    tarball_path = tarball_files(
        src_files=src_files,
        dest_dir=dest_dir,
        dest_fname=dest_fname,
        cmprsn=compression_alg,
        retain_structure=retain_structure,
        logger=logger,
    )

    # Check if the tarball is created
    assert tarball_path.exists()

    tbfname = tarball_path.name
    suf = tbfname[tbfname.find(".") :]
    assert suf == expected_extension

    # Check the contents of the tarball
    mode = f"r:{compression_alg.value}" if compression_alg else "r"
    with tarfile.open(tarball_path, mode=mode) as tar:
        tar_names = tar.getnames()

        if retain_structure:
            expected_paths = (str(f)[1:] for f in src_files)
        else:
            expected_paths = [file.name for file in src_files]

        for path in expected_paths:
            assert path in tar_names


@pytest.mark.parametrize(
    "invalid_file_path, error_type",
    [
        (Path("/invalid/file/path.txt"), FileNotFoundError),  # Non-existent file path
        (
            lambda tmp_dir: tmp_dir / "some_directory",
            ValueError,
        ),  # Path exists, but is a directory
    ],
)
def test_tarball_files_invalid_files(tmp_path, invalid_file_path, error_type):
    """
    Test cases for invalid file paths, expecting errors.
    """
    dest_fname = "test_archive"
    dest_dir = tmp_path / "output_dir"
    dest_dir.mkdir()

    logger = logging.getLogger("test_logger")

    # Create the invalid path or directory as needed
    if callable(invalid_file_path):
        invalid_file_path = invalid_file_path(tmp_path)
        invalid_file_path.mkdir()  # This creates the directory

    src_files = [invalid_file_path]

    # Expect the appropriate exception based on the error case
    with pytest.raises(error_type):
        tarball_files(
            src_files=src_files,
            dest_dir=dest_dir,
            dest_fname=dest_fname,
            cmprsn=CompressAlg.GZIP,
            retain_structure=False,
            logger=logger,
        )
