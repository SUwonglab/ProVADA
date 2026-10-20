"""
provada/utils/datasets.py

Datasets a run needs but the repository cannot carry, fetched and cached on
first use.

A dataset is pinned by checksum, so a cached copy is the published copy rather
than whatever a URL served that day, and a fetch either produces the pinned
bytes or raises. Nothing partial is ever left in the cache.
"""

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from provada.utils.log import get_logger

logger = get_logger(__name__)

#: Bytes read per chunk while downloading and hashing.
CHUNK_SIZE = 1 << 20

#: Seconds to wait for the download to start responding.
TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class ManagedDataset:
    """A published dataset, pinned to a size and checksum.

    Attributes:
        name: Directory the dataset is cached under, and how a config names it.
        url: Where to download the archive from.
        md5: Checksum of the archive, as the publisher gives it.
        size: Size of the archive in bytes.
        citation: Where the dataset came from, quoted in error messages so a
            failed download can be completed by hand.
        description: What the dataset holds, for logs and error messages.
    """

    name: str
    url: str
    md5: str
    size: int
    citation: str
    description: str


#: Every 9mer in the human proteome, used to exclude self peptides from
#: immunogenicity scoring: a peptide the host already presents on HLA-I is part
#: of self recognition and generally provokes no response, so counting it would
#: penalize a design for an epitope the immune system tolerates.
HUMAN_PEPTIDOME_9MER = ManagedDataset(
    name="human_peptidome9",
    url="https://ndownloader.figshare.com/files/70156035",
    md5="c51aa2f180e8d0e8425d482720cd0c4f",
    size=69258431,
    citation="figshare, doi:10.6084/m9.figshare.34517100.v1 (CC BY 4.0)",
    description="human 9mer peptidome",
)

#: Datasets a config can ask for by name.
MANAGED_DATASETS = {dataset.name: dataset for dataset in (HUMAN_PEPTIDOME_9MER,)}


def dataset_cache_dir() -> Path:
    """
    Directory cached datasets live in.

    ``PROVADA_DATA_DIR`` overrides it, which is how a shared or scratch
    filesystem gets used instead of a home directory.
    """
    override = os.environ.get("PROVADA_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "provada"


def resolve_dataset(dataset: ManagedDataset) -> Path:
    """
    Return the path to a dataset's file, fetching it on first use.

    Args:
        dataset: The dataset to resolve.

    Returns:
        Path to the single file the dataset's archive holds.

    Raises:
        RuntimeError: If the download fails, serves something other than the
            pinned bytes, or holds anything other than one file.
    """
    target_dir = dataset_cache_dir() / dataset.name
    cached = _cached_file(target_dir)
    if cached is not None:
        return cached

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Downloading the {dataset.description} ({dataset.size / 1e6:.0f} MB) to "
        f"{target_dir}. This happens once."
    )

    with tempfile.TemporaryDirectory(dir=target_dir) as staging:
        archive = Path(staging) / "archive.tar.gz"
        download(dataset, archive)
        with tarfile.open(archive) as tar:
            # Members are extracted as plain files under the target, so a
            # published archive cannot write outside it.
            tar.extractall(staging, filter="data")
        archive.unlink()

        extracted = sorted(path for path in Path(staging).rglob("*") if path.is_file())
        if len(extracted) != 1:
            raise RuntimeError(
                f"Expected the {dataset.description} archive to hold one file, found "
                f"{len(extracted)}: {[path.name for path in extracted]}. Extract it by "
                f"hand from {dataset.citation} and point the config at the file instead."
            )

        # Moved into place only once it is complete, so an interrupted fetch
        # cannot leave a partial file that later runs would trust.
        resolved = target_dir / extracted[0].name
        shutil.move(str(extracted[0]), resolved)

    logger.info(f"Cached the {dataset.description} at {resolved}")
    return resolved


def download(dataset: ManagedDataset, destination: Path) -> None:
    """
    Download a dataset's archive, and keep it only if it is the pinned archive.

    Args:
        dataset: The dataset to download.
        destination: Where to write the archive.

    Raises:
        RuntimeError: If the request fails, returns a status other than ``200``,
            or returns bytes that do not match the dataset's size and checksum.
    """
    try:
        response = urllib.request.urlopen(dataset.url, timeout=TIMEOUT_SECONDS)
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeError(
            f"Could not download the {dataset.description} from {dataset.url}: {error}. "
            f"Fetch it by hand from {dataset.citation} and point the config at the file "
            "instead."
        ) from error

    with response:
        status = getattr(response, "status", None)
        # A 202 is the publisher saying the download is still being prepared,
        # and comes with an empty body. It is not an error status, so it has to
        # be rejected by name rather than left to a status check.
        if status != 200:
            raise RuntimeError(
                f"Downloading the {dataset.description} from {dataset.url} returned HTTP "
                f"{status}, not 200. A 202 means the publisher is still preparing the "
                f"download; try again later, or fetch it by hand from {dataset.citation} "
                "and point the config at the file instead."
            )

        digest = hashlib.md5()
        written = 0
        with open(destination, "wb") as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)

    if written != dataset.size or digest.hexdigest() != dataset.md5:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"The {dataset.description} downloaded from {dataset.url} is not the "
            f"published file: got {written} bytes with md5 {digest.hexdigest()}, "
            f"expected {dataset.size} bytes with md5 {dataset.md5}. The download was "
            "discarded."
        )


def _cached_file(target_dir: Path) -> Optional[Path]:
    """Return the dataset file already cached in ``target_dir``, if there is one."""
    if not target_dir.is_dir():
        return None
    files = sorted(path for path in target_dir.iterdir() if path.is_file())
    return files[0] if len(files) == 1 else None
