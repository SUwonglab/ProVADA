"""Tests for fetching and caching the datasets a run downloads on first use.

Nothing here touches the network: every test serves the archive from memory, so
what is covered is the checking around the download rather than the download.
"""

import hashlib
import io
import tarfile

import pytest

from provada.utils import datasets
from provada.utils.datasets import (
    HUMAN_PEPTIDOME_9MER,
    ManagedDataset,
    resolve_dataset,
)

PEPTIDES = "AAAWYLWEV\nGILGFVFTL\nNLVPMVATV\n"


def make_archive(names=("peptides.txt",), content: str = PEPTIDES) -> bytes:
    """A gzipped tar holding one file per name, like the published archive."""
    raw = content.encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in names:
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()


class FakeResponse:
    """Stands in for an HTTP response, which is read in chunks."""

    def __init__(self, body: bytes, status: int = 200):
        self.status = status
        self.body = io.BytesIO(body)

    def read(self, size: int = -1) -> bytes:
        return self.body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def served(monkeypatch, body: bytes, status: int = 200) -> list:
    """Serve ``body`` for every download, and record the URLs requested."""
    requested = []

    def fake_urlopen(url, timeout=None):
        requested.append(url)
        return FakeResponse(body, status=status)

    monkeypatch.setattr(datasets.urllib.request, "urlopen", fake_urlopen)
    return requested


def pinned_to(archive: bytes, name: str = "test_peptidome") -> ManagedDataset:
    """A dataset pinned to the size and checksum of ``archive``."""
    return ManagedDataset(
        name=name,
        url="https://example.invalid/archive.tar.gz",
        md5=hashlib.md5(archive).hexdigest(),
        size=len(archive),
        citation="a test fixture",
        description="test peptidome",
    )


@pytest.fixture(autouse=True)
def cache_in_tmp_path(monkeypatch, tmp_path):
    """Keep every test's cache to itself, and off the real one."""
    monkeypatch.setenv("PROVADA_DATA_DIR", str(tmp_path / "data"))


def test_resolve_downloads_once_and_reuses_the_cache(monkeypatch):
    """The second call reads the cache rather than the network."""
    archive = make_archive()
    requested = served(monkeypatch, archive)
    dataset = pinned_to(archive)

    resolved = resolve_dataset(dataset)
    assert resolved.read_text() == PEPTIDES
    assert resolved.parent.name == dataset.name
    assert len(requested) == 1

    assert resolve_dataset(dataset) == resolved
    assert len(requested) == 1, "a cached dataset must not be downloaded again"


@pytest.mark.parametrize("truncate", [False, True])
def test_resolve_keeps_nothing_that_is_not_the_published_file(monkeypatch, truncate):
    """A wrong checksum, or a body that stops early, is discarded rather than cached."""
    archive = make_archive()
    dataset = pinned_to(archive)
    if truncate:
        served(monkeypatch, archive[: len(archive) // 2])
    else:
        served(monkeypatch, archive)
        dataset = ManagedDataset(**{**dataset.__dict__, "md5": "0" * 32})

    with pytest.raises(RuntimeError, match="not the published file"):
        resolve_dataset(dataset)

    cached = datasets.dataset_cache_dir() / dataset.name
    assert not any(cached.iterdir()), "nothing may be left behind to be trusted later"


def test_resolve_rejects_a_download_that_is_not_ready(monkeypatch):
    """A 202 with an empty body is the publisher still preparing the file.

    It is not an error status, so a status check alone would accept it and cache
    nothing, which is the failure this guards against.
    """
    served(monkeypatch, b"", status=202)

    with pytest.raises(RuntimeError, match="returned HTTP 202"):
        resolve_dataset(pinned_to(make_archive()))


def test_resolve_reports_a_failed_request(monkeypatch):
    """A refused or failing request says where to get the file by hand."""

    def fake_urlopen(url, timeout=None):
        raise datasets.urllib.error.HTTPError(url, 500, "Server Error", {}, None)

    monkeypatch.setattr(datasets.urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Could not download"):
        resolve_dataset(pinned_to(make_archive()))


def test_resolve_rejects_an_archive_holding_more_than_one_file(monkeypatch):
    """An archive whose layout changed is reported rather than guessed at."""
    archive = make_archive(names=("one.txt", "two.txt"))
    served(monkeypatch, archive)

    with pytest.raises(RuntimeError, match="hold one file, found 2"):
        resolve_dataset(pinned_to(archive))


def test_the_human_peptidome_is_pinned():
    """The published asset's identity is recorded, so a cached copy is checkable."""
    assert HUMAN_PEPTIDOME_9MER.md5 == "c51aa2f180e8d0e8425d482720cd0c4f"
    assert HUMAN_PEPTIDOME_9MER.size == 69258431
    assert "10.6084/m9.figshare.34517100" in HUMAN_PEPTIDOME_9MER.citation


def test_base_variant_loads_the_peptides_to_treat_as_self(monkeypatch, tmp_path):
    """A configured file is read; with no path, the managed peptidome is fetched."""
    from provada.base_variant import BaseVariant

    path = tmp_path / "peptides.txt"
    path.write_text(PEPTIDES + "\n")

    variant = BaseVariant(sequence="ACDEFGHIK")
    variant.load_self_peptide_db(str(path))
    assert variant.self_peptide_db == {"AAAWYLWEV", "GILGFVFTL", "NLVPMVATV"}

    archive = make_archive()
    served(monkeypatch, archive)
    monkeypatch.setattr(
        datasets, "HUMAN_PEPTIDOME_9MER", pinned_to(archive, name="human_peptidome9")
    )

    fetched = BaseVariant(sequence="ACDEFGHIK")
    fetched.load_self_peptide_db()
    assert fetched.self_peptide_db == variant.self_peptide_db


def test_base_variant_rejects_a_database_of_the_wrong_length(tmp_path):
    """A database that covers no 9mer would exclude nothing, silently."""
    from provada.base_variant import BaseVariant

    path = tmp_path / "peptides.txt"
    path.write_text("AAAWYLWE\nGILGFVFTLM\n")

    variant = BaseVariant(sequence="ACDEFGHIK")
    with pytest.raises(ValueError, match="none of them 9"):
        variant.load_self_peptide_db(str(path))


def test_exclusion_only_removes_self_peptides_that_would_score():
    """A self peptide below the floor stays counted, which is what the floor is for.

    Exclusion is checked only for peptides whose score passes 0.05, so a self
    peptide predicted as a non-binder contributes its (negligible) score like any
    other. A strong self binder is dropped from both aggregates.
    """
    from provada.components.evaluator import score_epitope_scan

    class FakeBinding:
        def __init__(self, peptide, rank_ba):
            self.peptide = peptide
            self.rank_ba = rank_ba

    class FakeScan:
        def __init__(self, bindings, sequence_length):
            self.bindings = bindings
            self.sequence_length = sequence_length

    # One strong self binder, one weak self peptide, one strong foreign binder.
    scan = FakeScan(
        [
            FakeBinding("FLWDDSMAL", 0.01),
            FakeBinding("GGGGSGGGG", 94.0),
            FakeBinding("KLQCVQLHV", 0.02),
        ],
        sequence_length=100,
    )
    self_peptides = {"FLWDDSMAL", "GGGGSGGGG"}

    counted = score_epitope_scan(scan, num_alleles=1, self_peptide_db=None)
    excluded = score_epitope_scan(scan, num_alleles=1, self_peptide_db=self_peptides)

    # The strong self binder is worth ~1.0 and is the only thing removed.
    assert counted["immunogenicity_score"] - excluded["immunogenicity_score"] == pytest.approx(
        0.98, abs=0.02
    )
    # Only the two strong binders are positive; excluding one leaves one.
    assert counted["allele_averaged_positive_pep_ratio"] == pytest.approx(2 / 100)
    assert excluded["allele_averaged_positive_pep_ratio"] == pytest.approx(1 / 100)
