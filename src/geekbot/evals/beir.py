"""BEIR datasets: download them once, check they did not change, read them (step S2.4, PLAN.md §9).

BEIR is a public benchmark of retrieval datasets. The plan uses two as *standardized retrieval
evidence*: SciFact (5,183 documents, 300 test queries) and NFCorpus (3,633 documents, 323 test
queries). Each comes as a zip with the same layout::

    scifact/corpus.jsonl      {"_id": "4983", "title": "...", "text": "..."}   one document a line
    scifact/queries.jsonl     {"_id": "1", "text": "..."}                      one query a line
    scifact/qrels/test.tsv    query-id <TAB> corpus-id <TAB> score             a header row, then judgements

The ``qrels`` ("query relevance judgements") say which documents answer which query, with a grade:
SciFact's are 0 or 1, NFCorpus's go higher (``evals.metrics`` handles both). A split's queries are
only those that have judgements in that split's file, which is how BEIR's own evaluator counts them.

**Which split for what** (D-065): choices made by trying values use SciFact's ``train`` part and
NFCorpus's ``dev`` part, and the headline numbers use ``test``. ``TUNING_SPLIT`` and
``HEADLINE_SPLIT`` hold that, so no run spells it by hand.

**Integrity: trust on first use.** BEIR publishes the zips but this code has no list of their
checksums to compare with (none was fetched from here). So the first download records its SHA-256
in ``checksums.json`` beside the data, and every later download of the same dataset must match it;
a change stops the run instead of silently scoring against different data. That protects against a
dataset changing under you, not against a bad first download: if the first fetch is suspect, compare
its recorded hash with one published by BEIR before relying on it. The archive is also bounded
(size cap, no member may leave the folder) because it is downloaded from the network.

Nothing here runs a retrieval; S2.6 does that.
"""

import hashlib
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx

__all__ = [
    "DATASET_NAMES",
    "HEADLINE_SPLIT",
    "TUNING_SPLIT",
    "BeirDataset",
    "BeirDocument",
    "BeirError",
    "FetchResult",
    "available_splits",
    "fetch_beir",
    "load_beir",
]

# BEIR's published location (its README). Not checked from this PC: the first `geekbot eval fetch`
# is the first test of it, and a wrong address fails with a clear error.
BEIR_URL: Final = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DATASET_NAMES: Final = ("scifact", "nfcorpus")
TUNING_SPLIT: Final = {"scifact": "train", "nfcorpus": "dev"}
HEADLINE_SPLIT: Final = "test"

CHECKSUMS_FILE: Final = "checksums.json"
MAX_DOWNLOAD_BYTES: Final = 200 * 1024 * 1024  # the real zips are a few MB
MAX_UNPACKED_BYTES: Final = 1024 * 1024 * 1024
READ_BLOCK_BYTES: Final = 1 << 16
QRELS_COLUMNS: Final = 3


class BeirError(RuntimeError):
    """A dataset could not be downloaded, did not match its recorded checksum, or is malformed."""


@dataclass(frozen=True)
class BeirDocument:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class BeirDataset:
    """One dataset and one split: documents, the queries with judgements, and the judgements."""

    name: str
    split: str
    corpus: Mapping[str, BeirDocument]
    queries: Mapping[str, str]
    qrels: Mapping[str, Mapping[str, int]]

    def summary(self) -> str:
        judgements = sum(len(rows) for rows in self.qrels.values())
        return (
            f"{self.name} {self.split}: {len(self.corpus)} documents, "
            f"{len(self.queries)} queries with judgements, {judgements} judgements"
        )


@dataclass(frozen=True)
class FetchResult:
    """What ``fetch_beir`` did. ``sha256`` is that of the zip, as recorded (``None`` if unknown)."""

    name: str
    folder: Path
    downloaded: bool
    sha256: str | None
    size_bytes: int
    new_checksum: bool


# --- downloading -----------------------------------------------------------------------------


def _is_extracted(folder: Path) -> bool:
    return (
        (folder / "corpus.jsonl").is_file()
        and (folder / "queries.jsonl").is_file()
        and (folder / "qrels").is_dir()
    )


def _read_checksums(destination: Path) -> dict[str, str]:
    path = destination / CHECKSUMS_FILE
    if not path.is_file():
        return {}
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BeirError(f"{path} is not readable JSON: {error}") from error
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise BeirError(f"{path} must map dataset names to SHA-256 hex strings")
    return dict(data)


def _download(client: httpx.Client, url: str, part: Path) -> tuple[str, int]:
    """Stream ``url`` into ``part``; return its SHA-256 and size. ``part`` is removed on failure."""
    digest = hashlib.sha256()
    size = 0
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with part.open("wb") as handle:
                for block in response.iter_bytes(READ_BLOCK_BYTES):
                    size += len(block)
                    if size > MAX_DOWNLOAD_BYTES:
                        raise BeirError(
                            f"{url} is larger than {MAX_DOWNLOAD_BYTES} bytes; refusing to continue"
                        )
                    digest.update(block)
                    handle.write(block)
    except (httpx.HTTPError, OSError) as error:
        part.unlink(missing_ok=True)
        raise BeirError(f"could not download {url}: {error}") from error
    except BeirError:
        part.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), size


def _extract(archive: Path, destination: Path) -> None:
    """Unpack into ``destination`` after checking the total size and that no member escapes it."""
    root = destination.resolve()
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if sum(member.file_size for member in members) > MAX_UNPACKED_BYTES:
                raise BeirError(f"{archive.name} unpacks to more than {MAX_UNPACKED_BYTES} bytes")
            for member in members:
                if not (root / member.filename).resolve().is_relative_to(root):
                    raise BeirError(f"{archive.name}: {member.filename!r} is outside the folder")
            bundle.extractall(root)
    except zipfile.BadZipFile as error:
        raise BeirError(f"{archive.name} is not a zip file") from error


def fetch_beir(
    name: str, destination: Path, *, client: httpx.Client, force: bool = False
) -> FetchResult:
    """Make sure ``destination/name`` holds the dataset, downloading it if it does not.

    Already unpacked and not ``force``: nothing is downloaded. Otherwise the zip is streamed to a
    temporary file, its SHA-256 compared with the one recorded for this dataset (recorded now if it
    is the first download, see the module docstring), and only then moved into place and unpacked.
    ``BeirError`` for a failed download, a changed checksum, an unsafe or oversized archive, or one
    that does not contain the three expected parts. ``ValueError`` for an unknown dataset name.
    """
    if name not in DATASET_NAMES:
        raise ValueError(f"unknown BEIR dataset {name!r}; choose from {', '.join(DATASET_NAMES)}")
    folder = destination / name
    if _is_extracted(folder) and not force:
        return FetchResult(name, folder, False, _read_checksums(destination).get(name), 0, False)
    destination.mkdir(parents=True, exist_ok=True)
    checksums = _read_checksums(destination)  # before the download: a damaged record stops early
    known = checksums.get(name)
    part = destination / f"{name}.zip.part"
    digest, size = _download(client, BEIR_URL.format(name=name), part)
    if known is not None and known != digest:
        part.unlink(missing_ok=True)
        raise BeirError(
            f"{name}: the download's SHA-256 {digest} differs from the one recorded in "
            f"{destination / CHECKSUMS_FILE} ({known}). The dataset changed or the download is "
            "damaged; nothing was unpacked."
        )
    archive = destination / f"{name}.zip"
    part.replace(archive)
    try:
        _extract(archive, destination)
        if not _is_extracted(folder):
            raise BeirError(f"{archive.name} lacks {name}/corpus.jsonl, queries.jsonl or qrels/")
    except BeirError:
        archive.unlink(missing_ok=True)  # a rejected archive must not become the reference
        raise
    if known is None:  # recorded only now, so a bad first download is never the reference
        checksums[name] = digest
        (destination / CHECKSUMS_FILE).write_text(
            json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return FetchResult(name, folder, True, digest, size, known is None)


# --- reading ---------------------------------------------------------------------------------


def available_splits(folder: Path) -> list[str]:
    """The splits that have a judgements file (``qrels/<split>.tsv``), sorted."""
    return sorted(path.stem for path in (folder / "qrels").glob("*.tsv"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row: object = json.loads(raw)
        except json.JSONDecodeError as error:
            raise BeirError(f"{path.name} line {number}: not valid JSON ({error.msg})") from error
        if not isinstance(row, dict) or "_id" not in row:
            raise BeirError(f"{path.name} line {number}: expected an object with an '_id'")
        rows.append(row)
    return rows


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split("\t")
        if len(parts) != QRELS_COLUMNS:
            raise BeirError(f"{path.name} line {number}: expected 3 tab-separated columns")
        try:
            grade = int(parts[2])
        except ValueError:
            if number == 1:  # the header row: query-id, corpus-id, score
                continue
            message = f"{path.name} line {number}: score {parts[2]!r} is not a whole number"
            raise BeirError(message) from None
        qrels.setdefault(parts[0], {})[parts[1]] = grade
    return qrels


def load_beir(folder: Path, split: str) -> BeirDataset:
    """Read one split of an unpacked dataset (``folder`` is e.g. ``<data>/beir/scifact``).

    The queries returned are those with judgements in ``split``. ``BeirError`` if the split has no
    judgements file (the message lists the ones that exist), a line is malformed, or a judgement
    names a query that ``queries.jsonl`` does not have. A judgement naming a document that is not in
    the corpus is kept, as BEIR's evaluator does: that document can never be retrieved.
    """
    qrels_path = folder / "qrels" / f"{split}.tsv"
    if not qrels_path.is_file():
        raise BeirError(
            f"{folder.name} has no judgements for the split {split!r}; "
            f"it has: {', '.join(available_splits(folder)) or 'none'}"
        )
    corpus = {
        str(row["_id"]): BeirDocument(
            doc_id=str(row["_id"]),
            title=str(row.get("title") or ""),
            text=str(row.get("text") or ""),
        )
        for row in _jsonl(folder / "corpus.jsonl")
    }
    every_query = {
        str(row["_id"]): str(row.get("text") or "") for row in _jsonl(folder / "queries.jsonl")
    }
    qrels = _read_qrels(qrels_path)
    missing = sorted(set(qrels) - set(every_query))
    if missing:
        raise BeirError(
            f"{qrels_path.name} judges {len(missing)} queries that queries.jsonl lacks, "
            f"e.g. {', '.join(missing[:3])}"
        )
    return BeirDataset(
        name=folder.name,
        split=split,
        corpus=corpus,
        queries={qid: every_query[qid] for qid in qrels},
        qrels=qrels,
    )
