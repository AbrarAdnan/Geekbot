"""Tests for step S2.4: BEIR datasets, with a tiny fake dataset and a fake server (no network)."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx
import pytest

from geekbot.evals import beir
from geekbot.evals.beir import (
    DATASET_NAMES,
    HEADLINE_SPLIT,
    TUNING_SPLIT,
    BeirError,
    available_splits,
    fetch_beir,
    load_beir,
)

CORPUS = [
    {"_id": "10", "title": "Alpha", "text": "first text"},
    {"_id": "11", "title": "", "text": "second"},
    {"_id": "12", "text": "no title key"},
]
QUERIES = [
    {"_id": "q1", "text": "what is alpha"},
    {"_id": "q2", "text": "second?"},
    {"_id": "q3", "text": "judged in train only"},
]
TEST_QRELS = "query-id\tcorpus-id\tscore\nq1\t10\t1\nq1\t11\t0\nq2\t12\t2\n"
TRAIN_QRELS = "query-id\tcorpus-id\tscore\nq3\t10\t1\n"


def jsonl(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def members(name: str = "scifact") -> dict[str, str]:
    return {
        f"{name}/corpus.jsonl": jsonl(CORPUS),
        f"{name}/queries.jsonl": jsonl(QUERIES),
        f"{name}/qrels/test.tsv": TEST_QRELS,
        f"{name}/qrels/train.tsv": TRAIN_QRELS,
    }


def make_zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for member, text in files.items():
            bundle.writestr(member, text)
    return buffer.getvalue()


def write_dataset(root: Path, name: str = "scifact") -> Path:
    for member, text in members(name).items():
        target = root / member
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return root / name


def serving(content: bytes, requests: list[httpx.Request] | None = None) -> httpx.Client:
    def handle(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        return httpx.Response(200, content=content)

    return httpx.Client(transport=httpx.MockTransport(handle))


def failing(status: int) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(status)))


def never_called() -> httpx.Client:
    def handle(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("no download was expected")

    return httpx.Client(transport=httpx.MockTransport(handle))


# --- constants -------------------------------------------------------------------------------


def test_the_datasets_and_the_splits_follow_the_plan() -> None:
    assert DATASET_NAMES == ("scifact", "nfcorpus")
    assert TUNING_SPLIT == {"scifact": "train", "nfcorpus": "dev"}  # D-065
    assert HEADLINE_SPLIT == "test"


# --- reading ---------------------------------------------------------------------------------


def test_a_split_has_its_documents_its_judged_queries_and_its_judgements(tmp_path: Path) -> None:
    data = load_beir(write_dataset(tmp_path), "test")
    assert (data.name, data.split) == ("scifact", "test")
    assert {k: (v.title, v.text) for k, v in data.corpus.items()} == {
        "10": ("Alpha", "first text"),
        "11": ("", "second"),
        "12": ("", "no title key"),
    }
    assert data.queries == {"q1": "what is alpha", "q2": "second?"}  # q3 has no test judgement
    assert data.qrels == {"q1": {"10": 1, "11": 0}, "q2": {"12": 2}}  # grades kept, 0 included
    assert data.summary() == "scifact test: 3 documents, 2 queries with judgements, 3 judgements"


def test_another_split_has_other_queries(tmp_path: Path) -> None:
    data = load_beir(write_dataset(tmp_path), "train")
    assert data.queries == {"q3": "judged in train only"}


def test_the_available_splits_are_the_qrels_files(tmp_path: Path) -> None:
    assert available_splits(write_dataset(tmp_path)) == ["test", "train"]


def test_a_missing_split_says_which_exist(tmp_path: Path) -> None:
    with pytest.raises(BeirError, match=r"no judgements for the split 'dev'.*it has: test, train"):
        load_beir(write_dataset(tmp_path), "dev")


def test_a_qrels_file_without_a_header_is_read_too(tmp_path: Path) -> None:
    folder = write_dataset(tmp_path)
    (folder / "qrels" / "test.tsv").write_text("q1\t10\t1\n", encoding="utf-8")
    assert load_beir(folder, "test").qrels == {"q1": {"10": 1}}


def test_a_judgement_for_a_document_not_in_the_corpus_is_kept(tmp_path: Path) -> None:
    folder = write_dataset(tmp_path)
    (folder / "qrels" / "test.tsv").write_text("q1\t999\t1\n", encoding="utf-8")
    assert load_beir(folder, "test").qrels == {"q1": {"999": 1}}  # never retrievable, as in BEIR


def test_numeric_ids_are_read_as_text(tmp_path: Path) -> None:
    folder = write_dataset(tmp_path)
    (folder / "corpus.jsonl").write_text('{"_id": 5, "text": "x"}\n', encoding="utf-8")
    assert "5" in load_beir(folder, "test").corpus


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (
            "query-id\tcorpus-id\tscore\nq1\t10\n",
            "test.tsv line 2: expected 3 tab-separated columns",
        ),
        ("query-id\tcorpus-id\tscore\nq1\t10\tmany\n", "test.tsv line 2: score 'many' is not"),
        (
            "query-id\tcorpus-id\tscore\nq9\t10\t1\n",
            "judges 1 queries that queries.jsonl lacks, e.g. q9",
        ),
    ],
)
def test_a_malformed_qrels_file_is_refused(tmp_path: Path, text: str, fragment: str) -> None:
    folder = write_dataset(tmp_path)
    (folder / "qrels" / "test.tsv").write_text(text, encoding="utf-8")
    with pytest.raises(BeirError) as raised:
        load_beir(folder, "test")
    assert fragment in str(raised.value)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ('{"_id": "1"}\nnot json\n', "corpus.jsonl line 2: not valid JSON"),
        ('{"text": "no id"}\n', "corpus.jsonl line 1: expected an object with an '_id'"),
        ('["list"]\n', "corpus.jsonl line 1: expected an object"),
    ],
)
def test_a_malformed_corpus_file_is_refused(tmp_path: Path, text: str, fragment: str) -> None:
    folder = write_dataset(tmp_path)
    (folder / "corpus.jsonl").write_text(text, encoding="utf-8")
    with pytest.raises(BeirError) as raised:
        load_beir(folder, "test")
    assert fragment in str(raised.value)


# --- downloading -----------------------------------------------------------------------------


def test_a_first_download_is_unpacked_and_its_checksum_recorded(tmp_path: Path) -> None:
    payload = make_zip(members())
    requests: list[httpx.Request] = []
    with serving(payload, requests) as client:
        result = fetch_beir("scifact", tmp_path / "beir", client=client)
    digest = hashlib.sha256(payload).hexdigest()
    assert (result.downloaded, result.new_checksum, result.sha256) == (True, True, digest)
    assert result.size_bytes == len(payload)
    assert result.folder == tmp_path / "beir" / "scifact"
    assert (result.folder / "corpus.jsonl").is_file()
    assert json.loads((tmp_path / "beir" / "checksums.json").read_text("utf-8")) == {
        "scifact": digest
    }
    assert not list((tmp_path / "beir").glob("*.part"))
    assert [(r.url.host, r.url.path) for r in requests] == [
        ("public.ukp.informatik.tu-darmstadt.de", "/thakur/BEIR/datasets/scifact.zip")
    ]
    assert load_beir(result.folder, "test").summary().startswith("scifact test: 3 documents")


def test_a_dataset_already_unpacked_is_not_downloaded_again(tmp_path: Path) -> None:
    destination = tmp_path / "beir"
    payload = make_zip(members())
    with serving(payload) as client:
        first = fetch_beir("scifact", destination, client=client)
    with never_called() as client:
        again = fetch_beir("scifact", destination, client=client)
    assert (again.downloaded, again.new_checksum) == (False, False)
    assert again.sha256 == first.sha256  # the recorded one


def test_force_downloads_again_and_the_same_bytes_match_the_record(tmp_path: Path) -> None:
    destination = tmp_path / "beir"
    payload = make_zip(members())
    with serving(payload) as client:
        fetch_beir("scifact", destination, client=client)
        again = fetch_beir("scifact", destination, client=client, force=True)
    assert (again.downloaded, again.new_checksum) == (True, False)


def test_a_changed_download_is_refused_and_nothing_is_replaced(tmp_path: Path) -> None:
    destination = tmp_path / "beir"
    original = make_zip(members())
    with serving(original) as client:
        fetch_beir("scifact", destination, client=client)
    changed = make_zip({**members(), "scifact/corpus.jsonl": jsonl([{"_id": "1", "text": "x"}])})
    with (
        serving(changed) as client,
        pytest.raises(BeirError, match="differs from the one recorded"),
    ):
        fetch_beir("scifact", destination, client=client, force=True)
    assert (destination / "scifact.zip").read_bytes() == original  # still the reference archive
    assert "first text" in (destination / "scifact" / "corpus.jsonl").read_text("utf-8")
    assert not list(destination.glob("*.part"))
    recorded = json.loads((destination / "checksums.json").read_text("utf-8"))
    assert recorded == {"scifact": hashlib.sha256(original).hexdigest()}


def test_a_failed_request_is_a_beir_error_and_leaves_nothing_behind(tmp_path: Path) -> None:
    destination = tmp_path / "beir"
    with failing(404) as client, pytest.raises(BeirError, match="could not download"):
        fetch_beir("scifact", destination, client=client)
    assert not list(destination.glob("*.part"))
    assert not (destination / "checksums.json").exists()


def test_an_unknown_dataset_name_is_refused() -> None:
    with (
        pytest.raises(ValueError, match="unknown BEIR dataset 'msmarco'"),
        never_called() as client,
    ):
        fetch_beir("msmarco", Path("unused"), client=client)


def test_a_download_over_the_size_cap_is_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(beir, "MAX_DOWNLOAD_BYTES", 10)
    destination = tmp_path / "beir"
    with serving(make_zip(members())) as client, pytest.raises(BeirError, match="larger than 10"):
        fetch_beir("scifact", destination, client=client)
    assert not list(destination.glob("*.part"))


def test_an_archive_that_is_not_a_zip_is_rejected_and_not_recorded(tmp_path: Path) -> None:
    destination = tmp_path / "beir"
    with serving(b"<html>not a zip</html>") as client, pytest.raises(BeirError, match="not a zip"):
        fetch_beir("scifact", destination, client=client)
    assert not (destination / "checksums.json").exists()
    assert not (destination / "scifact.zip").exists()


def test_a_member_that_would_leave_the_folder_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "beir"
    evil = make_zip({**members(), "../evil.txt": "pwned"})
    with serving(evil) as client, pytest.raises(BeirError, match="outside the folder"):
        fetch_beir("scifact", destination, client=client)
    assert not (tmp_path / "evil.txt").exists()
    assert not (destination / "scifact").exists()  # nothing was unpacked at all
    assert not (destination / "checksums.json").exists()


def test_an_archive_without_the_expected_parts_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "beir"
    incomplete = {k: v for k, v in members().items() if not k.endswith("corpus.jsonl")}
    with serving(make_zip(incomplete)) as client, pytest.raises(BeirError, match="lacks scifact"):
        fetch_beir("scifact", destination, client=client)
    assert not (destination / "checksums.json").exists()


def test_a_huge_unpacked_size_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(beir, "MAX_UNPACKED_BYTES", 5)
    with serving(make_zip(members())) as client, pytest.raises(BeirError, match="unpacks to more"):
        fetch_beir("scifact", tmp_path / "beir", client=client)


@pytest.mark.parametrize(
    ("content", "fragment"),
    [("not json", "not readable JSON"), ("[1]", "must map dataset names")],
)
def test_a_damaged_checksum_file_stops_the_fetch(
    tmp_path: Path, content: str, fragment: str
) -> None:
    destination = tmp_path / "beir"
    destination.mkdir()
    (destination / "checksums.json").write_text(content, encoding="utf-8")
    with serving(make_zip(members())) as client, pytest.raises(BeirError, match=fragment):
        fetch_beir("scifact", destination, client=client)


def test_each_dataset_has_its_own_record(tmp_path: Path) -> None:
    destination = tmp_path / "beir"
    handlers: dict[str, bytes] = {
        "/thakur/BEIR/datasets/scifact.zip": make_zip(members("scifact")),
        "/thakur/BEIR/datasets/nfcorpus.zip": make_zip(members("nfcorpus")),
    }

    def pick(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=handlers[request.url.path])

    with httpx.Client(transport=httpx.MockTransport(pick)) as client:
        fetch_beir("scifact", destination, client=client)
        fetch_beir("nfcorpus", destination, client=client)
    recorded = json.loads((destination / "checksums.json").read_text("utf-8"))
    assert set(recorded) == {"scifact", "nfcorpus"}
    assert recorded["scifact"] != recorded["nfcorpus"]
