import json
from pathlib import Path

import pytest

from eval.fidelity import ensure_manual_userdict
from retrieval import Bm25Index, Document, Hit, tokenize

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "corpus" / "sample_public.jsonl"
_PADS = (
    Document(id="pad-m", text="码头"),
    Document(id="pad-b", text="布庄"),
    Document(id="pad-c", text="祠堂"),
)


def _sample_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in SAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows.append(
            {
                "id": str(payload["id"]),
                "vernacular": str(payload["vernacular"]),
                "original": str(payload["original"]),
            }
        )
    return rows


def _ids(hits: list[Hit]) -> list[str]:
    return [hit.id for hit in hits]


SAMPLE_ROWS = _sample_rows()


@pytest.mark.parametrize("row", SAMPLE_ROWS, ids=[row["id"] for row in SAMPLE_ROWS])
def test_sample_public_top1_is_the_paired_original(row: dict[str, str]) -> None:
    documents = [Document(id=item["id"], text=item["original"]) for item in SAMPLE_ROWS]
    hits = Bm25Index(documents).search(row["vernacular"], k=1)
    assert _ids(hits) == [row["id"]]
    assert hits[0].rank == 1
    assert hits[0].score > 0


def test_stopword_only_query_is_empty() -> None:
    documents = [
        Document(id="function", text="他的了在是"),
        Document(id="content", text="粮仓"),
        *_PADS,
    ]
    assert Bm25Index(documents).search("他的了在是，。！", k=5) == []


def test_appending_stopwords_does_not_change_id_order() -> None:
    documents = [
        Document(id="content", text="粮仓"),
        Document(id="bait", text="码头他的了在是"),
        *_PADS[1:],
    ]
    index = Bm25Index(documents)
    plain = index.search("粮仓", k=5)
    decorated = index.search("他的了粮仓在是，。吗", k=5)
    assert _ids(plain) == _ids(decorated) == ["content"]


def test_document_sharing_only_stopwords_is_absent() -> None:
    documents = [
        Document(id="content", text="粮仓"),
        Document(id="function", text="他的了在是，。"),
        *_PADS,
    ]
    hits = Bm25Index(documents).search("粮仓的了在是", k=5)
    assert "function" not in _ids(hits)
    assert _ids(hits) == ["content"]
    assert all(hit.score > 0 for hit in hits)


def test_manual_names_stay_whole() -> None:
    assert tokenize("陈佐千和梅珊") == ["陈佐千", "梅珊"]
    assert tokenize("飞浦会来") == ["飞浦", "会", "来"]
    assert tokenize("燮国") == ["燮国"]


def test_tokenize_ignores_the_global_jieba_userdict() -> None:
    before = tokenize("陈佐千和梅珊飞浦会来燮国")
    ensure_manual_userdict()
    assert tokenize("陈佐千和梅珊飞浦会来燮国") == before


def test_zero_score_documents_are_omitted() -> None:
    documents = [
        Document(id="content", text="粮仓"),
        Document(id="other", text="布庄"),
        *_PADS[:2],
    ]
    hits = Bm25Index(documents).search("粮仓", k=5)
    assert _ids(hits) == ["content"]
    assert hits[0].score > 0


def test_ties_break_by_id_ascending() -> None:
    documents = [
        Document(id="b", text="粮仓"),
        Document(id="a", text="粮仓"),
        *_PADS,
    ]
    hits = Bm25Index(documents).search("粮仓", k=5)
    assert _ids(hits) == ["a", "b"]
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].score == hits[1].score


def test_exclude_ids_renumbers_ranks_from_one() -> None:
    documents = [
        Document(id="b", text="粮仓"),
        Document(id="a", text="粮仓"),
        *_PADS,
    ]
    index = Bm25Index(documents)
    kept = index.search("粮仓", k=5, exclude_ids=["a", "missing"])
    assert [(hit.id, hit.rank) for hit in kept] == [("b", 1)]
    assert index.search("粮仓", k=5, exclude_ids=["a", "b"]) == []


def test_k_larger_than_the_hit_list_returns_what_exists() -> None:
    documents = [Document(id="content", text="粮仓"), *_PADS]
    hits = Bm25Index(documents).search("粮仓", k=10)
    assert _ids(hits) == ["content"]
    assert hits[0].rank == 1


def test_non_positive_k_is_rejected() -> None:
    index = Bm25Index([Document(id="content", text="粮仓"), *_PADS])
    with pytest.raises(ValueError, match="k 必须是正整数"):
        index.search("粮仓", k=0)
    with pytest.raises(ValueError, match="k 必须是正整数"):
        index.search("粮仓", k=-1)


def test_empty_index_searches_to_empty() -> None:
    assert Bm25Index([]).search("粮仓", k=3) == []


def test_index_of_only_stopwords_searches_to_empty() -> None:
    documents = [
        Document(id="a", text="的了。"),
        Document(id="b", text="在是，"),
    ]
    index = Bm25Index(documents)
    assert index.search("粮仓", k=3) == []
    assert index.search("的了", k=3) == []


def test_duplicate_document_id_is_rejected() -> None:
    documents = [Document(id="a", text="粮仓"), Document(id="a", text="码头")]
    with pytest.raises(ValueError, match="重复的文档 id"):
        Bm25Index(documents)
