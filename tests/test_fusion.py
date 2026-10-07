import pytest

from retrieval.fusion import fuse
from retrieval.types import Hit


def _hit(doc_id: str, rank: int) -> Hit:
    return Hit(id=doc_id, rank=rank, score=0.0)


def _ranked(ids: list[str]) -> list[Hit]:
    return [_hit(doc_id, rank) for rank, doc_id in enumerate(ids, start=1)]


def _at_rank(doc_id: str, rank: int, width: int, prefix: str) -> list[Hit]:
    hits: list[Hit] = []
    for index in range(1, width + 1):
        name = doc_id if index == rank else f"{prefix}-{index}"
        hits.append(_hit(name, index))
    return hits


def test_scores_match_hand_calculation_and_ties_break_by_id() -> None:
    routes = {
        "bm25": _ranked(["a", "b"]),
        "dense": _ranked(["b", "c"]),
        "style": _ranked(["c"]),
    }
    fused = fuse(routes, k=3)
    by_id = {hit.id: hit for hit in fused}
    assert by_id["a"].score == pytest.approx(1 / 61)
    assert by_id["b"].score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["c"].score == pytest.approx(1 / 62 + 1 / 61)
    assert [hit.id for hit in fused] == ["b", "c", "a"]
    assert [hit.rank for hit in fused] == [1, 2, 3]
    assert by_id["b"].sources == {"bm25": 2, "dense": 1}
    assert by_id["c"].sources == {"dense": 2, "style": 1}
    assert by_id["a"].sources == {"bm25": 1}


def test_equal_weights_bury_style_and_half_content_weights_reverse_it() -> None:
    content = _at_rank("content", 30, 30, "pad")
    style = _ranked(["style-top"])
    routes = {"bm25": content, "dense": content, "style": style}
    equal = fuse(routes, k=31)
    equal_ids = [hit.id for hit in equal]
    assert equal_ids.index("content") < equal_ids.index("style-top")
    by_id = {hit.id: hit for hit in equal}
    assert by_id["content"].score == pytest.approx(2 / 90)
    assert by_id["style-top"].score == pytest.approx(1 / 61)

    halved = fuse(routes, k=31, weights={"bm25": 0.5, "dense": 0.5, "style": 1.0})
    halved_ids = [hit.id for hit in halved]
    assert halved_ids.index("style-top") < halved_ids.index("content")
    halved_by_id = {hit.id: hit for hit in halved}
    assert halved_by_id["content"].score == pytest.approx(1 / 90)
    assert halved_by_id["style-top"].score == pytest.approx(1 / 61)


def test_default_weights_match_explicit_ones() -> None:
    routes = {"bm25": _ranked(["a"]), "dense": _ranked(["b"]), "style": _ranked(["a"])}
    explicit = {"bm25": 1.0, "dense": 1.0, "style": 1.0}
    assert fuse(routes, k=2) == fuse(routes, k=2, weights=explicit)


def test_omitted_weight_defaults_to_one() -> None:
    routes = {"bm25": _ranked(["a", "b"]), "style": _ranked(["a"])}
    fused = fuse(routes, k=1, weights={"style": 2.0})
    assert fused[0].id == "a"
    assert fused[0].score == pytest.approx(1 / 61 + 2 / 61)


def test_empty_routes_do_not_crash() -> None:
    one_empty = {"bm25": [], "dense": _ranked(["a"]), "style": _ranked(["b"])}
    assert [hit.id for hit in fuse(one_empty, k=5)] == ["a", "b"]
    two_empty = {"bm25": [], "dense": [], "style": _ranked(["a"])}
    assert [hit.id for hit in fuse(two_empty, k=5)] == ["a"]
    assert fuse({"bm25": [], "dense": [], "style": []}, k=5) == []


def test_k_and_rrf_k_boundaries() -> None:
    routes = {"bm25": _ranked(["a"])}
    with pytest.raises(ValueError, match="k"):
        fuse(routes, k=0)
    with pytest.raises(ValueError, match="k"):
        fuse(routes, k=-1)
    assert len(fuse(routes, k=5)) == 1
    with pytest.raises(ValueError, match="rrf_k"):
        fuse(routes, k=1, rrf_k=0)
    with pytest.raises(ValueError, match="rrf_k"):
        fuse(routes, k=1, rrf_k=-3)
    assert fuse(routes, k=1, rrf_k=10)[0].score == pytest.approx(1 / 11)


def test_zero_weight_route_is_ignored() -> None:
    routes = {
        "bm25": _ranked(["both", "only-bm25"]),
        "style": _ranked(["both", "only-style"]),
    }
    fused = fuse(routes, k=5, weights={"style": 0.0})
    assert [hit.id for hit in fused] == ["both", "only-bm25"]
    assert fused[0].sources == {"bm25": 1}
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)


def test_zero_weight_route_is_still_checked() -> None:
    routes = {
        "bm25": _ranked(["a"]),
        "style": [_hit("a", 1), _hit("a", 2)],
    }
    with pytest.raises(ValueError, match="重复"):
        fuse(routes, k=1, weights={"style": 0.0})


def test_illegal_inputs() -> None:
    ok = _ranked(["a"])
    with pytest.raises(ValueError, match="重复"):
        fuse({"bm25": [_hit("a", 1), _hit("a", 2)]}, k=1)
    with pytest.raises(ValueError, match="连续"):
        fuse({"bm25": [_hit("a", 1), _hit("b", 3)]}, k=1)
    with pytest.raises(ValueError, match="连续"):
        fuse({"bm25": [_hit("a", 2)]}, k=1)
    with pytest.raises(ValueError, match="未知"):
        fuse({"bm25": ok}, k=1, weights={"dense": 1.0})
    with pytest.raises(ValueError, match="负"):
        fuse({"bm25": ok}, k=1, weights={"bm25": -0.1})
    with pytest.raises(ValueError, match="有限"):
        fuse({"bm25": ok}, k=1, weights={"bm25": float("nan")})


def test_route_insertion_order_does_not_change_output() -> None:
    shared = "shared"
    bm25 = _at_rank(shared, 1, 13, "b")
    dense = _at_rank(shared, 7, 13, "d")
    style = _at_rank(shared, 13, 13, "s")
    left = {"style": style, "bm25": bm25, "dense": dense}
    right = {"dense": dense, "bm25": bm25, "style": style}
    assert fuse(left, k=40) == fuse(right, k=40)
    hit = next(item for item in fuse(left, k=40) if item.id == shared)
    assert list(hit.sources) == ["bm25", "dense", "style"]
    assert hit.sources == {"bm25": 1, "dense": 7, "style": 13}
