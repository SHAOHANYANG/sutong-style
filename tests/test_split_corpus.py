from scripts.chunk_corpus import CorpusChunk
from scripts.split_corpus import stratified_split


def test_split_is_deterministic_stratified_and_complete() -> None:
    chunks = [
        CorpusChunk(id=f"作品{work}_{idx:04d}", work=f"作品{work}", idx=idx, original="正文")
        for work in range(6)
        for idx in range(1, 21)
    ]

    first = stratified_split(chunks)
    second = stratified_split(chunks)

    assert first == second
    assert len(first.eval) == 12
    assert set(first.train).isdisjoint(first.eval)
    assert set(first.train + first.eval) == {chunk.id for chunk in chunks}
