import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from retrieval.dense import DenseCacheError, save_dense_cache, text_sha256
from retrieval.query_embedder import CachedQueryEmbedder
from retrieval.types import Document, Embedder
from scripts.build_query_cache import eval_query_documents
from scripts.split_corpus import CorpusSplit
from scripts.train import Pair

ROOT = Path(__file__).resolve().parents[1]


def _pair(doc_id: str, split: str, vernacular: str, original: str) -> Pair:
    return Pair(
        id=doc_id, work="自编", idx=1, vernacular=vernacular, original=original, split=split
    )


def _write_cache(
    directory: Path,
    texts: list[str],
    *,
    model_id: str = "BAAI/bge-m3",
    revision: str = "rev-a",
    max_length: int = 512,
) -> np.ndarray:
    documents = [Document(id=f"d{index}", text=text) for index, text in enumerate(texts)]
    matrix = np.arange(len(documents) * 4, dtype=np.float64).reshape(len(documents), 4) + 1.0
    save_dense_cache(
        documents,
        matrix,
        model_id=model_id,
        revision=revision,
        max_length=max_length,
        directory=directory,
    )
    return matrix


def test_cached_embedder_returns_the_stored_row(tmp_path: Path) -> None:
    query_dir = tmp_path / "query"
    index_dir = tmp_path / "index"
    text = "张三去井边打水"
    matrix = _write_cache(query_dir, [text, "李四买了米"])
    _write_cache(index_dir, ["索引里的另一段"])
    embedder: Embedder = CachedQueryEmbedder(query_dir, index_dir)
    found = embedder.encode([text])
    assert found.shape == (1, 4)
    assert found[0] == pytest.approx(matrix[0])
    assert embedder.encode([]).shape == (0, 4)


def test_cached_embedder_rejects_a_miss(tmp_path: Path) -> None:
    query_dir = tmp_path / "query"
    index_dir = tmp_path / "index"
    _write_cache(query_dir, ["张三去井边打水"])
    _write_cache(index_dir, ["索引里的另一段"])
    embedder = CachedQueryEmbedder(query_dir, index_dir)
    missing = "这里没有向量"
    with pytest.raises(ValueError, match=text_sha256(missing)):
        embedder.encode([missing])


@pytest.mark.parametrize(
    ("field", "value"),
    [("model_id", "other/model"), ("revision", "rev-b"), ("max_length", 256)],
)
def test_cached_embedder_rejects_encoder_mismatch(
    tmp_path: Path, field: str, value: str | int
) -> None:
    query_dir = tmp_path / "query"
    index_dir = tmp_path / "index"
    _write_cache(query_dir, ["张三去井边打水"])
    model_id = value if field == "model_id" and isinstance(value, str) else "BAAI/bge-m3"
    revision = value if field == "revision" and isinstance(value, str) else "rev-a"
    max_length = value if field == "max_length" and isinstance(value, int) else 512
    _write_cache(
        index_dir,
        ["索引里的另一段"],
        model_id=model_id,
        revision=revision,
        max_length=max_length,
    )
    with pytest.raises(DenseCacheError, match="不一致"):
        CachedQueryEmbedder(query_dir, index_dir)


def test_cached_embedder_rejects_a_broken_cache(tmp_path: Path) -> None:
    query_dir = tmp_path / "query"
    index_dir = tmp_path / "index"
    _write_cache(query_dir, ["张三去井边打水"])
    _write_cache(index_dir, ["索引里的另一段"])
    matrix_path = query_dir / "embeddings.npy"
    matrix_path.unlink()
    with pytest.raises(DenseCacheError, match="不完整"):
        CachedQueryEmbedder(query_dir, index_dir)
    np.save(matrix_path, np.zeros((1, 3), dtype=np.float64))
    with pytest.raises(DenseCacheError, match="形状"):
        CachedQueryEmbedder(query_dir, index_dir)
    matrix_path.write_bytes(b"not-a-matrix")
    with pytest.raises(DenseCacheError, match="无法读取"):
        CachedQueryEmbedder(query_dir, index_dir)
    (query_dir / "embeddings.meta.json").write_text("{", encoding="utf-8")
    with pytest.raises(DenseCacheError, match="无法读取"):
        CachedQueryEmbedder(query_dir, index_dir)
    with pytest.raises(DenseCacheError, match="不完整"):
        CachedQueryEmbedder(tmp_path / "absent", index_dir)


def test_eval_queries_keep_vernacular_text() -> None:
    rows = [
        _pair("a", "train", "训练白话", "训练原文"),
        _pair("e", "eval", "张三去井边打水", "井边那个人"),
    ]
    split = CorpusSplit(seed=42, eval_ratio=0.08, train=["a"], eval=["e"])
    documents = eval_query_documents(rows, split)
    assert [(document.id, document.text) for document in documents] == [("e", "张三去井边打水")]


def test_eval_queries_reject_a_split_mismatch() -> None:
    rows = [_pair("e", "train", "张三去井边打水", "井边那个人")]
    split = CorpusSplit(seed=42, eval_ratio=0.08, train=[], eval=["e"])
    with pytest.raises(ValueError, match=r"split\.eval"):
        eval_query_documents(rows, split)


def test_query_script_import_does_not_load_eval_stack() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.build_query_cache\n"
            "import sys\n"
            "bad = [\n"
            "    name for name in sys.modules\n"
            "    if name in {'eval', 'stylometry', 'cn2an', 'openai'}\n"
            "    or name.startswith('eval.')\n"
            "    or name.startswith('stylometry.')\n"
            "    or name.startswith('cn2an.')\n"
            "    or name.startswith('openai.')\n"
            "]\n"
            "assert not bad, bad\n",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_query_cache_dry_run_writes_nothing_and_skips_torch(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    split_path = tmp_path / "split.json"
    output = tmp_path / "query_cache"
    rows = [
        _pair("b", "train", "李四买米", "米在门口"),
        _pair("e", "eval", "张三去井边打水", "井边那个人"),
        _pair("a", "train", "王五看天", "天色暗了"),
    ]
    pairs_path.write_text("".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8")
    split_path.write_text(
        CorpusSplit(seed=42, eval_ratio=0.08, train=["a", "b"], eval=["e"]).model_dump_json(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")])
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from scripts.build_query_cache import main; "
            "main(['--pairs', sys.argv[1], '--split', sys.argv[2], "
            "'--output-dir', sys.argv[3], '--dry-run']); "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
            str(pairs_path),
            str(split_path),
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert not output.exists()
    assert '"documents": 1' in proc.stdout
    assert "query_cache_dry_run" in proc.stdout
