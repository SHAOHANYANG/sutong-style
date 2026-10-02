from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_do_not_contain_hard_coded_api_keys() -> None:
    secret_prefix = "sk" + "-"
    offenders = [
        path
        for path in ROOT.rglob("*.py")
        if not {".venv", ".uv-cache", ".git"}.intersection(path.parts)
        and secret_prefix in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_copyrighted_jsonl_files_are_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "corpus/*.jsonl" in gitignore
    assert "!corpus/sample_public.jsonl" in gitignore
    assert "!corpus/human_eval.jsonl" in gitignore
