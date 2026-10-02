from scripts.validate_human_eval import HumanEvalRow, cjk_length, validate_rows


def make_rows(*, filled: bool = False) -> list[HumanEvalRow]:
    rows: list[HumanEvalRow] = []
    for index in range(30):
        text = ""
        if filled:
            text = "张明和李华在南京等了3天。" + "他们把事情从头到尾说了一遍。" * 8
        rows.append(
            HumanEvalRow(
                id=f"he_{index + 1:03d}",
                domain=f"domain_{index // 5}",
                seed="情境",
                vernacular=text,
            )
        )
    return rows


def test_empty_seed_file_is_reported_as_pending_not_invalid() -> None:
    summary = validate_rows(make_rows())

    assert summary.pending == 30
    assert summary.valid


def test_filled_rows_enforce_length_entities_and_numerals() -> None:
    rows = make_rows(filled=True)
    summary = validate_rows(rows)

    assert summary.pending == 0
    assert summary.valid


def test_invalid_domain_balance_is_rejected() -> None:
    rows = make_rows()
    rows[0] = rows[0].model_copy(update={"domain": "额外领域"})

    summary = validate_rows(rows)

    assert any("domain" in error for error in summary.errors)


def test_cjk_length_ignores_punctuation_and_digits() -> None:
    assert cjk_length("甲乙，12。") == 2
