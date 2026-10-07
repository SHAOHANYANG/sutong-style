from eval.exemplar_diagnostics import exemplar_copy_ratio, exemplar_entity_leak


def test_entity_leak_is_the_share_that_came_from_exemplars() -> None:
    leaked = exemplar_entity_leak("屋里安静", "颂莲来了", [("门开了", "颂莲坐着")], set())
    assert leaked == 1.0


def test_output_without_entities_scores_zero() -> None:
    score = exemplar_entity_leak("屋里安静", "屋里安静", [("颂莲来了", "颂莲坐着")], set())
    assert score == 0.0


def test_empty_exemplars_score_zero() -> None:
    assert exemplar_entity_leak("屋里安静", "颂莲来了", [], set()) == 0.0
    assert exemplar_copy_ratio("门开着", []) == 0.0


def test_copy_ratio_is_the_closest_exemplar_original() -> None:
    assert exemplar_copy_ratio("门开着", ["门开着"]) == 1.0
    assert exemplar_copy_ratio("门开着", ["井边有人", "门开着"]) == 1.0
