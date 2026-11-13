from grasp.eval import metrics


def test_normalize_answer():
    assert metrics.normalize_answer("The Beatles!") == "beatles"
    assert metrics.normalize_answer("  An   apple, a day ") == "apple day"


def test_exact_match_any_gold():
    assert metrics.exact_match("the Eiffel Tower", ["Eiffel tower", "La Tour"]) == 1.0
    assert metrics.exact_match("Eiffel", ["Eiffel tower"]) == 0.0


def test_f1_partial_and_max_over_golds():
    assert metrics.f1("Eiffel", ["Eiffel tower"]) == 2 * (1 * 0.5) / 1.5
    assert metrics.f1("Paris France", ["Lyon", "Paris"]) == 2 * (0.5 * 1) / 1.5


def test_f1_yes_no_must_match_exactly():
    assert metrics.f1("yes", ["no"]) == 0.0
    assert metrics.f1("yes it is", ["yes"]) == 0.0
    assert metrics.f1("Yes.", ["yes"]) == 1.0


def test_context_recall():
    assert metrics.context_recall(["a", "b"], ["a", "c"]) == 0.5
    assert metrics.context_recall(["a"], []) == 0.0
