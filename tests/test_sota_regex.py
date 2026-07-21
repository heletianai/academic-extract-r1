"""claims_sota 正则单测（checklist 边界案例 6 的代码版）。"""

from src.data.sota_regex import claims_sota_by_regex as R


class TestAssertions:
    def test_achieves_sota(self):
        assert R("Our method achieves state-of-the-art results on MMLU.")

    def test_new_sota_abbrev(self):
        assert R("establishing a new SOTA among small models")

    def test_outperforms_all(self):
        assert R("outperforms all existing baselines")

    def test_surpasses_all(self):
        assert R("surpassing all prior methods")

    def test_previous_best(self):
        assert R("exceeds the previous best by a wide margin")


class TestComparisons:
    def test_comparable_to(self):
        assert not R("Performance is comparable to state-of-the-art systems.")

    def test_close_to(self):
        assert not R("results close to the state of the art")

    def test_matches(self):
        assert not R("matches SOTA performance at half the cost")

    def test_approaching(self):
        assert not R("approaching state-of-the-art quality")

    def test_on_par(self):
        assert not R("on par with the state-of-the-art")


class TestNegativeAndMixed:
    def test_strong_baselines_only(self):
        assert not R("outperforms strong baselines on three tasks")

    def test_competitive(self):
        assert not R("competitive results without any pretraining")

    def test_empty(self):
        assert not R("")

    def test_mixed_one_assertion_wins(self):
        assert R("comparable to SOTA on X, and achieves state-of-the-art on Y")


class TestRedTeamFixes:
    def test_negation_blocked(self):
        assert not R("does not outperform all baselines")
        assert not R("fails to reach the previous best")

    def test_interjection_defeats_old_window(self):
        assert not R("broadly comparable, in some settings, to the state-of-the-art")

    def test_clause_boundary_preserves_assertion(self):
        assert R("comparable to SOTA on X, and achieves state-of-the-art on Y")

    def test_not_only_construction(self):
        assert R("not only simpler, but also achieves state-of-the-art accuracy")
