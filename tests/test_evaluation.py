"""Tests for evaluation metrics: exact_match, ref_in_pred, bleu, rouge_l_f1, embedding_similarity."""

import pytest

from src.evaluation import exact_match, ref_in_pred, bleu, rouge_l_f1


class TestExactMatch:
    def test_identical(self):
        assert exact_match("65 mph", "65 mph") is True

    def test_case_insensitive(self):
        assert exact_match("65 MPH", "65 mph") is True

    def test_whitespace_normalized(self):
        assert exact_match(" 65  mph ", "65 mph") is True

    def test_different(self):
        assert exact_match("65 mph", "55 mph") is False

    def test_empty_ref(self):
        assert exact_match("anything", "") is False
        assert exact_match("", "") is True

    def test_empty_pred(self):
        assert exact_match("", "65 mph") is False


class TestRefInPred:
    def test_ref_contained(self):
        assert ref_in_pred("The limit is 65 mph.", "65 mph") is True

    def test_ref_not_contained(self):
        assert ref_in_pred("The limit is 55.", "65 mph") is False

    def test_case_insensitive(self):
        assert ref_in_pred("The limit is 65 MPH.", "65 mph") is True

    def test_empty_ref(self):
        assert ref_in_pred("anything", "") is True

    def test_empty_pred(self):
        assert ref_in_pred("", "65 mph") is False

    def test_ref_equals_pred(self):
        assert ref_in_pred("65 mph", "65 mph") is True

    def test_whitespace_in_ref(self):
        # ref "65 mph" normalized to "65 mph"; pred "65  mph" normalized to "65 mph"
        assert ref_in_pred("Speed is 65  mph here.", "65 mph") is True


class TestBleu:
    def test_identical(self):
        assert bleu("65 mph", "65 mph") == 1.0

    def test_empty_ref(self):
        assert bleu("anything", "") == 1.0
        assert bleu("", "") == 1.0

    def test_empty_pred(self):
        assert bleu("", "65 mph") == 0.0

    def test_similar(self):
        s = bleu("The speed limit is 65 mph.", "65 mph")
        assert 0 <= s <= 1


class TestRougeL:
    def test_identical(self):
        assert rouge_l_f1("65 mph", "65 mph") == 1.0

    def test_ref_contained(self):
        assert rouge_l_f1("The limit is 65 mph on highways.", "65 mph") > 0.5

    def test_empty(self):
        assert rouge_l_f1("", "") == 1.0
        assert rouge_l_f1("x", "") == 0.0
        assert rouge_l_f1("", "x") == 0.0

    def test_no_overlap(self):
        assert rouge_l_f1("abc", "xyz") == 0.0


@pytest.mark.slow
class TestEmbeddingSimilarity:
    """Slow: loads embedding model. Run with pytest -m 'not slow' to skip."""

    def test_identical(self):
        from src.evaluation import embedding_similarity
        assert embedding_similarity("65 mph", "65 mph") == 1.0

    def test_similar(self):
        from src.evaluation import embedding_similarity
        s = embedding_similarity("The speed limit is 65 mph.", "65 mph limit")
        assert 0.5 <= s <= 1.0

    def test_empty(self):
        from src.evaluation import embedding_similarity
        assert embedding_similarity("", "") == 1.0
        assert embedding_similarity("x", "") == 0.0
