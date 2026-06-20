# -*- coding: utf-8 -*-
"""Sanity tests for diactc.metrics (DER, WER, SER).

These use the public package API, which fills in the Arabic letter / diacritic
class lists from diactc.constants.
"""

import pytest

from diactc.metrics import (
    calculate_der,
    calculate_wer,
    calculate_ser,
    calculate_der_strict,
    calculate_wer_strict,
    calculate_ser_strict,
    diac_coverage_rate,
    evaluate,
)


class TestDER:
    def test_perfect_match_is_zero(self):
        assert calculate_der("بِسْمِ", "بِسْمِ") == 0.0

    def test_perfect_match_regardless_of_case_ending(self):
        assert calculate_der("بِسْمِ", "بِسْمِ", case_ending=True) == 0.0
        assert calculate_der("بِسْمِ", "بِسْمِ", case_ending=False) == 0.0

    def test_character_deletion_counts_as_error(self):
        # ب matches, س is deleted -> 1/2 -> 50%
        assert calculate_der("بِسْ", "بِ") == 50.0

    def test_mismatch_in_range(self):
        der = calculate_der("بِسْمِ", "بَسْمِ")
        assert 0.0 <= der <= 100.0

    def test_empty_inputs(self):
        assert calculate_der("", "") == 0.0


class TestWER:
    def test_perfect_match_is_zero(self):
        assert calculate_wer("بِسْمِ", "بِسْمِ") == 0.0

    def test_single_word_all_errors_is_100(self):
        assert calculate_wer("بِسْمِ", "بَسْمُ") == 100.0

    def test_two_words_one_error_is_50(self):
        assert calculate_wer("بِسْمِ اللَّهِ", "بِسْمِ اللَّهَ") == 50.0

    def test_word_deletion_is_50(self):
        assert calculate_wer("بِسْمِ اللَّهِ", "بِسْمِ") == 50.0


class TestSER:
    def test_perfect_match_is_zero(self):
        assert calculate_ser("بِسْمِ", "بِسْمِ") == 0.0

    def test_single_sentence_all_errors_is_100(self):
        assert calculate_ser("بِسْمِ", "بَسْمُ") == 100.0

    def test_two_sentences_one_error_is_50(self):
        ref = "بِسْمِ\nاللَّهِ"
        hyp = "بِسْمِ\nاللَّهَ"
        assert calculate_ser(ref, hyp) == 50.0


class TestStrict:
    def test_strict_der_perfect_match(self):
        assert calculate_der_strict("بِسْمِ", "بِسْمِ") == 0.0

    def test_strict_der_single_diacritic_error(self):
        # one of three diacritized chars differs -> 1/3 -> 33.33%
        assert calculate_der_strict("بِسْمِ", "بَسْمِ") == pytest.approx(33.33, abs=0.01)

    def test_strict_wer_two_words_one_error(self):
        assert calculate_wer_strict("بِسْمِ اللَّهِ", "بِسْمِ اللَّهَ") == 50.0

    def test_strict_ser_perfect_match(self):
        assert calculate_ser_strict("بِسْمِ", "بِسْمِ") == 0.0

    def test_strict_der_raises_on_misaligned_chars(self):
        with pytest.raises(ValueError):
            calculate_der_strict("بِسْمِ", "بِ")

    def test_strict_wer_raises_on_word_count_mismatch(self):
        with pytest.raises(ValueError):
            calculate_wer_strict("بِسْمِ اللَّهِ", "بِسْمِ")

    def test_evaluate_strict_returns_same_shape(self):
        results = evaluate("بِسْمِ", "بِسْمِ", strict=True)
        assert set(results) == {"der", "wer", "ser", "coverage"}
        for metric in ("der", "wer", "ser"):
            assert all(v == 0.0 for v in results[metric].values())


class TestCoverage:
    def test_fully_diacritized_is_one(self):
        assert diac_coverage_rate("بِسْمِ") == 1.0

    def test_undiacritized_is_zero(self):
        assert diac_coverage_rate("بسم") == 0.0

    def test_partial_coverage(self):
        # 2 of 3 base characters carry a diacritic
        assert diac_coverage_rate("بِسْم") == pytest.approx(2 / 3)

    def test_no_arabic_raises(self):
        with pytest.raises(ValueError):
            diac_coverage_rate("hello")


class TestEvaluate:
    def test_evaluate_returns_all_variants(self):
        results = evaluate("بِسْمِ اللَّهِ", "بِسْمِ اللَّهِ")
        assert set(results) == {"der", "wer", "ser", "coverage"}
        for metric in ("der", "wer", "ser"):
            assert set(results[metric]) == {
                "with_case_with_no_diac",
                "without_case_with_no_diac",
                "with_case_without_no_diac",
                "without_case_without_no_diac",
            }
            assert all(v == 0.0 for v in results[metric].values())

    def test_evaluate_reports_coverage(self):
        results = evaluate("بِسْمِ", "بِسْمِ")
        assert results["coverage"]["reference"] == 1.0
        assert results["coverage"]["hypothesis"] == 1.0

    def test_evaluate_coverage_reflects_undiacritized_hypothesis(self):
        results = evaluate("بِسْمِ", "بسم")
        assert results["coverage"]["reference"] == 1.0
        assert results["coverage"]["hypothesis"] == 0.0

    def test_list_input_matches_string_input(self):
        ref = ["بِسْمِ", "اللَّهِ"]
        hyp = ["بِسْمِ", "اللَّهِ"]
        assert calculate_der(ref, hyp) == 0.0
        assert calculate_wer(ref, hyp) == 0.0
        assert calculate_ser(ref, hyp) == 0.0
