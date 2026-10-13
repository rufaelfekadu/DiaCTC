# -*- coding: utf-8 -*-
"""Tests for diactc.lexicon: Buckwalter conversion and pronunciation alignment.

The pronunciations are entries of the QCRI ar-ar lexicon.
"""

import pytest

from diactc.lexicon import (
    DiacritizationLexicon,
    align_pronunciation,
    allowed_diacritics,
    arabic_to_buckwalter,
    buckwalter_to_arabic,
    diacritic_classes,
    matches_variant,
)
from diactc.utils.text import FATHA, FATHATAN, KASRA, SHADDA, SUKUN


def align(word, pronunciation):
    return align_pronunciation(word, pronunciation.split())


class TestBuckwalter:
    @pytest.mark.parametrize(
        "buckwalter, arabic",
        [("ktb", "كتب"), ("Al$ms", "الشمس"), ("<lY", "إلى"), ("|mn", "آمن"), ("mA}p", "مائة"), ("h*A", "هذا"), ("$y'", "شيء")],
    )
    def test_round_trip(self, buckwalter, arabic):
        assert buckwalter_to_arabic(buckwalter) == arabic
        assert arabic_to_buckwalter(arabic) == buckwalter


class TestAlignPronunciation:
    def test_short_vowels(self):
        assert align("ktb", "k a t a b a") == {("a", "a", "a")}
        assert align("ktb", "k u t i b") == {("u", "i", "0")}

    def test_tanween(self):
        assert align("ktb", "k u t u b u n") == {("u", "u", "un")}

    def test_long_a_puts_fatha_on_previous_letter(self):
        assert align("ktAb", "k i t A b") == {("i", "a", "0", "0")}

    def test_fathatan_before_final_alef(self):
        assert align("ktAbA", "k i t A b a n") == {("i", "a", "0", "an", "0")}

    def test_article_before_sun_letter(self):
        assert align("Al$ms", "A $ a m s u") == {("0", "0", "a", "0", "u")}

    def test_ta_marbuta(self):
        assert align("mdrsp", "m a d r a s a") == {("a", "0", "a", "a", "0")}
        assert align("mdrsp", "m a d r a s a t u n") == {("a", "0", "a", "a", "un")}

    def test_unwritten_long_a(self):
        assert align("h*A", "h A V A") == {("a", "a", "0")}

    def test_alef_madda(self):
        assert ("0", "a", "0") in align("|mn", "G A m a n")

    def test_alef_maqsura(self):
        assert align("<lY", "G i l a") == {("i", "a", "0")}
        assert align("mstwY", "m u s t a w a a n") == {("u", "0", "a", "an", "0")}

    def test_alef_wasla_with_vowel(self):
        assert align("AnETAf", "A i n E i T A f") == {("i", "0", "i", "a", "0", "0")}

    def test_final_h_written_for_ta_marbuta(self):
        assert align("EAzlh", "E A z i l a t i n") == {("a", "0", "i", "a", "in")}

    def test_unalignable_pronunciation(self):
        assert align("ktb", "q a l a m") == set()
        assert align("ktb", "k a t a b a n a") == set()


class TestDiacriticClasses:
    def test_classes(self):
        letters, classes = diacritic_classes("كَتْب")
        assert letters == "كتب"
        assert classes == ["a", "o", ""]

    def test_shadda_is_ignored(self):
        assert diacritic_classes("مُحَمَّد")[1] == ["u", "a", "a", ""]

    def test_fathatan_on_final_alef_moves_to_previous_letter(self):
        assert diacritic_classes("كِتَابًا")[1] == diacritic_classes("كِتَاباً")[1] == ["i", "a", "", "an", ""]


class TestMatchesVariant:
    def test_exact_match(self):
        assert matches_variant(["a", "a", "a"], ("a", "a", "a"))
        assert not matches_variant(["a", "i", "a"], ("a", "a", "a"))

    def test_undiacritized_letter_always_matches(self):
        assert matches_variant(["a", "", "a"], ("a", "u", "a"))

    def test_sukun_needs_a_letter_without_vowel(self):
        assert matches_variant(["a", "o", "a"], ("a", "0", "a"))
        assert not matches_variant(["a", "o", "a"], ("a", "a", "a"))

    def test_free_final_ignores_last_letter(self):
        assert not matches_variant(["a", "a", "u"], ("a", "a", "a"))
        assert matches_variant(["a", "a", "u"], ("a", "a", "a"), free_final=True)

    def test_length_mismatch(self):
        assert not matches_variant(["a", "a"], ("a", "a", "a"))


class TestAllowedDiacritics:
    def test_shadda_is_optional(self):
        assert allowed_diacritics("a") == (FATHA, FATHA + SHADDA)
        assert allowed_diacritics("an") == (FATHATAN, FATHATAN + SHADDA)

    def test_no_vowel(self):
        assert allowed_diacritics("0") == ("", SUKUN, SHADDA)

    def test_any_vowel_allows_everything(self):
        assert KASRA in allowed_diacritics("*")
        assert "" in allowed_diacritics("*")


class TestDiacritizationLexicon:
    @pytest.fixture
    def lexicon(self, tmp_path):
        path = tmp_path / "lexicon.txt"
        path.write_text("# comment\nktb k a t a b a\nktb k u t u b\nktb q a l a m\n", encoding="utf-8")
        return DiacritizationLexicon.from_file(str(path))

    def test_variants(self, lexicon):
        assert lexicon.variants("كتب") == {("a", "a", "a"), ("u", "u", "0")}

    def test_unaligned_pronunciations_are_counted(self, lexicon):
        lexicon.variants("كتب")
        assert lexicon.num_aligned == 2
        assert lexicon.num_unaligned == 1

    def test_unknown_word_has_no_variants(self, lexicon):
        assert "قلم" not in lexicon
        assert lexicon.variants("قلم") == set()
