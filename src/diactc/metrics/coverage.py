"""Diacritization coverage rate.

The fraction of diacritizable Arabic base characters that actually carry at
least one diacritic. A value of 1.0 means every Arabic letter is diacritized;
0.0 means none are.
"""

from diactc.utils.text import (
    ARABIC_CHARACTERS_TO_BE_DIACRITIZED,
    get_groups_of_characters_with_diacritics,
    preprocess_text,
)


def diac_coverage_rate(text):
    """Return the diacritic coverage rate of ``text`` in [0.0, 1.0].

    Raises ValueError if the text contains no diacritizable Arabic characters.
    """
    text = preprocess_text(text)
    chars_diacs_pair = get_groups_of_characters_with_diacritics(text)

    # keep only diacritizable Arabic base characters
    chars_diac_pair = [c for c in chars_diacs_pair if c[0] in ARABIC_CHARACTERS_TO_BE_DIACRITIZED]
    if len(chars_diac_pair) == 0:
        raise ValueError("No diacritizable Arabic characters found in the text")

    chars_no_diac = [c for c in chars_diac_pair if c[1].strip() == ""]

    return 1.0 - (len(chars_no_diac) / len(chars_diac_pair))
