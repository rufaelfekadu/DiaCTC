"""Self-contained evaluation constants.

The original evaluation code (adapted from
https://github.com/AliOsm/arabic-text-diacritization) loaded the Arabic letter
list and the diacritic-class list from external pickle files. To keep DiaCTC
self-contained, we derive equivalent lists from :mod:`diactc.utils.text`.

`ARABIC_LETTERS` are the base characters that may carry diacritics.
`DIACRITIC_CLASSES` are the diacritic symbols used to bucket each base character
into a class: the eight single diacritics plus the valid shadda combinations in
both orders, so doubly-diacritized characters are distinguished.
"""

from diactc.utils.text import (
    ARABIC_CHARACTERS_TO_BE_DIACRITIZED,
    BASE_DIACRITICS,
    COMBINED_DIACRITICS,
)

ARABIC_LETTERS = list(ARABIC_CHARACTERS_TO_BE_DIACRITIZED)

DIACRITIC_CLASSES = (
    list(BASE_DIACRITICS)
    + list(COMBINED_DIACRITICS)
    + [combination[::-1] for combination in COMBINED_DIACRITICS]
)

__all__ = ["ARABIC_LETTERS", "DIACRITIC_CLASSES"]
