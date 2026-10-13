"""Diacritization variants of a word, derived from a pronunciation lexicon.

The lexicon (e.g. the QCRI ``ar-ar_lexicon``) maps an undiacritized word in
Buckwalter transliteration to one or more phoneme sequences::

    ktb k a t a b
    ktb k u t i b a

Each pronunciation is aligned to the letters of its word, which gives one
vowel class per letter. The lexicon does not mark gemination, so a class says
nothing about shadda; see :func:`allowed_diacritics`.
"""

import functools
from collections import defaultdict

import pyarabic.trans as trans

from diactc.utils.text import (
    FATHA,
    DAMMA,
    KASRA,
    FATHATAN,
    DAMMATAN,
    KASRATAN,
    SHADDA,
    SUKUN,
    VALID_DIACRITICS_COMBINATIONS,
    get_groups_of_characters_with_diacritics,
)

# Vowel classes of a letter
NO_VOWEL = "0"
ANY_VOWEL = "*"  # the pronunciation does not tell
VOWEL_TO_DIACRITIC = {
    "a": FATHA,
    "i": KASRA,
    "u": DAMMA,
    "an": FATHATAN,
    "in": KASRATAN,
    "un": DAMMATAN,
}
DIACRITIC_TO_VOWEL = {v: k for k, v in VOWEL_TO_DIACRITIC.items()}

# Classes of a letter in a diacritized text that carries no vowel
UNDIACRITIZED = ""
SUKUN_CLASS = "o"

SHORT_VOWEL_PHONES = "aiu"
HAMZA_PHONE = "G"
LONG_A_PHONE = "A"

# Buckwalter letter -> phone, for the letters that are always pronounced as one consonant
CONSONANT_PHONES = {letter: letter for letter in "btjHxdrzs$SDTZEgfqklmnhwyv"}
CONSONANT_PHONES["*"] = "V"
for _hamza in "'><&}":
    CONSONANT_PHONES[_hamza] = HAMZA_PHONE

SUN_LETTERS = set("tvd*rzs$SDTZln")
ALEF_LETTERS = "A{"
HAMZA_ON_ALEF_LETTERS = "><"
ALEF_MADDA = "|"
# the lexicon also writes the alef of the article with a hamza or a madda
ARTICLE_ALEF_LETTERS = ALEF_LETTERS + HAMZA_ON_ALEF_LETTERS + ALEF_MADDA
ALEF_MAQSURA = "Y"
TA_MARBUTA = "p"
TA_MARBUTA_PHONES = "hpt"

# Mark a letter read as a long a, without and with tanween; resolved in _resolve_long_vowels
_LONG = "long"
_LONG_TANWEEN = "long+n"


def buckwalter_to_arabic(text):
    return trans.convert(text, "tim", "utf")


def arabic_to_buckwalter(text):
    return trans.convert(text, "utf", "tim")


def _vowel_options(phones, j, allow_tanween=True):
    """Ways to read the phones from position j as the vowel of a consonant: (next position, class)."""
    options = [(j, NO_VOWEL)]
    if j < len(phones) and phones[j] in SHORT_VOWEL_PHONES:
        options.append((j + 1, phones[j]))
        # tanween is only written at the end of a word
        if allow_tanween and j + 2 == len(phones) and phones[j + 1] == "n":
            options.append((j + 2, phones[j] + "n"))
    # long a that is not written (as in h*A, Allh)
    if j < len(phones) and phones[j] == LONG_A_PHONE:
        options.append((j + 1, "a"))
    return options


def _resolve_long_vowels(classes):
    """A letter read as a long a carries no vowel, and the letter before it carries a fatha."""
    classes = list(classes)
    for i, cls in enumerate(classes):
        if cls == _LONG:
            classes[i] = NO_VOWEL
            if i > 0 and classes[i - 1] == NO_VOWEL:
                classes[i - 1] = "a"
        elif cls == _LONG_TANWEEN:
            classes[i] = NO_VOWEL
            if i > 0 and classes[i - 1] in (NO_VOWEL, "a"):
                classes[i - 1] = "an"
    return tuple(classes)


def align_pronunciation(word, phones):
    """Align a pronunciation to the letters of its word.

    Args:
        word: undiacritized word in Buckwalter transliteration
        phones: sequence of phones
    Returns:
        the set of possible alignments, each a tuple with one vowel class per
        letter (``"0"`` for no vowel, ``"*"`` for unknown, else one of a, i, u,
        an, in, un). Empty if the pronunciation cannot be aligned to the word.
    """
    phones = tuple(phones)

    @functools.lru_cache(maxsize=None)
    def align_from(i, j):
        if i == len(word):
            return {()} if j == len(phones) else set()

        letter = word[i]
        phone = phones[j] if j < len(phones) else None
        alignments = set()

        def take(next_j, cls):
            for rest in align_from(i + 1, next_j):
                alignments.add((cls,) + rest)

        def take_alef_maqsura():
            take(j, NO_VOWEL)
            if phone in (LONG_A_PHONE, "a"):
                take(j + 1, _LONG)
                # fathatan before the alef maqsura
                if phones[j + 1:] == ("n",):
                    take(j + 2, _LONG_TANWEEN)

        def take_ta_marbuta():
            take(j, NO_VOWEL)
            if phone is not None and phone in TA_MARBUTA_PHONES:
                for next_j, cls in _vowel_options(phones, j + 1):
                    take(next_j, cls)

        is_final = i == len(word) - 1

        if letter in CONSONANT_PHONES:
            if phone == CONSONANT_PHONES[letter]:
                for next_j, cls in _vowel_options(phones, j + 1):
                    take(next_j, cls)
            # the l of the article is not pronounced before a sun letter
            if letter == "l" and i > 0 and word[i - 1] in ARTICLE_ALEF_LETTERS + "l" and i + 1 < len(word) and word[i + 1] in SUN_LETTERS:
                take(j, NO_VOWEL)
            # long vowels written with their own phone
            if (letter == "w" and phone == "U") or (letter == "y" and phone == "I"):
                take(j + 1, NO_VOWEL)
            # the lexicon also lists alef with hamza without its hamza
            if letter in HAMZA_ON_ALEF_LETTERS and phone == LONG_A_PHONE:
                take(j + 1, ANY_VOWEL)
                if j + 1 < len(phones) and phones[j + 1] in SHORT_VOWEL_PHONES:
                    take(j + 2, phones[j + 1])
            # final h and y written for ta marbuta and alef maqsura
            if is_final and letter == "h":
                take_ta_marbuta()
            if is_final and letter == "y":
                take_alef_maqsura()

        elif letter in ALEF_LETTERS:
            if phone == LONG_A_PHONE:
                take(j + 1, _LONG)
                # alef wasla followed by its vowel
                if j + 1 < len(phones) and phones[j + 1] in SHORT_VOWEL_PHONES:
                    take(j + 2, phones[j + 1])
            if phone == HAMZA_PHONE:
                for next_j, cls in _vowel_options(phones, j + 1, allow_tanween=False):
                    take(next_j, cls)
            # silent: after tanween, after the w of the plural
            take(j, NO_VOWEL)

        elif letter == ALEF_MADDA:
            if phone == HAMZA_PHONE and j + 1 < len(phones) and phones[j + 1] in (LONG_A_PHONE, "a"):
                take(j + 2, NO_VOWEL)
            # the lexicon also lists alef madda as a plain or a hamza alef
            if phone == HAMZA_PHONE:
                for next_j, cls in _vowel_options(phones, j + 1, allow_tanween=False):
                    take(next_j, ANY_VOWEL)
            if phone == LONG_A_PHONE:
                take(j + 1, ANY_VOWEL)
                if j + 1 < len(phones) and phones[j + 1] in SHORT_VOWEL_PHONES:
                    take(j + 2, ANY_VOWEL)

        elif letter == ALEF_MAQSURA:
            take_alef_maqsura()
            if phone == "y":
                for next_j, cls in _vowel_options(phones, j + 1):
                    take(next_j, cls)

        elif letter == TA_MARBUTA:
            take_ta_marbuta()

        return alignments

    return {_resolve_long_vowels(alignment) for alignment in align_from(0, 0)}


def allowed_diacritics(cls):
    """Diacritics a letter of the given vowel class may carry.

    The lexicon does not mark gemination, so shadda is always optional.
    """
    if cls == ANY_VOWEL:
        return ("",) + tuple(VALID_DIACRITICS_COMBINATIONS)
    if cls == NO_VOWEL:
        return ("", SUKUN, SHADDA)
    diacritic = VOWEL_TO_DIACRITIC[cls]
    return (diacritic, diacritic + SHADDA)


def diacritic_classes(word):
    """Split a diacritized Arabic word into its letters and the class of each letter.

    The class is a vowel class (a, i, u, an, in, un), ``"o"`` for a sukun, or
    ``""`` when the letter carries no vowel mark at all.
    """
    groups = get_groups_of_characters_with_diacritics(word)
    letters = "".join(letter for letter, _ in groups)
    classes = []
    for _, diacritics in groups:
        vowel = diacritics.replace(SHADDA, "")
        if vowel == SUKUN:
            classes.append(SUKUN_CLASS)
        else:
            classes.append(DIACRITIC_TO_VOWEL.get(vowel, UNDIACRITIZED))

    # fathatan written on a final alef belongs to the letter before it
    for i in range(1, len(classes)):
        if classes[i] == "an" and letters[i] in "اى" and classes[i - 1] == UNDIACRITIZED:
            classes[i - 1], classes[i] = "an", UNDIACRITIZED

    return letters, classes


def matches_variant(classes, variant, free_final=False):
    """Whether the classes of a diacritized word are allowed by a lexicon variant.

    A letter without any vowel mark is always allowed. With ``free_final``, the
    last letter (which carries the case ending) is not checked.
    """
    if len(classes) != len(variant):
        return False
    if free_final:
        classes, variant = classes[:-1], variant[:-1]
    for cls, allowed in zip(classes, variant):
        if cls == UNDIACRITIZED or allowed == ANY_VOWEL:
            continue
        if cls == SUKUN_CLASS:
            if allowed != NO_VOWEL:
                return False
        elif cls != allowed:
            return False
    return True


class DiacritizationLexicon:
    """Looks up the diacritization variants of undiacritized Arabic words."""

    def __init__(self, pronunciations):
        """
        Args:
            pronunciations: dict mapping a Buckwalter word to its list of pronunciations
        """
        self.pronunciations = pronunciations
        self._variants = {}
        self.num_aligned = 0
        self.num_unaligned = 0

    @classmethod
    def from_file(cls, path):
        """Load a lexicon with one ``word phone phone ...`` entry per line; lines starting with # are skipped."""
        pronunciations = defaultdict(list)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                pronunciations[parts[0]].append(tuple(parts[1:]))
        return cls(dict(pronunciations))

    def __contains__(self, word):
        return arabic_to_buckwalter(word) in self.pronunciations

    def variants(self, word):
        """Return the set of vowel-class sequences of an undiacritized Arabic word.

        Empty if the word is not in the lexicon or none of its pronunciations
        could be aligned.
        """
        buckwalter_word = arabic_to_buckwalter(word)
        if buckwalter_word not in self._variants:
            variants = set()
            for phones in self.pronunciations.get(buckwalter_word, []):
                alignments = align_pronunciation(buckwalter_word, phones)
                if alignments:
                    self.num_aligned += 1
                else:
                    self.num_unaligned += 1
                variants.update(alignments)
            self._variants[buckwalter_word] = frozenset(variants)
        return self._variants[buckwalter_word]
