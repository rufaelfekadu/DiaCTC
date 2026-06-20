import re
import unicodedata
from diactc.config import WILDCARD_TOKEN, NO_DIAC_TOKEN, UNK_DIAC_TOKEN

FATHATAN = "\u064b"
DAMMATAN = "\u064c"
KASRATAN = "\u064d"
FATHA = "\u064e"
DAMMA = "\u064f"
KASRA = "\u0650"
SHADDA = "\u0651"
SUKUN = "\u0652"
BASE_DIACRITICS = [FATHATAN, DAMMATAN, KASRATAN, FATHA, DAMMA, KASRA, SHADDA, SUKUN]
BASE_DIACRITICS_STR = "".join(BASE_DIACRITICS)
assert len(BASE_DIACRITICS) == 8

# Valid combined diacritics
FATHA_SHADDA = FATHA + SHADDA
DAMMA_SHADDA = DAMMA + SHADDA
KASRA_SHADDA = KASRA + SHADDA
FATHATAN_SHADDA = FATHATAN + SHADDA
DAMMATAN_SHADDA = DAMMATAN + SHADDA
KASRATAN_SHADDA = KASRATAN + SHADDA

COMBINED_DIACRITICS = [FATHA_SHADDA, DAMMA_SHADDA, KASRA_SHADDA, FATHATAN_SHADDA, DAMMATAN_SHADDA, KASRATAN_SHADDA]

VALID_DIACRITICS_COMBINATIONS = [
    unicodedata.normalize("NFC", combination)
    for combination in [
        FATHATAN,
        DAMMATAN,
        KASRATAN,
        FATHA,
        DAMMA,
        KASRA,
        SUKUN,
        SHADDA,  # Added because it exists on its own in the dataset, not sure if that's an annotation error or not
        FATHA_SHADDA,
        DAMMA_SHADDA,
        KASRA_SHADDA,
        FATHATAN_SHADDA,
        DAMMATAN_SHADDA,
        KASRATAN_SHADDA,
    ]
]

MERGED_DIAC_TO_TOKEN_MAP = {
    FATHA_SHADDA: "\uE001",
    DAMMA_SHADDA: "\uE002",
    KASRA_SHADDA: "\uE003",
    FATHATAN_SHADDA: "\uE004",
    DAMMATAN_SHADDA: "\uE005",
    KASRATAN_SHADDA: "\uE006",
}

REVERSE_MERGED_DIAC_TO_TOKEN_MAP = {v: k for k, v in MERGED_DIAC_TO_TOKEN_MAP.items()}

INCOMPLETE_COMBINATIONS = [SHADDA]  # -> To be mapped to <UNK_DIAC>

# TODO: Currently includes alef(s) with hamza
# Note: this list does not include "TATWEEL"
ARABIC_CHARACTERS_TO_BE_DIACRITIZED = [
    chr(c) for c in range(ord("\u0621"), ord("\u063a") + 1)
] + [chr(c) for c in range(ord("\u0641"), ord("\u064a") + 1)]

ARABIC_NUMERALS = {
    "٠": "0",
    "١": "1",
    "٢": "2",
    "٣": "3",
    "٤": "4",
    "٥": "5",
    "٦": "6",
    "٧": "7",
    "٨": "8",
    "٩": "9",
}

ARABIC_CHARACTES_CATEGORICALLY_FOLLOWED_BY_NO_DIAC = ["ا", "آ", "ى"]


def preprocess_text(text):
    # Drop some quranic non-phonetic punctuation marks
    dropped_quranic_punctuation_text = re.sub(r"[\u06D6\u06DD\u06DA]", " ", text)

    no_excess_space_text = re.sub(r"\s{2,}", " ", dropped_quranic_punctuation_text)

    normalized_text = unicodedata.normalize("NFKC", no_excess_space_text)

    # TODO: How to deal with Tatweel? -> 26 times only in arvoice, and once is diacritized!!
    return normalized_text


def get_groups_of_characters_with_diacritics(text):
    # Form groups of single non-diacritic characters,
    # each followed by any number of diacritics.
    return re.findall(rf"([^{BASE_DIACRITICS_STR}])([{BASE_DIACRITICS_STR}]*)", text)


def tokenize_text(text, use_special_tokens=False, is_diacritized_dataset=True, no_diac_token=NO_DIAC_TOKEN, unk_diac_token=UNK_DIAC_TOKEN):
    characters_with_succeeding_diacritics = get_groups_of_characters_with_diacritics(
        text
    )
    tokenized_text = []

    for character, succeeding_diacritics in characters_with_succeeding_diacritics:
        tokenized_text.append(character)

        # The character has succeeding diacritics
        if succeeding_diacritics:
            # Use the combination of diacritics as is
            if not use_special_tokens:
                tokenized_text.append(succeeding_diacritics)

            else:
                # If the combination is valid -> add it as is
                if succeeding_diacritics in VALID_DIACRITICS_COMBINATIONS:
                    tokenized_text.append(succeeding_diacritics)
                # Otherwise, map it to the unknown diacritic token
                else:
                    tokenized_text.append(unk_diac_token)

        # The character does not have succeeding diacritics
        else:
            if is_diacritized_dataset:
                if character in ARABIC_CHARACTERS_TO_BE_DIACRITIZED:
                    tokenized_text.append(no_diac_token)
            else:
                # The dataset is undiacritized at all
                if character in ARABIC_CHARACTERS_TO_BE_DIACRITIZED:
                    if character not in ARABIC_CHARACTES_CATEGORICALLY_FOLLOWED_BY_NO_DIAC:
                        tokenized_text.append(unk_diac_token)
                    else:
                        tokenized_text.append(no_diac_token)

        try:
            assert len(succeeding_diacritics) <= 2
        except AssertionError:
            print(
                f"Warning - a sequence of more than two successive diacritics exist in '{text}'"
            )

    return tokenized_text


def form_wildcard_pattern(text, wildcard_token=WILDCARD_TOKEN):

    characters_with_succeeding_diacritics = get_groups_of_characters_with_diacritics(
        text
    )
    pattern = []
    for character, succeeding_diacritics in characters_with_succeeding_diacritics:
        pattern.append(character)

        # TODO: Should long vowels be removed from the list of characters to be diacritized?
        if character in ARABIC_CHARACTERS_TO_BE_DIACRITIZED:
            pattern.append(wildcard_token)

    return pattern
