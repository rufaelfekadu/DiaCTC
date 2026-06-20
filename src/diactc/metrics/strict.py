"""Strict DER / WER / SER.

These are the original positional metrics from
https://github.com/AliOsm/arabic-text-diacritization: diacritic classes are
compared position-by-position with no alignment, so the reference and hypothesis
must share the same base-character sequence. A mismatch in the number of base
characters (per line, per word, or in the word count) raises ``ValueError``.

Use the relaxed (DP-aligned) variants in :mod:`diactc.metrics.der` /
:mod:`diactc.metrics.wer` when reference and hypothesis may differ in length.
"""

from diactc.metrics._common import read_pair
from diactc.utils.metrics import clear_line, get_diacritics_classes


def calculate_der(
    original,
    target,
    arabic_letters,
    diacritic_classes,
    style="Fadel",
    case_ending=True,
    no_diacritic=True,
):
    original_content, target_content = read_pair(original, target)

    equal = 0
    not_equal = 0
    for line_idx, (original_line, target_line) in enumerate(zip(original_content, target_content)):
        if style == "Fadel":
            original_line = clear_line(original_line, arabic_letters, diacritic_classes)
            target_line = clear_line(target_line, arabic_letters, diacritic_classes)

        original_classes = get_diacritics_classes(
            original_line, case_ending, arabic_letters, diacritic_classes, style
        )
        target_classes = get_diacritics_classes(
            target_line, case_ending, arabic_letters, diacritic_classes, style
        )
        if len(original_classes) != len(target_classes):
            raise ValueError(
                f"Line {line_idx + 1}: base-character count mismatch "
                f"({len(original_classes)} vs {len(target_classes)}). "
                "Strict DER requires aligned reference and hypothesis."
            )

        for original_class, target_class in zip(original_classes, target_classes):
            if not no_diacritic and original_class == 0:
                continue
            if original_class == -1 and target_class == -1:
                continue

            equal += original_class == target_class
            not_equal += original_class != target_class

    return round(not_equal / max(1, (equal + not_equal)) * 100, 2)


def calculate_wer(
    original,
    target,
    arabic_letters,
    diacritic_classes,
    style="Fadel",
    case_ending=True,
    no_diacritic=True,
):
    original_content, target_content = read_pair(original, target)

    equal = 0
    not_equal = 0
    for line_idx, (original_line, target_line) in enumerate(zip(original_content, target_content)):
        if style == "Fadel":
            original_line = clear_line(original_line, arabic_letters, diacritic_classes)
            target_line = clear_line(target_line, arabic_letters, diacritic_classes)

        original_words = original_line.split()
        target_words = target_line.split()
        if len(original_words) != len(target_words):
            raise ValueError(
                f"Line {line_idx + 1}: word count mismatch "
                f"({len(original_words)} vs {len(target_words)}). "
                "Strict WER requires aligned reference and hypothesis."
            )

        for original_word, target_word in zip(original_words, target_words):
            original_classes = get_diacritics_classes(
                original_word, case_ending, arabic_letters, diacritic_classes, style
            )
            target_classes = get_diacritics_classes(
                target_word, case_ending, arabic_letters, diacritic_classes, style
            )
            if len(original_classes) != len(target_classes):
                raise ValueError(
                    f"Line {line_idx + 1}: base-character count mismatch within a word "
                    f"({len(original_classes)} vs {len(target_classes)})."
                )

            if len(original_classes) == 0:
                continue

            equal_classes = 0
            for original_class, target_class in zip(original_classes, target_classes):
                if not no_diacritic and original_class == 0:
                    equal_classes += 1
                    continue
                equal_classes += original_class == target_class

            equal += equal_classes == len(original_classes)
            not_equal += equal_classes != len(original_classes)

    return round(not_equal / max(1, (equal + not_equal)) * 100, 2)


def calculate_ser(
    original,
    target,
    arabic_letters,
    diacritic_classes,
    style="Fadel",
    case_ending=True,
    no_diacritic=True,
):
    original_content, target_content = read_pair(original, target)

    equal = 0
    not_equal = 0
    for line_idx, (original_line, target_line) in enumerate(zip(original_content, target_content)):
        if style == "Fadel":
            original_line = clear_line(original_line, arabic_letters, diacritic_classes)
            target_line = clear_line(target_line, arabic_letters, diacritic_classes)

        original_words = original_line.split()
        target_words = target_line.split()
        if len(original_words) != len(target_words):
            raise ValueError(
                f"Line {line_idx + 1}: word count mismatch "
                f"({len(original_words)} vs {len(target_words)}). "
                "Strict SER requires aligned reference and hypothesis."
            )

        equal_words = True
        for original_word, target_word in zip(original_words, target_words):
            original_classes = get_diacritics_classes(
                original_word, case_ending, arabic_letters, diacritic_classes, style
            )
            target_classes = get_diacritics_classes(
                target_word, case_ending, arabic_letters, diacritic_classes, style
            )
            if len(original_classes) != len(target_classes):
                raise ValueError(
                    f"Line {line_idx + 1}: base-character count mismatch within a word "
                    f"({len(original_classes)} vs {len(target_classes)})."
                )

            if len(original_classes) == 0:
                continue

            equal_classes = 0
            for original_class, target_class in zip(original_classes, target_classes):
                if not no_diacritic and original_class == 0:
                    equal_classes += 1
                    continue
                equal_classes += original_class == target_class

            if equal_classes != len(original_classes):
                equal_words = False

        equal += equal_words
        not_equal += not equal_words

    return round(not_equal / max(1, (equal + not_equal)) * 100, 2)
