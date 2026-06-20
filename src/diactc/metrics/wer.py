"""Word Error Rate (WER) and Sentence Error Rate (SER) for diacritization.

A word/sentence is correct only if every diacritic class matches after aligning
base characters with dynamic programming.
Adapted from https://github.com/AliOsm/arabic-text-diacritization.
"""

from diactc.metrics._common import read_pair
from diactc.utils.metrics import (
    clear_line,
    extract_base_characters,
    align_chars_dp,
    get_aligned_diacritic_classes,
)


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
    for original_line, target_line in zip(original_content, target_content):
        if style == "Fadel":
            original_line = clear_line(original_line, arabic_letters, diacritic_classes)
            target_line = clear_line(target_line, arabic_letters, diacritic_classes)

        original_words = original_line.split()
        target_words = target_line.split()

        # Align words by comparing their base character sequences using DP
        orig_word_bases = []
        for idx, word in enumerate(original_words):
            base_chars = extract_base_characters(word, arabic_letters, diacritic_classes, style)
            base_str = ''.join([c[1] for c in base_chars])
            orig_word_bases.append((idx, base_str))

        target_word_bases = []
        for idx, word in enumerate(target_words):
            base_chars = extract_base_characters(word, arabic_letters, diacritic_classes, style)
            base_str = ''.join([c[1] for c in base_chars])
            target_word_bases.append((idx, base_str))

        _, _, _, _, word_alignment = align_chars_dp(orig_word_bases, target_word_bases)

        # Process aligned word pairs
        for orig_word_idx, target_word_idx in word_alignment:
            if orig_word_idx is None or target_word_idx is None:
                # Word insertion/deletion - count as error
                not_equal += 1
                continue

            original_word = original_words[orig_word_idx]
            target_word = target_words[target_word_idx]

            original_base_chars = extract_base_characters(original_word, arabic_letters, diacritic_classes, style)
            target_base_chars = extract_base_characters(target_word, arabic_letters, diacritic_classes, style)

            _, _, _, _, char_alignment = align_chars_dp(original_base_chars, target_base_chars)

            original_classes, target_classes = get_aligned_diacritic_classes(
                original_word, target_word, char_alignment, original_base_chars, target_base_chars,
                case_ending, arabic_letters, diacritic_classes
            )

            if len(original_classes) == 0:
                continue

            if len(original_classes) != len(target_classes):
                not_equal += 1
                continue

            equal_classes = 0
            total_classes = 0
            for original_class, target_class in zip(original_classes, target_classes):
                if original_class == -2 or target_class == -2 or original_class == -3 or target_class == -3:
                    continue

                if not no_diacritic and original_class == 0:
                    equal_classes += 1
                    total_classes += 1
                    continue

                total_classes += 1
                equal_classes += original_class == target_class

            if total_classes == 0:
                continue

            equal += equal_classes == total_classes
            not_equal += equal_classes != total_classes

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
    for original_line, target_line in zip(original_content, target_content):
        if style == "Fadel":
            original_line = clear_line(original_line, arabic_letters, diacritic_classes)
            target_line = clear_line(target_line, arabic_letters, diacritic_classes)

        original_words = original_line.split()
        target_words = target_line.split()

        # Align words by comparing their base character sequences using DP
        orig_word_bases = []
        for idx, word in enumerate(original_words):
            base_chars = extract_base_characters(word, arabic_letters, diacritic_classes, style)
            base_str = ''.join([c[1] for c in base_chars])
            orig_word_bases.append((idx, base_str))

        target_word_bases = []
        for idx, word in enumerate(target_words):
            base_chars = extract_base_characters(word, arabic_letters, diacritic_classes, style)
            base_str = ''.join([c[1] for c in base_chars])
            target_word_bases.append((idx, base_str))

        _, _, _, _, word_alignment = align_chars_dp(orig_word_bases, target_word_bases)

        equal_words = True
        for orig_word_idx, target_word_idx in word_alignment:
            if orig_word_idx is None or target_word_idx is None:
                equal_words = False
                break

            original_word = original_words[orig_word_idx]
            target_word = target_words[target_word_idx]

            original_base_chars = extract_base_characters(original_word, arabic_letters, diacritic_classes, style)
            target_base_chars = extract_base_characters(target_word, arabic_letters, diacritic_classes, style)

            _, _, _, _, char_alignment = align_chars_dp(original_base_chars, target_base_chars)

            original_classes, target_classes = get_aligned_diacritic_classes(
                original_word, target_word, char_alignment, original_base_chars, target_base_chars,
                case_ending, arabic_letters, diacritic_classes
            )

            if len(original_classes) == 0:
                continue

            if len(original_classes) != len(target_classes):
                equal_words = False
                break

            equal_classes = 0
            for original_class, target_class in zip(original_classes, target_classes):
                if original_class == -2 or target_class == -2 or original_class == -3 or target_class == -3:
                    equal_words = False
                    break

                if not no_diacritic and original_class == 0:
                    equal_classes += 1
                    continue
                equal_classes += original_class == target_class

            if equal_classes != len(original_classes):
                equal_words = False
                break

        equal += equal_words
        not_equal += not equal_words

    return round(not_equal / max(1, (equal + not_equal)) * 100, 2)
