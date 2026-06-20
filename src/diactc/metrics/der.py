"""Diacritic Error Rate (DER).

Base characters are aligned with dynamic programming before comparing diacritic
classes, so reference and hypothesis lines need not have identical lengths.
Adapted from https://github.com/AliOsm/arabic-text-diacritization.
"""

from diactc.metrics._common import read_pair
from diactc.utils.metrics import (
    clear_line,
    extract_base_characters,
    align_chars_dp,
    get_aligned_diacritic_classes,
)


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
    for original_line, target_line in zip(original_content, target_content):
        if style == "Fadel":
            original_line = clear_line(original_line, arabic_letters, diacritic_classes)
            target_line = clear_line(target_line, arabic_letters, diacritic_classes)

        # Extract base characters
        original_base_chars = extract_base_characters(original_line, arabic_letters, diacritic_classes, style)
        target_base_chars = extract_base_characters(target_line, arabic_letters, diacritic_classes, style)

        # Align base characters using DP
        _, _, _, _, alignment = align_chars_dp(original_base_chars, target_base_chars)

        # Get aligned diacritic classes
        original_classes, target_classes = get_aligned_diacritic_classes(
            original_line, target_line, alignment, original_base_chars, target_base_chars,
            case_ending, arabic_letters, diacritic_classes
        )
        assert len(original_classes) == len(target_classes)

        for original_class, target_class in zip(original_classes, target_classes):
            # Handle special markers for insertions/deletions
            if original_class == -2 or target_class == -2:
                not_equal += 1
                continue
            if original_class == -3 or target_class == -3:
                not_equal += 1
                continue

            if not no_diacritic and original_class == 0:
                continue
            if original_class == -1 and target_class != -1:
                not_equal += 1
                continue
            if original_class != -1 and target_class == -1:
                not_equal += 1
                continue
            if original_class == -1 and target_class == -1:
                continue

            equal += original_class == target_class
            not_equal += original_class != target_class

    return round(not_equal / max(1, (equal + not_equal)) * 100, 2)
