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


def diac_precision_recall(reference, hypothesis):
    """Return precision and recall of the diacritics in ``hypothesis``.

    ``reference`` and ``hypothesis`` are lists of lines (or two single strings)
    sharing the same base characters. Over the diacritizable Arabic characters:

    - precision: fraction of the diacritics emitted in the hypothesis that match the reference
    - recall: fraction of the diacritized reference characters whose diacritics are recovered

    The returned dict also holds the underlying counts and the number of line
    pairs skipped because their base characters differ. Precision / recall are
    None when their denominator is zero.
    """
    if isinstance(reference, str):
        reference, hypothesis = [reference], [hypothesis]

    emitted = correct = reference_diacritized = positions = skipped = 0
    for reference_line, hypothesis_line in zip(reference, hypothesis, strict=True):
        ref_pairs = get_groups_of_characters_with_diacritics(preprocess_text(reference_line).strip())
        hyp_pairs = get_groups_of_characters_with_diacritics(preprocess_text(hypothesis_line).strip())
        if [c for c, _ in ref_pairs] != [c for c, _ in hyp_pairs]:
            skipped += 1
            continue

        for (character, ref_diacs), (_, hyp_diacs) in zip(ref_pairs, hyp_pairs):
            if character not in ARABIC_CHARACTERS_TO_BE_DIACRITIZED:
                continue
            positions += 1
            reference_diacritized += bool(ref_diacs)
            emitted += bool(hyp_diacs)
            correct += bool(hyp_diacs) and sorted(hyp_diacs) == sorted(ref_diacs)

    return {
        "precision": correct / emitted if emitted else None,
        "recall": correct / reference_diacritized if reference_diacritized else None,
        "emitted": emitted,
        "correct": correct,
        "reference_diacritized": reference_diacritized,
        "positions": positions,
        "skipped": skipped,
    }
