"""Measure how well a pronunciation lexicon would constrain diacritization.

For every word of the reference, looks up the diacritization variants the
lexicon allows and reports how often the reference falls inside them. With a
hypothesis file, also reports how many of its wrong words a lexicon constraint
could fix and how many of its correct words it would exclude.

Two constraint shapes are reported: the whole word, and the stem only (the
last letter, which carries the case ending, left free).

Example:
    python scripts/analyze_lexicon.py \
        --lexicon ar-ar_lexicon_2014-03-17.txt \
        -ref outputs/wav2vec/wfst/constrained/clartts/gt.txt \
        -hyp outputs/wav2vec/wfst/constrained/clartts/pred.txt
"""

import argparse
import logging
from collections import Counter

from diactc.lexicon import DiacritizationLexicon, diacritic_classes, matches_variant
from diactc.metrics._common import read_pair
from diactc.utils.text import preprocess_text, get_groups_of_characters_with_diacritics

SHAPES = {"whole_word": False, "stem_only": True}


def _allowed(classes, variants, free_final):
    return any(matches_variant(classes, variant, free_final=free_final) for variant in variants)


def _same_diacritics(reference_word, hypothesis_word, free_final):
    reference_groups = get_groups_of_characters_with_diacritics(reference_word)
    hypothesis_groups = get_groups_of_characters_with_diacritics(hypothesis_word)
    if free_final:
        reference_groups, hypothesis_groups = reference_groups[:-1], hypothesis_groups[:-1]
    return reference_groups == hypothesis_groups


def analyze(lexicon, reference_lines, hypothesis_lines=None):
    """Return word-level counts for each constraint shape.

    Counts per shape:
        reference_allowed: covered words whose reference is inside the lexicon variants
        fixable: hypothesis wrong and excluded by the lexicon, reference allowed
        broken: hypothesis correct, reference excluded by the lexicon
        wrong_allowed: hypothesis wrong but inside the lexicon variants
        wrong_reference_excluded: hypothesis wrong, reference excluded as well
    """
    results = {"tokens": 0, "covered": 0, "skipped_lines": 0, "shapes": {shape: Counter() for shape in SHAPES}}
    excluded_references = Counter()

    if hypothesis_lines is None:
        hypothesis_lines = [None] * len(reference_lines)

    for reference_line, hypothesis_line in zip(reference_lines, hypothesis_lines):
        reference_words = preprocess_text(reference_line).split()
        hypothesis_words = preprocess_text(hypothesis_line).split() if hypothesis_line is not None else [None] * len(reference_words)
        if len(reference_words) != len(hypothesis_words):
            results["skipped_lines"] += 1
            continue

        for reference_word, hypothesis_word in zip(reference_words, hypothesis_words):
            results["tokens"] += 1
            letters, reference_classes = diacritic_classes(reference_word)
            variants = lexicon.variants(letters)
            if not variants:
                continue
            results["covered"] += 1

            for shape, free_final in SHAPES.items():
                counts = results["shapes"][shape]
                reference_allowed = _allowed(reference_classes, variants, free_final)
                counts["reference_allowed"] += reference_allowed
                if shape == "whole_word" and not reference_allowed:
                    excluded_references[reference_word] += 1

                if hypothesis_word is None:
                    continue
                _, hypothesis_classes = diacritic_classes(hypothesis_word)
                hypothesis_allowed = _allowed(hypothesis_classes, variants, free_final)
                correct = _same_diacritics(reference_word, hypothesis_word, free_final)
                if correct:
                    counts["correct"] += 1
                    counts["broken"] += not reference_allowed
                else:
                    counts["wrong"] += 1
                    if not reference_allowed:
                        counts["wrong_reference_excluded"] += 1
                    elif hypothesis_allowed:
                        counts["wrong_allowed"] += 1
                    else:
                        counts["fixable"] += 1

    results["shapes"] = {shape: dict(counts) for shape, counts in results["shapes"].items()}
    results["excluded_references"] = excluded_references.most_common(20)
    results["pronunciations_aligned"] = lexicon.num_aligned
    results["pronunciations_unaligned"] = lexicon.num_unaligned
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze how a pronunciation lexicon would constrain diacritization")
    parser.add_argument("--lexicon", required=True, help="Path to the lexicon (Buckwalter word followed by its phones, one entry per line)")
    parser.add_argument("-ref", "--reference-file-path", required=True, help="Path to ground truth file")
    parser.add_argument("-hyp", "--hypothesis-file-path", default=None, help="Path to predictions file")
    parser.add_argument("--log_file", default=None, help="Path to the log file")
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)
    logger = logging.getLogger(__name__)

    lexicon = DiacritizationLexicon.from_file(args.lexicon)

    if args.hypothesis_file_path:
        reference_lines, hypothesis_lines = read_pair(args.reference_file_path, args.hypothesis_file_path)
    else:
        with open(args.reference_file_path, "r", encoding="utf-8") as f:
            reference_lines, hypothesis_lines = f.readlines(), None

    results = analyze(lexicon, reference_lines, hypothesis_lines)

    tokens, covered = results["tokens"], results["covered"]
    logger.info(f"Analyzing: {args.reference_file_path} vs {args.hypothesis_file_path}")
    logger.info(f"Word tokens: {tokens}; with lexicon variants: {covered} ({covered / max(1, tokens):.1%})")
    unaligned, aligned = results["pronunciations_unaligned"], results["pronunciations_aligned"]
    logger.info(f"Pronunciations that could not be aligned: {unaligned} of {aligned + unaligned}")
    if results["skipped_lines"]:
        logger.info(f"Lines skipped (word count differs between reference and hypothesis): {results['skipped_lines']}")

    for shape, counts in results["shapes"].items():
        logger.info("")
        logger.info(f"[{shape}]")
        logger.info(f"  reference inside lexicon variants: {counts.get('reference_allowed', 0)} ({counts.get('reference_allowed', 0) / max(1, covered):.1%} of covered)")
        if hypothesis_lines is not None:
            logger.info(f"  hypothesis wrong: {counts.get('wrong', 0)}")
            logger.info(f"    fixable (hypothesis excluded, reference allowed): {counts.get('fixable', 0)}")
            logger.info(f"    wrong but inside lexicon variants:               {counts.get('wrong_allowed', 0)}")
            logger.info(f"    reference excluded as well:                      {counts.get('wrong_reference_excluded', 0)}")
            logger.info(f"  hypothesis correct: {counts.get('correct', 0)}")
            logger.info(f"    broken (reference excluded):                     {counts.get('broken', 0)}")

    logger.info("")
    logger.info("Most frequent references outside the lexicon variants (whole word):")
    for word, count in results["excluded_references"]:
        logger.info(f"  {count:4d}  {word}  {sorted(lexicon.variants(diacritic_classes(word)[0]))[:4]}")

    return results


if __name__ == "__main__":
    main()
