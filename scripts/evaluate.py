"""Evaluate diacritization predictions with DER, WER, and SER.

Example:
    python scripts/evaluate.py \
        -ref outputs/wav2vec/wfst/constrained/clartts/gt.txt \
        -hyp outputs/wav2vec/wfst/constrained/clartts/pred.txt \
        --log_file outputs/wav2vec/wfst/constrained/clartts/eval.log
"""

import argparse
import logging

from diactc.metrics import evaluate


def _format_table(metric_name, variants):
    header = (
        "+---------------------------------------------------------------------------------------------+\n"
        "|       |  With case ending  | Without case ending |  With case ending  | Without case ending |\n"
        f"|  {metric_name:<4} |------------------------------------------+------------------------------------------|\n"
        "|       |          Including no diacritic          |          Excluding no diacritic          |\n"
        "|-------+------------------------------------------+------------------------------------------|\n"
    )
    row = "|   %%   |        %5.2f       |        %5.2f        |        %5.2f       |        %5.2f        |" % (
        variants["with_case_with_no_diac"],
        variants["without_case_with_no_diac"],
        variants["with_case_without_no_diac"],
        variants["without_case_without_no_diac"],
    )
    footer = "\n+---------------------------------------------------------------------------------------------+"
    return header + row + footer


def _format_coverage(value):
    return "n/a" if value is None else f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="Calculate DER, WER, and SER")
    parser.add_argument("-ref", "--reference-file-path", required=True, help="Path to ground truth file")
    parser.add_argument("-hyp", "--hypothesis-file-path", required=True, help="Path to predictions file")
    parser.add_argument(
        "-s", "--style", default="Fadel", choices=["Zitouni", "Fadel"],
        help="Evaluation style",
    )
    parser.add_argument("--log_file", default="eval.log", help="Path to the log file")
    parser.add_argument(
        "--strict", action="store_true",
        help="Use strict positional metrics (require aligned reference/hypothesis).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    logger.info(
        f"Evaluating: {args.reference_file_path} vs {args.hypothesis_file_path} "
        f"(style={args.style}, strict={args.strict})"
    )
    results = evaluate(
        args.reference_file_path, args.hypothesis_file_path,
        style=args.style, strict=args.strict,
    )

    for metric in ("der", "wer", "ser"):
        logger.info(_format_table(metric.upper(), results[metric]))
        logger.info("")

    coverage = results["coverage"]
    logger.info(
        "Diacritic coverage: reference=%s, hypothesis=%s",
        _format_coverage(coverage["reference"]),
        _format_coverage(coverage["hypothesis"]),
    )

    return results


if __name__ == "__main__":
    main()
