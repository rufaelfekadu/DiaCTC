"""Diacritization evaluation metrics (DER, WER, SER).

Public functions take a reference and a hypothesis (file paths, lists of lines,
or multi-line strings) and fill in the Arabic letter / diacritic class lists from
:mod:`diactc.constants`, so callers do not need to manage those constants.
"""

from diactc.constants import ARABIC_LETTERS, DIACRITIC_CLASSES
from diactc.metrics._common import read_pair
from diactc.metrics.coverage import diac_coverage_rate
from diactc.metrics.der import calculate_der as _calculate_der
from diactc.metrics.wer import (
    calculate_wer as _calculate_wer,
    calculate_ser as _calculate_ser,
)
from diactc.metrics.strict import (
    calculate_der as _calculate_der_strict,
    calculate_wer as _calculate_wer_strict,
    calculate_ser as _calculate_ser_strict,
)

__all__ = [
    "calculate_der",
    "calculate_wer",
    "calculate_ser",
    "calculate_der_strict",
    "calculate_wer_strict",
    "calculate_ser_strict",
    "diac_coverage_rate",
    "evaluate",
]


def calculate_der(reference, hypothesis, style="Fadel", case_ending=True, no_diacritic=True):
    return _calculate_der(
        reference, hypothesis, ARABIC_LETTERS, DIACRITIC_CLASSES,
        style, case_ending, no_diacritic,
    )


def calculate_wer(reference, hypothesis, style="Fadel", case_ending=True, no_diacritic=True):
    return _calculate_wer(
        reference, hypothesis, ARABIC_LETTERS, DIACRITIC_CLASSES,
        style, case_ending, no_diacritic,
    )


def calculate_ser(reference, hypothesis, style="Fadel", case_ending=True, no_diacritic=True):
    return _calculate_ser(
        reference, hypothesis, ARABIC_LETTERS, DIACRITIC_CLASSES,
        style, case_ending, no_diacritic,
    )


def calculate_der_strict(reference, hypothesis, style="Fadel", case_ending=True, no_diacritic=True):
    """Strict DER: positional comparison, requires aligned reference/hypothesis."""
    return _calculate_der_strict(
        reference, hypothesis, ARABIC_LETTERS, DIACRITIC_CLASSES,
        style, case_ending, no_diacritic,
    )


def calculate_wer_strict(reference, hypothesis, style="Fadel", case_ending=True, no_diacritic=True):
    """Strict WER: positional comparison, requires aligned reference/hypothesis."""
    return _calculate_wer_strict(
        reference, hypothesis, ARABIC_LETTERS, DIACRITIC_CLASSES,
        style, case_ending, no_diacritic,
    )


def calculate_ser_strict(reference, hypothesis, style="Fadel", case_ending=True, no_diacritic=True):
    """Strict SER: positional comparison, requires aligned reference/hypothesis."""
    return _calculate_ser_strict(
        reference, hypothesis, ARABIC_LETTERS, DIACRITIC_CLASSES,
        style, case_ending, no_diacritic,
    )


# The four standard reporting variants: (case_ending, no_diacritic).
_VARIANTS = {
    "with_case_with_no_diac": (True, True),
    "without_case_with_no_diac": (False, True),
    "with_case_without_no_diac": (True, False),
    "without_case_without_no_diac": (False, False),
}


def _safe_coverage(text):
    try:
        return diac_coverage_rate(text)
    except ValueError:
        return None


def evaluate(reference, hypothesis, style="Fadel", strict=False):
    """Compute DER, WER, SER, and diacritic coverage.

    Returns a nested dict with one entry per metric. DER/WER/SER each map to the
    four standard reporting variants; "coverage" maps to the reference and
    hypothesis coverage rates.

    When ``strict`` is True, the positional (non-aligned) metrics are used, which
    require the reference and hypothesis to share the same base characters.
    """
    if strict:
        metric_fns = {
            "der": calculate_der_strict,
            "wer": calculate_wer_strict,
            "ser": calculate_ser_strict,
        }
    else:
        metric_fns = {"der": calculate_der, "wer": calculate_wer, "ser": calculate_ser}
    results = {}
    for metric, fn in metric_fns.items():
        results[metric] = {
            variant: fn(reference, hypothesis, style=style, case_ending=ce, no_diacritic=nd)
            for variant, (ce, nd) in _VARIANTS.items()
        }

    reference_content, hypothesis_content = read_pair(reference, hypothesis)
    results["coverage"] = {
        "reference": _safe_coverage("".join(reference_content)),
        "hypothesis": _safe_coverage("".join(hypothesis_content)),
    }
    return results
