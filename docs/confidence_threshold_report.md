# Confidence threshold for WFST decoding: analysis and results

12 October 2026 · branch `wfst-confidence-threshold` · model `rufaelfekadu/diactc-combined`

## Summary

We added an optional confidence threshold to constrained WFST decoding: a diacritic the model is unsure of is left out instead of guessed. We then searched for the best threshold on CLArTTS and ArVoice.

- **No threshold improves DER.** The references are fully diacritized, so a dropped diacritic is almost always an error. The lowest DER is at threshold 0, which is decoding without a threshold.
- **Precision rises slowly and expensively.** Reaching 99% precision takes a threshold of 0.995 on CLArTTS, which leaves only 64% of the reference diacritics recovered. ArVoice never reaches 99% on the grid tested.
- **The balance point is 0.25.** Up to there, a point of DER buys about a point of precision. Pooled precision rises from 92.3% to 92.7% and DER from 5.9% to 6.2%, so the gain is small.
- **Two decoding bugs were found and fixed along the way.** Both change results without a threshold.

The threshold was tuned on the test sets, because neither dataset has a dev split.

## How the threshold works

Constrained WFST decoding fixes the letters of the text and lets the model choose a diacritic for each letter from an allowed set (the diacritics, plus "no diacritic"). Normally it commits to the best path whatever the margin.

With `confidence_threshold` set, each decoded diacritic gets a confidence score, and any diacritic scoring below the threshold is dropped.

**The score is a lattice posterior.** For a given letter, it is the probability of the decoded diacritic among all the options allowed on that letter, summed over every alignment of the audio to the text (forward-backward on the decoding lattice). The options for a letter sum to 1, so the threshold is a plain probability between 0 and 1.

This was preferred to thresholding raw frame-level softmax values because it gives one number per letter, is normalized over only the allowed options, and accounts for alignment uncertainty.

**Implementation.** Each arc of the pattern graph carries an id that k2 propagates through composition and intersection, so the posterior of each lattice arc can be summed back onto the pattern arc it came from (`decode_wfst_with_confidence` in `src/diactc/models/base.py`).

**What happens to a dropped diacritic.** It is replaced by `<no_diac>` when the vocabulary has that token, and otherwise removed. Either way the letter comes out undiacritized. `rufaelfekadu/diactc-combined` has no `<no_diac>` token, so here it is always removed.

**Left untouched:** the base letters, and letters already decoded as `<no_diac>` or `<unk_diac>`. With no threshold set, the decode path is the same as before.

### Is the score informative?

Yes. On a probe of 376 letters from 10 CLArTTS utterances, accuracy rises steadily with the score:

| Posterior of the decoded diacritic | Letters | Accuracy |
| --- | --- | --- |
| below 0.5 | 7 | 57.1% |
| 0.5 to 0.7 | 11 | 72.7% |
| 0.7 to 0.9 | 18 | 88.9% |
| 0.9 to 0.99 | 60 | 96.7% |
| 0.99 to 0.999 | 139 | 97.8% |
| 0.999 and above | 141 | 100.0% |

The catch is in the counts: most errors sit among high-confidence predictions, because there are so many more of them. A threshold has to discard many correct diacritics to remove the wrong ones.

## Setup

| Item | Value |
| --- | --- |
| Model | `rufaelfekadu/diactc-combined`, constrained WFST decoding |
| CLArTTS | test split, 205 utterances, 0.25 h |
| ArVoice | test split (human speech), 248 utterances, 0.84 h |
| Thresholds | 0 to 0.95 in steps of 0.05, then 0.97, 0.98, 0.99, 0.995, 0.999 |

`scripts/tune_threshold.py` decodes each utterance once and applies every threshold to the stored confidences, so the model is not re-run per threshold.

Metrics:

- **Precision:** the share of emitted diacritics that match the reference.
- **Recall:** the share of reference diacritics that are recovered.
- **DER:** diacritic error rate with case endings. Two variants are reported: excluding letters the reference leaves undiacritized, and including them. The excluding variant equals 100 minus recall, so trading it against precision is a precision-recall trade-off.

## Results

Threshold 0 is decoding without a threshold. All values are percentages.

### CLArTTS

| Threshold | Precision | DER, excl. undiacritized letters | DER, incl. undiacritized letters |
| --- | --- | --- | --- |
| 0 | 96.39 | 3.81 | 3.80 |
| 0.1 | 96.39 | 3.84 | 3.83 |
| 0.2 | 96.52 | 4.04 | 3.92 |
| **0.25** | **96.59** | **4.13** | **3.96** |
| 0.3 | 96.69 | 4.22 | 3.98 |
| 0.4 | 96.77 | 4.39 | 4.05 |
| 0.5 | 97.00 | 4.64 | 4.18 |
| 0.6 | 97.31 | 5.15 | 4.54 |
| 0.7 | 97.51 | 5.83 | 5.04 |
| 0.8 | 97.78 | 7.05 | 5.98 |
| 0.9 | 98.05 | 9.40 | 7.74 |
| 0.95 | 98.35 | 12.91 | 10.48 |
| 0.99 | 98.86 | 25.71 | 20.49 |
| 0.999 | 99.45 | 64.04 | 50.72 |

### ArVoice

| Threshold | Precision | DER, excl. undiacritized letters | DER, incl. undiacritized letters |
| --- | --- | --- | --- |
| 0 | 91.06 | 6.53 | 8.69 |
| 0.1 | 91.14 | 6.57 | 8.69 |
| 0.2 | 91.31 | 6.72 | 8.72 |
| **0.25** | **91.48** | **6.87** | **8.75** |
| 0.3 | 91.64 | 7.21 | 8.92 |
| 0.4 | 92.05 | 8.03 | 9.36 |
| 0.5 | 92.56 | 9.27 | 10.06 |
| 0.6 | 93.10 | 10.90 | 11.03 |
| 0.7 | 93.67 | 13.06 | 12.34 |
| 0.8 | 94.33 | 17.01 | 14.91 |
| 0.9 | 95.20 | 25.61 | 20.72 |
| 0.95 | 96.09 | 37.45 | 28.79 |
| 0.99 | 97.26 | 70.13 | 51.79 |
| 0.999 | 98.39 | 95.77 | 70.14 |

### Pooled

| Threshold | Precision | DER, excl. undiacritized letters | DER, incl. undiacritized letters |
| --- | --- | --- | --- |
| 0 | 92.31 | 5.87 | 7.59 |
| 0.1 | 92.37 | 5.91 | 7.59 |
| 0.2 | 92.54 | 6.07 | 7.63 |
| **0.25** | **92.68** | **6.21** | **7.67** |
| 0.3 | 92.84 | 6.49 | 7.80 |
| 0.4 | 93.18 | 7.16 | 8.16 |
| 0.5 | 93.63 | 8.16 | 8.73 |
| 0.6 | 94.13 | 9.52 | 9.56 |
| 0.7 | 94.62 | 11.32 | 10.69 |
| 0.8 | 95.21 | 14.62 | 12.89 |
| 0.9 | 95.97 | 21.71 | 17.78 |
| 0.95 | 96.77 | 31.56 | 24.65 |
| 0.99 | 97.96 | 59.46 | 44.71 |
| 0.999 | 99.16 | 88.15 | 65.74 |

Pooled figures are dominated by ArVoice, which holds about three quarters of the emitted diacritics. All 25 thresholds are in `outputs/tune_threshold/der/results.json`.

## Choosing a threshold

The search script supports two objectives.

### Lowest DER

| Dataset | Selected threshold | DER (incl. undiacritized letters) |
| --- | --- | --- |
| CLArTTS | 0 | 3.80 |
| ArVoice | 0 | 8.69 |
| Pooled | 0.05 | 7.58, against 7.59 at 0 |

The pooled difference is at the rounding level. DER rises steadily from there, so for DER the answer is to use no threshold.

### Target precision of 99%

| Dataset | Selected threshold | Precision | Recall |
| --- | --- | --- | --- |
| CLArTTS | 0.995 | 99.24% | 64.1% |
| ArVoice | not reached; best is 0.999 | 98.39% | 4.2% |
| Pooled | 0.999 | 99.16% | 11.9% |

A 99% target is reachable only by discarding most diacritics.

### Balance between precision and DER

The balance point was taken as the highest threshold at which a point of DER still buys at least a point of precision, measured from decoding without a threshold. DER here excludes undiacritized letters.

| Threshold | Pooled precision gained (points) | Pooled DER paid (points) | DER points per precision point |
| --- | --- | --- | --- |
| 0.25 | 0.37 | 0.34 | 0.9 |
| 0.3 | 0.53 | 0.62 | 1.2 |
| 0.5 | 1.32 | 2.29 | 1.7 |
| 0.7 | 2.31 | 5.45 | 2.4 |
| 0.9 | 3.66 | 15.84 | 4.3 |

- **On pooled data the price crosses 1 just after 0.25.** F1 agrees: it peaks at 93.2% between 0.15 and 0.25 and falls from 0.3 on.
- **The rule holds on ArVoice**, where the price at 0.25 is 0.8.
- **On CLArTTS thresholding barely pays at all.** The price is already above 1 from 0.2, on a precision gain of only 0.2 points.

This rule weighs a point of precision and a point of DER equally. A different weighting moves the choice; the tables give the cost of each alternative.

## Bugs fixed along the way

Both fixes were ported from the upstream `diac-wfst` repository and change behaviour even without a threshold.

- **Shadda combinations were scored as errors.** `diacritize()` left the merged shadda+vowel tokens in its output as private-use characters (U+E001 to U+E006). They are now mapped back to their diacritics.
- **`--method wfst --device cuda` crashed.** `k2.compose` with epsilon arcs runs only on CPU. The WFST step now runs on CPU while the model forward pass stays on the configured device. Decoding costs about 0.1 s per utterance on CPU, against several seconds for a CPU forward pass.

## Caveats

- **Tuned on the test sets.** The threshold is one scalar fitted on about 23,000 diacritics, so the optimism should be small, but the choice should be confirmed on held-out data.
- **ArVoice precision may be capped by its references.** Without a threshold, 5.2% of the diacritics emitted on ArVoice (929 of 17,835) fall on letters the reference leaves undiacritized, more than the 3.7% (666) that are substitutions. Whether those are model errors or annotation gaps has not been checked. On CLArTTS the same figures are 1.0% and 2.6% of 5,481.
- **One model, two datasets**, both read or formal Modern Standard Arabic.

## Recommendation

Leave the threshold off when DER is the goal. Use it when a wrong diacritic costs more than a missing one: 0.25 is the balanced setting, and the tables give the price of going higher.

## Reproducing

Threshold search, with either objective:

```bash
python scripts/tune_threshold.py \
    --model_path rufaelfekadu/diactc-combined \
    --manifest_paths <clartts_test_metadata.json> <arvoice_test_metadata.json> \
    --objective der \
    --use_blank_token --use_no_diac_token --use_unk_diac_token \
    --output_path outputs/tune_threshold \
    --device cuda:0
```

Use `--objective precision --target_precision 0.99` for the precision target. Results are written to `outputs/tune_threshold/[der|precision_<target>]/results.json`.

Decoding at a chosen threshold:

```bash
python scripts/inference.py \
    --model_path rufaelfekadu/diactc-combined \
    --manifest_paths <clartts_test_metadata.json> <arvoice_test_metadata.json> \
    --method wfst \
    --use_blank_token --use_no_diac_token --use_unk_diac_token \
    --confidence_threshold 0.25 \
    --output_path outputs/decode \
    --device cuda:0
```

Outputs go to `outputs/decode/wav2vec/wfst/constrained/conf_<threshold>/<dataset>/`. Running `inference.py` at thresholds 0.05 and 0.999 reproduced the DER and coverage of the search table exactly.

Code: `src/diactc/models/base.py` (`decode_wfst`, `decode_wfst_with_confidence`, `apply_confidence_threshold`), `scripts/tune_threshold.py`, `diac_precision_recall` in `src/diactc/metrics/coverage.py`, and tests in `tests/test_decode_wfst.py` and `tests/test_metrics.py`.
