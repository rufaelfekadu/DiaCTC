# Lexicon-guided WFST decoding: analysis and results

## Summary

We tested whether a pronunciation lexicon can improve constrained WFST diacritization by limiting each word to the diacritizations the lexicon lists.

- **A hard constraint makes results worse.** Restricting words outright raises DER from 3.80 to 4.99 on CLArTTS and from 8.69 to 8.79 on ArVoice.
- **A soft constraint gives a small gain on ArVoice and none on CLArTTS.** With a penalty weight of 2, ArVoice DER falls from 8.69 to 8.51 and CLArTTS is unchanged within noise (3.80 to 3.78).
- **Any weight from 0.5 to 4 is at least as good as the baseline on both sets.** From 8 upward it is worse.
- **The ceiling is low.** About 72% of the baseline's wrong words are already inside the lexicon's variants, mostly case endings, where the lexicon cannot help.

The weight was tuned on the test sets, because neither dataset has a dev split, so the 0.2-point gain is an optimistic estimate.

## The lexicon

The lexicon is the QCRI `ar-ar_lexicon_2014-03-17.txt` (Apache 2.0): 526,119 words with 2,022,704 pronunciations, 3.8 per word on average.

Each line is an **undiacritized word in Buckwalter transliteration followed by a phoneme sequence**. It does not contain diacritized words:

```
ktb k a t a b
ktb k u t i b a
ktb k u t u b u n
```

Two steps are needed to turn it into allowed diacritics.

### Buckwalter to Arabic

Buckwalter is a fixed one-to-one character table, so no model is needed. `pyarabic.trans`, already a dependency, covers every character the lexicon uses except one stray `V` in a single entry. 526,116 of the 526,119 words convert to Arabic and back unchanged.

### Phonemes to diacritics

Each pronunciation is aligned to the letters of its word, which yields one vowel class per letter: none, `a`, `i`, `u`, or a tanween (`an`, `in`, `un`). For example `ktb → k u t i b a` gives (u, i, a).

The aligner (`align_pronunciation` in `src/diactc/lexicon.py`) follows the lexicon's own conventions:

- a consonant letter is its phone, followed by an optional short vowel or a word-final tanween;
- all hamza forms are the phone `G`;
- alef is a long `A` (the letter before it takes a fatha), a hamza plus vowel at the start of a word, or silent;
- the `l` of the article is silent before a sun letter;
- ta marbuta is silent, `h`, `p` or `t`, with an optional vowel;
- alef maqsura is silent or a long `a`;
- an unwritten long `A` is allowed after a consonant (هذا, الله);
- the lexicon's spelling variants are accepted: final `h` for ta marbuta, final `y` for alef maqsura, and the article's alef written with a hamza or madda.

On a 30,000-word sample, 0.04% of pronunciations could not be aligned (49 of 114,996).

### What the lexicon cannot tell us

- **Shadda.** Gemination is not marked (`mHmd → m u H a m a d`), so the lexicon only restricts the vowel. Shadda is always left optional.
- **Sukun versus no mark.** A letter with no vowel may carry a sukun or nothing.
- **Many case endings and forms.** Liaison vowels (`مِنَ`), some verb and passive forms (`جَهِلَ`, `حُكِيَ`), and proper nouns (`تُونُسَ`) are missing.

## Offline analysis

Before changing the decoder, we measured what a lexicon constraint could and could not do, using the baseline predictions (`scripts/analyze_lexicon.py`). Counts are word tokens.

Two constraint shapes were compared: restricting the **whole word**, and restricting the **stem only**, leaving the last letter free because it carries the case ending.

| | CLArTTS, whole word | CLArTTS, stem only | ArVoice, whole word | ArVoice, stem only |
| --- | --- | --- | --- | --- |
| Word tokens | 1,694 | 1,694 | 4,914 | 4,914 |
| Found in the lexicon | 95.1% | 95.1% | 98.5% | 98.5% |
| Reference is among the lexicon's variants | 89.0% | 94.1% | 92.6% | 96.0% |
| Baseline wrong words | 197 | 92 | 1,443 | 1,038 |
| …fixable at most (prediction excluded, reference allowed) | 38 | 17 | 203 | 149 |
| …wrong but already inside the variants | 113 | 55 | 1,069 | 779 |
| …reference excluded as well | 46 | 20 | 171 | 110 |
| Correct words whose reference is excluded | 131 | 75 | 185 | 85 |

In the stem-only columns a word counts as wrong only if its stem is wrong, which is why those totals are lower.

What this showed:

- **Whole word, pooled:** 241 fixable against 316 correct words excluded. A hard whole-word constraint would break more than it fixes.
- **Stem only, pooled:** 166 fixable against 160 excluded. Roughly break-even, positive on ArVoice (149 vs 85) and negative on CLArTTS (17 vs 75).
- **"Fixable" is an upper bound**, because the decoder still has to choose the right variant among several.
- **Most errors are out of reach.** Under the whole-word shape, 1,182 of the 1,640 baseline wrong words (72%) are already inside the lexicon's variants.

This led to two design choices: restrict the stem only, and make the constraint a tunable penalty rather than a hard rule.

## How the decoder uses the lexicon

The change is in `build_pattern_fsa` in `src/diactc/models/base.py`.

**Without a lexicon**, each word in the pattern graph is a straight chain: a letter, then a slot offering every allowed diacritic, then the next letter. All arcs cost nothing, so the audio alone decides.

**With a lexicon**, each known word gets a second set of paths between the same start and end points:

- **Lexicon paths, cost 0.** They spell the same letters, but each slot offers only the diacritics of one lexicon variant. For `كتب` the stems are (a, a), (u, i) and (u, u).
- **The original free chain, cost `lexicon_weight`.** It still allows anything, but its first arc carries the penalty.

A path's score is its acoustic log-probability, minus the weight if it used the free chain for that word. A reading the lexicon lists pays nothing; one it does not list pays the weight, so it wins only if the audio prefers it by more than the weight:

| Weight | Audio must favour the unlisted reading by a factor of |
| --- | --- |
| 0.5 | about 1.6 |
| 2 | about 7.4 |
| 4 | about 55 |
| 8 | about 3,000 |
| 1000 | effectively never (a hard constraint) |

The factor is e raised to the weight, since scores are natural-log probabilities.

Details that soften the constraint:

- The penalty is charged once per word, however many letters differ.
- The last letter is never restricted.
- Shadda is always optional.
- "No diacritic" is always allowed on any letter.
- Words the lexicon does not know, and one-letter words, are left untouched.
- A weight of 0 builds no lexicon paths, so decoding is identical to the baseline.

Side effect: the confidence scores behind `--confidence_threshold` are computed on this weighted graph, so with a lexicon they lean toward lexicon-listed readings. The two features have not been evaluated together.

## Results

Model `rufaelfekadu/diactc-combined`, constrained WFST decoding, on the CLArTTS test set (205 utterances) and the ArVoice test set (248 utterances, human speech).

### CLArTTS

| Lexicon weight | DER | DER, no case endings | DER, excl. no-diacritic letters | WER |
| --- | --- | --- | --- | --- |
| none (baseline) | 3.80 | 2.57 | 3.81 | 13.05% |
| 0.5 | 3.79 | 2.55 | 3.81 | 12.99% |
| 1 | 3.76 | 2.52 | 3.79 | 12.81% |
| 2 | 3.78 | 2.54 | 3.82 | 12.93% |
| 4 | 3.78 | 2.54 | 3.82 | 12.99% |
| 8 | 4.28 | 3.20 | 4.46 | 14.82% |
| 1000 | 4.99 | 4.12 | 5.35 | 16.65% |

### ArVoice

| Lexicon weight | DER | DER, no case endings | DER, excl. no-diacritic letters | WER |
| --- | --- | --- | --- | --- |
| none (baseline) | 8.69 | 7.56 | 6.53 | 30.36% |
| 0.5 | 8.61 | 7.47 | 6.47 | 30.10% |
| 1 | 8.55 | 7.39 | 6.45 | 30.08% |
| **2** | **8.51** | **7.34** | **6.43** | **29.96%** |
| 4 | 8.52 | 7.35 | 6.47 | 29.98% |
| 8 | 8.74 | 7.63 | 6.79 | 30.65% |
| 1000 | 8.79 | 7.69 | 6.87 | 30.75% |

DER is with case endings and includes no-diacritic positions unless the column says otherwise. WER is the word error rate on the diacritized text.

### Reading the results

- **ArVoice improves by about 0.2 DER points** at weights 2 to 4, across every DER variant.
- **CLArTTS does not move meaningfully.** The largest change, 0.04 points at weight 1, is within noise for 205 utterances.
- **High weights hurt, as the offline analysis predicted.** At weight 2 the lexicon overrides the model only where the audio was nearly undecided, which is where the model is most often wrong. At 8 and above it also overrides confident predictions, and the words whose correct form the lexicon lacks become new errors.
- **The measured gain is far below the offline upper bound**, consistent with "fixable" counting words the decoder could fix, not ones it does.

## Caveats

- **Tuned on the test sets.** The weight is one scalar, so overfitting is limited, but the gain should be confirmed on held-out data.
- **One model, two datasets.** Both are read or formal Modern Standard Arabic; the lexicon may behave differently on dialectal or conversational speech.
- **ArVoice references are partly undiacritized**, which lowers its scores in the "including no-diacritic" DER variants independently of the lexicon.
- **No interaction test** with the confidence threshold.

## Recommendation

Keep the lexicon off by default. For ArVoice-like text it is a safe, small improvement at `--lexicon_weight 2`. A larger gain would need a lexicon that covers case endings and marks gemination, since those are where the remaining errors are.

## Reproducing

Offline analysis, from a reference and prediction file:

```bash
python scripts/analyze_lexicon.py \
    --lexicon ar-ar_lexicon_2014-03-17.txt \
    -ref outputs/decode/wav2vec/wfst/constrained/clartts_test_metadata/gt.txt \
    -hyp outputs/decode/wav2vec/wfst/constrained/clartts_test_metadata/pred.txt
```

Lexicon-guided decoding at one weight:

```bash
python scripts/inference.py \
    --model_path rufaelfekadu/diactc-combined \
    --manifest_paths <clartts_test_metadata.json> <arvoice_test_metadata.json> \
    --method wfst \
    --use_blank_token --use_no_diac_token --use_unk_diac_token \
    --lexicon ar-ar_lexicon_2014-03-17.txt --lexicon_weight 2.0 \
    --output_path outputs/decode \
    --device cuda:0
```

Outputs go to `outputs/decode/wav2vec/wfst/constrained/lex_<weight>/<dataset>/`. One pass over both test sets takes about two minutes and peaks near 5.5 GB of RAM.

Code: `src/diactc/lexicon.py` (conversion and alignment), `src/diactc/models/base.py` (`build_pattern_fsa`), `scripts/analyze_lexicon.py`, and tests in `tests/test_lexicon.py` and `tests/test_decode_wfst.py`.
