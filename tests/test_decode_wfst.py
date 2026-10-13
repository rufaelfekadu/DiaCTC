# -*- coding: utf-8 -*-
"""Tests for DiacritizationModel.decode_wfst confidence thresholding.

These use a toy vocabulary and hand-built frame probabilities, so no
checkpoint is needed. They are skipped when k2 is not installed.
"""

import pytest

k2 = pytest.importorskip("k2")
import torch

from diactc.config import NO_DIAC_TOKEN, PAD_TOKEN, WILDCARD_TOKEN, WORD_DELIMITER_TOKEN
from diactc.lexicon import DiacritizationLexicon
from diactc.models.base import DiacritizationModel
from diactc.utils.text import DAMMA, FATHA

BA, TA = "ب", "ت"
PATTERN = [BA, WILDCARD_TOKEN, TA, WILDCARD_TOKEN]


class ToyModel(DiacritizationModel):
    def __init__(self, vocab):
        self._token2id = {token: i for i, token in enumerate(vocab)}
        self.device = torch.device("cpu")

    @property
    def token2id(self):
        return self._token2id

    @property
    def constrained_wildcard_ids(self):
        return [self._token2id[t] for t in (FATHA, DAMMA, NO_DIAC_TOKEN, PAD_TOKEN) if t in self._token2id]

    @property
    def unconstrained_wildcard_ids(self):
        return list(self._token2id.values())

    @property
    def word_delimiter_token(self):
        return WORD_DELIMITER_TOKEN

    def load_model(self):
        pass

    def get_logits(self):
        pass

    def diacritize(self, text, audio_path, constrained=True, method="wfst", confidence_threshold=None):
        pass

    def log_probs(self, frames, floor=0.002):
        """One row per frame; each frame is a {token: probability} dict."""
        probs = torch.full((len(frames), len(self._token2id)), floor)
        for t, frame in enumerate(frames):
            for token, p in frame.items():
                probs[t, self._token2id[token]] = p
        return torch.log(probs / probs.sum(dim=-1, keepdim=True))


# ب is followed by a confident fatha, ت by a damma that fatha nearly ties with.
FRAMES = [{BA: 1.0}, {FATHA: 1.0}, {TA: 1.0}, {DAMMA: 0.55, FATHA: 0.45}]


@pytest.fixture
def model():
    return ToyModel([PAD_TOKEN, BA, FATHA, DAMMA, NO_DIAC_TOKEN, TA])


def tokens(model, ids):
    id2token = {i: t for t, i in model.token2id.items()}
    return [id2token[i] for i in ids]


class TestConfidenceThreshold:
    def test_no_threshold_keeps_best_path(self, model):
        decoded = model.decode_wfst(model.log_probs(FRAMES), PATTERN)
        assert tokens(model, decoded) == [BA, FATHA, TA, DAMMA]

    def test_zero_threshold_matches_no_threshold(self, model):
        log_probs = model.log_probs(FRAMES)
        assert model.decode_wfst(log_probs, PATTERN, confidence_threshold=0.0) == model.decode_wfst(log_probs, PATTERN)

    def test_only_uncertain_diacritic_falls_back(self, model):
        decoded = model.decode_wfst(model.log_probs(FRAMES), PATTERN, confidence_threshold=0.8)
        assert tokens(model, decoded) == [BA, FATHA, TA, NO_DIAC_TOKEN]

    def test_threshold_below_posterior_keeps_diacritic(self, model):
        decoded = model.decode_wfst(model.log_probs(FRAMES), PATTERN, confidence_threshold=0.5)
        assert tokens(model, decoded) == [BA, FATHA, TA, DAMMA]

    def test_threshold_of_one_drops_all_diacritics_but_keeps_letters(self, model):
        decoded = model.decode_wfst(model.log_probs(FRAMES), PATTERN, confidence_threshold=1.0)
        assert tokens(model, decoded) == [BA, NO_DIAC_TOKEN, TA, NO_DIAC_TOKEN]

    def test_confidences_cover_wildcard_tokens_only(self, model):
        ids, confidences = model.decode_wfst_with_confidence(model.log_probs(FRAMES), PATTERN)
        assert tokens(model, ids) == [BA, FATHA, TA, DAMMA]
        assert confidences[0] is None and confidences[2] is None
        assert confidences[1] > 0.9
        assert 0.5 < confidences[3] < 0.6

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 0.8, 1.0])
    def test_split_decode_matches_decode_wfst(self, model, threshold):
        log_probs = model.log_probs(FRAMES)
        ids, confidences = model.decode_wfst_with_confidence(log_probs, PATTERN)
        assert model.apply_confidence_threshold(ids, confidences, threshold) == model.decode_wfst(log_probs, PATTERN, confidence_threshold=threshold)

    def test_uncertain_diacritic_is_dropped_without_no_diac_token(self):
        model = ToyModel([PAD_TOKEN, BA, FATHA, DAMMA, TA])
        decoded = model.decode_wfst(model.log_probs(FRAMES), PATTERN, confidence_threshold=0.8)
        assert tokens(model, decoded) == [BA, FATHA, TA]


# The first diacritic is a fatha that damma nearly ties with; the lexicon only knows the word with a damma.
LEXICON_FRAMES = [{BA: 1.0}, {FATHA: 0.55, DAMMA: 0.45}, {TA: 1.0}, {DAMMA: 1.0}]


class TestLexicon:
    @pytest.fixture
    def lexicon(self):
        return DiacritizationLexicon({"bt": [("b", "u", "t")]})

    def decode(self, model, lexicon, weight, **kwargs):
        model.lexicon, model.lexicon_weight = lexicon, weight
        return tokens(model, model.decode_wfst(model.log_probs(LEXICON_FRAMES), PATTERN, **kwargs))

    def test_without_lexicon(self, model):
        assert self.decode(model, None, 0.0) == [BA, FATHA, TA, DAMMA]

    def test_zero_weight_matches_no_lexicon(self, model, lexicon):
        assert self.decode(model, lexicon, 0.0) == [BA, FATHA, TA, DAMMA]

    def test_lexicon_variant_wins_when_the_weight_outweighs_the_margin(self, model, lexicon):
        assert self.decode(model, lexicon, 5.0) == [BA, DAMMA, TA, DAMMA]

    def test_acoustics_win_when_the_weight_is_below_the_margin(self, model, lexicon):
        assert self.decode(model, lexicon, 0.05) == [BA, FATHA, TA, DAMMA]

    def test_last_letter_is_not_restricted(self, model, lexicon):
        # the lexicon variant ends without a vowel, the audio has a damma
        assert self.decode(model, lexicon, 5.0)[-1] == DAMMA

    def test_unknown_word_is_unconstrained(self, model):
        lexicon = DiacritizationLexicon({"tb": [("t", "u", "b")]})
        assert self.decode(model, lexicon, 5.0) == [BA, FATHA, TA, DAMMA]

    def test_words_before_a_lexicon_word_are_kept(self, lexicon):
        model = ToyModel([PAD_TOKEN, BA, FATHA, DAMMA, NO_DIAC_TOKEN, TA, WORD_DELIMITER_TOKEN])
        model.lexicon, model.lexicon_weight = lexicon, 5.0
        pattern = [TA, WILDCARD_TOKEN, WORD_DELIMITER_TOKEN] + PATTERN
        frames = [{TA: 1.0}, {FATHA: 1.0}, {WORD_DELIMITER_TOKEN: 1.0}] + LEXICON_FRAMES
        decoded = model.decode_wfst(model.log_probs(frames), pattern)
        assert tokens(model, decoded) == [TA, FATHA, WORD_DELIMITER_TOKEN, BA, DAMMA, TA, DAMMA]

    def test_unconstrained_decoding_ignores_the_lexicon(self, model, lexicon):
        assert self.decode(model, lexicon, 5.0, constrained=False) == [BA, FATHA, TA, DAMMA]

    def test_confidence_sums_over_lexicon_and_free_paths(self, model, lexicon):
        model.lexicon, model.lexicon_weight = lexicon, 5.0
        ids, confidences = model.decode_wfst_with_confidence(model.log_probs(LEXICON_FRAMES), PATTERN)
        assert tokens(model, ids) == [BA, DAMMA, TA, DAMMA]
        assert 0.9 < confidences[1] <= 1.0
        assert 0.9 < confidences[3] <= 1.0
