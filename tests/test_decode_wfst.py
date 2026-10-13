# -*- coding: utf-8 -*-
"""Tests for DiacritizationModel.decode_wfst confidence thresholding.

These use a toy vocabulary and hand-built frame probabilities, so no
checkpoint is needed. They are skipped when k2 is not installed.
"""

import pytest

k2 = pytest.importorskip("k2")
import torch

from diactc.config import NO_DIAC_TOKEN, PAD_TOKEN, WILDCARD_TOKEN
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
