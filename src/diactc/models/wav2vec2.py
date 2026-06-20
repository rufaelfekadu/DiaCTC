from transformers import Wav2Vec2Processor, AutoModelForCTC
import torch
import numpy as np
import librosa
import time
import re

from diactc.models.base import DiacritizationModel
from diactc.utils.text import (
    BASE_DIACRITICS_STR,
    VALID_DIACRITICS_COMBINATIONS,
    REVERSE_MERGED_DIAC_TO_TOKEN_MAP,
    preprocess_text,
    form_wildcard_pattern,
)
from diactc.config import (
    NO_DIAC_TOKEN,
    UNK_DIAC_TOKEN,
    UNK_TOKEN,
)


class Wav2Vec2DiacritizationModel(DiacritizationModel):
    '''
    Wav2Vec2 diacritization model. implements
    - load_model
    - get_logits
    - diacritize ctc
    - diacritize wfst
    '''

    def __init__(self, model_name, device="cpu", use_blank_token=True, use_no_diac_token=True, use_unk_diac_token=True):
        self.model_name = model_name
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.sr = self.processor.feature_extractor.sampling_rate or 16000
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.device = torch.device(device)
        self.use_blank_token = use_blank_token
        self.use_no_diac_token = use_no_diac_token
        self.use_unk_diac_token = use_unk_diac_token
        self.to(self.device)
        self.model.eval()

    @property
    def token2id(self):
        return dict(self.processor.tokenizer.get_vocab().items())

    @property
    def id2token(self):
        return {v: k for k, v in self.token2id.items()}

    @property
    def constrained_wildcard_ids(self):
        merged_diacs = list(REVERSE_MERGED_DIAC_TO_TOKEN_MAP.keys())
        constrained_wildcard_ids = [self.token2id[ch] for ch in set(VALID_DIACRITICS_COMBINATIONS + merged_diacs) if ch in self.processor.tokenizer.get_vocab()]

        if self.use_blank_token:
            constrained_wildcard_ids.append(self.token2id[self.processor.tokenizer.pad_token])

        if NO_DIAC_TOKEN in self.processor.tokenizer.get_vocab() and self.use_no_diac_token:
            constrained_wildcard_ids.append(self.token2id[NO_DIAC_TOKEN])

        if UNK_DIAC_TOKEN in self.processor.tokenizer.get_vocab() and self.use_unk_diac_token:
            constrained_wildcard_ids.append(self.token2id[UNK_DIAC_TOKEN])

        return constrained_wildcard_ids

    @property
    def unconstrained_wildcard_ids(self):
        return [v for k, v in self.processor.tokenizer.get_vocab().items()]

    @property
    def word_delimiter_token(self):
        return self.processor.tokenizer.word_delimiter_token

    def load_model(self):
        pass

    def _replace_unk_with_source(self, decoded_text: str, source_text: str) -> str:
        """
        Replace base vocabulary UNK tokens (e.g. '<unk>') in the decoded text
        with the corresponding characters from the original undiacritized text.

        Args:
            decoded_text: String obtained by concatenating decoded tokens
                          (may contain diacritics and special tokens like '<unk>').
            source_text:  Original undiacritized text used to build the pattern
                          (after preprocessing and space -> word_delimiter, so it
                          aligns character-wise with the decoded sequence).
        """
        if UNK_TOKEN not in decoded_text:
            return decoded_text

        decoded_text_split = []
        last_char = None
        for group in re.finditer(UNK_TOKEN, decoded_text):
            st, en = group.span()
            if last_char is None and decoded_text[:st]:
                decoded_text_split.append(decoded_text[:st])
            elif last_char and decoded_text[last_char:st]:
                decoded_text_split.append(decoded_text[last_char:st])

            last_char = en
            decoded_text_split.append(UNK_TOKEN)
        if last_char < len(decoded_text):
            decoded_text_split.append(decoded_text[last_char:])

        result = []
        src_chars = list(source_text)
        src_idx = 0
        decoded_text_token_index = 0

        while decoded_text_token_index < len(decoded_text_split):
            decoded_text_token = decoded_text_split[decoded_text_token_index]

            # Replace unknown character using original text
            if decoded_text_token == UNK_TOKEN:
                # Skip diacritics of src_characters
                while src_idx < len(src_chars) and src_chars[src_idx] in BASE_DIACRITICS_STR:
                    src_idx += 1

                # Map this UNK to the next character from the original text
                assert src_idx < len(src_chars)
                result.append(src_chars[src_idx])
                src_idx += 1
            else:
                # The strings must match
                token_index = 0
                while token_index < len(decoded_text_token):
                    # Skip diacritics of src_characters
                    while src_idx < len(src_chars) and src_chars[src_idx] in BASE_DIACRITICS_STR:
                        src_idx += 1

                    if decoded_text_token[token_index] in BASE_DIACRITICS_STR:
                        result.append(decoded_text_token[token_index])
                    else:
                        assert decoded_text_token[token_index] == src_chars[src_idx]
                        result.append(decoded_text_token[token_index])
                        src_idx += 1

                    token_index += 1
            decoded_text_token_index += 1

        return "".join(result)

    def get_logits(self, wav):
        '''
        Get logits from the model.
        Args:
            wav: wav file
        Returns:
            logits: logits from the model
        '''
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        elif isinstance(wav, torch.Tensor):
            wav = wav
        else:
            raise ValueError(f"Unsupported wav type: {type(wav)}")

        inputs = self.processor(wav, sampling_rate=self.sr, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs

    def diacritize(self, text, audio_path, constrained=True, method="wfst"):
        # preprocess text
        text = preprocess_text(text)
        # Use the same form here and in the pattern so that UNK replacement can
        # align to these characters (including word delimiter tokens).
        text_for_pattern = text.replace(" ", self.word_delimiter_token)

        # read audio and get audio array
        audio, sr = librosa.load(audio_path)
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            sr = self.sr

        # convert to mono
        if audio.ndim == 2:
            audio = audio[:, 0]

        audio = np.array(audio, dtype=np.float32)
        audio_duration = len(audio) / sr
        start_time = time.time()

        # get logits
        log_probs = self.get_logits(audio)
        pattern = form_wildcard_pattern(text_for_pattern)

        if method == "wfst":
            decoded = self.decode_wfst(log_probs, pattern, constrained=constrained)
        elif method == "ctc":
            decoded = self.decode_ctc(log_probs)
        elif method == "ctc_greedy":
            decoded = self.decode_ctc_greedy(log_probs)
        else:
            raise ValueError(f"Invalid method: {method}")

        decoded_text = "".join(self.id2token[i] for i in decoded)

        # skip <no_diac> and <unk_diac> tokens
        decoded_text = decoded_text.replace(NO_DIAC_TOKEN, "")
        decoded_text = decoded_text.replace(UNK_DIAC_TOKEN, "")

        # Replace base '<unk>' tokens with the corresponding original characters from the undiacritized text.
        decoded_text = self._replace_unk_with_source(decoded_text, text_for_pattern)

        # replace the word delimiter token with space
        if self.word_delimiter_token:
            decoded_text = decoded_text.replace(self.word_delimiter_token, " ")

        decoded_text = decoded_text.strip()

        end_time = time.time()
        rtf = (end_time - start_time) / audio_duration

        return decoded_text, rtf
