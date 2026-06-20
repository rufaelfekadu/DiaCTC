<div align="center">

# Constrained CTC Decoding for Efficient Diacritic Restoration

**DiaCTC** — constrained CTC / WFST decoding for Arabic diacritization with Wav2Vec2

[![Paper (arXiv)](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/TBD)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github)](https://github.com/rufaelfekadu/DiaCTC)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Rufael Marew · Amr Keleg · Hanan Aldarmaki · MBZUAI*

</div>

---

In this work, we address diacritic restoration for Arabic speech transcripts. Most speech data are undiacritized, limiting the ability of modeling fine-grained phonological distinctions. The speech modality has recently been explored as a way to complement text-based diacritic restoration efforts. We propose an efficient non-autoregressive approach for speech-to-text diacritization based on Connectionist Temporal Classification (CTC). Our method incorporates hard constraints during decoding by constructing a character-level diacritization lattice from an undiacritized transcript and restricting hypotheses to valid diacritized realizations. We evaluate on Classical Arabic and Modern Standard Arabic test sets (namely, ArVoice and ClArTTS) against a more computationally-complex multi-modal diacritic restoration baseline, and show statistically significant reductions in diacritic error rates in both, demonstrating that the proposed approach offers both performance and efficiency gains

> **Accepted at Interspeech 2026**

## Results

Diacritic restoration on **ClArTTS** (Classical Arabic) and **ArVoice** (MSA).
WER and DER (%) are from Table 3 in the paper; DER includes *no diacritic* and
word-ending diacritics. Base encoder:
[jonatasgrosman/wav2vec2-large-xlsr-53-arabic](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic).

| Training data | Test set | WER ↓ | DER ↓ | Model |
|:--------------|:---------|------:|------:|:------|
| ClArTTS | ClArTTS | 11.21 | 3.53 | [HF *(coming soon)*](#) |
| ClArTTS | ArVoice | 39.89 | 12.04 | [HF *(coming soon)*](#) |
| ArVoice | ClArTTS | 34.94 | 11.86 | [HF *(coming soon)*](#) |
| ArVoice | ArVoice | 27.87 | 7.73 | [HF *(coming soon)*](#) |
| ClArTTS + ArVoice | ClArTTS | 13.05 | 3.80 | [HF *(coming soon)*](#) |
| ClArTTS + ArVoice | ArVoice | 30.36 | 8.69 | [HF *(coming soon)*](#) |

For comparison, the **Text + ASR** baseline on combined training reaches 29.63 /
9.05 WER/DER on ClArTTS and 34.47 / 9.93 on ArVoice; DiaCTC improves DER on
both test sets with a single-stream CTC decoder (see paper for full baselines and
bootstrap significance).

## Installation

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install the package:

```bash
pip install -e .
```

3. Install [k2](https://k2-fsa.github.io/k2/installation/index.html) (required for
   WFST/CTC lattice decoding). It must match your CUDA and PyTorch versions:

```bash
# Example for CUDA 12.8 / PyTorch 2.9.1
pip install k2==1.24.4.dev20251118+cuda12.8.torch2.9.1 -f https://k2-fsa.github.io/k2/cuda.html
```

## Package layout

```
src/diactc/
  config.py        # special tokens
  constants.py     # Arabic letters / diacritic classes for evaluation
  models/          # DiacritizationModel (base) + Wav2Vec2DiacritizationModel
  utils/           # text processing + alignment helpers
  metrics/         # DER, WER, SER
scripts/
  prep_data.py     # build / manifest / split data preparation
  train.py         # Wav2Vec2 CTC fine-tuning
  inference.py     # WFST / CTC decoding over manifests
  evaluate.py      # DER / WER / SER report
```

## Pipeline

### 1. Prepare data

Manifests are JSONL with at least `audio_filepath` and `text` fields.

```bash
# (optional) build raw per-dataset manifests
python scripts/prep_data.py build clartts arvoice

# clean text, build vocab.json/charset, and combine into a training manifest
python scripts/prep_data.py manifest \
    -i data/clartts/raw/train/clartts_train_metadata.json \
       data/arvoice/raw/train/arvoice_train_metadata.json \
    --split combined --add_special -o outputs/finetune

# split into train/validation
python scripts/prep_data.py split \
    -i outputs/finetune/combined_manifest.json \
    -t outputs/finetune/train_manifest.json \
    -v outputs/finetune/val_manifest.json --val_ratio 0.1
```

### 2. Train

```bash
python scripts/train.py \
    --train_data_path outputs/finetune/train_manifest.json \
    --eval_data_path outputs/finetune/val_manifest.json \
    --model_name_or_path jonatasgrosman/wav2vec2-large-xlsr-53-arabic \
    --tokenizer_name_or_path outputs/finetune/vocab.json \
    --output_dir outputs/finetune/experiments/run-v1 \
    --text_column_name text_with_special \
    --num_train_epochs 25 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-4 \
    --warmup_steps 1500 \
    --eval_strategy steps --eval_steps 200 --save_steps 200 \
    --save_total_limit 3 --freeze_feature_encoder --gradient_checkpointing \
    --fp16 --group_by_length --report_to tensorboard \
    --load_best_model_at_end --metric_for_best_model eval_wer --greater_is_better False \
    --do_train --do_eval
```

### 3. Inference

```bash
python scripts/inference.py \
    --model_path outputs/finetune/experiments/run-v1 \
    --manifest_paths data/clartts/raw/test/clartts_test_metadata.json \
    --method wfst \
    --use_blank_token --use_no_diac_token --use_unk_diac_token \
    --output_path outputs/decode \
    --device cuda
```

`--method` is one of `wfst`, `ctc`, or `ctc_greedy`. Add `--unconstrained` to use
unconstrained WFST decoding. Outputs (`gt.txt`, `pred.txt`, `metrics.json`) are
written under `outputs/decode/wav2vec/<method>/[constrained|unconstrained]/<dataset>/`.

### 4. Evaluate

```bash
python scripts/evaluate.py \
    -ref outputs/decode/wav2vec/wfst/constrained/clartts_test_metadata/gt.txt \
    -hyp outputs/decode/wav2vec/wfst/constrained/clartts_test_metadata/pred.txt \
    --log_file eval.log
```

DER, WER, and SER are reported with/without case ending and including/excluding
no-diacritic positions (`Fadel` style by default; `Zitouni` is also supported).
The diacritic coverage rate (fraction of diacritizable Arabic characters that
carry a diacritic) is reported for both the reference and the hypothesis.

By default the metrics are *relaxed*: base characters are aligned with dynamic
programming, so reference and hypothesis lines may differ in length. Pass
`--strict` (CLI) or `strict=True` (API) for the *strict* positional metrics,
which require the reference and hypothesis to share the same base characters and
raise an error otherwise.

You can also call the metrics programmatically:

```python
from diactc.metrics import (
    calculate_der, calculate_der_strict, diac_coverage_rate, evaluate,
)

calculate_der(reference_lines, hypothesis_lines, case_ending=True, no_diacritic=True)
calculate_der_strict(reference_lines, hypothesis_lines)   # positional, aligned inputs
diac_coverage_rate("بِسْمِ")                  # -> 1.0
evaluate("gt.txt", "pred.txt")               # relaxed DER/WER/SER variants + coverage
evaluate("gt.txt", "pred.txt", strict=True)  # strict variants
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use DiaCTC in your research, please cite:

```bibtex
@article{marew2026constrained,
  title   = {Constrained CTC Decoding for Efficient Diacritic Restoration},
  author  = {Marew, Rufael and Keleg, Amr and Aldarmaki, Hanan},
  journal = {arXiv preprint arXiv:TBD},
  year    = {2026}
}
```
