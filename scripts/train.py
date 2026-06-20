#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fine-tune a Wav2Vec2 CTC model for Arabic diacritization."""

import functools
import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import evaluate
import torch
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoProcessor,
    Wav2Vec2CTCTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from diactc.config import UNK_DIAC_TOKEN, NO_DIAC_TOKEN


check_min_version("4.57.0.dev0")

require_version(
    "datasets>=1.18.0",
    "To fix: pip install -r requirements.txt",
)


logger = logging.getLogger(__name__)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={"help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_feature_encoder: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the feature encoder layers of the model."},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout ratio for the attention probabilities."},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout ratio for activations inside the fully connected layer."},
    )
    feat_proj_dropout: float = field(default=0.0, metadata={"help": "The dropout ratio for the projected features."})
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer."},
    )
    mask_time_prob: float = field(
        default=0.05,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen as the start of the vector "
                "span to be masked."
            )
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    mask_feature_prob: float = field(
        default=0.0,
        metadata={"help": "Probability of each feature vector along the feature axis to be chosen as the start of the span to be masked."},
    )
    mask_feature_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the feature axis."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    ctc_loss_reduction: str | None = field(
        default="mean",
        metadata={"help": "The way the ctc loss should be reduced. Should be one of 'mean' or 'sum'."},
    )
    ctc_zero_infinity: bool | None = field(
        default=False,
        metadata={"help": "Whether to zero infinite CTC losses and the associated gradients."},
    )
    add_adapter: bool | None = field(
        default=False,
        metadata={"help": "Whether a convolutional attention network should be stacked on top of the encoder."},
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    train_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "Path to the evaluation data."},
    )
    dataset_config_name: str = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use."},
    )
    train_split_name: str = field(
        default="train+validation",
        metadata={"help": "The name of the training data set split to use."},
    )
    eval_split_name: str = field(
        default="test",
        metadata={"help": "The name of the evaluation data set split to use."},
    )
    audio_column_name: str = field(
        default="audio_filepath",
        metadata={"help": "The name of the dataset column containing the audio data."},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "The name of the dataset column containing the text data."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: int | None = field(
        default=None,
        metadata={"help": "Truncate the number of training examples to this value if set."},
    )
    max_eval_samples: int | None = field(
        default=None,
        metadata={"help": "Truncate the number of validation examples to this value if set."},
    )
    chars_to_ignore: list[str] | None = list_field(
        default=None,
        metadata={"help": "A list of characters to remove from the transcripts."},
    )
    eval_metrics: list[str] = list_field(
        default=["wer", "cer"],
        metadata={"help": "A list of metrics the model should be evaluated on. E.g. `'wer cer'`"},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={"help": "Filter audio files that are longer than `max_duration_in_seconds` seconds."},
    )
    min_duration_in_seconds: float = field(
        default=0.0,
        metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds."},
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={"help": "Whether to only do data preprocessing and skip training."},
    )
    token: str = field(
        default=None,
        metadata={"help": "The token to use as HTTP bearer authorization for remote files."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust the execution of code from datasets/models defined on the Hub."},
    )
    unk_token: str = field(
        default="<unk>",
        metadata={"help": "The unk token for the tokenizer"},
    )
    pad_token: str = field(
        default="<pad>",
        metadata={"help": "The padding token for the tokenizer"},
    )
    word_delimiter_token: str = field(
        default="|",
        metadata={"help": "The word delimiter token for the tokenizer"},
    )
    phoneme_language: str | None = field(
        default=None,
        metadata={"help": "The target language that should be passed to the tokenizer for tokenization."},
    )


@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that will dynamically pad the inputs received."""

    processor: AutoProcessor
    padding: bool | str = "longest"
    pad_to_multiple_of: int | None = None
    pad_to_multiple_of_labels: int | None = None
    feature_extractor_input_name: str | None = "input_values"

    def __call__(self, features: list[dict[str, list[int] | torch.Tensor]]) -> dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [
            {self.feature_extractor_input_name: feature[self.feature_extractor_input_name]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        labels = labels.masked_fill(labels_batch.input_ids.eq(self.processor.tokenizer.vocab[UNK_DIAC_TOKEN]), -100)

        batch["labels"] = labels
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch


def extract_characters_from_data(datasets: DatasetDict):
    """Extract all unique characters from the dataset."""
    def extract_all_chars(batch):
        all_text = " ".join(batch["target_text"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = datasets.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=datasets["train"].column_names,
    )

    vocab_set = functools.reduce(
        lambda vocab_1, vocab_2: set(vocab_1["vocab"][0]) | set(vocab_2["vocab"][0]),
        vocabs.values(),
    )

    return vocab_set


def create_vocabulary_from_data(
    datasets: DatasetDict,
    word_delimiter_token: str | None = None,
    unk_token: str | None = None,
    pad_token: str | None = None,
):
    vocab_set = extract_characters_from_data(datasets)

    vocab_dict = {}
    idx = 0

    if pad_token is not None:
        vocab_dict[pad_token] = idx
        idx += 1

    if unk_token is not None:
        vocab_dict[unk_token] = idx
        idx += 1

    if word_delimiter_token is not None:
        vocab_set.discard(" ")
        vocab_dict[word_delimiter_token] = idx
        idx += 1

    for char in sorted(vocab_set):
        vocab_dict[char] = idx
        idx += 1

    return vocab_dict


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_process_index) else logging.WARN)

    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    if is_main_process(training_args.local_process_index):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    # 1. Load the dataset
    raw_datasets = DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_dataset(
            "json",
            data_files=data_args.train_data_path,
        )["train"]

        if data_args.audio_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--audio_column_name '{data_args.audio_column_name}' not found in dataset."
                f" One of {', '.join(raw_datasets['train'].column_names)}."
            )

        if data_args.text_column_name not in raw_datasets["train"].column_names:
            raise ValueError(
                f"--text_column_name {data_args.text_column_name} not found in dataset. "
                f"One of {', '.join(raw_datasets['train'].column_names)}."
            )

        if data_args.max_train_samples is not None:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    if training_args.do_eval:
        raw_datasets["eval"] = load_dataset(
            "json",
            data_files=data_args.eval_data_path,
        )["train"]

        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

    # preprocess the text: replace spaces with the word delimiter token
    def preprocess_text(example):
        example[data_args.text_column_name] = example[data_args.text_column_name].replace(" ", "|")
        return example

    raw_datasets["train"] = raw_datasets["train"].map(preprocess_text)
    raw_datasets["eval"] = raw_datasets["eval"].map(preprocess_text)

    # 2. Load config
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    # 3. Tokenizer (from the prebuilt vocab.json) and feature extractor
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=model_args.tokenizer_name_or_path,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
    )

    config.update(
        {
            "feat_proj_dropout": model_args.feat_proj_dropout,
            "attention_dropout": model_args.attention_dropout,
            "hidden_dropout": model_args.hidden_dropout,
            "final_dropout": model_args.final_dropout,
            "mask_time_prob": model_args.mask_time_prob,
            "mask_time_length": model_args.mask_time_length,
            "mask_feature_prob": model_args.mask_feature_prob,
            "mask_feature_length": model_args.mask_feature_length,
            "gradient_checkpointing": training_args.gradient_checkpointing,
            "layerdrop": model_args.layerdrop,
            "ctc_loss_reduction": model_args.ctc_loss_reduction,
            "ctc_zero_infinity": model_args.ctc_zero_infinity,
            "pad_token_id": tokenizer.pad_token_id,
            "vocab_size": len(tokenizer),
            "activation_dropout": model_args.activation_dropout,
            "add_adapter": model_args.add_adapter,
        }
    )

    # 4. Create model
    model = AutoModelForCTC.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        config=config,
        token=data_args.token,
        trust_remote_code=data_args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    # 5. Resample audio to the feature extractor's sampling rate
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name,
        datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate),
    )

    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    text_column_name = data_args.text_column_name
    num_workers = data_args.preprocessing_num_workers
    feature_extractor_input_name = feature_extractor.model_input_names[0]

    phoneme_language = data_args.phoneme_language

    def prepare_dataset(batch):
        sample = batch[audio_column_name]

        inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
        batch[feature_extractor_input_name] = getattr(inputs, feature_extractor_input_name)[0]
        batch["input_length"] = len(sample["array"].squeeze())

        additional_kwargs = {}
        if phoneme_language is not None:
            additional_kwargs["phonemizer_lang"] = phoneme_language

        batch["labels"] = tokenizer(batch[text_column_name], **additional_kwargs).input_ids
        if audio_column_name in batch:
            del batch[audio_column_name]

        return batch

    with training_args.main_process_first(desc="dataset map preprocessing"):
        vectorized_datasets = raw_datasets.map(
            prepare_dataset,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=num_workers,
            desc="preprocess datasets",
        )

        def is_audio_in_length_range(length):
            return length > min_input_length and length < max_input_length

        vectorized_datasets = vectorized_datasets.filter(
            is_audio_in_length_range,
            num_proc=num_workers,
            input_columns=["input_length"],
        )

    # 6. Metrics
    eval_metrics = {metric: evaluate.load(metric, cache_dir=model_args.cache_dir) for metric in data_args.eval_metrics}

    if data_args.preprocessing_only:
        logger.info(f"Data preprocessing finished. Files cached at {vectorized_datasets.cache_files}")
        return

    def preprocess_logits_for_metrics(logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels

    def compute_metrics(pred):
        pred_ids = pred.predictions[0]
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)

        # drop the unk and no_diac tokens for metric computation
        pred_str = [s.replace(UNK_DIAC_TOKEN, "").replace(NO_DIAC_TOKEN, "") for s in pred_str]
        label_str = [s.replace(UNK_DIAC_TOKEN, "").replace(NO_DIAC_TOKEN, "") for s in label_str]

        return {k: v.compute(predictions=pred_str, references=label_str) for k, v in eval_metrics.items()}

    # 7. Save processor components so a single processor can be reloaded
    with training_args.main_process_first():
        if is_main_process(training_args.local_process_index):
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    data_collator = DataCollatorCTCWithPadding(
        processor=processor, feature_extractor_input_name=feature_extractor_input_name
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=vectorized_datasets["train"] if training_args.do_train else None,
        eval_dataset=vectorized_datasets["eval"] if training_args.do_eval else None,
        processing_class=processor,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # 8. Training
    if training_args.do_train:
        if os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(vectorized_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(vectorized_datasets["train"]))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 9. Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(vectorized_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(vectorized_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    config_name = data_args.dataset_config_name if data_args.dataset_config_name is not None else "na"
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "automatic-speech-recognition",
        "dataset_args": (
            f"Config: {config_name}, Training split: {data_args.train_split_name}, Eval split:"
            f" {data_args.eval_split_name}"
        ),
    }

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()
