"""Data preparation for DiaCTC (Wav2Vec2).

Three stages, selectable with the first positional argument:

  build     Download/extract raw datasets into per-dataset JSONL manifests
            (audio_filepath, text, duration, source, ...).
  manifest  Clean text, build vocab.json / charset, and combine one or more
            raw manifests into a single split manifest.
  split     Split a combined manifest into train/validation manifests.

Examples:
  python scripts/prep_data.py build clartts arvoice
  python scripts/prep_data.py manifest -i data/clartts/raw/train/clartts_train_metadata.json \
      --split combined --add_special -o outputs/finetune
  python scripts/prep_data.py split -i outputs/finetune/combined_manifest.json \
      -t outputs/finetune/train_manifest.json -v outputs/finetune/val_manifest.json
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from functools import partial

import librosa
import numpy as np
import soundfile
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

from diactc.config import SPECIAL_TOKENS, NO_DIAC_TOKEN, UNK_DIAC_TOKEN
from diactc.utils.text import (
    ARABIC_CHARACTERS_TO_BE_DIACRITIZED,
    ARABIC_NUMERALS,
    VALID_DIACRITICS_COMBINATIONS,
    get_groups_of_characters_with_diacritics,
    preprocess_text,
)

TARGET_SR = 16000
NUM_PROC = 16


# --------------------------------------------------------------------------- #
# Stage 1: raw dataset builders
# --------------------------------------------------------------------------- #

DATASET_CONFIGS = {
    "clartts": {
        "name": "clartts",
        "hf_dataset": "MBZUAI/clartts",
        "output_dir": os.path.abspath("data/clartts/raw"),
        "splits": ["train", "test"],
        "audio_col": "audio",
        "text_col": "text",
        "filename_col": "file",
        "sampling_rate": 16000,
    },
    "fleurs": {
        "name": "fleurs",
        "hf_dataset": "google/fleurs",
        "data_name": "ar_eg",
        "output_dir": os.path.abspath("data/fleurs/raw"),
        "splits": ["train", "validation", "test"],
        "audio_col": "audio",
        "text_col": "transcription",
        "filename_col": "audio_path",
        "sampling_rate": 16000,
    },
    "nadi": {
        "name": "nadi",
        "hf_dataset": "MBZUAI/NADI-2025-Sub-task-3-test",
        "output_dir": os.path.abspath("data/nadi/raw-test"),
        "splits": ["test"],
        "audio_col": "audio",
        "text_col": "transcription",
        "filename_col": None,
        "sampling_rate": 16000,
    },
    "arvoice": {
        "name": "arvoice",
        "hf_dataset": "csv",
        "base_dir": "/l/ArVoice/v1",
        "csv_files": [
            "/l/ArVoice/v1/part-1/metadata_{}.csv",
            "/l/ArVoice/v1/part-2/metadata_{}.csv",
        ],
        "output_dir": os.path.abspath("data/arvoice/raw"),
        "splits": ["train", "test"],
        "audio_col": "file_name",
        "text_col": "transcription",
        "filename_col": "file_name",
        "sampling_rate": 16000,
    },
}


def process_item_hf(item, split, config, idx=None):
    def parse_file_name():
        if config["filename_col"] in item:
            return item[config["filename_col"]]
        return str(idx) + ".wav"  # default to id if no filename column is found

    item["id"] = idx

    audio = item[config["audio_col"]]
    if isinstance(audio, dict) and "array" in audio:
        audio_array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]
    elif isinstance(audio, str):
        audio_array, sr = librosa.load(audio)
    elif isinstance(audio, (np.ndarray, list)):
        audio_array = np.array(audio, dtype=np.float32)
        sr = item.get("sampling_rate", config["sampling_rate"])
    else:
        audio_array = np.array(audio["array"], dtype=np.float32)
        sr = audio["sampling_rate"]

    if sr != TARGET_SR:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    audio_path = parse_file_name()
    audio_path = os.path.join(config["output_dir"], split, "clips", audio_path)
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    item["audio_filepath"] = audio_path
    item["sampling_rate"] = TARGET_SR
    item["duration"] = librosa.get_duration(y=audio_array, sr=TARGET_SR)

    if not os.path.isfile(audio_path):
        soundfile.write(audio_path, audio_array, TARGET_SR, format="wav")
    if "audio" in item:
        del item["audio"]

    return item


def process_hf_dataset(config):
    dataset = None
    for split in config["splits"]:
        dataset = load_dataset(
            config["hf_dataset"],
            name=config.get("data_name"),
            split=split,
            num_proc=NUM_PROC,
        )

        if config.get("max_samples_per_split") is not None:
            dataset = dataset.shuffle().select(range(config["max_samples_per_split"]))

        dataset = dataset.map(
            lambda x, idx: process_item_hf(x, split=split, config=config, idx=idx),
            with_indices=True,
            num_proc=NUM_PROC,
            desc=f"processing {split} dataset",
        )
        dataset = dataset.add_column("source", [config["name"]] * len(dataset))

        if config["text_col"] != "text":
            dataset = dataset.rename_column(config["text_col"], "text")

        to_keep = ["audio_filepath", "text", "sampling_rate", "duration", "id", "source"]
        dataset = dataset.remove_columns(set(dataset.column_names) - set(to_keep))

        split_metadata_path = os.path.join(config["output_dir"], split, f"{config['name']}_{split}_metadata.json")
        os.makedirs(os.path.dirname(split_metadata_path), exist_ok=True)
        with open(split_metadata_path, "w") as f:
            for item in tqdm(dataset):
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

    return dataset


def process_item_arvoice(item, split, config):
    audio, sr = librosa.load(item["file_name"])
    duration = librosa.get_duration(y=audio, sr=sr)

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    rel_path = os.path.relpath(item["file_name"], config["base_dir"])
    audio_path = os.path.join(config["output_dir"], split, rel_path)
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    item["duration"] = duration
    item["sampling_rate"] = sr
    item["audio_filepath"] = audio_path

    if not os.path.isfile(audio_path):
        soundfile.write(audio_path, audio, sr, format="wav")

    return item


def process_arvoice_dataset(config):
    for split in config["splits"]:
        total_arvoice = []
        preprocess_fn = partial(process_item_arvoice, split=split, config=config)
        for csv_file in config["csv_files"]:
            arvoice = load_dataset("csv", data_files=csv_file.format(split), cache_dir=config["output_dir"])["train"]
            arvoice = arvoice.map(lambda x: {"source": x["file_name"].split("/")[-3]}, num_proc=NUM_PROC)
            if split == "test":
                arvoice = arvoice.filter(lambda x: x["source"] != "khaleej")
            arvoice = arvoice.map(preprocess_fn, num_proc=NUM_PROC, desc=f"Processing {split} ArVoice dataset")
            total_arvoice.append(arvoice)
        total_arvoice = concatenate_datasets(total_arvoice)
        total_arvoice = total_arvoice.rename_column("transcription", "text")

        split_metadata_path = os.path.join(config["output_dir"], split, f"arvoice_{split}_metadata.json")
        os.makedirs(os.path.dirname(split_metadata_path), exist_ok=True)
        with open(split_metadata_path, "w") as f:
            for item in tqdm(total_arvoice, desc=f"Dumping {split} ArVoice dataset"):
                if "audio" in item:
                    del item["audio"]
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")


def run_build(args):
    for dataset in args.datasets:
        config = DATASET_CONFIGS[dataset]
        if dataset == "arvoice":
            process_arvoice_dataset(config)
        else:
            process_hf_dataset(config)


# --------------------------------------------------------------------------- #
# Stage 2: manifest cleaning + vocabulary building
# --------------------------------------------------------------------------- #

chars_to_ignore_regex = r'[,\?\.\!\-\;\:\"\“%\‘\”�…{}\【\】・。『』、ー〜\[\]<●•\\&﴾﴿>/*👍😍()،؛؟ۗ’+#»«٪=‹›]'
chars_to_ignore_additional = r'[\u0640\u200b\u0009\ufeff\u202b]'


def add_special_tokens(text, partial_diac=True):
    """Add special tokens to text based on diacritization status.

    partial_diac=True: add <unk_diac> after every Arabic char without diacritics.
    partial_diac=False: add <no_diac> after every Arabic char without diacritics.
    """
    characters_with_diacritics = get_groups_of_characters_with_diacritics(text)
    result_parts = []

    for character, succeeding_diacritics in characters_with_diacritics:
        result_parts.append(character)
        if succeeding_diacritics:
            result_parts.append(succeeding_diacritics)
        elif character in ARABIC_CHARACTERS_TO_BE_DIACRITIZED:
            result_parts.append(UNK_DIAC_TOKEN if partial_diac else NO_DIAC_TOKEN)

    return "".join(result_parts)


def build_charset(manifest_paths, extra_chars=None, output_path=None):
    char_counts = Counter()
    for mf in manifest_paths:
        with open(mf, "r", encoding="utf-8") as f:
            for line in f:
                char_counts.update(list(json.loads(line)["text"]))

    if extra_chars:
        for char in extra_chars:
            char_counts[char] += 1

    char_counts.pop("\n", None)
    char_counts.pop("\t", None)
    char_counts.pop(" ", None)

    chars = sorted(set(list(char_counts.keys()) + VALID_DIACRITICS_COMBINATIONS))

    vocab = {char: idx for idx, char in enumerate(SPECIAL_TOKENS + chars)}

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)

    return chars, char_counts


def normalize_arabic_numerals(text):
    for k, v in ARABIC_NUMERALS.items():
        text = text.replace(k, v)
    return text


def clean_arabic(data):
    data["text"] = preprocess_text(data["text"])
    return data


def clean_english(data):
    data["text"] = re.sub(chars_to_ignore_regex, "", data["text"])
    data["text"] = re.sub(chars_to_ignore_additional, "", data["text"])
    data["text"] = data["text"].replace("—", " ").replace("–", " ")
    data["text"] = data["text"].lower()
    data["text"] = data["text"].replace("\n", " ")
    return data


def keep_cols(data):
    return {
        "audio_filepath": data["audio_filepath"],
        "source": data["source"],
        "text": data["text"],
        "duration": data["duration"],
    }


def preprocess_manifest(manifest_paths, add_special=False):
    partial_sources = {"masc", "khaleej", "arzen", "mixat"}
    processed_manifest_paths = []
    for mf in manifest_paths:
        processed_mf = mf.replace(".json", "_preprocessed.json")
        with open(mf, "r", encoding="utf-8") as f, open(processed_mf, "w", encoding="utf-8") as f_out:
            for line in f:
                item = json.loads(line)
                item = clean_arabic(item)
                item = clean_english(item)
                partial_diac = item["source"] in partial_sources
                item = keep_cols(item)
                if add_special:
                    item["text_with_special"] = add_special_tokens(item["text"], partial_diac=partial_diac)
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        processed_manifest_paths.append(processed_mf)
    return processed_manifest_paths


def run_manifest(args):
    os.makedirs(args.output_path, exist_ok=True)
    processed_manifest_paths = preprocess_manifest(args.input_paths, args.add_special)

    vocab_output = os.path.join(args.output_path, "vocab.json") if args.split != "test" else None
    chars, char_counts = build_charset(processed_manifest_paths, args.extra_chars, vocab_output)
    print(f"Char set ({len(chars)}): {chars}")

    if args.split != "test":
        with open(os.path.join(args.output_path, "charset.txt"), "w", encoding="utf-8") as f:
            for c in chars:
                f.write(c + "\n")

        char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
        with open(os.path.join(args.output_path, "charset_counts.txt"), "w", encoding="utf-8") as f:
            for c, count in char_counts:
                f.write(f"{c}\t{count}\n")

    combined_manifest_path = os.path.join(args.output_path, f"{args.split}_manifest.json")
    with open(combined_manifest_path, "w", encoding="utf-8") as f:
        for mf in processed_manifest_paths:
            with open(mf, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    f.write(line)
    print(f"Combined manifest saved to: {combined_manifest_path}")


# --------------------------------------------------------------------------- #
# Stage 3: train/validation split
# --------------------------------------------------------------------------- #

def run_split(args):
    items = []
    with open(args.input_manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"Total items in manifest: {len(items)}")

    if not args.no_shuffle:
        random.seed(args.seed)
        random.shuffle(items)

    val_size = int(len(items) * args.val_ratio)
    train_items = items[: len(items) - val_size]
    val_items = items[len(items) - val_size:]
    print(f"Train items: {len(train_items)}, Validation items: {len(val_items)}")

    for path, split_items in ((args.train_output, train_items), (args.val_output, val_items)):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in split_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():
    parser = argparse.ArgumentParser(description="DiaCTC data preparation")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    p_build = subparsers.add_parser("build", help="Build raw per-dataset manifests")
    p_build.add_argument("datasets", nargs="+", choices=sorted(DATASET_CONFIGS.keys()), help="Datasets to process")
    p_build.set_defaults(func=run_build)

    p_manifest = subparsers.add_parser("manifest", help="Clean text, build vocab, and combine manifests")
    p_manifest.add_argument("--input_paths", "-i", nargs="+", required=True, help="List of raw manifest file paths.")
    p_manifest.add_argument("--split", "-s", default="combined", help="Split name (use 'test' to skip vocab building).")
    p_manifest.add_argument("--extra_chars", "-e", nargs="+", help="Extra characters to add to the charset.")
    p_manifest.add_argument("--output_path", "-o", default="outputs/finetune", help="Directory to save outputs.")
    p_manifest.add_argument("--add_special", "-a", action="store_true", help="Add special tokens to the text.")
    p_manifest.set_defaults(func=run_manifest)

    p_split = subparsers.add_parser("split", help="Split a combined manifest into train/validation")
    p_split.add_argument("--input_manifest", "-i", required=True, help="Input combined manifest (JSONL).")
    p_split.add_argument("--train_output", "-t", required=True, help="Output train manifest path.")
    p_split.add_argument("--val_output", "-v", required=True, help="Output validation manifest path.")
    p_split.add_argument("--val_ratio", "-r", type=float, default=0.1, help="Validation ratio (default 0.1).")
    p_split.add_argument("--no_shuffle", action="store_true", help="Don't shuffle before splitting.")
    p_split.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_split.set_defaults(func=run_split)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.stage == "split" and not 0 < args.val_ratio < 1:
        parser.error("val_ratio must be between 0 and 1")
    args.func(args)


if __name__ == "__main__":
    main()
