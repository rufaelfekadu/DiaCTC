"""Run Wav2Vec2 diacritization inference over one or more manifests.

For each manifest, writes gt.txt, pred.txt, and metrics.json under
{output_path}/wav2vec/{method}[/constrained|unconstrained]/{dataset_name}/.
"""

import argparse
import os
import json
import logging

import pandas as pd
from tqdm import tqdm
from pyarabic import araby
from jiwer import wer as wer_fn, cer as cer_fn

from diactc.models import Wav2Vec2DiacritizationModel
from diactc.utils.text import preprocess_text
from diactc.metrics import calculate_der, diac_coverage_rate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text):
    return preprocess_text(text)


def load_manifest(manifest_path):
    base_name = os.path.basename(manifest_path)
    _, ext = os.path.splitext(base_name)

    if ext == ".tsv":
        df = pd.read_csv(manifest_path, sep="\t")
    elif ext == ".json":
        with open(manifest_path, "r") as infile:
            content = infile.read().strip()
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    data = [data]
            except Exception:
                # fallback to line-delimited JSON
                data = []
                with open(manifest_path, "r") as inf:
                    for line in inf:
                        line = line.strip()
                        if line:
                            data.append(json.loads(line))
        df = pd.DataFrame(data)
    else:
        raise ValueError("Unsupported input file format. Only .tsv and .json are supported.")

    return df


def pred_manifest(args, model, manifest_path):
    df = load_manifest(manifest_path)
    df["text"] = df["text"].apply(clean_text)

    dataset_name = os.path.basename(manifest_path).split(".")[0]
    output_path = os.path.join(args.output_path, dataset_name)
    os.makedirs(output_path, exist_ok=True)

    gt_path = os.path.join(output_path, "gt.txt")
    pred_path = os.path.join(output_path, "pred.txt")
    metrics_path = os.path.join(output_path, "metrics.json")

    predictions = []
    with open(pred_path, "w") as pred_f, open(gt_path, "w") as gt_f:
        for _, entry in tqdm(df.iterrows(), total=df.shape[0]):
            audio_path = entry["audio_filepath"]
            text = entry["text"]
            text_no_diac = araby.strip_diacritics(text)

            diacritized_text, rtf = model.diacritize(text_no_diac, audio_path, not args.unconstrained, args.method)

            if len(diacritized_text) != 0:
                pred_f.write(f"{diacritized_text}\n")
                gt_f.write(f"{text}\n")
                predictions.append(
                    {
                        "hypothesis": diacritized_text,
                        "reference": text,
                        "wer": wer_fn(hypothesis=diacritized_text, reference=text),
                        "wer_no_diac": wer_fn(hypothesis=araby.strip_diacritics(diacritized_text), reference=text_no_diac),
                        "cer": cer_fn(hypothesis=diacritized_text, reference=text),
                        "cer_no_diac": cer_fn(hypothesis=araby.strip_diacritics(diacritized_text), reference=text_no_diac),
                        "rtf": rtf,
                        "diac_coverage_pred": diac_coverage_rate(diacritized_text),
                        "diac_coverage_ref": diac_coverage_rate(text),
                    }
                )
            else:
                logger.warning(f"Empty diacritized text for {audio_path}: {text}")

    metadata = {
        "model": args.model_path,
        "model_type": "wav2vec",
        "decoding_strategy": args.method,
        "dataset": dataset_name,
        "constrained": not args.unconstrained,
        "device": args.device,
    }

    hypothesis_texts = [m["hypothesis"] for m in predictions]
    reference_texts = [m["reference"] for m in predictions]

    hypothesis_texts_no_diac = [araby.strip_diacritics(h) for h in hypothesis_texts]
    reference_texts_no_diac = [araby.strip_diacritics(r) for r in reference_texts]

    average_metrics = {
        "overall_wer": wer_fn(hypothesis=hypothesis_texts, reference=reference_texts),
        "overall_wer_no_diac": wer_fn(hypothesis=hypothesis_texts_no_diac, reference=reference_texts_no_diac),
        "overall_cer": cer_fn(hypothesis=hypothesis_texts, reference=reference_texts),
        "overall_cer_no_diac": cer_fn(hypothesis=hypothesis_texts_no_diac, reference=reference_texts_no_diac),
        "avg_rtf": sum([m["rtf"] for m in predictions]) / len(df),
        "avg_diac_coverage_pred": sum([m["diac_coverage_pred"] for m in predictions]) / len(df),
        "avg_diac_coverage_ref": sum([m["diac_coverage_ref"] for m in predictions]) / len(df),
        "overall_diac_coverage_pred": diac_coverage_rate("".join(hypothesis_texts)),
        "overall_diac_coverage_ref": diac_coverage_rate("".join(reference_texts)),
        # DER variants
        "overall_der_with_case_with_no_diac": calculate_der(reference_texts, hypothesis_texts, case_ending=True, no_diacritic=True),
        "overall_der_without_case_with_no_diac": calculate_der(reference_texts, hypothesis_texts, case_ending=False, no_diacritic=True),
        "overall_der_with_case_without_no_diac": calculate_der(reference_texts, hypothesis_texts, case_ending=True, no_diacritic=False),
        "overall_der_without_case_without_no_diac": calculate_der(reference_texts, hypothesis_texts, case_ending=False, no_diacritic=False),
    }

    output_payload = {
        "metadata": metadata,
        "average_metrics": average_metrics,
        "predictions": predictions,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=4, ensure_ascii=False)


def main(args):
    model = Wav2Vec2DiacritizationModel(
        args.model_path,
        device=args.device,
        use_blank_token=args.use_blank_token,
        use_no_diac_token=args.use_no_diac_token,
        use_unk_diac_token=args.use_unk_diac_token,
    )
    model.to(args.device)
    for manifest in args.manifest_paths:
        logger.info(f"Processing {manifest}")
        pred_manifest(args, model, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model id or path to a fine-tuned Wav2Vec2 checkpoint")
    parser.add_argument("--manifest_paths", nargs="+", type=str, required=True, help="One or more manifest files (.json or .tsv)")
    parser.add_argument("--unconstrained", action="store_true", help="Use unconstrained decoding if set.")
    parser.add_argument("--method", type=str, default="wfst", choices=["wfst", "ctc", "ctc_greedy"])
    parser.add_argument("--output_path", type=str, default="outputs/decode/")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_blank_token", action="store_true", help="Use blank token if set.")
    parser.add_argument("--use_no_diac_token", action="store_true", help="Use no diac token if set.")
    parser.add_argument("--use_unk_diac_token", action="store_true", help="Use unk diac token if set.")
    args = parser.parse_args()

    constrained = "constrained" if not args.unconstrained else "unconstrained"
    args.output_path = os.path.join(args.output_path, "wav2vec", args.method)
    args.output_path = os.path.join(args.output_path, constrained) if args.method == "wfst" else args.output_path

    os.makedirs(args.output_path, exist_ok=True)

    logger.info(f"Output path: {args.output_path}")
    file_handler = logging.FileHandler(os.path.join(args.output_path, "inference.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    main(args)
