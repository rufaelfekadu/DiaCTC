"""Search for the confidence threshold of constrained WFST decoding.

Each utterance is decoded once; every candidate threshold is then applied to
the decoded tokens and their confidences, so the model is not re-run per
threshold. For each manifest, and for all manifests pooled, writes the
per-threshold metrics and the selected threshold to
{output_path}/[der|precision_{target_precision}]/results.json.

Objectives:
    der         lowest diacritic error rate
    precision   lowest threshold at which the emitted diacritics reach
                --target_precision, i.e. the most coverage at that precision
"""

import argparse
import os
import json
import logging

from tqdm import tqdm
from pyarabic import araby

from diactc.models import Wav2Vec2DiacritizationModel
from diactc.utils.text import preprocess_text, form_wildcard_pattern
from diactc.metrics import calculate_der, diac_coverage_rate, diac_precision_recall

from inference import load_manifest, clean_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = [round(0.05 * i, 2) for i in range(20)] + [0.97, 0.98, 0.99, 0.995, 0.999]


def decode_manifest(args, model, manifest_path):
    df = load_manifest(manifest_path)
    df["text"] = df["text"].apply(clean_text)
    if args.max_samples is not None:
        df = df.head(args.max_samples)

    utterances = []
    for _, entry in tqdm(df.iterrows(), total=df.shape[0]):
        audio_path = entry["audio_filepath"]
        text = entry["text"]
        text_for_pattern = preprocess_text(araby.strip_diacritics(text)).replace(" ", model.word_delimiter_token)

        log_probs = model.get_logits(model.load_audio(audio_path))
        ids, confidences = model.decode_wfst_with_confidence(log_probs, form_wildcard_pattern(text_for_pattern))

        if len(ids) != 0:
            utterances.append(
                {
                    "reference": text,
                    "text_for_pattern": text_for_pattern,
                    "ids": ids,
                    "confidences": confidences,
                }
            )
        else:
            logger.warning(f"Empty decoding for {audio_path}: {text}")

    return utterances


def evaluate_threshold(model, utterances, threshold):
    reference_texts = [u["reference"] for u in utterances]
    hypothesis_texts = [
        model.decoded_to_text(model.apply_confidence_threshold(u["ids"], u["confidences"], threshold), u["text_for_pattern"])
        for u in utterances
    ]

    precision_recall = diac_precision_recall(reference_texts, hypothesis_texts)

    return {
        "threshold": threshold,
        "der": calculate_der(reference_texts, hypothesis_texts, case_ending=True, no_diacritic=True),
        "der_without_case": calculate_der(reference_texts, hypothesis_texts, case_ending=False, no_diacritic=True),
        "der_without_no_diac": calculate_der(reference_texts, hypothesis_texts, case_ending=True, no_diacritic=False),
        "precision": precision_recall["precision"],
        "recall": precision_recall["recall"],
        "coverage": diac_coverage_rate("".join(hypothesis_texts)),
        "skipped": precision_recall["skipped"],
    }


def select_threshold(table, objective, target_precision):
    if objective == "der":
        return dict(min(table, key=lambda row: (row["der"], row["threshold"])))

    # precision: the lowest threshold reaching the target keeps the most diacritics
    scored = [row for row in table if row["precision"] is not None]
    reaching = [row for row in scored if row["precision"] >= target_precision]
    if reaching:
        selected = min(reaching, key=lambda row: row["threshold"])
    else:
        selected = max(scored, key=lambda row: (row["precision"], -row["threshold"]))

    return {**selected, "target_precision": target_precision, "target_reached": bool(reaching)}


def log_table(name, table, selected):
    logger.info(f"{name}")
    logger.info(f"{'threshold':>9} {'DER':>7} {'DER-nocase':>10} {'DER-nonodiac':>12} {'precision':>9} {'recall':>7} {'coverage':>8}")
    for row in table:
        precision = f"{row['precision']:.4f}" if row["precision"] is not None else "-"
        marker = "  <- selected" if row["threshold"] == selected["threshold"] else ""
        logger.info(
            f"{row['threshold']:>9} {row['der']:>7.2f} {row['der_without_case']:>10.2f} {row['der_without_no_diac']:>12.2f} {precision:>9} {row['recall']:>7.4f} {row['coverage']:>8.4f}{marker}"
        )


def main(args):
    model = Wav2Vec2DiacritizationModel(
        args.model_path,
        device=args.device,
        use_blank_token=args.use_blank_token,
        use_no_diac_token=args.use_no_diac_token,
        use_unk_diac_token=args.use_unk_diac_token,
    )
    model.to(args.device)

    decoded = {}
    for manifest in args.manifest_paths:
        logger.info(f"Decoding {manifest}")
        dataset_name = os.path.basename(manifest).split(".")[0]
        decoded[dataset_name] = decode_manifest(args, model, manifest)
    decoded["pooled"] = [u for utterances in decoded.values() for u in utterances]

    results = {}
    for name, utterances in decoded.items():
        table = [evaluate_threshold(model, utterances, threshold) for threshold in args.thresholds]
        selected = select_threshold(table, args.objective, args.target_precision)
        log_table(f"{name} ({len(utterances)} utterances)", table, selected)
        if args.objective == "precision" and not selected["target_reached"]:
            logger.warning(f"{name}: no threshold reaches precision {args.target_precision}; selected the most precise one")
        results[name] = {"num_utterances": len(utterances), "selected": selected, "table": table}

    output_payload = {
        "metadata": {
            "model": args.model_path,
            "manifests": args.manifest_paths,
            "objective": args.objective,
            "target_precision": args.target_precision if args.objective == "precision" else None,
            "thresholds": args.thresholds,
            "max_samples": args.max_samples,
            "device": args.device,
        },
        "results": results,
    }

    results_path = os.path.join(args.output_path, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=4, ensure_ascii=False)
    logger.info(f"Selected threshold (pooled): {results['pooled']['selected']['threshold']}")
    logger.info(f"Results written to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="HuggingFace model id or path to a fine-tuned Wav2Vec2 checkpoint")
    parser.add_argument("--manifest_paths", nargs="+", type=str, required=True, help="One or more manifest files (.json or .tsv)")
    parser.add_argument("--objective", type=str, default="precision", choices=["precision", "der"])
    parser.add_argument("--target_precision", type=float, default=0.99, help="Precision the emitted diacritics must reach (precision objective only).")
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS, help="Candidate thresholds (0-1).")
    parser.add_argument("--max_samples", type=int, default=None, help="Only use the first N utterances of each manifest.")
    parser.add_argument("--output_path", type=str, default="outputs/tune_threshold/")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_blank_token", action="store_true", help="Use blank token if set.")
    parser.add_argument("--use_no_diac_token", action="store_true", help="Use no diac token if set.")
    parser.add_argument("--use_unk_diac_token", action="store_true", help="Use unk diac token if set.")
    args = parser.parse_args()

    if not all(0.0 <= threshold <= 1.0 for threshold in args.thresholds):
        parser.error("--thresholds must be between 0 and 1")
    args.thresholds = sorted(set(args.thresholds))

    objective = f"precision_{args.target_precision}" if args.objective == "precision" else args.objective
    args.output_path = os.path.join(args.output_path, objective)
    os.makedirs(args.output_path, exist_ok=True)

    logger.info(f"Output path: {args.output_path}")
    file_handler = logging.FileHandler(os.path.join(args.output_path, "tune_threshold.log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    main(args)
