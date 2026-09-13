"""Calculate per-conversion MCD with an explicitly selected reference protocol."""

import argparse
import csv
from pathlib import Path
from statistics import mean

from pymcd.mcd import Calculate_MCD


def load_parallel_map(path):
    references = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["source_speaker"], row["source_sentence"], row["target_speaker"])
            if key in references:
                raise ValueError(f"Duplicate parallel reference for {key}")
            references[key] = Path(row["reference_wav"])
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--protocol", choices=["submitted", "parallel"], required=True)
    parser.add_argument("--parallel-map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.protocol == "parallel" and args.parallel_map is None:
        parser.error("--parallel-map is required for the parallel protocol")
    references = load_parallel_map(args.parallel_map) if args.parallel_map else {}
    calculator = Calculate_MCD(MCD_mode="dtw")
    scores = []
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = Path(row["output_wav"])
            source_id = Path(row["source_wav"]).stem.split("_")[1]
            target_id = Path(row["reference_wav"]).stem.split("_")[1]
            target = row["target_speaker"]
            if args.protocol == "submitted":
                reference = args.reference_root / target / f"{target}_{target_id}.wav"
            else:
                reference = references[(row["source_speaker"], source_id, target)]
            if not converted.is_file() or not reference.is_file():
                raise FileNotFoundError(f"Missing converted/reference WAV: {converted}, {reference}")
            score = calculator.calculate_mcd(str(reference), str(converted))
            scores.append({"source_speaker": row["source_speaker"],
                           "target_speaker": target, "source_sentence": source_id,
                           "reference_sentence": target_id,
                           "reference_wav": str(reference),
                           "converted_wav": str(converted), "mcd": score})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    print(f"count={len(scores)} mean_mcd={mean(row['mcd'] for row in scores):.6f}")


if __name__ == "__main__":
    main()
