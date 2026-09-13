"""Generate the 12-direction, 30-by-29 conversion grid used in the paper."""

import argparse
import csv
from itertools import permutations
from pathlib import Path

SPEAKERS = ("p236", "p239", "p259", "p263")


def rows(wav_root, output_root, speakers=SPEAKERS):
    wav_root = Path(wav_root)
    output_root = Path(output_root)
    by_speaker = {}
    for speaker in speakers:
        files = sorted((wav_root / speaker).glob(f"{speaker}_*.wav"))
        by_speaker[speaker] = {path.stem.split("_")[1]: path for path in files}
        if len(by_speaker[speaker]) != 30:
            raise ValueError(f"Expected 30 test WAVs for {speaker}, got {len(files)}")
    for source, target in permutations(speakers, 2):
        for source_id, source_wav in sorted(by_speaker[source].items()):
            for target_id, reference_wav in sorted(by_speaker[target].items()):
                if source_id == target_id:
                    continue
                yield {
                    "source_speaker": source,
                    "target_speaker": target,
                    "source_wav": str(source_wav),
                    "reference_wav": str(reference_wav),
                    "output_wav": str(output_root / f"{source}_to_{target}" /
                                      f"{source_id}_to_{target_id}.wav"),
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_speaker", "target_speaker",
                                                    "source_wav", "reference_wav", "output_wav"])
        writer.writeheader()
        count = 0
        for row in rows(args.wav_root, args.output_root):
            writer.writerow(row)
            count += 1
    print(f"Wrote {count} conversions to {args.manifest}")


if __name__ == "__main__":
    main()
