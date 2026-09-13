"""Prepare the four-speaker VCTK experiment layout from licensed local WAVs.

Input: <wav-root>/<speaker>/<speaker>_<sentence>.wav
Output: <output>/VCTK_2F2M_{train,valid,test}/{wavs,mels,embeds,textgrids,mels_mode}/...
"""

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tgt
import torch
from scipy.stats import mode

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "speaker_encoder"))
from encoder import inference as spk_encoder

from scripts.audio_features import get_mel


SPEAKERS = ("p236", "p239", "p259", "p263")
TEST_IDS = {f"{i:03d}" for i in range(2, 34)} - {"008", "022"}
VALID_IDS = {f"{i:03d}" for i in range(34, 64)}


def split_for(sentence_id):
    if sentence_id in TEST_IDS:
        return "test"
    if sentence_id in VALID_IDS:
        return "valid"
    return "train"


def _intervals(textgrid_path, frames):
    grid = tgt.io.read_textgrid(str(textgrid_path)).get_tier_by_name("phones")
    for interval in grid:
        start = max(0, int(interval.start_time * 22050) // 256)
        end = min(frames, int(interval.end_time * 22050) // 256 + 1)
        if end > start and interval.text:
            yield interval.text, start, end


def build_mode_mels(train_root):
    """Fit phone templates on training utterances only, then write train targets."""
    examples = defaultdict(list)
    mel_files = sorted((train_root / "mels").glob("*/*_mel.npy"))
    for path in mel_files:
        speaker, stem = path.parent.name, path.name[:-8]
        grid = train_root / "textgrids" / speaker / f"{stem}.TextGrid"
        if not grid.is_file():
            continue
        mel = np.load(path)
        for phone, start, end in _intervals(grid, mel.shape[1]):
            examples[phone].append(np.round(np.median(mel[:, start:end], axis=1), 1))
    if not examples:
        raise ValueError("No training TextGrids found; cannot construct mels_mode")
    templates = {phone: mode(np.stack(vectors), axis=0).mode[0]
                 for phone, vectors in examples.items()}
    for path in mel_files:
        speaker, stem = path.parent.name, path.name[:-8]
        grid = train_root / "textgrids" / speaker / f"{stem}.TextGrid"
        if not grid.is_file():
            continue
        mel = np.load(path)
        avg = mel.copy()
        for phone, start, end in _intervals(grid, avg.shape[1]):
            if phone in templates:
                avg[:, start:end] = templates[phone][:, None]
        out = train_root / "mels_mode" / speaker / f"{stem}_avgmel.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, avg)
    return len(templates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speaker-encoder", type=Path, required=True)
    parser.add_argument("--textgrid-root", type=Path)
    parser.add_argument("--with-mode-mels", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spk_encoder.load_model(args.speaker_encoder, device=device)
    counts = defaultdict(int)
    manifest = []
    for speaker in SPEAKERS:
        wavs = sorted((args.wav_root / speaker).glob(f"{speaker}_*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No WAVs found for {speaker}")
        for wav_path in wavs:
            parts = wav_path.stem.split("_")
            if len(parts) < 2 or parts[0] != speaker:
                raise ValueError(f"Unexpected filename: {wav_path}")
            split = split_for(parts[1])
            root = args.output / f"VCTK_2F2M_{split}"
            out_wav = root / "wavs" / speaker / wav_path.name
            out_mel = root / "mels" / speaker / f"{wav_path.stem}_mel.npy"
            out_embed = root / "embeds" / speaker / f"{wav_path.stem}_embed.npy"
            for path in (out_wav, out_mel, out_embed):
                path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav_path, out_wav)
            np.save(out_mel, get_mel(wav_path).astype(np.float32))
            processed = spk_encoder.preprocess_wav(str(wav_path))
            np.save(out_embed, spk_encoder.embed_utterance(processed).astype(np.float32))
            if args.textgrid_root:
                grid = args.textgrid_root / speaker / f"{wav_path.stem}.TextGrid"
                if grid.is_file():
                    destination = root / "textgrids" / speaker / grid.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(grid, destination)
            counts[(speaker, split)] += 1
            manifest.append((split, speaker, parts[1], str(wav_path.resolve())))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "speaker", "sentence_id", "source_wav"])
        writer.writerows(manifest)
    if args.with_mode_mels:
        templates = build_mode_mels(args.output / "VCTK_2F2M_train")
        print(f"Built {templates} training-only phone templates")
    for (speaker, split), count in sorted(counts.items()):
        print(f"{speaker} {split}: {count}")


if __name__ == "__main__":
    main()
