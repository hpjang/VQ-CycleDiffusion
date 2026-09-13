"""Batched Table 9 CD inference for S2T-HF and S2T-WS.

Loads the VC model, vocoder, speaker encoder, codebooks, and count map once per
speaker pair. Only one reverse-diffusion pass is needed per converted utterance.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hifi-gan"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "speaker_encoder"))
from env import AttrDict
from models import Generator as HiFiGAN
from encoder import inference as spk_encoder

import params
from scripts.audio_features import get_mel
from model.codebook import VectorQuantizer
from model.utils import sequence_mask
from model.vc_vq import DiffVC


def _noise_median_smoothing(x, w=1):
    y = np.copy(x)
    padded = np.pad(x, w, "edge")
    for i in range(len(y)):
        med = np.median(padded[i:i + 2 * w + 1])
        y[i] = min(padded[i + w + 1], med)
    return y


def legacy_postfilter(mel_synth, mel_reference):
    """Preserve the submitted inference post-filter reference selection."""
    mel_len = mel_reference.shape[-1]
    silence_window = 5
    if mel_len <= silence_window:
        return mel_synth
    energies = [np.sum(np.exp(2.0 * mel_reference[:, i:i + silence_window]))
                for i in range(mel_len - silence_window)]
    i_min = int(np.argmin(energies))
    noise = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min + silence_window]), axis=-1)
    noise = _noise_median_smoothing(noise, w=1)
    denoised = np.copy(mel_synth)
    for i in range(mel_len):
        signal = np.maximum(np.exp(2.0 * mel_synth[:, i]) - noise, 0.02 * noise)
        denoised[:, i] = np.log(np.sqrt(signal))
    return denoised


def mapped_vectors(counts, source_indices, target_weights, method):
    """Return (B, D, T) vectors using the original count-map equations."""
    if counts.ndim != 2 or counts.shape[1] != target_weights.shape[0]:
        raise ValueError("Count-map and target codebook dimensions do not match")
    flat = source_indices.reshape(-1)
    if method == "s2t-hf":
        vectors = target_weights[counts.argmax(dim=1)[flat]]
    elif method == "s2t-ws":
        rows = counts[flat].float()
        vectors = (rows / (rows.sum(dim=1, keepdim=True) + 1e-8)) @ target_weights
    else:
        raise ValueError(f"Unsupported method: {method}")
    batch, frames = source_indices.shape
    return vectors.reshape(batch, frames, -1).permute(0, 2, 1)


class Converter:
    def __init__(self, root, method, size, decoder_root, steps, postfilter):
        self.root = Path(root)
        self.method = method
        self.size = size
        self.decoder_root = Path(decoder_root)
        self.steps = steps
        self.postfilter = postfilter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DiffVC(params.n_mels, params.channels, params.filters, params.heads,
                            params.layers, params.kernel, params.dropout, params.window_size,
                            params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
                            params.beta_min, params.beta_max).to(self.device).eval()
        with open(self.root / "hifi-gan" / "config.json", encoding="utf-8") as handle:
            config = AttrDict(json.load(handle))
        self.vocoder = HiFiGAN(config).to(self.device).eval()
        weights = torch.load(self.root / "hifi-gan" / "generator_universal.pth",
                             map_location=self.device)
        self.vocoder.load_state_dict(weights.get("generator", weights))
        self.vocoder.remove_weight_norm()
        spk_encoder.load_model(self.root / "checkpts" / "spk_encoder" / "pretrained.pt",
                               device=str(self.device))
        self._quantizers = {}
        self._pair = None
        self._counts = None
        self._target = None

    def _quantizer(self, speaker):
        if speaker not in self._quantizers:
            path = (self.root / "log" / "codebook_stock_255_exclude" /
                    f"{speaker}_exclude" / f"codebook_stock_{speaker}_{self.size}.pt")
            quantizer = VectorQuantizer(self.size, params.embedding_dim, 0.25).to(self.device)
            quantizer.load_state_dict(torch.load(path, map_location=self.device))
            quantizer.eval().requires_grad_(False)
            self._quantizers[speaker] = quantizer
        return self._quantizers[speaker]

    def _select_pair(self, source, target):
        if (source, target) == self._pair:
            return
        if target != self._target:
            path = self.decoder_root / target / "best_model.pt"
            if not path.is_file():
                raise FileNotFoundError(f"CD decoder checkpoint required: {path}")
            checkpoint = torch.load(path, map_location=self.device)
            state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
            self.model.load_state_dict(state)
            self.model.eval()
            self._target = target
        self._quantizer(source)
        self._quantizer(target)
        path = (self.root / "mappings" / str(self.size) / "indv2indv_count" /
                f"count_matrix_{source}_to_{target}.pt")
        self._counts = torch.load(path, map_location=self.device)
        if self._counts.shape != (self.size, self.size):
            raise ValueError(f"Unexpected count-map shape: {path}")
        self._pair = (source, target)

    @torch.no_grad()
    def convert(self, source, target, source_wav, reference_wav, output_wav):
        self._select_pair(source, target)
        source_mel = torch.from_numpy(get_mel(source_wav)).float().unsqueeze(0).to(self.device)
        target_mel = torch.from_numpy(get_mel(reference_wav)).float().unsqueeze(0).to(self.device)
        source_lengths = torch.tensor([source_mel.shape[-1]], device=self.device)
        target_lengths = torch.tensor([target_mel.shape[-1]], device=self.device)
        target_embed = spk_encoder.embed_utterance(
            spk_encoder.preprocess_wav(str(reference_wav)))
        target_embed = torch.from_numpy(target_embed).float().unsqueeze(0).to(self.device)
        source_mask = sequence_mask(source_lengths).unsqueeze(1).to(source_mel.dtype)
        target_mask = sequence_mask(target_lengths).unsqueeze(1).to(target_mel.dtype)

        # The legacy generator.forward() ran an unused full diffusion pass to get
        # these two encoder outputs. Direct encoding leaves the VQ path unchanged.
        mean = self.model.encoder(source_mel, source_mask)
        mean_ref = self.model.encoder(target_mel, target_mask)
        indices = self._quantizer(source).get_code_indices(mean)
        target_vq = self._quantizer(target)
        quantized = mapped_vectors(self._counts, indices, target_vq.embeddings.weight,
                                   self.method)
        quantized_ref, _, _, _, _ = target_vq(mean_ref)
        _, converted_mel = self.model.forward_vq(
            source_mel, source_lengths, target_mel, target_lengths, target_embed,
            quantized, quantized_ref, n_timesteps=self.steps, mode="ml")
        if self.postfilter:
            mel_np = converted_mel.squeeze(0).cpu().numpy()
            reference_np = (source_mel.squeeze(0).cpu().numpy()
                            if self.method == "s2t-hf" else mel_np)
            mel_np = legacy_postfilter(mel_np, reference_np)
            converted_mel = torch.from_numpy(mel_np).float().unsqueeze(0).to(self.device)
        audio = self.vocoder(converted_mel).squeeze().clamp(-1, 1).cpu().numpy()
        output_wav = Path(output_wav)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_wav, audio, params.sampling_rate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--method", choices=["s2t-hf", "s2t-ws"], required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--decoder-root", type=Path,
                        default=Path("log/Decoder_cycle_only_10.0/local/512_diff_1e-08"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--no-postfilter", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    decoder_root = (args.decoder_root if args.decoder_root.is_absolute()
                    else root / args.decoder_root)
    converter = Converter(root, args.method, args.size, decoder_root,
                          args.steps, not args.no_postfilter)
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        required = {"source_speaker", "target_speaker", "source_wav",
                    "reference_wav", "output_wav"}
        if not required.issubset(rows.fieldnames or []):
            raise ValueError(f"Manifest requires columns: {sorted(required)}")
        for row in rows:
            output = Path(row["output_wav"])
            if args.skip_existing and output.exists():
                continue
            converter.convert(row["source_speaker"], row["target_speaker"],
                              row["source_wav"], row["reference_wav"], output)


if __name__ == "__main__":
    main()
