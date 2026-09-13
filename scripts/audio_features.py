"""Log-mel extraction matching the inherited VC inference scripts."""

import librosa
import numpy as np
from librosa.filters import mel as librosa_mel_fn


MEL_BASIS = librosa_mel_fn(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000)


def get_mel(path):
    wav, _ = librosa.load(path, sr=22050, mono=True)
    wav = wav[: (len(wav) // 256) * 256]
    if len(wav) < 512:
        raise ValueError(f"Waveform too short: {path}")
    wav = np.pad(wav, 384, mode="reflect")
    stft = librosa.stft(wav, n_fft=1024, hop_length=256, win_length=1024,
                        window="hann", center=False)
    magnitude = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + 1e-9)
    return np.log(np.clip(MEL_BASIS @ magnitude, a_min=1e-5, a_max=None))
