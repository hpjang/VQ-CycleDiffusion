"""Reusable, frozen codebooks and compact argmax count-map lookups."""

from pathlib import Path

import torch

from model.codebook import VectorQuantizer


class FrozenAssets:
    def __init__(self, size, embedding_dim, device, root=Path(".")):
        self.size = size
        self.embedding_dim = embedding_dim
        self.device = device
        self.root = Path(root)
        self._quantizers = {}
        self._lookups = {}

    def quantizer(self, speaker, commitment_cost=0.25):
        key = (speaker, commitment_cost)
        if key not in self._quantizers:
            path = (self.root / "log" / "codebook_stock_255_exclude" /
                    f"{speaker}_exclude" / f"codebook_stock_{speaker}_{self.size}.pt")
            quantizer = VectorQuantizer(self.size, self.embedding_dim,
                                        commitment_cost).to(self.device)
            quantizer.load_state_dict(torch.load(path, map_location=self.device))
            quantizer.requires_grad_(False)
            quantizer.eval()
            self._quantizers[key] = quantizer
        return self._quantizers[key]

    def argmax(self, source, target):
        pair = (source, target)
        if pair not in self._lookups:
            path = (self.root / "mappings" / str(self.size) / "indv2indv_count" /
                    f"count_matrix_{source}_to_{target}.pt")
            counts = torch.load(path, map_location="cpu")
            if counts.shape != (self.size, self.size):
                raise ValueError(f"Unexpected count-map shape in {path}: {tuple(counts.shape)}")
            self._lookups[pair] = counts.argmax(dim=1).to(self.device)
            del counts
        return self._lookups[pair]
