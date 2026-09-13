def get_homogeneous_speaker_pair(batch, context="cycle batch"):
    """Return a single (src_spk, tgt_spk) pair and fail fast on mixed target batches.

    The cycle training scripts load one mapping/codebook pair per batch.  The dataset can
    sample a different target speaker for each item, so batch_size > 1 may silently apply
    the first item's mapping to the rest of the batch.  Keeping this check explicit makes
    the inherited issue visible instead of training on mismatched codebooks.
    """
    src_spks = list(batch["src_spk"])
    tgt_spks = list(batch["tgt_spk"])
    src_unique = sorted(set(src_spks))
    tgt_unique = sorted(set(tgt_spks))
    if len(src_unique) != 1 or len(tgt_unique) != 1:
        raise ValueError(
            f"{context} contains mixed speaker pairs: "
            f"src={src_spks}, tgt={tgt_spks}. "
            "Use batch_size=1 or implement pair-grouped batching/per-item mapping."
        )
    return src_unique[0], tgt_unique[0]
