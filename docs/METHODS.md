# Method and implementation notes

The encoder maps an 80-bin source mel sequence to framewise latent vectors.
Speaker-specific (`C^s`, `C^t`) and universal (`C^u`) codebooks are KMeans
centroid tables. Count matrices record how often two codebooks select a given
index for the same encoded frame. The matrices are constructed on training
data, then kept fixed at conversion and CD fine-tuning time.

| Paper subsection | Method | Original inference module | Input lookup |
| --- | --- | --- | --- |
| 3.3.1 | S2T-HF | `convert/inference/inference_vq_mapping_without_global_argmax.py` | source codebook to target codebook, row argmax |
| 3.3.2 | S2T-WS | `convert/inference/inference_vq_mapping_without_global_weightedsum.py` | source codebook to target codebook, normalized row weighted sum |
| 3.3.3 | U2T-HF | `convert/inference/inference_vq_mapping_without_source_argmax.py` | universal codebook to target codebook, row argmax |
| 3.3.4 | U2T-WS | `convert/inference/inference_vq_mapping_without_source_weightedsum.py` | universal codebook to target codebook, normalized row weighted sum |
| 3.3.5 | S2U2T-HF | `convert/inference/inference_vq_mapping_argmax.py` | source to universal to target, highest-frequency path |
| 3.3.6 | S2U2T-WS | `convert/inference/inference_vq_mapping_weightedsum.py` | source to universal to target, weighted composition |
| 3.3.7 | E2T | `convert/inference/inference_vq_mapping_tgt.py` | encoder output to target codebook |

For direct S2T-HF, `j = argmax_j M[i,j]` and the target centroid `C^t[j]`
replaces each source codeword with index `i`. Direct S2T-WS uses
`sum_j M[i,j] C^t[j] / (sum_j M[i,j] + 1e-8)`. The Table 9 CD runner in
`scripts/convert_cd.py` supports these two methods only and uses the same
equations as `convert/inference_vq_train/`.

The 3.3.5 highest-frequency path in the original source chooses the strongest
source-to-universal-to-target path; it is not simply an argmax of the product
of two row-normalized matrices. Consult the actual module before implementing
an independent reproduction. The original scripts also have an unseen-row
repair heuristic in some initial-model paths; the CD direct S2T scripts do not
apply it. Do not silently merge these behaviors.

## Optimizations in this release

- The direct count builders replace per-frame Python/GPU indexing with
  `torch.bincount`. This preserves exact integer co-occurrence counts.
- The recommended CD trainer caches frozen speaker codebooks and compact
  argmax lookup vectors. It no longer deserializes two large count matrices
  and two codebooks every batch.
- `scripts/convert_cd.py` computes the two mel-encoder outputs directly. The
  original inference code ran a full unused 30-step `generator.forward()` to
  obtain them, then ran `forward_vq()` for another 30 steps. The optimized
  runner keeps only the required VQ pass and reuses loaded models across WAVs.
- These are implementation optimizations, not a claim of bitwise-identical
  WAV output. Differences in floating-point order, package versions, and
  checkpoint choice must be checked against archived output before reporting
  reproduced scores.

The two historical CD inference scripts use different mel post-filter
references: S2T-HF uses the source mel and S2T-WS uses the generated mel.
The optimized runner preserves this distinction by default.

For experiment tracking, record the Git commit, data split manifest, asset
hashes, method, `K`, checkpoint, seed, sampler, reverse steps, post-filter
setting, GPU, package versions, and per-utterance outputs.
