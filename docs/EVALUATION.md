# Evaluation protocol and limitations

## MCD

`scripts/evaluate_mcd.py` writes one row per conversion using `pymcd` DTW
MCD. The evaluation protocol must be chosen explicitly:

```bash
python -m scripts.evaluate_mcd --manifest outputs/s2t_hf_manifest.csv \
  --reference-root VCTK_2F2M_valid/wavs --protocol submitted \
  --output outputs/s2t_hf_mcd.csv
```

`submitted` reproduces the original reference-selection rule: a converted
file named `034_to_035.wav` is compared with the target speaker's `035` WAV.
The converted content comes from source utterance `034`, however, and VCTK
speakers' same-numbered utterances need not share a transcript. DTW aligns
feature sequences; it does not make different sentences text-equivalent.
Thus, this MCD is a historical diagnostic, not a clean measure of spectral
distortion against a parallel target utterance. The fact that every method
uses the same rule does not remove this content confound.

For a new parallel evaluation, create a text-verified map with columns
`source_speaker,source_sentence,target_speaker,reference_wav` and run:

```bash
python -m scripts.evaluate_mcd --manifest outputs/s2t_hf_manifest.csv \
  --reference-root VCTK_2F2M_valid/wavs --protocol parallel \
  --parallel-map /assets/parallel_reference_map.csv \
  --output outputs/s2t_hf_parallel_mcd.csv
```

Build the map from actual normalized transcripts, not sentence numbers alone.
Use the same map and software environment for all compared systems. If no
matching target recording exists, do not invent a parallel reference.

## Speaker similarity and predicted MOS

The manuscript's i-vector and x-vector cosine scores require the original
Kaldi VoxCeleb recipes, extractor checkpoints, enrollment/reference selection,
and aggregation protocol. The Python `kaldiio` package is not a substitute for
those binaries and models. They are not distributed in this repository.
Do not treat cosine similarity as direct proof of speaker disentanglement.

UTMOS and DNSMOS are model-predicted MOS measures, not human listening scores.
Their external predictor weights and exact versions must be recorded before
comparing to published results. A subjective MOS/CMOS study remains necessary
for strong naturalness or high-fidelity claims. As with MCD, paired tests
should use per-utterance or speaker-pair observations, not only table means.

The original evaluation has four speakers and 12 ordered conversion pairs.
Results therefore support a limited proof of concept, not general robustness
or performance on unseen speakers. Trained checkpoints and generated audio
must be released separately with the appropriate rights if exact score
reproduction is required.
