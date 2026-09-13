# VQ-CycleDiffusion

Research implementation of vector-quantized, cycle-consistent diffusion voice
conversion. The paper studies speaker-specific and universal codebooks and
seven codeword-transformation strategies. This release provides the model,
training and inference source, a four-speaker data-preparation utility, and
scripts for the Table 9 diffusion-only fine-tuning setting.

## Scope and status

The repository contains code, not the VCTK audio, forced alignments,
preprocessed arrays, or trained checkpoints. The current release has been
syntax-checked and its manifest-generation test has been run. End-to-end
training and output-score replication require the original assets and a CUDA
GPU; no numerical equivalence to the published tables is claimed without
those checks. See [Reproduction](docs/REPRODUCTION.md) before using the code.

The recommended CD fine-tuning script corrects a cycle-output assignment and
rejects mixed speaker-pair batches. The historical submitted script is retained
at `legacy/train_decoder_cycle_indv_original.py` for audit. These versions are
not expected to produce identical checkpoints. The optimized S2T inference
script removes one unused 30-step diffusion pass and reuses loaded models,
while retaining each method's post-filter choice.

## Quick start

```bash
git clone https://github.com/hpjang/VQ-CycleDiffusion.git
cd VQ-CycleDiffusion
docker build -t vq-cyclediffusion .
docker run --gpus all --rm -it --shm-size=8g \
  -v "$PWD:/workspace/VQ-CycleDiffusion" \
  -v "/absolute/path/to/your/assets:/assets" \
  vq-cyclediffusion
```

The commands below run from the repository root inside the container. Mount
licensed data and trusted checkpoints into the container and arrange them as
shown in [Reproduction](docs/REPRODUCTION.md).

```bash
python -m scripts.prepare_vctk \
  --wav-root /assets/vctk_wavs --output /workspace/VQ-CycleDiffusion \
  --speaker-encoder checkpts/spk_encoder/pretrained.pt \
  --textgrid-root /assets/vctk_textgrids --with-mode-mels

python -m train.init_codebook_stock_indv --spk p236 --size 512
python -m train.counting_map_script --size 512 \
  --out_dir mappings/512/indv2indv_count
python -m train.train_decoder_cycle_indv --size 512 --batch_size 1

python -m scripts.make_manifest \
  --wav-root VCTK_2F2M_valid/wavs \
  --output-root outputs/s2t_hf \
  --manifest outputs/s2t_hf_manifest.csv
python -m scripts.convert_cd --manifest outputs/s2t_hf_manifest.csv \
  --method s2t-hf --size 512 --skip-existing
```

Generate a separate manifest and run `--method s2t-ws` for the weighted-sum
variant. Both methods use the same CD fine-tuned decoder; only the inference
mapping differs. The default decoder root is for `K=512`; override it for any
other size or training run.

## Repository map

| Path | Purpose |
| --- | --- |
| `model/` | Mel encoder, VQ, and diffusion decoder |
| `train/` | Codebook initialization, count maps, and fine-tuning variants |
| `convert/` | Original seven transformation and conversion scripts |
| `scripts/prepare_vctk.py` | VCTK split, mel and speaker embeddings, optional phone-mode targets |
| `scripts/convert_cd.py` | Optimized, reusable S2T-HF/S2T-WS batch inference for Table 9 CD |
| `speaker_encoder/`, `hifi-gan/` | Third-party speaker encoder and vocoder source |
| `legacy/` | Historical CD fine-tuning script kept for audit |

## Documentation

- [Data, checkpoints, training, and inference](docs/REPRODUCTION.md)
- [Transformation methods and implementation notes](docs/METHODS.md)
- [Evaluation and known limitations](docs/EVALUATION.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

Use converted speech only with the consent and authorization appropriate to
the speaker and dataset. This experimental code is not an identity-verification
system.
