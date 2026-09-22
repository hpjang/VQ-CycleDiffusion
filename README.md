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

This is a documented reproduction release based on the [experiment author's
repository](https://github.com/kusmin1363/VQ-CycleDiffusion). Its stage names
follow that implementation, but the recommended CD trainer and S2T converter
include the corrections and optimizations described above. The two releases
should not be assumed to produce identical checkpoints.

## Download pretrained assets

The [pretrained weights and preprocessed four-speaker VCTK data](https://drive.google.com/drive/folders/165Y5OZqV2XXWRX_XqTtytrv03aCspOpQ?usp=sharing)
are distributed separately from Git. Download them on the host, outside the
checkout, using [gdown 6.2.0](https://github.com/wkentaro/gdown) and Python
3.10 or newer. The Docker image uses Python 3.8, so do this before starting
the container.

```bash
git clone https://github.com/hpjang/VQ-CycleDiffusion.git
cd VQ-CycleDiffusion
python3 -m venv /absolute/path/to/gdown-venv  # python3 must be >= 3.10
/absolute/path/to/gdown-venv/bin/python -m pip install 'gdown==6.2.0'
ASSETS=/absolute/path/to/vq-assets
/absolute/path/to/gdown-venv/bin/gdown --continue \
  'https://drive.google.com/drive/folders/1PPmIn9Jtu87OCe1UEYPxHUVqFwv862Ra?usp=sharing' \
  -O "$ASSETS"
python3 scripts/setup_drive_assets.py --asset-root "$ASSETS"
```

The setup script checks the downloaded layout and creates ignored symlinks
for the VCTK splits, speaker encoder, vocoder, supplied codebook files, and
baseline `vc_255.pt`. It does not overwrite existing assets. The Drive folder does
not contain `K=512` codebooks or counting maps; generate those before running
the Table 9 CD setting. The historical CD fine-tuned checkpoints are also
not included. See [Reproduction](docs/REPRODUCTION.md) for the exact layout,
training stages, and split caveats.

## Quick start

```bash
docker build -t vq-cyclediffusion .
docker run --gpus all --rm -it --shm-size=8g \
  -v "$PWD:/workspace/VQ-CycleDiffusion" \
  -v "$ASSETS:$ASSETS:ro" \
  vq-cyclediffusion
```

The asset mount must use the same absolute path on the host and in the
container so the symlinks remain valid. The commands below run from the
repository root inside the container, using the downloaded preprocessed data.

```bash
for spk in p236 p239 p259 p263; do
  python -m train.init_codebook_stock_indv --spk "$spk" --size 512
done
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
| `scripts/setup_drive_assets.py` | Validate and link the separately downloaded Drive assets |
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
