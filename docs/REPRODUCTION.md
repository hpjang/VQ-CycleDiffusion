# Reproduction guide

Run commands from the repository root. The Dockerfile is the historical Python
3.8/CUDA environment. It pins `pymcd 0.2.1` and `pyworld 0.3.4`; the submitted
MCD environment may differ. Do not mix means from different package versions.

## Assets

The experiment author's [repository](https://github.com/kusmin1363/VQ-CycleDiffusion)
describes the original staged workflow. A separately hosted [Drive asset
folder](https://drive.google.com/drive/folders/1PPmIn9Jtu87OCe1UEYPxHUVqFwv862Ra?usp=sharing)
contains pretrained weights and preprocessed VCTK arrays. Its top level was
checked when this guide was written:

```text
checkpts/spk_encoder/pretrained.pt
hifi-gan/generator_universal.pth
log/codebook_stock_255_exclude/{global,p236_exclude,p239_exclude,p259_exclude,p263_exclude}/
VCTK_2F2M/{wavs,mels,mels_mode,embeds,textgrids,txt}/
VCTK_2F2M_{train,valid,test}/{wavs,mels,mels_mode,embeds,textgrids,txt}/
vc_255.pt
```

The top-level `vc_255.pt` is the pretrained CycleDiffusion backbone. The code
expects it at `log/log_Gunhee/vc_255.pt`. The downloaded `log/` contains stock
codebooks, not that nested backbone path. `scripts/setup_drive_assets.py`
creates the required link. It links the supplied codebook **files** into a
local `log/` directory so new `K=512` codebooks and training checkpoints can
be written without changing the downloaded assets.

Download with [gdown 6.2.0](https://github.com/wkentaro/gdown) on the host.
This version requires Python 3.10 or newer; older gdown releases can fail
on folders containing more than 50 files. The model Docker image remains
Python 3.8. Use a host-side Python 3.10+ interpreter, and choose an asset
directory outside the Git checkout:

```bash
python3 --version  # must be at least 3.10
python3 -m venv /absolute/path/to/gdown-venv
/absolute/path/to/gdown-venv/bin/python -m pip install 'gdown==6.2.0'
ASSETS=/absolute/path/to/vq-assets
/absolute/path/to/gdown-venv/bin/gdown --continue \
  'https://drive.google.com/drive/folders/1PPmIn9Jtu87OCe1UEYPxHUVqFwv862Ra?usp=sharing' \
  -O "$ASSETS"
python3 scripts/setup_drive_assets.py --asset-root "$ASSETS"
```

The `-O` path has no trailing slash so its contents go directly into
`$ASSETS`, without an extra `checkpoints/` directory. `--continue` skips
completed files and resumes partial ones. The command can take a long time
because the VCTK folders contain many files. Do not commit these assets;
weights and data are excluded by `.gitignore`. If download access fails,
check sharing permissions in a browser. Do not disable TLS verification.

Inside Docker, mount the checkout and the asset directory at the same
absolute asset path. This preserves the host-created symlink targets while
keeping downloaded files read-only:

```bash
docker build -t vq-cyclediffusion .
docker run --gpus all --rm -it --shm-size=8g \
  -v "$PWD:/workspace/VQ-CycleDiffusion" \
  -v "$ASSETS:$ASSETS:ro" \
  vq-cyclediffusion
```

Verify the links inside the container:

```bash
test -f checkpts/spk_encoder/pretrained.pt
test -f hifi-gan/generator_universal.pth
test -f log/log_Gunhee/vc_255.pt
test -d VCTK_2F2M_train/mels
test -d VCTK_2F2M_valid/wavs
```

The Table 9 S2T-HF/S2T-WS setting needs these **additional generated paths**
for `K=512`:

```text
log/codebook_stock_255_exclude/<spk>_exclude/codebook_stock_<spk>_<K>.pt
mappings/<K>/indv2indv_count/count_matrix_<src>_to_<tgt>.pt
log/Decoder_cycle_only_10.0/local/<K>_diff_1e-08/<spk>/best_model.pt
```

The U2T and S2U2T variants additionally need
`log/codebook_stock_255_exclude/global/codebook_stock_<K>.pt` and
`mappings/<K>/count_matrix_<spk>_to_global.pt`.

The inspected Drive codebook folders contain speaker-specific `K=4096` and
`K=8192` weights; the global folder also contains `K=16384`. They do **not**
contain `K=512` stock codebooks, count maps, or the CD fine-tuned decoder
checkpoints. Follow the codebook, mapping, and CD training steps below for
Table 9. Providing this asset link does not imply that the submitted CD
checkpoint or its reported scores can be reproduced bit-for-bit with the
corrected trainer. Obtain and use VCTK data under its own terms. Never load
an untrusted PyTorch checkpoint: historical `torch.load` can execute pickle
code.

## Data preparation

The intended split is test IDs `002..033` excluding `008` and `022` (30),
validation IDs `034..063` (30), and the remaining available IDs for training.
The paper reported 410 training utterances per speaker. The archived scoring
grid uses the directory named `VCTK_2F2M_valid` for the 30 evaluation WAVs;
this split-name versus manuscript-test-label mismatch must be resolved in any
publication claim. Verify actual counts,
exception files, and the generated `split_manifest.csv` before fitting.

If you downloaded the preprocessed split folders above, skip
`scripts.prepare_vctk`. That command is only for reconstructing arrays from
raw WAVs and TextGrids:

```bash
python -m scripts.prepare_vctk --wav-root /assets/vctk_wavs --output . \
  --speaker-encoder checkpts/spk_encoder/pretrained.pt \
  --textgrid-root /assets/vctk_textgrids --with-mode-mels
```

The utility writes 22,050-Hz, 80-bin log mel-spectrograms with 1,024-point
FFT and 256-sample hop, plus speaker embeddings. Optional `mels_mode` targets
require Montreal Forced Aligner `phones` TextGrids. Phone templates are fit on
training utterances only to avoid cross-split leakage. This is a clean
reconstruction, not a verified byte-for-byte copy of the original arrays.
The four-speaker data alone does not regenerate the pretrained CycleDiffusion
backbone: `train_cyclediffusion_enc.py` expects broader VCTK data. Use the
exact `vc_255.pt` for strict checkpoint-dependent replication.

## Workflow and checkpoints

The original implementation organizes work as baseline encoder/decoder
training, k-means codebook initialization, optional codebook or joint updates,
count-map construction, diffusion fine-tuning, conversion, and scoring. This
release provides those entry points, but the Table 9 CD workflow starts from
the downloaded `vc_255.pt` instead of retraining the baseline. It initializes
four `K=512` speaker codebooks, builds direct source-to-target count maps,
fine-tunes the diffusion decoder only, and then converts with S2T-HF or
S2T-WS. Do not confuse the downloaded backbone checkpoint with a CD
fine-tuned checkpoint.

## Codebooks and count maps

```bash
for spk in p236 p239 p259 p263; do
  python -m train.init_codebook_stock_indv --spk "$spk" --size 512 \
    --max-iter 100 --n-init 10
done
python -m train.init_codebook_stock_global --size 512 \
  --max-iter 100 --n-init 10
python -m train.counting_map_script --size 512 \
  --out_dir mappings/512/indv2indv_count
python -m train.counting_map_spk2spk --codebooksize 512
```

KMeans uses `random_state=0`, 10 initializations, and at most 100 iterations
by default. Regenerate all dependent assets if `K` changes. The historical
filename `counting_map_spk2spk.py` actually writes speaker-to-universal maps;
`counting_map_script.py` writes direct source-to-target maps. The counters use
vectorized `torch.bincount` with the same frame-level co-occurrence definition.
Count maps are fixed after construction, including during CD fine-tuning.

## CD diffusion-only fine-tuning

```bash
python -m train.train_decoder_cycle_indv --size 512 --batch_size 1
```

This trains one decoder per target speaker from `vc_255.pt`, with frozen
encoder and codebooks: 100 epochs, Adam learning rate `1e-8`, cycle weight 10,
and three 6-step paths per batch (self-reconstruction, A-to-B, B-to-A cycle).
The best validation-loss checkpoint goes under the `Decoder_cycle_only_10.0`
path shown above. Training cost is not only a codeword lookup.

**Historical-result distinction.** The submitted CD script used batch size 2,
misassigned the first `forward_vq` output as the generated mel in the cycle
loss, and could apply one map to a mixed speaker-pair batch. The recommended
script corrects the output assignment and defaults to batch size 1. It cannot
recreate the historical checkpoint bit-for-bit. The original script is in
`legacy/` for audit, not as the recommended trainer. The historical best
checkpoint's selected epoch could not be recovered from the available archive.

## Conversion

```bash
python -m scripts.make_manifest --wav-root VCTK_2F2M_valid/wavs \
  --output-root outputs/s2t_hf --manifest outputs/s2t_hf_manifest.csv
python -m scripts.convert_cd --manifest outputs/s2t_hf_manifest.csv \
  --method s2t-hf --size 512 --skip-existing
```

Repeat with a distinct output directory and `--method s2t-ws`. Both methods
share the same CD-trained decoder. The manifest has 12 ordered pairs times 30
source utterances times 29 distinct target references, or 10,440 outputs.
The runner fails if a target-speaker CD checkpoint is absent; it never
substitutes the baseline without notice. Set `--decoder-root` explicitly when
using checkpoints outside the documented `K=512` path.
The optimized runner loads the vocoder/speaker encoder once, caches codebooks
and the current pair, and removes an unused 30-step baseline diffusion pass.
It retains 30-step `ml` VQ decoding and the legacy post-filter by default:
S2T-HF uses the source mel as reference, while S2T-WS uses the generated mel.
`--no-postfilter` is an ablation, not the submitted setting.

The seven initial transformations remain in `convert/`. Those original scripts
may reload models for every WAV. They use different checkpoints from Table 9
CD; do not combine metrics from both stages into a single row.
