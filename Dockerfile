# syntax=docker/dockerfile:1
#
# VQ-CycleDiffusion research image
#
# Build from the repository root:
#   docker build -t vq-cyclediffusion .
#
# Run with a GPU and only the licensed assets needed for your experiment:
#   docker run --gpus all -it --rm \
#     --shm-size=8G \
#     -v "$(pwd):/workspace/VQ-CycleDiffusion" \
#     -v "/path/to/your/assets:/assets" \
#     vq-cyclediffusion
#
# Note:
# - This image covers training/conversion/MCD/ASR-package dependencies used in this repo.
# - Kaldi binaries for the paper's i-vector/x-vector evaluation are not included; kaldiio
#   Python packages are installed, but a full Kaldi recipe/model should be mounted separately.

FROM python:3.8-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    HF_HOME=/tmp/huggingface \
    TORCH_HOME=/tmp/torch \
    WORKDIR=/workspace/VQ-CycleDiffusion

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libffi-dev \
    libgomp1 \
    libsndfile1 \
    libsndfile1-dev \
    libsox-dev \
    locales \
    make \
    pkg-config \
    sox \
    swig \
    unzip \
    wget \
    build-essential \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i 's/# ko_KR.UTF-8 UTF-8/ko_KR.UTF-8 UTF-8/' /etc/locale.gen \
    && sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN python -m pip install --upgrade "pip==24.0" "setuptools==68.0.0" "wheel==0.42.0"

# PyTorch stack pinned to the CUDA 11.6 wheels used by the submitted experiment env.
RUN python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu116 \
    "torch==1.13.0+cu116" \
    "torchaudio==0.13.0+cu116" \
    "torchvision==0.14.0+cu116"

# Build-sensitive audio/MCD packages. pyworld 0.3.5 can fail in slim images
# with a broken generated pyworld.cpp, so use 0.3.4 for a stable cp38 build.
RUN python -m pip install "numpy==1.21.6" "Cython==0.29.36" \
    && python -m pip install --no-build-isolation \
      "fastdtw==0.3.4" \
      "pyworld==0.3.4" \
      "pysptk==1.0.1" \
    && python -m pip install "Cython==3.0.12"

# Core numeric/audio/ML dependencies. Versions follow VQ-CycleDiffusion/requirements.txt
# where the freeze is directly usable; a few packages are added for imports present in code.
RUN python -m pip install \
    "absl-py==2.1.0" \
    "audioread==3.0.1" \
    "cachetools==5.5.2" \
    "certifi==2025.1.31" \
    "cffi==1.15.1" \
    "charset-normalizer==3.4.1" \
    "click==8.1.8" \
    "cycler==0.11.0" \
    "decorator==5.1.1" \
    "einops==0.3.0" \
    "et-xmlfile==1.1.0" \
    "filelock==3.12.2" \
    "fonttools==4.38.0" \
    "fsspec==2023.1.0" \
    "future==1.0.0" \
    "google-auth==2.39.0" \
    "google-auth-oauthlib==0.4.6" \
    "grpcio==1.62.3" \
    "huggingface-hub==0.16.4" \
    "HyperPyYAML==1.2.2" \
    "idna==3.10" \
    "importlib-metadata==6.7.0" \
    "importlib-resources==5.12.0" \
    "jiwer==3.0.5" \
    "joblib==1.3.2" \
    "kaldi-io==0.9.8" \
    "kaldiio==2.18.1" \
    "kiwisolver==1.4.5" \
    "lazy_loader==0.4" \
    "librosa==0.9.2" \
    "llvmlite==0.39.1" \
    "Markdown==3.4.4" \
    "MarkupSafe==2.1.5" \
    "matplotlib==3.5.3" \
    "msgpack==1.0.5" \
    "multiprocess==0.70.15" \
    "numba==0.56.4" \
    "oauthlib==3.2.2" \
    "openpyxl==3.1.3" \
    "packaging==24.0" \
    "pandas==1.3.5" \
    "Pillow==9.5.0" \
    "platformdirs==4.0.0" \
    "pooch==1.8.2" \
    "protobuf==3.20.3" \
    "pyasn1==0.5.1" \
    "pyasn1-modules==0.3.0" \
    "pycparser==2.21" \
    "pymcd==0.2.1" \
    "pyparsing==3.1.4" \
    "python-dateutil==2.9.0.post0" \
    "pytz==2025.2" \
    "PyYAML==6.0.1" \
    "rapidfuzz==3.4.0" \
    "regex==2024.4.16" \
    "requests==2.31.0" \
    "requests-oauthlib==2.0.0" \
    "resampy==0.4.3" \
    "rsa==4.9.1" \
    "ruamel.yaml==0.18.13" \
    "ruamel.yaml.clib==0.2.8" \
    "safetensors==0.2.8" \
    "scikit-learn==1.0.2" \
    "scipy==1.7.3" \
    "seaborn==0.12.2" \
    "sentencepiece==0.2.0" \
    "six==1.17.0" \
    "soundfile==0.13.1" \
    "soxr==0.3.7" \
    "speechbrain==0.5.14" \
    "tb-nightly==2.12.0a20230113" \
    "tensorboard-data-server==0.6.1" \
    "tensorboard-plugin-wit==1.8.1" \
    "tgt==1.5" \
    "threadpoolctl==3.1.0" \
    "tokenizers==0.13.2" \
    "tqdm==4.67.1" \
    "transformers==4.26.0" \
    "typing_extensions==4.7.1" \
    "urllib3==2.0.7" \
    "webrtcvad==2.0.10" \
    "Werkzeug==2.2.3" \
    "zipp==3.15.0"

# Optional utilities used by speaker_encoder visualizations/preprocess notebooks.
RUN python -m pip install \
    "umap-learn==0.5.5" \
    "visdom==0.2.4" \
    "inflect==6.0.5" \
    "Unidecode==1.3.8"

WORKDIR /workspace/VQ-CycleDiffusion
COPY . /workspace/VQ-CycleDiffusion

ENV PYTHONPATH=/workspace/VQ-CycleDiffusion:/workspace/VQ-CycleDiffusion/hifi-gan:/workspace/VQ-CycleDiffusion/speaker_encoder

RUN mkdir -p /tmp/matplotlib /tmp/numba /tmp/huggingface /tmp/torch \
    && python -c "import torch, librosa, soundfile, sklearn, tgt; from pymcd.mcd import Calculate_MCD; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

CMD ["/bin/bash"]
