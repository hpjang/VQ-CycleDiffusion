# python3 train.train_codebook

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.vc import FwdDiffusion
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from train.prepare_data_indv import VCEncDataset, VCEncBatchCollate, VCTKEncDataset_indv
import params
from model.data_cycle_4speakers import VCDecBatchCollate, VCTKDecDataset
from model.vc import DiffVC
from model.utils import FastGL
from model.codebook import VectorQuantizer, OnlineCodebookStocker # <-- 추가
import argparse

parser = argparse.ArgumentParser(description="Initialize one speaker codebook from encoder outputs")
parser.add_argument("--spk", required=True, choices=["p236", "p239", "p259", "p263"])
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--max-iter", type=int, default=100)
parser.add_argument("--n-init", type=int, default=10)
parser.add_argument("--data-dir", default="VCTK_2F2M_train")
parser.add_argument("--checkpoint", default="log/log_Gunhee/vc_255.pt")
argv = parser.parse_args()

# model 구성
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_mels = params.n_mels
sampling_rate = params.sampling_rate
n_fft = params.n_fft
hop_size = params.hop_size

channels = params.channels
filters = params.filters
layers = params.layers
kernel = params.kernel
dropout = params.dropout
heads = params.heads
window_size = params.window_size
enc_dim = params.enc_dim
embedding_dim = params.embedding_dim
dec_dim = params.dec_dim
spk_dim = params.spk_dim
use_ref_t = params.use_ref_t
beta_min = params.beta_min
beta_max = params.beta_max
num_embeddings = params.num_code
random_seed = params.seed
test_size = params.test_size

# encoder 불러오기
print("Loading encoder...")
model = DiffVC(n_mels, channels, filters, heads, layers, kernel, 
                dropout, window_size, enc_dim, spk_dim, use_ref_t, 
                dec_dim, beta_min, beta_max).cuda()

model.load_state_dict(torch.load(argv.checkpoint, map_location=device))
model.eval()
encoder = model.encoder.eval()
# Step 2: Load dataset
print("Loading dataset...")


data_dir = argv.data_dir
exc_file = 'filelists/exceptions_vctk.txt'
avg_type = 'mode'

spk = argv.spk

#train set에서 test data 제외하기 위해 사용
train_set = VCTKEncDataset_indv(data_dir, exc_file, avg_type='mode', speaker=spk)
collate_fn = VCEncBatchCollate()
train_loader = DataLoader(train_set, batch_size=1, collate_fn=None)

# Step 3: 즉시 z-stock 수집
print("Extracting and stocking z vectors...")

codebook = OnlineCodebookStocker(dim=enc_dim)
from model.utils import sequence_mask, fix_len_compatibility, mse_loss

with torch.no_grad():
    for batch in tqdm(train_loader):
        mel_source = batch['x'].to(device)         # (1, 80, T)
        mel_source_lengths = torch.LongTensor([mel_source.shape[-1]]).to(device)
        x_mask = sequence_mask(mel_source_lengths).unsqueeze(1).to(mel_source.dtype) # ([1, 1, 128])
        mean = model.encoder(mel_source, x_mask) # ([1, 80, 128])
        mean = mean.permute(0, 2, 1)
        codebook.add(mean)

print("get_all_before")

z_data = codebook.get_all()  # Codebook size = 233088, dim = 128
num_embeddings = z_data.shape[0]
embedding_dim = z_data.shape[1]
print(f"Codebook size = {num_embeddings}, dim = {embedding_dim}")

codebook_size = [argv.size]
import numpy as np
from sklearn.cluster import KMeans
import torch

for size in codebook_size:
    # Step 5: Run KMeans
    print("Running KMeans...", size)
    kmeans = KMeans(n_clusters=size, random_state=0, n_init=argv.n_init,
                    verbose=1, max_iter=argv.max_iter)
    print("fitting before")
    kmeans.fit(z_data)
    print("fiting done")
    # Save centroids to use in VectorQuantizer
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    # Step 4~5: 코드북 초기화 (샘플링 방식)
    vq = VectorQuantizer(
        num_embeddings=size,
        embedding_dim=embedding_dim,
        commitment_cost=0.25
    )
    vq.embeddings.weight.data.copy_(centroids)

    # 저장
    os.makedirs(f'log/codebook_stock_255_exclude/{spk}_exclude', exist_ok=True)
    torch.save(vq.state_dict(), f'log/codebook_stock_255_exclude/{spk}_exclude/codebook_stock_{spk}_{size}.pt')
    print("Saved raw-initialized embedding weights.")
