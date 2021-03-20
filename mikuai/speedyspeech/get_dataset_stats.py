"""Automatically calculate spectrogram statistics for dataset in HPText.dataset
"""

import argparse
import torch
from barbar import Bar  # progress bar

from mikuai.speedyspeech.datasets.AudioDataset import AudioDataset
from torch.utils.data import DataLoader
from mikuai.speedyspeech.main import Collate
from mikuai.speedyspeech.hparam import HPText

from mikuai.speedyspeech.utils.masked import masked_max, masked_min, mask, masked_mean, masked_std

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
parser.add_argument("--num_samples", default=4096, type=int, help="On how many samples to compute statistics")
parser.add_argument("--dataset", default=HPText.dataset, type=str, help="Path to dataset")
args = parser.parse_args()

collate = Collate('cuda' if torch.cuda.is_available() else 'cpu', standardize=False)
dl = DataLoader(AudioDataset(HPText.dataset, end_idx=args.num_samples), collate_fn=collate, batch_size=args.batch_size, shuffle=False)

maxi = float('-inf')
mini = float('inf')

mean = 0
std = 0
w = 0

for i, b in enumerate(Bar(dl), 1):
    s, slen, _, plen, _, _ = b
    m = mask(s, slen, dim=1)
    maxi = max(maxi, masked_max(s, m))
    mini = min(mini, masked_min(s, m))
    mean = mean + (masked_mean(s, m) - mean)/i
    std = std + (masked_std(s, m) - std)/i
    ww = sum([sl / pl for sl, pl in zip(slen, plen)])/len(slen)
    w = w + (ww - w)/i
