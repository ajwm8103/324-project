import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from train import fetch_arguments
from embedding import load_embedding_from_pickle, embedding_to_midi
from train import MidiDataset
import numpy as np


def test_model_from_file(dataloader, args, filename="model_1.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Transformer(d_model=5, nhead=args.nhead, num_encoder_layers=args.num_encoder_layers).to(device)
    
    model.load_state_dict(torch.load(filename, map_location=device))
    model.to(device)

    srcs, tgts = next(iter(dataloader))
    src = srcs[0].to(device)
    tgt = tgts[0].to(device)
    
    output = model(src, tgt)  # tgt[:-1] used as target input to predict tgt[1:]

    # Save to a midi file
    embedding = np.array([output.cpu().detach().numpy()])
    embedding_to_midi(embedding, filename='output.mid')

if __name__ == "__main__":
    args = fetch_arguments()

    data_embedding = load_embedding_from_pickle(args)

    dataset = MidiDataset(data_embedding, args.seq_len, args, args.stride_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_model_from_file(dataloader, args)