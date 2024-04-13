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
import random


def test_model_from_file(dataloader, args, filename="model_1_long.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Transformer(d_model=5, nhead=args.nhead, num_encoder_layers=args.num_encoder_layers).to(device)
    
    model.load_state_dict(torch.load(f"models/{filename}", map_location=device))
    model.to(device)

    batches = list(dataloader)
    for i in range(8):
        random_batch = random.choice(batches)
        srcs, tgts = random_batch
        src = srcs[0].to(device)
        tgt = tgts[0].to(device)
        
        output = model(src, tgt)  # tgt[:-1] used as target input to predict tgt[1:]
        args.log("Sample Input:", src[0:5])
        args.log("Sample Target:", tgt[0:5])
        args.log("Model Output:", output[0:5])

        # Save to a midi file
        embedding = output#.cpu().detach().numpy()
        embedding_to_midi(embedding, filename=f'good_outputs/model_output_{i}.mid')
        embedding = src#.cpu().detach().numpy()
        embedding_to_midi(embedding, filename=f'good_outputs/input_{i}.mid')
        embedding = tgt
        embedding_to_midi(embedding, filename=f'good_outputs/target_{i}.mid')

if __name__ == "__main__":
    args = fetch_arguments()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    data_embedding = load_embedding_from_pickle(args)

    dataset = MidiDataset(data_embedding, args.seq_len, args, args.stride_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    test_model_from_file(dataloader, args)