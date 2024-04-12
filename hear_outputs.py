import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from tqdm import tqdm

import note_seq, pretty_midi
from note_seq import midi_io
from embedding import load_embedding_from_pickle
import numpy as np

from train import MidiDataset, fetch_arguments
from embedding import embedding_to_midi

args = fetch_arguments()

data_embedding = load_embedding_from_pickle(args)
dataset = MidiDataset(data_embedding, args.seq_len)
print(dataset)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Transformer(d_model=5, nhead=args.nhead, num_encoder_layers=args.num_encoder_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


# Load the model:
model.load_state_dict(torch.load('transformer_midi_model.pth'))

model.eval()
with torch.no_grad():
    src, tgt = next(iter(dataloader))
    src, tgt = src.to(device), tgt.to(device)
    output = model(src, tgt[:-1])  # tgt[:-1] used as target input to predict tgt[1:]
    print("Sample input:", src[0])
    print("Model output:", output[1])
    print("True target:", tgt[1][0])

    # Convert output to MIDI
    embedding_to_midi(output[1], 'test_output.mid')

    

    

