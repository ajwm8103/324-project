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

def fetch_arguments():
    parser = argparse.ArgumentParser(description='Transformer Training Script')
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-t", "--tiny", action="store_true", help="Make dataset tiny")
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--nhead', type=int, default=5, help='Number of heads in the transformer model')
    parser.add_argument('--num_encoder_layers', type=int, default=12, help='Number of encoder layers in the transformer model') # TODO: Check this
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data', type=str, default='data', help='Name of data')

    args = parser.parse_args()

    args.log = print if args.verbose else lambda x: None
    return args

class MidiDataset(Dataset):
    # In order to capture the elements of music I am combining all tracks together. This means that the Transformer model will loose the "uniqueness" of any individual song.
    def __init__(self, data, seq_len, pad_value=0):
        self.max_len = max(len(x) for x in data)
        self.seq_len = seq_len
        self.pad_value = pad_value
        self.padded_data = torch.tensor([np.pad(x, ((0, self.max_len - len(x)), (0, 0)), 'constant', constant_values=self.pad_value) for x in data])
        self.padded_data = self.padded_data.reshape(-1, data[0].shape[1])  # Flatten the data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.padded_data = self.padded_data.to(self.device)

    def __len__(self):
        return len(self.padded_data) - self.seq_len

    def __getitem__(self, idx):
        x = self.padded_data[idx:idx + self.seq_len]
        y = self.padded_data[idx + 1:idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32, device=self.device)

def main():
    args = fetch_arguments()

    data_embedding = load_embedding_from_pickle(args)
    dataset = MidiDataset(data_embedding, args.seq_len)
    print(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on", device)
    model = nn.Transformer(d_model=5, nhead=args.nhead, num_encoder_layers=args.num_encoder_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for src, tgt in tqdm(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            #print(loss.item())

    # H2: Save our model:
    torch.save(model.state_dict(), 'transformer_midi_model.pth')

    model.eval()
    with torch.no_grad():
        src, tgt = next(iter(dataloader))
        src, tgt = src.to(device), tgt.to(device)
        output = model(src, tgt[:-1])  # tgt[:-1] used as target input to predict tgt[1:]
        print("Sample input:", src[0])
        print("Model output:", output[0])
        print("True target:", tgt[1][0])



    # Time to try training with a transformer model


if __name__ == '__main__':
    main()
