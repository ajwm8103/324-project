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
    parser.add_argument('--train_count', type=int, default=10000, help='Number of training iterations. Each epoch is about 480000 data points')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--nhead', type=int, default=5, help='Number of heads in the transformer model')
    parser.add_argument('--num_encoder_layers', type=int, default=12, help='Number of encoder layers in the transformer model') # TODO: Check this
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data', type=str, default='data', help='Name of data')

    args = parser.parse_args()

    args.log = print if args.verbose else lambda *x, **y: None
    return args

class MidiDataset(Dataset):
    # In order to capture the elements of music I am combining all tracks together. This means that the Transformer model will loose the "uniqueness" of any individual song.
    def __init__(self, data, seq_len, args, stride_length=1):
        self.seq_len = seq_len
        # slice the data into size seq_len using a sliding window:
        data_sliced = []
        for song in data:
            data_slice = np.array([song[i:i + seq_len + 1, :] for i in range(0, len(song) - seq_len, stride_length)])
            data_sliced.extend(data_slice)
        data_sliced = np.array(data_sliced)
        self.data = torch.tensor(data_sliced)
        args.log("Data shape:", self.data.shape)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32, device=self.device)

def main():
    args = fetch_arguments()

    data_embedding = load_embedding_from_pickle(args)
    dataset = MidiDataset(data_embedding, args.seq_len, args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Training on", device)
    model = nn.Transformer(d_model=5, nhead=args.nhead, num_encoder_layers=args.num_encoder_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for training_iteration in range(args.train_count // args.batch_size):
        model.train()
        t = tqdm(dataloader, desc='Loss: N/A')
        for src, tgt in t:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()
            #print(loss.item())
            t.set_description(f'Loss: {loss.item()}')
            t.refresh()


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
