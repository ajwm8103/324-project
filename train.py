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
    parser.add_argument('--stride_length', type=int, default=64, help='Stride length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--nhead', type=int, default=8, help='Number of heads in the transformer model')
    parser.add_argument('--num_encoder_layers', type=int, default=12, help='Number of encoder layers in the transformer model') # TODO: Check this
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data', type=str, default='data', help='Name of data')
    parser.add_argument('--model', type=str, default='model_1', help='Model name')
    parser.add_argument('--embedding', type=str, default='continuous', help='Embedding style')
    parser.add_argument('--from_model', type=str, default=None, help='Load model from file')

    args = parser.parse_args()

    args.log = print if args.verbose else lambda *x, **y: None
    return args

class MidiDataset(Dataset):
    # In order to capture the elements of music I am combining all tracks together. This means that the Transformer model will loose the "uniqueness" of any individual song.
    def __init__(self, data, seq_len, args, stride_length=100):
        self.seq_len = seq_len
        # slice the data into size seq_len using a sliding window:
        self.data = []
        for song in data:
            data_slice = np.array([song[i:i + seq_len + 1, :] for i in range(0, len(song) - seq_len, stride_length)])
            self.data.extend(data_slice)
        self.data = np.array(self.data)
        self.data = torch.tensor(self.data)
        args.log("Data shape:", self.data.shape)
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32, device=self.device)

class MidiDatasetToken(Dataset):
    def __init__(self, data, seq_len, args, stride_length=100):
        self.seq_len = seq_len
        self.data = []
        self.targets = []

        for song_tensor in data:
            # Ensure the tensor is on CPU for slicing (if not already)
            song_tensor = song_tensor.cpu()
            # Generate slices of `seq_len + 1` size
            for start_index in range(0, song_tensor.size(0) - seq_len, stride_length):
                end_index = start_index + seq_len + 1
                slice_ = song_tensor[start_index:end_index]
                if slice_.size(0) == seq_len + 1:
                    self.data.append(slice_[:-1])  # Input sequence
                    self.targets.append(slice_[1:])  # Target sequence

        self.data = torch.stack(self.data)  # Stack all the data tensors
        self.targets = torch.stack(self.targets)  # Stack all the target tensors

        args.log("Data shape:", self.data.shape)
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx].to(dtype=torch.long, device=self.device)  # Ensure data type and device
        y = self.targets[idx].to(dtype=torch.long, device=self.device)
        return x, y

def initialize_model(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Training on", device)
    d_model = 5 if args.embedding == 'continuous' else 1
    model = nn.Transformer(d_model=d_model, nhead=args.nhead, num_encoder_layers=args.num_encoder_layers).to(device)
    return model, device

def train_for_x_iterations(model, dataloader, device, iterations, args):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    try:
        for training_iteration in range(iterations):
            model.train()
            t = tqdm(dataloader, desc='Loss: N/A')
            for src, tgt in t:
                src, tgt = src.to(device), tgt.to(device)
                optimizer.zero_grad()
                output = model(src, tgt)
                loss = criterion(output, tgt)
                loss.backward()
                optimizer.step()
                t.set_description(f'Loss: {loss.item()}')
                t.refresh()
    except KeyboardInterrupt:
        print("Training interrupted.")


def main():
    args = fetch_arguments()

    # Get embedding
    data_embedding = load_embedding_from_pickle(args)

    # Dataset
    if args.embedding == 'continuous':
        dataset = MidiDataset(data_embedding, args.seq_len, args, args.stride_length)
    elif args.embedding == 'token':
        dataset = MidiDatasetToken(data_embedding, args.seq_len, args, args.stride_length)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model, device = initialize_model(args)
    if not(args.from_model == None):
        print("Load model from file")
        model.load_state_dict(torch.load(f'models/{args.from_model}.pth', map_location=device))
        
    
    num_iterations_to_train = args.train_count // args.batch_size
    train_for_x_iterations(model, dataloader, device, num_iterations_to_train, args)

    # H2: Save our model:
    torch.save(model.state_dict(), f'models/{args.model}.pth')

    model.eval()
    with torch.no_grad():
        src, tgt = next(iter(dataloader))
        src, tgt = src.to(device), tgt.to(device)
        output = model(src, tgt)  # tgt[:-1] used as target input to predict tgt[1:]
        print("Sample input:", src[0])
        print("Model output:", output[0])
        print("True target:", tgt[1][0])



    # Time to try training with a transformer model


if __name__ == '__main__':
    main()
