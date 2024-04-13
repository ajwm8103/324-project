import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from tqdm import tqdm

import note_seq, pretty_midi
from note_seq import midi_io
from embedding import load_embedding_from_pickle
from transformer import TransformerModel
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def fetch_arguments():
    parser = argparse.ArgumentParser(description='Transformer Training Script')
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-t", "--tiny", action="store_true", help="Make dataset tiny")
    parser.add_argument('--train_count', type=int, default=10000, help='Number of training iterations. Each epoch is about 480000 data points')
    parser.add_argument('--seq_len', type=int, default=64, help='Sequence length')
    parser.add_argument('--stride_length', type=int, default=64, help='Stride length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--nhead', type=int, default=5, help='Number of heads in the transformer model')
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
    
def collate_fn(batch):
    inputs, targets = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    return inputs_padded, targets_padded

def calculate_loss(model, data):
    criterion = nn.MSELoss()
    model.eval() # Switch to evaluation mode
    k = 0
    loss_total = 0
    with torch.no_grad():
        for src, tgt in data:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt)
            loss = criterion(output, tgt)
            loss_total += loss.item()
            k += 1
            if k > 30: break
    model.train() # Switch back to training mode
    return loss_total / k

def initialize_model(args):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Training on", device)
    d_model = 5 if args.embedding == 'continuous' else 1
    model = nn.Transformer(d_model=d_model, nhead=args.nhead, num_encoder_layers=args.num_encoder_layers).to(device)
    return model, device


def train_continuous(args):
    # Get embedding
    data_embedding = load_embedding_from_pickle(args)


    data_train, data_test = train_test_split(data_embedding, test_size=0.10, random_state=42)
    # Dataset
    dataset_train = MidiDataset(data_train, args.seq_len, args, args.stride_length)
    dataset_test = MidiDataset(data_test, args.seq_len, args, args.stride_length)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model, device = initialize_model(args)
    if not(args.from_model == None):
        print("Load model from file")
        model.load_state_dict(torch.load(f'models/{args.from_model}.pth', map_location=device))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    
    num_iterations_to_train = args.train_count // args.batch_size
    try:
        iters, train_loss, train_acc, test_acc = [], [], [], []
        iter_count = 0
        for training_iteration in range(num_iterations_to_train):
            model.train()
            t = tqdm(dataloader_train, desc='Loss: N/A')
            for src, tgt in t:
                src, tgt = src.to(device), tgt.to(device)
                optimizer.zero_grad()
                output = model(src, tgt)
                loss = criterion(output, tgt)
                loss.backward()
                optimizer.step()
                #print(loss.item())
                
                iter_count += 1
                if iter_count % 100 == 0:
                    t.set_description(f'Loss: {loss.item()}')
                    t.refresh()
                    iters.append(iter_count)
                    ta = calculate_loss(model, dataloader_train)
                    va = calculate_loss(model, dataloader_test)
                    train_loss.append(float(loss))
                    train_acc.append(ta)
                    test_acc.append(va)
                    print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)
    except KeyboardInterrupt:
        print("Training interrupted.")

    # H2: Save our model:
    torch.save(model.state_dict(), f'models/{args.model}.pth')

    model.eval()
    with torch.no_grad():
        src, tgt = next(iter(dataloader_train))
        src, tgt = src.to(device), tgt.to(device)
        output = model(src, tgt)  # tgt[:-1] used as target input to predict tgt[1:]
        print("Sample input:", src[0])
        print("Model output:", output[0])
        print("True target:", tgt[1][0])
    
    plt.figure()
    plt.plot(iters[:len(train_loss)], train_loss)
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig('figures/continuous/train loss.png')

    plt.figure()
    plt.plot(iters[:len(train_acc)], train_acc)
    plt.plot(iters[:len(test_acc)], test_acc)
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(["Train", "Test"])
    plt.savefig('figures/continuous/train test acc.png')


def calculate_loss_token(model, data):
    criterion = nn.CrossEntropyLoss()
    model.eval() # Switch to evaluation mode
    k = 0
    loss_total = 0
    with torch.no_grad():
        for src, tgt in data:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            output_flat = output.view(-1, 388)
            loss = criterion(output_flat, tgt.reshape(-1).contiguous())
            loss_total += loss.item()

            k += 1
            if k > 30: break
    model.train() # Switch back to training mode
    return loss_total / k

def train_token(args):
    # Get embedding
    data_embedding = load_embedding_from_pickle(args)

    data_train, data_test = train_test_split(data_embedding, test_size=0.10, random_state=42)

    # Dataset
    dataset_train = MidiDatasetToken(data_train, args.seq_len, args, args.stride_length)
    dataset_test= MidiDatasetToken(data_test, args.seq_len, args, args.stride_length)

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)

    # Other params
    d_model = 5 if args.embedding == 'continuous' else 1

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Training on", device)
    ntokens = 388  # size of vocabulary
    emsize = 64  # embedding dimension
    d_hid = 128 # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 4  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 4  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    try:
        model.train()
        t = tqdm(enumerate(dataloader_train), desc='Loss1: N/A', total=args.train_count // args.batch_size)
        iters, train_loss, train_acc, test_acc = [], [], [], []
        iter_count = 0
        for epoch in range(10000):
            for batch, (data, targets) in t:
                data, targets = data.to(device), targets.to(device)
                iter_count += 1
            
                output = model(data)
                output_flat = output.view(-1, ntokens)
                #print('data', data.shape, 'targets', targets.shape, 'output', output.shape, 'flat', output_flat.shape)
                loss = criterion(output_flat, targets.reshape(-1).contiguous())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                if iter_count % 100 == 0:
                    t.set_description(f'Loss: {loss.item()}')
                    t.refresh()
                    iters.append(iter_count)
                    ta = calculate_loss_token(model, dataloader_train)
                    va = calculate_loss_token(model, dataloader_test)
                    train_loss.append(float(loss))
                    train_acc.append(ta)
                    test_acc.append(va)
                    print(iter_count, epoch, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)

            torch.save(model.state_dict(), f'models/latest_{args.model}_token.pth')
    except KeyboardInterrupt:
        pass

    # H2: Save our model:
    torch.save(model.state_dict(), f'models/{args.model}_token.pth')

    plt.figure()
    plt.plot(iters[:len(train_loss)], train_loss)
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig('figures/token/train loss.png')

    plt.figure()
    plt.plot(iters[:len(train_acc)], train_acc)
    plt.plot(iters[:len(test_acc)], test_acc)
    plt.title("Loss over iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend(["Train", "Test"])
    plt.savefig('figures/token/train test acc.png')

    model.eval()
    with torch.no_grad():
        src, tgt = next(iter(dataloader_train))
        src, tgt = src.to(device), tgt.to(device)
        output = model(src)  # tgt[:-1] used as target input to predict tgt[1:]
        print("Sample input:", src[0])
        print("Model output:", output[0])
        print("True target:", tgt[1][0])

        # Generating an output
        src = torch.zeros((10,1), dtype=torch.long, device=device)
        #src = torch.randint(0, 388, (10, 32))  # Example input data [seq_len, batch_size]
        src_mask = model.generate_square_subsequent_mask(src.size(0)).to(model.device)
        decoded = model.decode(src, max_len=400)
        print(decoded)

def main():
    args = fetch_arguments()
    if args.embedding == 'token':
        train_token(args)
    elif args.embedding == 'continuous':
        train_continuous(args)

if __name__ == '__main__':
    main()
