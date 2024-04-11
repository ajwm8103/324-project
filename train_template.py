import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from tqdm import tqdm

import note_seq, pretty_midi
from note_seq import midi_io

# Dummy dataset that returns random src and tgt tensors
class RandomDataset(Dataset):
    def __init__(self, data_size, seq_len, num_classes):
        """
        Args:
            data_size: Total number of samples to generate
            seq_len: Length of each sequence
            num_classes: Vocab size
        """
        self.data_size = data_size
        self.seq_len = seq_len
        self.num_classes = num_classes

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        src = torch.rand(self.seq_len, self.num_classes)
        #tgt = torch.rand(self.seq_len, self.num_classes)
        tgt = torch.randint(0, self.num_classes, (self.seq_len,))
        return src, tgt

def train(args):
    # Get train data
    #train_data = torch.rand((100, 32, 512))
    train_data = RandomDataset(args.data_size, args.seq_len, args.num_classes)

    # Create dataloader
    train_loader = DataLoader(train_data,
                            batch_size=args.batch_size,
                            shuffle=True) # reshuffle minibatches every epoch

    


    # Create model (Transformer)
    model = nn.Transformer(d_model=args.num_classes, nhead=args.nhead, num_encoder_layers=args.num_encoder_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #src = torch.rand((10, 32, 512))
    #tgt = torch.rand((20, 32, 512))
    #out = model(src, tgt)

    # Create optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train loop
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    for epoch in tqdm(range(args.epochs)):
        model.train()

        for i, (src, tgt) in enumerate(train_loader):
            optimizer.zero_grad()

            # Assuming tgt input is shifted by one for the transformer's expected input
            tgt_input = tgt[:-1, :]
            outputs = model(src, tgt_input)
            # Compute loss; we'll need to adjust dimensions for CrossEntropy
            tgt_out = tgt[1:, :]  # Shifted by one for the expected output

            #loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_out.reshape(-1))
            # Assuming outputs is (sequence_length, batch_size, num_classes)
            # and tgt_out is (sequence_length, batch_size) with class indices
            print('outputs', outputs.shape, 'tgt_out', tgt_out.shape)
            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_out.view(-1))

            loss.backward()

            optimizer.step()
            total_loss += loss.item()

            iter_count += 1
            if iter_count % args.plot_every == 0:
                iters.append(iter_count)
                #ta = accuracy(model, train_data)
                #va = accuracy(model, val_data)
                ta, va = 0, 0
                train_loss.append(float(loss))
                train_acc.append(ta)
                val_acc.append(va)
                print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)

        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader)}")

def generate(model, src, max_len=20):
    model.eval()
    with torch.no_grad():
        outputs = [src]
        for _ in range(max_len):
            output = model(src, torch.cat(outputs, dim=0))
            outputs.append(output[-1, :, :].unsqueeze(0))
        return torch.cat(outputs, dim=0)

def main():
    parser = argparse.ArgumentParser(description='Transformer Training Script')
    parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--data_size', type=int, default=1000, help='Size of the dataset')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
    parser.add_argument('--num_classes', type=int, default=512, help='Size of feature vector')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--nhead', type=int, default=16, help='Number of heads in the transformer model')
    parser.add_argument('--num_encoder_layers', type=int, default=12, help='Number of encoder layers in the transformer model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()
    
    print('hi')

    if args.verbose:
        print("verbose")
    else:
        print("not verbose")
    
    train(args)

if __name__ == '__main__':
    main()
