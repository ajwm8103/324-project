import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import note_seq, pretty_midi
from note_seq import midi_io

def train(args):
    # Get train data
    train_data = torch.rand((100, 32, 512))

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True) # reshuffle minibatches every epoch

    # Create model (Transformer)
    model = nn.Transformer(nhead=16, num_encoder_layers=12)
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    out = model(src, tgt)

    # Create optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train loop
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0 # count the number of iterations that has passed

    for epoch in tqdm(range(args.num_epochs)):
        model.train()

        for i, (src, tgt) in enumerate(train_loader):
            optimizer.zero_grad()

            # Assuming tgt input is shifted by one for the transformer's expected input
            tgt_input = tgt[:-1, :]
            outputs = model(src, tgt_input)
            # Compute loss; we'll need to adjust dimensions for CrossEntropy
            tgt_out = tgt[1:, :]  # Shifted by one for the expected output

            loss = criterion(outputs.view(-1, outputs.size(-1)), tgt_out.reshape(-1))
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
                    

    args = parser.parse_args()
    
    print('hi')

    if args.verbose:
        print("verbose")
    else:
        print("not verbose")
    
    train(args)
