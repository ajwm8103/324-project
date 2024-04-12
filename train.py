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


def main():
    args = fetch_arguments()

    data_embedding = load_embedding_from_pickle(args)
    for i in range(len(data_embedding)):
        print(data_embedding[i].shape)
    
    

    # Time to try training with a transformer model


def fetch_arguments():
    parser = argparse.ArgumentParser(description='Transformer Training Script')
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--nhead', type=int, default=16, help='Number of heads in the transformer model')
    parser.add_argument('--num_encoder_layers', type=int, default=12, help='Number of encoder layers in the transformer model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    args = parser.parse_args()

    args.log = print if args.verbose else lambda x: None
    return args

if __name__ == '__main__':
    main()
