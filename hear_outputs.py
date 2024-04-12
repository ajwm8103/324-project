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
import random



def generate_sequence(model, src, steps=50, seq_length=128):
    model.eval()
    generated = src
    for step in range(steps):
        with torch.no_grad():
            output = model(generated, generated[:, :-1, :]) 
            
            if output.size(1) == 0:  # Adjust dimension check for batch_first
                raise ValueError(f"Output tensor is empty at step {step}. Check model configuration and input dimensions.")
            
            new_seq = output[:, -1, :].unsqueeze(1)  # Adjust indexing for batch_first
            generated = torch.cat((generated, new_seq), dim=1)  # Append along the sequence dimension for batch_first
            
            if generated.size(1) > seq_length:  # Check sequence length for batch_first
                generated = generated[:, -seq_length:, :]  # Keep only the most recent sequences

            # print(f"Step {step}: Generated shape: {generated.shape}")
    print(generated)
    return generated


# def generate_sequence(model, src, steps=50):
#     model.eval()
#     generated = src
#     for step in range(steps):
#         with torch.no_grad():
#             output = model(generated, generated[:, :-1, :]) 
            
#             if output.size(1) == 0:  # Adjust dimension check for batch_first
#                 raise ValueError(f"Output tensor is empty at step {step}. Check model configuration and input dimensions.")
            
#             new_seq = output[:, -1, :].unsqueeze(1)  # Adjust indexing for batch_first
#             generated = torch.cat((generated, new_seq), dim=1)  # Append along the sequence dimension for batch_first
            
#             print(f"Step {step}: Generated shape: {generated.shape}")
#     return generated


args = fetch_arguments()
data_embedding = load_embedding_from_pickle(args)
dataset = MidiDataset(data_embedding, args.seq_len)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Ensure batch_first is set correctly
model = nn.Transformer(
    d_model=5,
    nhead=args.nhead,
    num_encoder_layers=args.num_encoder_layers,
    batch_first=True  # Set batch_first to True
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Load the model:
model.load_state_dict(torch.load('transformer_midi_model.pth'))

# # # Generate a longer sequence
src, tgt = next(iter(dataloader))
src, tgt = src.to(device), tgt.to(device)

# Load all initial sequences into a list (if not too large, or adjust accordingly)
# initial_sequences = list(dataloader)

# Select a random initial sequence
# random_index = random.randint(0, len(initial_sequences) - 1)
# src, tgt = initial_sequences[random_index]
# src, tgt = src.to(device), tgt.to(device)

# Before starting the generation, check input dimensions
print("Input src shape:", src.shape)
if src.nelement() == 0:
    raise ValueError("Input src tensor is empty, check your DataLoader and dataset.")


generated_seq = generate_sequence(model, src, steps=50)  # Adjust steps for desired length

# Convert output to MIDI
embedding_to_midi(generated_seq, 'long_output_128steps.mid')
