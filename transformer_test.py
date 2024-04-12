import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import mido  # MIDI library for Python
import math
from torch.nn.utils.rnn import pad_sequence


def midi_to_tokens(midi_path):
    midi = mido.MidiFile(midi_path)
    tokens = []
    for track in midi.tracks:
        for msg in track:
            if not msg.is_meta and msg.type in ['note_on', 'note_off']:
                # Encode 'note_on' as 1 and 'note_off' as 0
                msg_type = 1 if msg.type == 'note_on' else 0
                tokens.append((msg_type, msg.note, msg.velocity, msg.time))
    return tokens

class MIDIDataset(Dataset):
    def __init__(self, midi_files):
        self.data = [midi_to_tokens(file) for file in midi_files]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Tokenize each MIDI message into integers
        token_data = [item for sublist in self.data[idx] for item in sublist]
        return token_data

    @staticmethod
    def collate_fn(batch):
        batch = [torch.tensor(item, dtype=torch.long) for item in batch]
        batch = pad_sequence(batch, batch_first=True, padding_value=0)  # Add padding value as needed
        return batch

# Example usage
midi_files = ['data/maestro-v3.0.0/2018/MIDI-Unprocessed_Schubert10-12_MID--AUDIO_20_R2_2018_wav.midi', 'data/maestro-v3.0.0/2018/MIDI-Unprocessed_Schubert10-12_MID--AUDIO_18_R2_2018_wav.midi']
dataset = MIDIDataset(midi_files)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=MIDIDataset.collate_fn)

import torch.nn as nn

class MidiTransformer(nn.Module):
    def __init__(self, num_tokens, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(MidiTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, batch_first=True)
        self.fc_out = nn.Linear(d_model, num_tokens)

    def make_src_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer(src, src, src_mask)
        output = self.fc_out(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

from torch.optim import Adam

def train(model, data_loader, epochs=10):
    model.train()
    optimizer = Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(data_loader):
            print(f"Processing batch {i+1}/{len(data_loader)} for epoch {epoch+1}")
            optimizer.zero_grad()
            src_mask = model.make_src_mask(batch.size(1))
            output = model(batch, src_mask)
            loss = criterion(output.view(-1, num_tokens), batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")

# Example usage
# model = MidiTransformer(num_tokens=256)  # Assuming 256 different tokens
model = MidiTransformer(num_tokens=256, d_model=256, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=1024)
train(model, loader)

