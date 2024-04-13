import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from train import fetch_arguments
from embedding import load_embedding_from_pickle, embedding_to_midi
from train import MidiDataset, MidiDatasetToken, collate_fn
from transformer import TransformerModel
import numpy as np
import note_seq, pretty_midi

def test_token_model_from_file(args, filename='models/latest_model_1_token.pth'):
    print('Testing!')
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    ntokens = 388  # size of vocabulary
    emsize = 64  # embedding dimension
    d_hid = 128 # dimension of the feedforward network model in ``nn.TransformerEncoder``
    nlayers = 4  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
    nhead = 4  # number of heads in ``nn.MultiheadAttention``
    dropout = 0.2  # dropout probability
    model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
    
    model.load_state_dict(torch.load(filename, map_location=device))
    model.to(device)

    src = torch.zeros((10,1), dtype=torch.long, device=device)
    decoded = model.decode(src, max_len=400)
    print(decoded)

    # Create decoder, loop over songs and events
    encoder_decoder = note_seq.OneHotIndexEventSequenceEncoderDecoder(note_seq.PerformanceOneHotEncoding())
    for index_out in decoded:
        note_performance = note_seq.Performance(steps_per_second=100)
        for index in index_out:
            event = encoder_decoder.class_index_to_event(class_index=index.item(), events=[])
            note_performance.append(event)
            #print(index, index_out)
        #encoder_decoder.decode_event()
    print(note_performance)

    generated_sequence = note_performance.to_sequence(
        max_note_duration=5.0)

    print(generated_sequence, type(generated_sequence))

    note_seq.sequence_proto_to_midi_file(generated_sequence, 'outputs/token/generated_sequence.mid')

def test_model_from_file(dataloader, args, filename="models/latest_model_1.pth"):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = nn.Transformer(d_model=5, nhead=args.nhead, num_encoder_layers=args.num_encoder_layers).to(device)
    
    model.load_state_dict(torch.load(filename, map_location=device))
    model.to(device)

    srcs, tgts = next(iter(dataloader))
    src = srcs[0].to(device)
    tgt = tgts[0].to(device)
    
    output = model(src, tgt)  # tgt[:-1] used as target input to predict tgt[1:]
    args.log("Sample Input:", src[0:5])
    args.log("Sample Target:", tgt[0:5])
    output[:, 0] = output[:, 0]
    output[:, 1] = output[:, 1]
    output[:, 2] = output[:, 2]
    output[:, 3] = output[:, 3] * 127
    output[:, 4] = output[:, 4] * 127
    args.log("Model Output:", output[0:5])

    # Save to a midi file
    embedding = output#.cpu().detach().numpy()
    embedding_to_midi(embedding, filename='outputs/continuous/model_output_2.mid')

if __name__ == "__main__":
    args = fetch_arguments()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    data_embedding = load_embedding_from_pickle(args)

    if args.embedding == 'token':
        test_token_model_from_file(args)
    else:
        dataset = MidiDataset(data_embedding, args.seq_len, args, args.stride_length)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        test_model_from_file(dataloader, args)