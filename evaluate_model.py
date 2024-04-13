import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from train import fetch_arguments
from embedding import load_embedding_from_pickle, embedding_to_midi
from train import MidiDataset, MidiDatasetToken, collate_fn
from pipelines import PerformanceExtractor, extract_performances, Quantizer, TranspositionPipeline, EncoderPipeline
from transformer import TransformerModel
import numpy as np
import random
import note_seq, pretty_midi

def tokenize_note_sequence(note_sequence):
    '''
    Set up the encoder and decoder, and return the tokenized sequence
    '''
    if isinstance(note_sequence, note_seq.NoteSequence):
        note_sequence = [note_sequence]
    perf_extractor = PerformanceExtractor(
        min_events=32,
        max_events=512,
        num_velocity_bins=0,
        note_performance=False,
    )
    note_sequence = [perf_extractor.transform(n) for n in note_sequence]
    note_sequence = [n for ns in note_sequence for n in ns] # Flatten

    #print(len(note_sequence), type(note_sequence[0]))

    encoder_decoder = note_seq.OneHotIndexEventSequenceEncoderDecoder(note_seq.PerformanceOneHotEncoding())
    #encoder_decoder = note_seq.EventSequenceEncoderDecoder(note_seq.PerformanceOneHotEncoding())
    encoder_pipeline = EncoderPipeline(encoder_decoder=encoder_decoder, control_signals=None,
                                        optional_conditioning=False)
    note_sequence = [encoder_pipeline.transform(n) for n in note_sequence]

    print(note_sequence)

    return note_sequence
    
def token_seq_to_midi(token_seq, filename='outputs/token/converted.mid'):
    '''
    Convert the tokenized sequence back to a midi file
    '''
    encoder_decoder = note_seq.OneHotIndexEventSequenceEncoderDecoder(note_seq.PerformanceOneHotEncoding())
    for index_out in token_seq:
        note_performance = note_seq.Performance(steps_per_second=100)
        for index in index_out:
            event = encoder_decoder.class_index_to_event(class_index=index.item(), events=[])
            note_performance.append(event)
            #print(index, index_out)
        #encoder_decoder.decode_event()

    generated_sequence = note_performance.to_sequence(
        max_note_duration=5.0)

    note_seq.sequence_proto_to_midi_file(generated_sequence, filename)

def test_token_model_from_file(emb, args, filename='models/latest_model_1_token.pth', i=0):
    '''
    Takes in a transformer model and generates midi files for your pleasure
    '''
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
    src = emb.to(device)
    # Cast src as long tensor
    src = src.long()
    src = src.unsqueeze(0)
    decoded = model.decode(src, max_len=400)
    #print(decoded)

    print(src.shape, decoded.shape)
    src_and_decoded = torch.cat((src, decoded), dim=1)

    # Just the suffix of the midi
    token_seq_to_midi(decoded, f'outputs/token/generated_sequence_{i}_suffix.mid')

    # Extended midi
    token_seq_to_midi(src_and_decoded, f'outputs/token/generated_sequence_{i}_extended.mid')

    token_seq_to_midi(src, f'outputs/token/generated_sequence_{i}_src.mid')
    

def test_model_from_file(dataloader, args, filename="models/latest_model_1.pth"):
    '''
    Takes in a transformer model trained on continuous tokens and generates midi files, as will as their inputs (to see how the model works)
    '''
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = nn.Transformer(d_model=5, nhead=args.nhead, num_encoder_layers=args.num_encoder_layers).to(device)
    
    model.load_state_dict(torch.load(f"models/{filename}", map_location=device))
    model.to(device)

    batches = list(dataloader)
    for i in range(8):
        random_batch = random.choice(batches)
        srcs, tgts = random_batch
        src = srcs[0].to(device)
        tgt = tgts[0].to(device)
        
        output = model(src, tgt)  # tgt[:-1] used as target input to predict tgt[1:]
        args.log("Sample Input:", src[0:5])
        args.log("Sample Target:", tgt[0:5])
        args.log("Model Output:", output[0:5])

    # Save to a midi file
    embedding = output#.cpu().detach().numpy()
    embedding_to_midi(embedding, filename='outputs/continuous/model_output_2.mid')

if __name__ == "__main__":
    '''
    This file is for testing models
    '''
    args = fetch_arguments()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    data_embedding = load_embedding_from_pickle(args)

    if args.embedding == 'token':

        for i in range(10):
            test_token_model_from_file(data_embedding[i], args, i=i)
    else:
        dataset = MidiDataset(data_embedding, args.seq_len, args, args.stride_length)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        test_model_from_file(dataloader, args)