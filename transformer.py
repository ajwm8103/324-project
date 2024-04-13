import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

    def get_src_embeddings(self, src):
        src_x = self.src_embed(src)
        if self.src_pe is not None:
            src_x = self.src_pe(src_x)
        return src_x

    def get_tgt_embeddings(self, tgt):
        tgt_x = self.tgt_embed(tgt)
        if self.tgt_pe is not None:
            tgt_x = self.tgt_pe(tgt_x)
        return tgt_x

    def greedy_decode(self, src, target_sos, target_eos, max_len=100):
        """
        Do not call the encoder more than once, or you will lose marks.
        The model calls must be batched and the only loop should be over the sequence length, or you will lose marks.
        It will also make evaluating the debugging the model difficult if you do not follow these instructions.

        Hint: use torch.argmax to get the most likely token at each step and
        concatenate_generation_sequence to add it to the sequence.

        src: torch.Tensor, [batch_size, src_seq_len]
        target_sos: int, the start of sequence token
        target_eos: int, the end of sequence token
        max_len: int, the maximum length of the output sequence
        return: torch.Tensor, [batch_size, seq_len]
            Such that each sequence is padded with the padding token after the end of sequence token (if it exists)
            Hint: use the pad_generation_sequence function
        """
        # Encode source
        src_emb = self.get_src_embeddings(src)
        src_mask = self.create_pad_mask(src)
        src_x = self.encoder(src_emb, src_mask)

        # Create output array
        # tgt_generation: shape [batch_size, 1] full of target_sos
        tgt_generation = torch.zeros(src.shape[0], 1).fill_(target_sos).type_as(src).to(src.device)
        tgt_emb = self.get_tgt_embeddings(tgt_generation)

        for _ in range(max_len-1):
            # Update target mask
            tgt_pad_mask = self.create_pad_mask(tgt_generation)
            tgt_casual_mask = self.create_causal_mask(tgt_generation)
            # Combine padding and casual masks for decoder's self-attention
            tgt_mask = tgt_pad_mask | tgt_casual_mask
            
            # Get results from decoder
            #tgt_emb = self.get_tgt_embeddings(tgt_generation)
            # decoder_out: shape [batch_size, tgt_seq_len, vocab_size]
            #print(f'tgt_emb {tgt_emb.shape}, tgt_mask {tgt_mask.shape}, src_x {src_x.shape}, src_mask {src_mask.shape}')
            decoder_out = self.decoder(
                tgt_emb, tgt_mask,
                src_x, src_mask, True
            )

            # Find argmax token
            next_token = torch.argmax(decoder_out[:, -1, :], dim=-1)
            next_token = next_token.unsqueeze(1)

            # Concat to target generation
            # tgt_generation becomes: shape [batch_size, i+2]
            tgt_generation = self.concatenate_generation_sequence(
                tgt_generation, next_token
            )
            # Concat to target embeddings
            next_token_emb = self.get_tgt_embeddings(next_token)
            tgt_emb = torch.cat((tgt_emb, next_token_emb), dim=1)

            if self.all_finished(tgt_generation, target_eos, max_len):
                break
        
        # Pad with padding token after eos_token
        tgt_generation = self.pad_generation_sequence(tgt_generation, target_eos)
        return tgt_generation

    def decode(self, src, max_len=100):
        print(src.shape)
        #ys = torch.full(src.size(1), target_sos).type_as(src.data)
        ys = src
        for i in range(max_len-1):
            out = self.forward(ys, self.generate_square_subsequent_mask(ys.size(0)).to(self.device))
            #print(out.shape)
            prob = out[:, -1, :]
            next_word = torch.argmax(prob, dim=-1)
            #print(next_word, next_word.shape)
            next_word = next_word.unsqueeze(1)
            ys = torch.cat([ys, next_word.type_as(src.data)], dim=1)
        return ys

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask