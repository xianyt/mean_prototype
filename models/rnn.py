#! -*- coding: utf-8 -*-
import torch

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .embed import EmbeddingLayer


class RNN(nn.Module):

    def __init__(self, rnn_type, embed_dim, hid_dim, dropout, layer_num, pooling, rnn_bidir):
        super(RNN, self).__init__()

        self.rnn_type = rnn_type
        self.rnn_bidir = rnn_bidir
        self.layer_num = layer_num
        self.hid_dim = hid_dim
        self.pooling = pooling

        rnn_drop = dropout if layer_num > 1 else 0
        rnn = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn(embed_dim, hid_dim, dropout=rnn_drop, num_layers=layer_num, bidirectional=rnn_bidir)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, inp, inp_len, mask):
        if self.rnn_bidir:
            inp_len, idx = torch.sort(inp_len, descending=True)
            inp = torch.index_select(inp, dim=1, index=idx)

            inp = pack_padded_sequence(inp, inp_len)
            outp, _ = self.rnn(inp)
            outp, _ = pad_packed_sequence(outp)

            _, rev_idx = torch.sort(idx, descending=False)
            outp = torch.index_select(outp, dim=1, index=rev_idx)
        else:
            outp, _ = self.rnn(inp)

        if self.dropout:
            outp = self.dropout(outp)

        max_idx = None
        if self.pooling == 'mean':
            pool_val = torch.mean(outp, 0).squeeze()
        elif self.pooling == 'max':
            # opt 1:
            # pool_val, max_idx = torch.max(outp, 0)
            # opt 2:
            # outp = F.adaptive_max_pool1d(outp.permute(1, 2, 0), 1).view(inp.size(1), -1)

            # adaptive max pooling by hand
            # collect all the non padded elements of the batch and then take max of them
            # outp = torch.cat(
            #     [torch.max(i[:l], dim=0)[0].view(1, -1) for i, l in zip(outp.permute(1, 0, 2), inp_len)],
            #     dim=0)

            # opt 3:
            mask = ((1 - mask) * 1e10).unsqueeze(-1)
            pool_val, max_idx = torch.max(outp - mask, 0)
        else:
            raise Exception('Error when initializing Classifier')

        return pool_val, max_idx


class RnnEncoder(nn.Module):

    def __init__(self, args, pre_trained_embed):
        super(RnnEncoder, self).__init__()

        self.embedding = EmbeddingLayer(args, pre_trained_embed)

        self.rnn = RNN(args.rnn_type, args.embed_dim, args.rnn_dim, args.rnn_dropout,
                       args.rnn_layer_num, args.rnn_pool, args.rnn_bidir)

        # self.dropout = nn.Dropout(args.rnn_dropout) if args.rnn_dropout > 0 else None

        self.rnn_dim = args.rnn_dim
        self.rnn_bidir = args.rnn_bidir

    @property
    def enc_dim(self):
        # return self.rnn_dim
        return self.rnn_dim * (2 if self.rnn_bidir else 1)

    def forward(self, inputs, inputs_len):
        mask = (inputs > 0).float()
        embed = self.embedding(inputs, inputs_len)
        enc, outp = self.rnn(embed, inputs_len, mask)

        # if self.dropout:
        #     enc = self.dropout(enc)

        return enc, outp
