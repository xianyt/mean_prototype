#! -*- coding: utf-8 -*-
import torch

import torch.nn as nn
from torch.distributions import Uniform
from constants import PAD


class EmbeddingLayer(nn.Module):

    def __init__(self, args, pre_trained_embed):
        super(EmbeddingLayer, self).__init__()
        self.word_dropout = args.word_dropout
        self.word_permute = args.word_permute
        self.uniform = Uniform(torch.tensor([0.0]), torch.tensor([1.0]))

        if pre_trained_embed is not None:
            weights = torch.FloatTensor(pre_trained_embed)
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=PAD)
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embed_dim, padding_idx=PAD)

        if args.embed_dropout > 0:
            self.embed_drop = nn.Dropout(args.embed_dropout)
        else:
            self.embed_drop = None

    def _drop_words(self, inputs, inputs_len):
        mask = torch.zeros_like(inputs)
        if inputs.get_device() >= 0:
            mask = mask.cuda()

        for i, ll in enumerate(inputs_len):
            ll = int(ll.item())
            drop = self.uniform.sample((ll,)) < self.word_dropout
            mask[:ll, i] = torch.squeeze(drop, dim=-1)

        return torch.where(mask > 0, mask, inputs)

    def _rand_perm_with_constraint(self, inputs, inputs_len, k):
        """
        Randomly permutes words ensuring that words are no more than k positions
        away from their original position.
        """
        device = 'cuda' if inputs.get_device() >= 0 else None
        for i, l in enumerate(inputs_len):
            length = int(l.item())
            offset = torch.squeeze(self.uniform.sample((length,)), dim=-1) * (k + 1)
            if inputs.get_device() >= 0:
                offset = offset.cuda()
            new_pos = torch.arange(length, dtype=torch.float, device=device) + offset
            inputs[:length, i] = torch.take(inputs[:length, i], torch.argsort(new_pos))
        return inputs

    def forward(self, inputs, inputs_len):
        if self.word_dropout > 0 and self.training:
            inputs = self._drop_words(inputs, inputs_len)

        if self.word_permute > 0 and self.training:
            inputs = self._rand_perm_with_constraint(inputs, inputs_len, self.word_permute)

        embed = self.embedding(inputs)
        if self.embed_drop:
            embed = self.embed_drop(embed)

        return embed
