#! -*- coding: utf-8 -*-
import torch

import torch.nn as nn
from torch.nn import functional as F

from .rnn import RnnEncoder
from .losses import euclidean_dist


class ProtoNet(nn.Module):
    def __init__(self, args, pre_trained_embed=None):
        super(ProtoNet, self).__init__()

        self.class_num = args.class_num
        self.proto_dim = args.proto_dim

        self.support_num = args.support_num
        self.samples_per_class = args.samples_per_class
        self.query_num = self.samples_per_class - self.support_num

        self.prototypes = nn.Parameter(
            torch.zeros([args.class_num, args.proto_dim]),
            requires_grad=False)
        self.proto_decay = args.proto_decay

        self.encoder = RnnEncoder(args, pre_trained_embed)
        self.fc = nn.Linear(self.encoder.enc_dim, args.proto_dim)
        self.tanh = nn.Tanh()

    @property
    def protos(self):
        return self.prototypes

    def forward(self, inputs, inputs_len, alpha=None):

        enc, att = self.encoder(inputs, inputs_len)
        enc = self.tanh(self.fc(enc))

        if self.training:
            # inputs: sorted by class label
            # inputs: [class_num * (support_num + query_num), dim]
            assert enc.size()[0] == self.class_num * (self.support_num + self.query_num)

            # targets: [class_num * (support_num + query_num)]

            # [class_num, support_num, dim]
            supports = enc.view(self.class_num, self.samples_per_class, -1)[:, :self.support_num, :]
            # [class_num, query_num, dim]
            queries = enc.view(self.class_num, self.samples_per_class, -1)[:, -self.query_num:, :]

            # [class_num, dim]
            protos = torch.mean(supports, dim=1)
            self.prototypes = nn.Parameter(alpha * self.prototypes + (1 - alpha) * protos, requires_grad=False)
            # self.prototypes.detach_()

            # [class_num * query_num, dim]
            dists = euclidean_dist(queries.contiguous().view(self.class_num * self.query_num, -1), self.prototypes)
            cls_prob = F.softmax(-dists, dim=1)
        else:
            # [samples_per_episode, dim]
            dists = euclidean_dist(enc, self.prototypes)
            cls_prob = F.softmax(-dists, dim=1)

        return cls_prob, self.prototypes, enc, att
