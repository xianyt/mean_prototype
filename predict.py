#!/usr/bin/python
#  -*- coding: utf-8 -*-

import argparse
import json
import logging
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from losses import proto_loss
from meters import Meters
from models.proto_net import ProtoNet
from train import create_data_iter, parameters_string
from utils import load_vocab, load_idx_data, pad_seq

from mpl_toolkits.mplot3d import Axes3D


def parse_args():
    parser = argparse.ArgumentParser()

    """
    inputs
    """
    parser.add_argument('--dir', type=str, required=True, help='path to dataset')
    parser.add_argument('--vocab', type=str, default='vocab', help='vocabulary file')
    parser.add_argument('--test', type=str, default='test.idx', help='test file contains word indices'),
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint')
    parser.add_argument('--max-seq-len', type=int, default=-1, help='max length of text')
    parser.add_argument('--log-freq', type=int, default=1, help='log frequency')

    parser.add_argument('--plt', action='store_true', help='plot')
    parser.add_argument('--plt-show', action='store_true', help='plot')
    parser.add_argument('--plt-name', type=str, default='fig.png', help='plot fig name')

    parser.add_argument('--att-json', type=str, default=None, help='attention JSON file')

    """
    test
    """
    parser.add_argument('--cuda', action='store_true', help='use cuda device')
    parser.add_argument('--samples-per-class', type=int, default=32, help='number of examples per class')
    parser.add_argument('--support-num', type=int, default=16, help='number of support examples per class')

    return parser.parse_args()


def load_test_data(args):
    word2idx, vocab = load_vocab(path.join(args.dir, args.vocab))
    test_x, test_y = load_idx_data(path.join(args.dir, args.test))
    return test_x, test_y, word2idx, vocab


def predict(model, args, ckpt_args, test_seqs, test_labels, word2idx):
    # switch to eval mode
    model.eval()

    meters = Meters()

    fig, ax = None, None
    if args.plt and (ckpt_args.proto_dim == 2 or ckpt_args.proto_dim == 3):
        fig = plt.figure()
        if ckpt_args.proto_dim == 2:
            ax = fig.add_subplot()
        else:
            ax = fig.add_subplot(111, projection='3d')

        # colors = np.arange(start=1, stop=ckpt_args.class_num + 1) / (ckpt_args.class_num + 1)
        colors = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
            'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
        ]
        if args.plt_show:
            fig.show()

    att_json = None
    if args.att_json:
        att_json = []

    # eval loop in an epoch
    batch_size = args.samples_per_class * ckpt_args.class_num
    data_iter = create_data_iter(test_seqs, test_labels, batch_size)

    for x_eval, y_eval, batch, batch_num in data_iter:
        with torch.no_grad():
            x_eval_len = [
                len(s)
                if args.max_seq_len <= 0 or len(s) <= args.max_seq_len
                else args.max_seq_len
                for s in x_eval
            ]
            max_seq_len = max(x_eval_len)
            x_eval_len = Variable(torch.FloatTensor(x_eval_len))

            x_eval = [pad_seq(s, max_seq_len) for s in x_eval]
            x_eval = Variable(torch.LongTensor(x_eval).t())
            if args.cuda:
                x_eval = x_eval.cuda()
                x_eval_len = x_eval_len.cuda()

            y_eval = Variable(torch.LongTensor(y_eval))
            if args.cuda:
                y_eval = y_eval.cuda(async=True)

            # prototypical loss
            prob, protos, feats, att = model(x_eval, x_eval_len)

            # all samples are query sample
            p_loss, p_acc = proto_loss(prob, y_eval, protos, ckpt_args.disable_reg)
            meters.update('loss', p_loss)
            meters.update('acc', p_acc)

            if (batch + 1) % args.log_freq == 0 or batch + 1 == batch_num:
                msg = ' TEST [{:3d}/{:3d}] {:.4f}'
                log.info(msg.format(batch + 1, batch_num, meters))

            if args.plt and (ckpt_args.proto_dim == 2 or ckpt_args.proto_dim == 3):
                c = [colors[int(l.item())] for l in y_eval]
                if args.cuda and torch.cuda.is_available():
                    feats = feats.cpu()
                feats = feats.numpy()
                if ckpt_args.proto_dim == 2:
                    ax.scatter(feats[:, 0], feats[:, 1], c=c, s=10, alpha=0.4)
                else:
                    ax.scatter(feats[:, 0], feats[:, 2], feats[:, 1], c=c, s=10, alpha=0.4)
                if args.plt_show:
                    fig.show()

            if att_json is not None:
                p, y_pred = torch.max(prob, dim=-1)
                for i, pp in enumerate(p.tolist()):
                    if y_pred[i] == y_eval[i]:
                        seq = x_eval[:, i].tolist()
                        seq = [word2idx.get(w, '<unk>') for w in seq if w > 0]
                        seq_att = [a.tolist() for a in att[i]]

                        att_json.append({
                            "seq": seq,
                            "att": seq_att,
                            "y_pred": y_pred[i].item(),
                            "y_prob": pp,
                            "y": y_eval[i].item()
                        })

    if args.plt and (ckpt_args.proto_dim == 2 or ckpt_args.proto_dim == 3):
        protos = model.protos
        if args.cuda and torch.cuda.is_available():
            protos = protos.cpu()
        protos = protos.numpy()
        if ckpt_args.proto_dim == 2:
            proto_x, proto_y = protos[:, 0], protos[:, 1]
            ax.scatter(proto_x, proto_y, marker="P", c="black", s=50, alpha=0.9)
        else:
            proto_x, proto_y, proto_z = protos[:, 0], protos[:, 2], protos[:, 1]
            ax.scatter(proto_x, proto_y, proto_z, marker="P", c="black", s=100, alpha=0.9)
        fig.savefig(args.plt_name)
        if args.plt_show:
            fig.show()

    if att_json is not None:
        with open(args.att_json, 'w', encoding='utf-8') as f_json:
            f_json.write("const att = ")
            f_json.write(json.dumps(att_json, indent=2))

    return meters.val('loss'), meters.val('acc')


def main(args):
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")

    log.info("creating model ...")
    ckpt = torch.load(args.ckpt)
    ckpt_args = ckpt['args']
    model = ProtoNet(ckpt_args)
    model = model.to(device)

    print(model)
    log.info(parameters_string(model))

    log.info("loading model parameters from {}".format(args.ckpt))
    model.load_state_dict(ckpt['model_state_dict'])

    test_x, test_y, word2idx, vocab = load_test_data(args)
    idx2word = {kv[1]: kv[0] for kv in word2idx.items()}

    predict(model, args, ckpt_args, test_x, test_y, idx2word)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('main')

    torch.manual_seed(0)
    np.random.seed(0)

    main(parse_args())
    exit(0)
