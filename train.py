#!/usr/bin/python
#  -*- coding: utf-8 -*-
import argparse
import itertools
import math
import random
from os import path

from torch import optim, nn
from torch.autograd import Variable

from models.losses import proto_loss
from meters import Meters
from models.proto_net import ProtoNet
from utils import *

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('main')


def parse_args():
    parser = argparse.ArgumentParser()

    """
    inputs
    """
    parser.add_argument('--dir', type=str, required=True, help='path to dataset')
    parser.add_argument('--vocab', type=str, default='vocab', help='vocabulary file')
    parser.add_argument('--train', type=str, default='dev_train.idx', help='train file contains word indices')
    parser.add_argument('--dev', type=str, default='dev.idx', help='dev file contains word indices')
    parser.add_argument('--test', type=str, default='test.idx', help='test file contains word indices'),
    parser.add_argument('--class-num', type=int, default=2, help='class number of dataset')

    """
    model
    """
    parser.add_argument('--proto-dim', type=int, default=64, help='dimension of prototype')
    parser.add_argument('--proto-decay', type=float, default=0.99, help='prototype decay')

    parser.add_argument('--max-seq-len', type=int, default=-1, help='max length limit of text')
    parser.add_argument('--word-dropout', type=float, default=0,
                        help='randomly set a word to “UNK” with a probability')
    parser.add_argument('--word-permute', type=int, default=0,
                        help='randomly permutes words ensuring that words are no more than k positions')

    parser.add_argument('--embed-dim', type=int, default=100, help='embedding dimension')
    parser.add_argument('--embed-dropout', type=float, default=0.5, help='embedding dropout')

    parser.add_argument('--rnn-type', type=str, default="gru", choices=['gru', 'lstm'])
    parser.add_argument('--rnn-dim', type=int, default=384, help='RNN hidden dimension')
    parser.add_argument('--rnn-pool', type=str, default='max', choices=['max', 'mean'])
    parser.add_argument('--rnn-dropout', type=float, default=0.1, help='RNN dropout')
    parser.add_argument('--rnn-layer-num', type=int, default=1, help='layer number of RNN')
    parser.add_argument('--rnn-bidir', action='store_true', help='use bi-direction RNN')

    """
    train
    """
    parser.add_argument('--cuda', action='store_true', help='use cuda device')
    parser.add_argument('--epochs', type=int, default=50, help='max training epochs')
    parser.add_argument('--samples-per-class', type=int, default=64, help='number of examples per class')
    parser.add_argument('--support-num', type=int, default=32, help='number of support examples per class')
    parser.add_argument('--pre-trained-embed', type=str, required=False, help='pre-trained word embeddings')

    parser.add_argument('--early-stop-patience', type=int, default=5, help='early stop patience')
    parser.add_argument('--early-stop-monitor', type=str, default='acc', choices=['acc', 'loss'])

    parser.add_argument('--grad-clip', action='store_true', help='perform gradient clipping (maximum L2 norm of 1)')
    parser.add_argument('--disable-reg', action='store_true', help='disable reg')

    parser.add_argument('--optimizer', type=str, default="Adam", help='optimizers, Adam or SGD')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    parser.add_argument('--log-freq', type=int, default=100, help='log frequency')
    parser.add_argument('--ckpt', type=str, default='./ckpt', help='checkpoint directory')

    return parser.parse_args()


def load_data(args):
    word2idx, vocab = load_vocab(path.join(args.dir, args.vocab))

    train_x, train_y = load_idx_data(path.join(args.dir, args.train))

    dev_file = path.join(args.dir, args.dev)
    if path.isfile(dev_file):
        dev_x, dev_y = load_idx_data(dev_file)
    else:
        dev_x, dev_y = None, None

    test_x, test_y = load_idx_data(path.join(args.dir, args.test))

    return vocab, train_x, train_y, test_x, test_y, dev_x, dev_y


def accuracy(output, target):
    preds = torch.max(output, 1)[1]
    correct = torch.sum((preds == target).float())
    return correct.item() / len(target)


def create_episode_iter(seqs, labels, args):
    label_idx = [(i, l) for i, l in enumerate(labels)]
    label_idx = sorted(label_idx, key=lambda kv: kv[1])

    groups = {}
    for k, g in itertools.groupby(label_idx, key=lambda kv: kv[1]):
        groups[k] = [kv[0] for kv in g]

    assert len(groups) == args.class_num

    for batch in range(args.iters_per_epoch):
        batch_seqs = []
        batch_labels = []
        for k in groups.keys():
            idxs = random.sample(groups[k], args.samples_per_class)
            batch_seqs += [seqs[i] for i in idxs]
            batch_labels += [k for _ in idxs]

        # batch_seqs: [class_num * samples_per_class]
        yield batch_seqs, batch_labels, batch, args.iters_per_epoch


def create_data_iter(seqs, labels, batch_size, sort_by_len=False, repeat=False):
    batch_num = math.ceil(len(seqs) / batch_size)
    while True:
        shuf_idx = [i for i in range(len(seqs))]
        random.shuffle(shuf_idx)
        seqs = [seqs[i] for i in shuf_idx]
        labels = [labels[i] for i in shuf_idx]

        batch = 0
        while batch < batch_num:
            offset = batch * batch_size
            batch_seqs = seqs[offset: offset + batch_size]
            batch_labels = labels[offset: offset + batch_size]

            if sort_by_len:
                seqs_len = [len(s) for s in batch_seqs]
                len_idx = zip(seqs_len, range(len(batch_seqs)))
                sorted_len_idx = sorted(len_idx, key=lambda it: it[0], reverse=True)
                sorted_idx = [idx for _, idx in sorted_len_idx]

                batch_seqs = [batch_seqs[i] for i in sorted_idx]
                batch_labels = [batch_labels[i] for i in sorted_idx]

            yield batch_seqs, batch_labels, batch, batch_num
            batch += 1
        if repeat:
            continue
        else:
            break


def train(model, optimizer, args, epoch, train_x, train_y):
    # switch to train mode
    model.train()

    meters = Meters()

    # training loop in an epoch
    train_iter = create_episode_iter(train_x, train_y, args)
    for x_train, y_train, batch, batch_num in train_iter:

        # labeled data
        x_train_len = [
            len(s)
            if args.max_seq_len <= 0 or len(s) <= args.max_seq_len
            else args.max_seq_len
            for s in x_train
        ]
        max_seq_len = max(x_train_len)
        x_train_len = Variable(torch.FloatTensor(x_train_len))

        x_train = [pad_seq(s, max_seq_len) for s in x_train]
        x_train = Variable(torch.LongTensor(x_train).t())
        if args.cuda:
            x_train = x_train.cuda()
            x_train_len = x_train_len.cuda()

        y_train = Variable(torch.LongTensor(y_train))
        if args.cuda:
            y_train = y_train.cuda()

        # training losses
        losses = []

        # prototypical loss
        step = epoch * batch_num + batch
        # alpha = min(1 - 1 / max((step + 1) - 100, 1), args.proto_decay)
        alpha = min(1 - 1 / (step + 1), args.proto_decay)

        prob, protos, feats, att = model(x_train, x_train_len, alpha)

        query_num = args.samples_per_class - args.support_num
        y_query = y_train.view(args.class_num, args.samples_per_class)[:, -query_num:]
        y_query = y_query.contiguous().view(-1)

        if args.disable_reg:
            loss_val, acc_val = proto_loss(prob, y_query, protos, True)
            meters.update('proto', loss_val)
            losses.append(loss_val)
        else:
            loss_val, reg_val, acc_val = proto_loss(prob, y_query, protos)
            meters.update('proto', loss_val)
            losses.append(loss_val)
            meters.update('reg', reg_val)
            losses.append(reg_val)

        # sum all losses
        loss = sum(losses)
        meters.update('loss', loss)
        meters.update('acc', acc_val)

        if (batch + 1) % args.log_freq == 0 or batch + 1 == batch_num:
            msg = 'epoch: [{:2d}/{:2d}] [{:3d}/{:3d}] {:.4f}'
            log.info(msg.format(epoch + 1, args.epochs, batch + 1, batch_num, meters))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()


def evaluate(model, args, epoch, dev_seqs, dev_labels, tag='Eval'):
    # switch to eval mode
    model.eval()

    meters = Meters()

    # eval loop in an epoch
    batch_size = args.samples_per_class * args.class_num
    data_iter = create_data_iter(dev_seqs, dev_labels, batch_size)

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
                y_eval = y_eval.cuda()

            # prototypical loss
            prob, protos, feats, att = model(x_eval, x_eval_len)

            losses = []
            if args.disable_reg:
                loss_val, acc_val = proto_loss(prob, y_eval, protos, True)
                meters.update('proto', loss_val)
                losses.append(loss_val)
            else:
                loss_val, reg_val, acc_val = proto_loss(prob, y_eval, protos)
                meters.update('proto', loss_val)
                losses.append(loss_val)
                meters.update('reg', reg_val)
                losses.append(reg_val)

            # sum all losses
            loss = sum(losses)
            meters.update('loss', loss)
            meters.update('acc', acc_val)

            if (batch + 1) % args.log_freq == 0 or batch + 1 == batch_num:
                msg = ' {} epoch: [{:2d}/{:2d}] [{:3d}/{:3d}] {:.4f}'
                log.info(msg.format(tag, epoch + 1, args.epochs, batch + 1, batch_num, meters))

    return meters.val('loss'), meters.val('acc')


def main(args):
    log.info("loading data from: %s", args.dir)
    vocab, train_x, train_y, test_x, test_y, dev_x, dev_y = load_data(args)

    # set vocab_size
    args.vocab_size = len(vocab)

    # set iters_per_epoch
    args.iters_per_epoch = len(train_y) // (args.samples_per_class * args.class_num)

    log.info("vocab: %d, train: %d, dev: %d, test: %d",
             len(vocab), len(train_y), len(dev_y), len(test_y))

    embed_weights = None
    if args.pre_trained_embed:
        embed_weights = load_embeddings(vocab, args.pre_trained_embed, args.embed_dim)

    # create model
    model = ProtoNet(args, embed_weights)
    device = torch.device("cuda:0" if args.cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # model = nn.DataParallel(model)

    # print settings
    print_table([(k, str(v)) for k, v in vars(args).items()])

    # print(model)
    log.info(parameters_string(model))

    # build optimizer from args
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
    else:
        raise Exception('For other optimizers, please add it yourself. '
                        'supported ones are: SGD and Adam.')

    eval_best = {'acc': 0.0, 'loss': 0.0, 'epoch': 0}
    early_stop_patience = args.early_stop_patience
    for epoch in range(args.epochs):

        # train and evaluate an epoch
        train(model, optimizer, args, epoch, train_x, train_y)
        eval_loss, eval_acc = evaluate(model, args, epoch, dev_x, dev_y)

        # early stopping and save best model
        # if eval_acc > eval_best['acc']:
        if ((args.early_stop_monitor == 'loss' and (epoch == 0 or eval_loss < eval_best['loss'])) or
                (args.early_stop_monitor == 'acc' and eval_acc > eval_best['acc'])):

            msg = '{}\n\tLast best\tepoch: {:2d} loss: {:.4f} acc: {:.4f}\n' \
                  ' \tBEST\t\tepoch: {:2d} loss: {:.4f} acc: {:.4f}'
            log.info(msg.format('-' * 80, eval_best['epoch'] + 1, eval_best['loss'], eval_best['acc'],
                                epoch + 1, eval_loss, eval_acc))
            log.info('-' * 80)

            # save best meters
            eval_best['acc'] = eval_acc
            eval_best['loss'] = eval_loss
            eval_best['epoch'] = epoch

            # save best checkpoint
            if args.ckpt:
                state = {
                    'args': args,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                save_checkpoint(state, True, args.ckpt, epoch)

            early_stop_patience = args.early_stop_patience
        else:
            log.info('-' * 80)
            msg = ' BEST epoch: {:2d} loss: {:.4f} acc: {:.4f}\n' \
                  '\t        epoch: {:2d} loss: {:.4f} acc: {:.4f}'
            log.info(msg.format(eval_best['epoch'] + 1, eval_best['loss'], eval_best['acc'],
                                epoch + 1, eval_loss, eval_acc))
            log.info('-' * 80)

            # stop training
            early_stop_patience -= 1
            if early_stop_patience == 0:
                break

    # load model from best checkpoint
    best_ckpt = os.path.join(args.ckpt, 'best.ckpt')
    log.info("loading model from {}".format(best_ckpt))
    model = ProtoNet(args)
    model = model.to(device)
    ckpt = torch.load(best_ckpt)
    model.load_state_dict(ckpt['model_state_dict'])

    test_loss, test_acc = evaluate(model, args, eval_best['epoch'], test_x, test_y, tag='Test')
    log.info('-' * 80)
    msg = ' Test: loss: {:.4f} acc: {:.4f}'
    log.info(msg.format(test_loss, test_acc))


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    main(parse_args())
    exit(0)
