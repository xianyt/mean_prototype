#!/usr/bin/python
#  -*- coding: utf-8 -*-
import argparse
import logging
import random
import re
import sys
from os import path


from utils import build_vocab, text_to_idx, load_vocab


def parse_args():
    parser = argparse.ArgumentParser()

    """
    inputs
    """
    parser.add_argument('--dir', type=str, required=True,
                        help='path to dataset')
    parser.add_argument('--vocab', type=str, default='vocab',
                        help='vocab file name')
    parser.add_argument('--train', type=str, default='train.csv',
                        help='filename of training data in dataset dir')
    parser.add_argument('--unlabel', type=str, default='unlabel.csv',
                        help='filename of unlabeled training data in dataset dir')
    parser.add_argument('--dev', type=str, default='dev.csv',
                        help='filename of validation data in dataset dir')
    parser.add_argument('--test', type=str, default='test.csv',
                        help='filename of validation data in dataset dir')

    parser.add_argument('--min-freq', type=int, default=2,
                        help='minimize frequency of word in vocab')
    parser.add_argument('--one-based-class', action='store_true', help='class id is one based')

    return parser.parse_args()


def split_train_dev(data, data_dir, rate):
    """
    split train data to dev_train and dev sets
    """
    with open(data, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        random.shuffle(lines)
        dev_idx = int(len(lines) * rate)

    f_train = path.join(data_dir, 'dev_train.csv')
    f_dev = path.join(data_dir, 'dev.csv')

    with open(f_dev, 'w', encoding='utf-8') as f:
        f.write(''.join(lines[0:dev_idx]))

    with open(f_train, 'w', encoding='utf-8') as f:
        f.write(''.join(lines[dev_idx:]))

    return f_train, f_dev


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('main')

    args = parse_args()

    train_file = path.join(args.dir, args.train)
    unlabel_file = path.join(args.dir, args.unlabel)
    dev_file = path.join(args.dir, args.dev)
    test_file = path.join(args.dir, args.test)

    if not path.isfile(dev_file):
        # dev file does not exists, split from train file
        log.info("dev file does not exists, split from train file")
        train_file, dev_file = split_train_dev(train_file, args.dir, 0.1)

    vocab_file = path.join(args.dir, args.vocab)
    if not path.exists(vocab_file):
        text_files = [train_file]
        if path.isfile(unlabel_file):
            text_files.append(unlabel_file)

        log.info("Building vocabulary from training file ...")
        word2idx, vocab = build_vocab(text_files, vocab_file, args.min_freq)
    else:
        log.info("Loading vocabulary from vocab file.")
        word2idx, vocab = load_vocab(vocab_file)

    log.info("Converting train file into idx file.")
    text_to_idx(train_file, re.sub(r'\.\w+$', ".idx", train_file), word2idx, args.one_based_class)

    if path.isfile(unlabel_file):
        log.info("Converting unlabeled train file into idx file.")
        text_to_idx(unlabel_file, re.sub(r'\.\w+$', ".idx", unlabel_file), word2idx, args.one_based_class)
    else:
        log.info("unlabeled train file is not exists: %s", unlabel_file)

    if path.isfile(dev_file):
        log.info("Converting dev file into idx file.")
        text_to_idx(dev_file, re.sub(r'\.\w+$', ".idx", dev_file), word2idx, args.one_based_class)
    else:
        log.info("dev file is not exists: %s", dev_file)

    log.info("Converting test file into idx file.")
    text_to_idx(test_file, re.sub(r'\.\w+$', ".idx", test_file), word2idx, args.one_based_class)

    sys.exit(0)
