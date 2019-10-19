#!/usr/bin/python
#  -*- coding: utf-8 -*-
import argparse
import logging
import random
import re
import sys
from os import path

from utils import build_vocab, text_to_idx, load_vocab, build_glove_corpus


def parse_args():
    parser = argparse.ArgumentParser()

    """
    inputs
    """
    parser.add_argument('--dir', type=str, required=True,
                        help='path to dataset')
    parser.add_argument('--train', type=str, default='train.csv',
                        help='filename of training data in dataset dir')

    return parser.parse_args()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('main')

    args = parse_args()

    train_file = path.join(args.dir, args.train)

    log.info("Converting train file into txt file.")
    build_glove_corpus(train_file, re.sub(r'\.\w+$', ".txt", train_file))

    sys.exit(0)
