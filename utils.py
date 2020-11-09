#  -*- coding: utf-8 -*-

import re
import os
import csv
import sys
import shutil
import logging
import numpy as np

import torch
from typing import Dict, Any, Union

from constants import *


def clean_str(string):
    """
    Tokenization and cleaning string
    Original copy from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z0-9(),!?.\'`\u4E00-\u9FA5]", " ", string)

    # string = re.sub(r"[,;)(!'.]", " ", string)

    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " \. ", string)
    string = re.sub(r"\"", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r" \d+(\.\d\+)? ", " <num> ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def build_vocab(text_files, vocab_file='vocab', stop_words=[], min_frq=3, delimiter=','):
    """
    build vocab from data
    :param text_files: input train data files
    :param vocab_file: vocabulary filename
    :param min_frq: words that appears less than {min_frq} times will be replaced as <unk>
    :param delimiter: delimiter for CSV file
    :return: vocabulary dictionary, (key: word, value: idx)
    """

    csv.field_size_limit(sys.maxsize)
    token_count: Dict[str, Union[int, Any]] = {}
    for file in text_files:
        with open(file, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=delimiter)
            for line in csv_reader:
                text = clean_str(" ".join(line[1:]))

                # word only counts once in a document
                tokens = set(text.split(' '))
                for tok in tokens:
                    token_count[tok] = token_count.get(tok, 0) + 1

    # remove token when count < min_frq
    filtered_token_count = {k: v for k, v in token_count.items() if v >= min_frq}

    # sort token by frequency descent
    word_count = sorted(filtered_token_count.items(), key=lambda kv: -kv[1])
    vocab = SPECIAL_WORDS + [w for w, _ in word_count]

    with open(vocab_file, 'w', encoding='utf-8') as f:
        for w in vocab:
            f.write('{0}\n'.format(w))

    return {w: i for i, w in enumerate(vocab)}, vocab


def load_vocab(vocab_file='vocab'):
    """
    load vocabulary from file
    :param vocab_file: vocabulary filename
    :return: vocabulary dict
    """
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = [v.strip() for v in f]
    return {w: i for i, w in enumerate(vocab)}, vocab


def text_to_idx(text_file, idx_file, word2idx, one_based_class=False, delimiter=','):
    """
    convert csv file into vocabulary index file
    :param text_file: text data file
    :param idx_file: idx data file
    :param word2idx: vocabulary file
    :param delimiter: delimiter in data file
    :param one_based_class: class id is start from one
    :return:
    """
    with open(text_file, 'r', encoding='utf-8') as csv_file:
        with open(idx_file, 'w', encoding='utf-8') as idx_file:
            csv_reader = csv.reader(csv_file, delimiter=delimiter)
            for line in csv_reader:
                if one_based_class:
                    # one-based label index
                    label = str(int(line[0]) - 1)
                else:
                    # zero-based label index
                    label = str(int(line[0]))

                text = clean_str(" ".join(line[1:]))
                tokens = text.split(' ')
                token_idx = [word2idx.get(t, UNK) for t in tokens]

                idx_file.write('{0}{1}{2}\n'.format(
                    label, delimiter, ' '.join(map(lambda t: str(t), token_idx))))


def load_idx_data(idx_file, sep=','):
    """
    Load idx file
    :param idx_file: data file
    :param sep: separator between label and text indices
    :return:
    """
    labels = []
    seqs = []
    with open(idx_file, 'r', encoding='utf-8') as file:
        for line in file:
            label, seq = line.split(sep, maxsplit=1)
            labels.append(int(label))
            seqs.append([int(t) for t in seq.strip().split(' ') if len(t) > 0])

    return seqs, labels


def drop_words(words, prob):
    """Drops words with the given probability."""
    length = len(words)
    keep_prob = np.random.uniform(size=length)
    keep = np.random.uniform(size=length) > prob
    if np.count_nonzero(keep) == 0:
        ind = np.random.randint(0, length)
        keep[ind] = True
    words = np.take(words, keep.nonzero())[0]
    return words


def pad_seq(seq, max_len):
    """
    padding or truncate sequence to fixed length
    :param seq: input sequence
    :param max_len: max length
    :return: padded sequence
    """
    if max_len < len(seq):
        seq = seq[:max_len]  # trunc
    else:
        for j in range(max_len - len(seq)):
            seq.append(0)  # padding
    return seq


def load_embeddings(vocab, embed_file, embed_dim):
    """
    Load pre-trained embeddings
    :param vocab: vocabulary
    :param embed_file: pre-trained embedding file
    :param embed_dim: dimension of pre-trained embeddings
    :return: pre-trained word embeddings
    """

    print("loading pre-trained word vectors form %s" % embed_file)
    word2embed = {}
    with open(embed_file) as f:
        for line in f:
            values = line.split(' ')
            if len(values) == embed_dim + 1:
                word = values[0]
                embed = np.asarray(values[1:], dtype='float32')
                word2embed[word] = embed

    # prepare embedding matrix
    num_token = len(vocab)
    count = 0
    weights = np.random.rand(num_token, embed_dim)
    for i, word in enumerate(vocab):
        if word == PAD_WORD:
            embedding_vector = np.zeros(shape=(1, embed_dim), dtype='float32')
        else:
            embedding_vector = word2embed.get(word)

        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            weights[i] = embedding_vector
            count = count + 1

    print("%d vectors loaded." % count)
    return weights


def build_glove_corpus(f_csv, f_txt, delimiter=','):
    csv.field_size_limit(sys.maxsize)
    with open(f_csv, 'r', encoding='utf-8') as csv_file:
        with open(f_txt, 'w', encoding='utf-8') as txt_file:
            csv_reader = csv.reader(csv_file, delimiter=delimiter)
            for line in csv_reader:
                text = clean_str(" ".join(line[1:]))
                txt_file.write(text + "\n")


def print_table(tab):
    col_width = [max(len(x) for x in col) for col in zip(*tab)]
    print("+-" + "-+-".join("{:-^{}}".format('-', col_width[i]) for i, x in enumerate(tab[0])) + "-+")
    for line in tab:
        print("| " + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)) + " |")
    print("+-" + "-+-".join("{:-^{}}".format('-', col_width[i]) for i, x in enumerate(tab[0])) + "-+")


def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 75)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


def save_checkpoint(state, is_best, ckpt_dir, epoch):
    os.makedirs(ckpt_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger('util')

    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(ckpt_dir, filename)
    best_path = os.path.join(ckpt_dir, 'best.ckpt')
    torch.save(state, checkpoint_path)
    log.info(" checkpoint saved to %s " % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        log.info(" checkpoint copied to %s " % best_path)
