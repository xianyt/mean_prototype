#!/usr/bin/python
#  -*- coding: utf-8 -*-

import torch


def euclidean_dist(x, y):
    """
    Euclidean distance between matrix rows
    :param x: matrix with size [n, dim]
    :param y: matrix with size [m, dim]
    :return: distance matrix with size [n, m]
    """
    assert x.size(1) == y.size(1)

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def proto_loss(cls_prob, targets, protos, disable_reg=False):
    """
    :type protos:
    :param cls_prob:    [bsz, class_num]
    :param targets:     [bsz]
    :param disable_reg:
    :return:
    """
    # [bsz, class_num]
    class_num = cls_prob.size(1)
    y_prob = torch.gather(cls_prob, 1, targets.unsqueeze(-1).expand(-1, class_num))[:, 0]
    loss_val = -torch.log(y_prob).view(-1).mean()

    _, y_hat = cls_prob.max(-1)
    acc_val = y_hat.eq(targets).float().mean()

    if disable_reg:
        return loss_val, acc_val
    else:
        reg_val = torch.mean(torch.exp(-euclidean_dist(protos, protos))) - 1.0 / class_num
        return loss_val, reg_val, acc_val
