#  -*- coding: utf-8 -*-

import time
import sys


class AverageMeter:
    """Computes and stores the average"""

    def __init__(self, name):
        self.name = name
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def __format__(self, format_spec):
        return "{self.name}: {self.avg:{format}}".format(self=self, format=format_spec)


class Meters(object):
    def __init__(self):
        self.names = []
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, val):
        if name not in self.meters:
            self.meters[name] = AverageMeter(name)
            self.names.append(name)

        self.meters[name].update(val)

    def val(self, name):
        return self.meters[name].avg

    def __format__(self, format_spec):
        return " ".join([
            "{self:{format}}".format(self=self.meters[n], format=format_spec) for n in self.names])
