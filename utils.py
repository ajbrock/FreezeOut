#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
import time
import json
import logging
import path
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


scale_fn = {'linear':lambda x: x,
                         'squared': lambda x: x**2,
                         'cubic': lambda x: x**3}
                         
def calc_speedup(growthRate,nDenseBlocks,t_0,how_scale):
    # Height*Width at each stage
    HW = [32**2, 16**2, 8**2]

    # FLOPs of first layer
    c = [3* (2*growthRate)*HW[0]*9]
    # num channels
    n = 2
    for i in range(3):
        for j in range(nDenseBlocks):
            # Calc flops for this layer
            c.append(n*(4*growthRate*growthRate)*HW[i] + 4*9*growthRate*growthRate*HW[i])
            n +=1
        n = math.floor(n*0.5)
    
    # Total computational cost for training run without freezeout
    C = 2*sum(c)

    # Computational Cost with FreezeOut
    C_f = sum(c)+sum([c_i*scale_fn[how_scale](
                    (t_0 + (1 - t_0) * float(index) / len(c) ))
                    for index,c_i in enumerate(c)])

    
    if how_scale=='linear':
        return 1.3*(1-float(C_f)/C)
    else:
        return 1-float(C_f)/C


def get_data_loader(which_dataset,augment=True,validate=True,batch_size=50):
    class CIFAR10(dset.CIFAR10):
        def __len__(self):
            if self.train:
                return len(self.train_data)
            else:
                return 10000


    class CIFAR100(dset.CIFAR100):
        def __len__(self):
            if self.train:
                return len(self.train_data)
            else:
                return 10000

    if which_dataset is 10:
        print('Loading CIFAR-10...')
        norm_mean = [0.49139968, 0.48215827, 0.44653124]
        norm_std = [0.24703233, 0.24348505, 0.26158768]
        dataset = CIFAR10

    elif which_dataset is 100:
        print('Loading CIFAR-100...')
        norm_mean = [0.50707519, 0.48654887, 0.44091785]
        norm_std = [0.26733428, 0.25643846, 0.27615049]
        dataset = CIFAR100

    # Prepare transforms and data augmentation
    norm_transform = transforms.Normalize(norm_mean, norm_std)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm_transform
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        norm_transform
    ])
    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_set = dataset(
        root='cifar',
        train=True,
        download=True,
        transform=train_transform if augment else test_transform)
    # If we're evaluating on the test set, load the test set
    if validate == 'test':
        test_set = dataset(root='cifar', train=False, download=True,
                           transform=test_transform)

    # If we're evaluating on the validation set, prepare validation set
    # as the last 5,000 samples in the training set.
    elif validate:
        test_set = dataset(root='cifar', train=True, download=True,
                           transform=test_transform)
        test_set.train_data = test_set.train_data[-5000:]
        test_set.train_labels = test_set.train_labels[-5000:]
        train_set.train_data = train_set.train_data[:-5000]
        train_set.train_labels = train_set.train_labels[:-5000]

    # Prepare data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, **kwargs)
    return train_loader, test_loader
    

class MetricsLogger(object):

    def __init__(self, fname, reinitialize=False):
        self.fname = path.Path(fname)
        self.reinitialize = reinitialize
        if self.fname.exists():
            if self.reinitialize:
                logging.warn('{} exists, deleting'.format(self.fname))
                self.fname.remove()

    def log(self, record=None, **kwargs):
        """
        Assumption: no newlines in the input.
        """
        if record is None:
            record = {}
        record.update(kwargs)
        record['_stamp'] = time.time()
        with open(self.fname, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=True)+'\n')


def read_records(fname):
    """ convenience for reading back. """
    skipped = 0
    with open(fname, 'rb') as f:
        for line in f:
            if not line.endswith('\n'):
                skipped += 1
                continue
            yield json.loads(line.strip())
        if skipped > 0:
            logging.warn('skipped {} lines'.format(skipped))
            
"""
Very basic progress indicator to wrap an iterable in.

Author: Jan Schlüter
"""


def progress(items, desc='', total=None, min_delay=0.1):
    """
    Returns a generator over `items`, printing the number and percentage of
    items processed and the estimated remaining processing time before yielding
    the next item. `total` gives the total number of items (required if `items`
    has no length), and `min_delay` gives the minimum time in seconds between
    subsequent prints. `desc` gives an optional prefix text (end with a space).
    """
    total = total or len(items)
    t_start = time.time()
    t_last = 0
    for n, item in enumerate(items):
        t_now = time.time()
        if t_now - t_last > min_delay:
            print("\r%s%d/%d (%6.2f%%)" % (
                    desc, n+1, total, n / float(total) * 100), end=" ")
            if n > 0:
                t_done = t_now - t_start
                t_total = t_done / n * total
                print("(ETA: %d:%02d)" % divmod(t_total - t_done, 60), end=" ")
            sys.stdout.flush()
            t_last = t_now
        yield item
    t_total = time.time() - t_start
    print("\r%s%d/%d (100.00%%) (took %d:%02d)" % ((desc, total, total) +
                                                   divmod(t_total, 60)))
