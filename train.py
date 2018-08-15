#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
FreezeOut Training Function
Andy Brock, 2017

This script trains and tests a model using FreezeOut to accelerate training
by progressively freezing early layers and excluding them from the backward
pass. It has command-line options for defining the phase-out strategy, including
how far into training to start phasing out layers, whether to scale 
initial learning rates as a function of how long the layer is trained for,
and how the phase out schedule is defined for layers after the first (i.e. are
layers frozen at regular intervals or is cubically more time given to later
layers?)

Based on Jan Schlüter's DenseNet training code:
https://github.com/Lasagne/Recipes/blob/master/papers/densenet
'''

import os
import logging
import sys
from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import get_data_loader, MetricsLogger, progress
# Set the recursion limit to avoid problems with deep nets
sys.setrecursionlimit(5000)


def opts_parser():
    usage = 'Trains and tests a FreezeOut DenseNet on CIFAR.'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '-L', '--depth', type=int, default=76,
        help='Network depth in layers (default: %(default)s)')
    parser.add_argument(
        '-k', '--growth-rate', type=int, default=12,
        help='Growth rate in dense blocks (default: %(default)s)')
    parser.add_argument(
        '--dropout', type=float, default=0,
        help='Dropout rate (default: %(default)s)')
    parser.add_argument(
        '--augment', action='store_true', default=True,
        help='Perform data augmentation (enabled by default)')
    parser.add_argument(
        '--no-augment', action='store_false', dest='augment',
        help='Disable data augmentation')
    parser.add_argument(
        '--validate', action='store_true', default=True,
        help='Perform validation on validation set (ensabled by default)')
    parser.add_argument(
        '--no-validate', action='store_false', dest='validate',
        help='Disable validation')
    parser.add_argument(
        '--validate-test', action='store_const', dest='validate',
        const='test', help='Evaluate on test set after every epoch.')
    parser.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs (default: %(default)s)')
    parser.add_argument(
        '--t_0', type=float, default=0.8,
        help=('How far into training to start freezing. Note that this if using'
              +' cubic scaling then this is the uncubed value.'))
    parser.add_argument(
        '--scale_lr', type=bool, default=True,
        help='Scale each layer''s start LR as a function of its t_0 value?')
    parser.add_argument(
        '--no_scale', action='store_false', dest='scale_lr',
        help='Don''t scale each layer''s start LR as a function of its t_0 value')
    parser.add_argument(
        '--how_scale',type=str,default='cubic',
        help=('How to relatively scale the schedule of each subsequent layer.'
              +'options: linear, squared, cubic.'))
    parser.add_argument(
        '--const_time', type=bool, default=False,
        help='Scale the #epochs as a function of ice to match wall clock time.')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed to use.')
    parser.add_argument(
        '--which_dataset', type=int, default=100,
        help='Which Dataset to train on (default: %(default)s)')
    parser.add_argument(
        '--batch_size', type=int, default=50,
        help='Images per batch (default: %(default)s)')
    parser.add_argument(
        '--resume', type=bool, default=False,
        help='Whether or not to resume training')
    parser.add_argument(
        '--model', type=str, default='densenet', metavar='FILE',
        help='Which model to use')
    parser.add_argument(
        '--save-weights', type=str, default='default_save', metavar='FILE',
        help='Save network weights to given .pth file')
    return parser





def train_test(depth, growth_rate, dropout, augment,
               validate, epochs, save_weights, batch_size, 
               t_0, seed, scale_lr, how_scale, which_dataset, 
               const_time, resume, model):
    
    # Update save_weights:
    if save_weights=='default_save':
        save_weights = (model + '_k' + str(growth_rate) + 'L' + str(depth)
                        + '_ice' + str(int(100*t_0)) + '_'+how_scale + str(scale_lr)
                        + '_seed' + str(seed) + '_epochs' + str(epochs) 
                        + 'C' + str(which_dataset))
    # Seed RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Name of the file to which we're saving losses and errors.
    metrics_fname = 'logs/'+save_weights + '_log.jsonl'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s| %(message)s')
    logging.info('Metrics will be saved to {}'.format(metrics_fname))
    logging.info('Running with seed ' + str(seed) + ', t_0 of ' + str(t_0)
                + ', and the ' + how_scale + ' scaling method '
                + 'with learning rate scaling set to ' + str(scale_lr) + '.') 
    mlog = MetricsLogger(metrics_fname, reinitialize=(not resume))
   
    # Import the model module
    model_module = __import__(model)

    # Get information specific to each dataset
    train_loader,test_loader = get_data_loader(which_dataset, augment,
                                               validate, batch_size)

    # Build network, either by initializing it or loading a pre-trained
    # network.
    if resume:
        logging.info('loading network ' + save_weights + '...')
        net = torch.load(save_weights + '.pth')

        # Which epoch we're starting from
        start_epoch = net.epoch+1 if hasattr(net, 'epoch') else 0

    #  Get net
    else:
        logging.info('Instantiating network with model ' + model + '...')
        net = model_module.DenseNet(growth_rate, depth=depth,
                                    nClasses=which_dataset,
                                    epochs=epochs,
                                    t_0 = t_0,
                                    scale_lr = scale_lr,
                                    how_scale = how_scale,
                                    const_time = const_time)
        net = net.cuda()
        start_epoch = 0
    


    logging.info('Number of params: {}'.format(
                 sum([p.data.nelement() for p in net.parameters()]))
                 )

    # Training Function, presently only returns training loss
    # x: input data
    # y: target labels
    def train_fn(x, y):
        net.optim.zero_grad()
        output = net(x.cuda())
        loss = F.nll_loss(output, y.cuda())
        loss.backward()
        net.optim.step()
        return loss.item()

    # Testing function, returns test loss and test error for a batch
    # x: input data
    # y: target labels
    def test_fn(x, y):
        with torch.no_grad():
            output = net(x.cuda())
            test_loss = F.nll_loss(output, y.cuda()).item()

            # Get the index of the max log-probability as the prediction.
            pred = output.data.max(1)[1].cpu()
            test_error = pred.ne(y).sum()

        return test_loss, test_error

    # Finally, launch the training loop.
    logging.info('Starting training at epoch '+str(start_epoch)+'...')
    for epoch in range(start_epoch, net.epochs):

        # Pin the current epoch on the network.
        net.epoch = epoch

        # shrink learning rate at scheduled intervals, if desired
        if 'epoch' in net.lr_sched and epoch in net.lr_sched['epoch']:

            logging.info('Annealing learning rate...')

            # Optionally checkpoint at annealing
            # if net.checkpoint_before_anneal:
                # torch.save(net, str(epoch) + '_' + save_weights + '.pth')

            for param_group in net.optim.param_groups:
                param_group['lr'] *= 0.1

        # List where we'll store training loss
        train_loss = []

        # Prepare the training data
        batches = progress(
            train_loader, desc='Epoch %d/%d, Batch ' % (epoch + 1, net.epochs),
            total=len(train_loader.dataset) // batch_size)

        # Put the network into training mode
        net.train()
    
        # Execute training pass
        for x, y in batches:
        
            # Update LR if using cosine annealing
            if 'itr' in net.lr_sched:
                net.update_lr()
                
            train_loss.append(train_fn(x, y))

        # Report training metrics
        train_loss = float(np.mean(train_loss))
        print('  training loss:\t%.6f' % train_loss)
        mlog.log(epoch=epoch, train_loss=float(train_loss))
        
        # Check how many layers are active
        actives = 0
        for m in net.modules():
            if hasattr(m,'active') and m.active:
                actives += 1
        logging.info('Currently have ' + str(actives) + ' active layers...')
        
        # Optionally, take a pass over the validation or test set.
        if validate:

            # Lists to store
            val_loss = []
            val_err = err = []

            # Set network into evaluation mode
            net.eval()

            # Execute validation pass
            for x, y in test_loader:
                loss, err = test_fn(x, y)
                val_loss.append(loss)
                val_err.append(err)

            # Report validation metrics
            val_loss = float(np.mean(val_loss))
            val_err =  100 * float(np.sum(val_err)) / len(test_loader.dataset)
            print('  validation loss:\t%.6f' % val_loss)
            print('  validation error:\t%.2f%%' % val_err)
            mlog.log(epoch=epoch, val_loss=val_loss, val_err=val_err)

        # Save weights for this epoch
        print('saving weights to ' + save_weights + '...')
        torch.save(net, save_weights + '.pth')

    # At the end of it all, save weights even if we didn't checkpoint.
    if save_weights:
        torch.save(net, save_weights + '.pth')


def main():
    # parse command line
    parser = opts_parser()
    args = parser.parse_args()
    train_test(**vars(args))

    # run
    # train_test(**vars(args))


if __name__ == '__main__':
    main()
