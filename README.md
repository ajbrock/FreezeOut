# FreezeOut
A simple technique to accelerate neural net training by progressively freezing layers.

![LRCURVE](http://i.imgur.com/yKE9pzG.gif)

This repository contains code for the extended abstract "[FreezeOut.](https://arxiv.org/abs/1706.04983)" 

FreezeOut directly accelerates training by annealing layer-wise learning rates to zero on a set schedule, and excluding layers from the backward pass once their learning rate bottoms out.

I had this idea while replying [to a reddit comment](https://www.reddit.com/r/MachineLearning/comments/6goyh5/r_forward_thinking_building_and_training_neural/dis12qh/) at 4AM. I threw it in an experiment, and it just worked out of the box (with linear scaling and t_0=0.5), so I went on a 96-hour SCIENCE binge, and now, here we are.

![DESIGNCURVE](http://i.imgur.com/lsa1pRq.png)

The exact speedup you get depends on how much error you can tolerate--higher speedups appear to come at the cost of an increase in error, but speedups below 20% should be within a 3% relative error envelope, and speedups around 10% seem to incur no error cost for Scaled Cubic and Unscaled Linear strategies.

## Installation
To run this script, you will need [PyTorch](http://pytorch.org) and a CUDA-capable GPU. If you wish to run it on CPU, just remove all the .cuda() calls.

## Running
To run with default parameters, simply call

```sh
python train.py
```

This will by default download CIFAR-100, split it into train, valid, and test sets, then train a k=12 L=76 DenseNet-BC using SGD with Nesterov Momentum.

This script supports command line arguments for a variety of parameters, with the FreezeOut specific parameters being:
- how_scale selects which annealing strategy to use, among linear, squared, and cubic. Cubic by default.
- scale_lr determines whether to scale initial learning rates based on t_i. True by default.
- t_0 is a float between 0 and 1 that decides how far into training to freeze the first layer. 0.8 (pre-cubed) by default.
- const_time is an experimental setting that increases the number of epochs based on the estimated speedup, in order to match the total training time against a non-FreezeOut baseline. I have not validated if this is worthwhile or not.

You can also set the name of the weights and the metrics log, which model to use, how many epochs to train for, etc.

If you want to calculate an estimated speedup for a given strategy and t_0 value, use the calc_speedup() function in utils.py.

## Notes
If you know how to implement this in a static-graph framework (specifically TensorFlow or Caffe2), shoot me an email! It's really easy to do with dynamic graphs, but I believe it to be possible with some simple conditionals in a static graph. 

There's (at least) one typo in the paper where it defines the learning rate schedule, there should be a 1/2 in front of alpha.

## Acknowledgments
- DenseNet code stolen in a daring midnight heist from Brandon Amos: https://github.com/bamos/densenet.pytorch
- Training and Progress code acquired in a drunken game of SpearPong with Jan Schlüter: https://github.com/Lasagne/Recipes/tree/master/papers/densenet
- Metrics Logging code extracted from ancient diary of Daniel Maturana: https://github.com/dimatura/voxnet
