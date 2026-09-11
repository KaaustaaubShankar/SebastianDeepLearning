---
title: Lecture 14 - CNN Architectures
type: note
date: 2025-11-11
last_modified_at: 2025-11-11
order: 14
tags:
  - deep-learning-with-sebastian
  - ml
  - deep-learning
  - cnn
draft: false
toc: true
---
## Padding
We add rows of pixels to maintain the size. 

$$
\begin{aligned}
o = \lfloor{\frac{i + 2p - k}{s}} \rfloor + 1
\end{aligned}
$$
To maintain the same size, calculate padding as such:
$$
p = \frac{k-1}{2}
$$
## Spatial Dropout and Batch Norm

BatchNorm1d normalizes for each feature:
$$
y = \frac{x-E[x]}{\sqrt{\text{Var}[x] + \epsilon}} * \gamma + \beta
$$
BatchNorm2d applies this for each channel (based off of group of $H * W$)

## Common Architectures
### VGG
This has 3x3 convs, stride = 1, padding to maintain size, and 2x2 max pooling. This results in 64  > 32 > 16 > 8 > 4 > 2 after 5 block of convolutions and max pooling. This is then resized to a 3x3 using AveragePooling. This is passed through a layer of 4096 twice and then finally to a softmax for classification.
### Resnet

The unique thing about this model is that they allow for features to influence the network directly instead of only through an activation. This is is through a skip connection. This makes the model deeper without adding more parameters. This makes the model work like so: 28 > 28 > 28 > into classifier.

For deeper networks to save on computation, we increase the number of channels and then go back down. This is done through a 1x1 conv, 3x3 conv, 1x1 conv.

## What happens when we replace max pooling with convolutional layers

If you use convolutional layers with stride 2, you are able to pool in a learnable way. This has been shown to SOTA results at the [paper](https://arxiv.org/pdf/1412.6806)'s time of publication.

## Convolutional Layer instead of Dense
Since a dense layer computes $y = W\cdot x + b$ , we can just do this in the channels. There is no real benefit.

## Transfer Learning
Cut off the dense layer and replace one that fits your model and train the dense layer. Empirically, training the last 3 layers works better.
