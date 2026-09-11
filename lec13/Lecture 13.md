---
title: Lecture 13 - Convolutional Networks
type: note
date: 2025-06-14
last_modified_at: 2025-06-14
order: 13
tags:
  - deep-learning-with-sebastian
  - ml
  - deep-learning
  - cnn
draft: false
toc: true
---

## What can CNNs do?
* Image Classification
* Object Detection
* Object Segmentation
* Face Recognition
* Image Synthesis

## Basics
Images are essentially an **arrangement** of pixels where the pixels are related to each other. Since they are related to each other, we can use an inductive base of **locality** to work with images.  

## How Does Convolution Work?

In convolutional neural networks (CNNs), we use the fact that **nearby pixels are often related**. To capture this, we slide small windows (called **kernels** or **filters**) over the input image.

Each filter learns to detect a specific feature (like edges, corners, etc.). The result of applying this sliding operation is a **feature map** (also called an activation map).

### How Do We Calculate the Output Size?

The size of the output feature map after applying a convolution is calculated using this formula:

$$
\left\lfloor \frac{\text{Input Size} - \text{Kernel Size} + 2 \cdot \text{Padding}}{\text{Stride}} \right\rfloor + 1
$$

Where:
- **Input Size** = width or height of the input image (like 32)
- **Kernel Size** = width/height of the filter (like 5)
- **Padding** = number of pixels added to each side of the input (usually 0 or "same")
- **Stride** = how many pixels the kernel moves at a time

### Example

If we have:
- Input size: `32`
- Kernel size: `5`
- Padding: `0`
- Stride: `1`

Then the output size is:

$$
\left\lfloor \frac{32 - 5 + 2 \cdot 0}{1} \right\rfloor + 1 = 28
$$

So, applying a `5×5` filter to a `32×32` image with no padding and stride 1 gives an output of **28×28**.

###  Rule of Thumb

| Setting       | Typical Value |
|---------------|---------------|
| Kernel Size   | 3, 5, or 7     |
| Stride        | 1 or 2         |
| Padding       | 0 ("valid") or padding to keep size same ("same") |

Use more filters and smaller strides to capture finer details; increase stride or pool to reduce spatial size later in the network.

## How does Pooling work?
We apply a small sliding window on to each channel and for each window, we either take the maximum value inside the window or the average value inside the window. This allows us to shrink our input feature map allowing us to reduce the number of parameters and noise in the activations by focusing on more dominant features.

<img 
  src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*gpkHl16U7ppl4-lBlnAYqw.gif" 
  alt="Pooling Illustration" 
  style="max-width: 100%; height: auto;" />
  

We can calculate this through the same formula above.

> Note: we dont usually pad in pooling layers because it goes against the principle behind pooling and if used, it can lead to distortions.

## What does the Kernel actually do?

It detects different features. The more kernels we have, the more features we can extract. The cool thing about kernels is that since kernels are able to detect features, they can be applied to any position since the kernel is applied to the entire input.

## Cross Correlation vs Convolution

Convolution in DL is actually Cross Correlation (when we do the sliding dot product over the image).  
The key difference is that convolution flips the kernel, while cross-correlation does not.  
In deep learning, the kernel is learned, so flipping isn't necessary. So cross-correlation is used but called convolution.

### How do we back propagate with kernels and everything?

In convolutional neural networks, **weight sharing** means the same kernel weights $w_1,w_2, ...$ are used across all spatial positions of the input.

Because each weight is reused multiple times (at different locations), the total gradient for a single weight is the sum of its gradients from every output position where it was applied:

$$
\frac{\partial l}{\partial w_1} = \sum_{i,j} \frac{\partial l}{\partial o_{i,j}} \cdot \frac{\partial o_{i,j}}{\partial w_1}
$$

Here, each term $\frac{\partial l}{\partial o_{i,j}} \cdot \frac{\partial o_{i,j}}{\partial w_1}$ represents the contribution from the output at position $(i,j)$.

Because $w_1$ is shared, it receives gradient signals from **all** those positions, and the backpropagation sums them to get the final update.

This is different from a fully connected layer where each weight connects to only one output neuron; in CNNs, weights are reused (shared), making this summation necessary.

### What do CNNs see?

CNNs attempt to extract features but the question is can we understand the features?

Features extractors early on in the model are more interpretable while later on they are more abstract like extracting edges.

