---

title: "Lecture 3 - Perceptron Learning"
type: "note"
date: 2025-06-11
last_modified_at: 2025-06-11
order: 3
tags:
- deep-learning-with-sebastian
- ml
- deep-learning
- neural-networks
- perceptron
- optimization
- backpropagation
draft: false
toc: true
---

## Pitt's Neuron Model

### Formula

The net input $z$ is computed as:

$$
z = \sum_{i} x_i \cdot w_i
$$

### Threshold Function

After calculating $z$, apply a threshold to determine the neuron's output:

$$
\hat{y} = \begin{cases}
1, & z \geq \theta \\
0, & z < \theta
\end{cases}
$$

## The Perceptron Learning Rule

### Terminology

* **Net Input ($z$)**: $z = \sum x_i w_i$
* **Activation Function ($\sigma(z)$)**: Typically a threshold
* **Predicted Output ($\hat{y}$)**: Final binary output after thresholding
* **Bias ($b$)**: Adjusts the threshold $\theta$, defined as $b = -\theta$

### Formula with Bias

The perceptron output with explicit bias is:

$$
\hat{y} = \sigma(\mathbf{x}^{\mathsf{T}}\mathbf{w} + b)
$$

Where:

$$
\sigma(z) = \begin{cases}
0, & z \leq 0 \\
1, & z > 0
\end{cases}
$$

## Perceptron Learning Algorithm

Given a dataset $D = \{(\mathbf{x}^{[1]}, y^{[1]}), \dots, (\mathbf{x}^{[n]}, y^{[n]})\}$:

1. **Initialize weights:**

   $$
   \mathbf{w} = \mathbf{0}^m
   $$

2. **Training Loop:**
   For each epoch, iterate over each example:

   * Predict:

     $$
     \hat{y}^{[i]} = \sigma(\mathbf{x}^{[i]\mathsf{T}}\mathbf{w})
     $$

   * Compute error:

     $$
     \text{error} = y^{[i]} - \hat{y}^{[i]}
     $$

   * Update weights:

     $$
     \mathbf{w} = \mathbf{w} + \text{(error)} \cdot \mathbf{x}^{[i]}
     $$
