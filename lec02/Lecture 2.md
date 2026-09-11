---

title: "Lecture 2 - Neural Networks Basics"
type: "note"
date: 2025-06-11
last_modified_at: 2025-06-11
order: 2
tags:
  - deep-learning-with-sebastian
  - ml
  - deep-learning
  - neural-networks
  - perceptron
  - optimization
  - cnn
  - rnn
draft: false
toc: true
---

## Artificial Neurons

### Pitt's Model

* **Structure:** Two inputs $x_1, x_2$, threshold $t_1$.
* **Operation:** Output is $1$ if $x_1 + x_2 \geq t_1$, else $0$.
* **Limitations:** Can handle basic logical operations (AND, OR, NOT) but **cannot solve XOR**.

### Rosenblatt Perceptron

* **Structure:** Weighted sum of inputs processed by an activation function.
* **Limitations:** Cannot solve XOR.

### Adaline (Adaptive Linear Neuron)

* **Characteristics:** Differentiable model, error is calculated before applying the activation (threshold) function.
* **Weight Updates:** Uses errors computed before thresholding.
* **Limitations:** Cannot solve XOR.

### XOR Problem

* **Issue:** Single-layer models fail XOR because the XOR problem requires multiple linear boundaries; a single linear boundary is insufficient.

## Multilayer Networks

* Also known as:

  * Feed-forward Neural Networks
  * Fully Connected Networks
  * Multilayer Perceptrons (MLP)

* **Characteristics:**

  * Multiple hidden layers
  * Non-linear activation functions to handle complex decision boundaries

* **Training Challenges:**

  * Difficult to train, solved by using **backpropagation**.

### Alternative Training Methods

* **Hebbian Learning:** Strengthens neurons based on repeated usage; no feedback used.
* **Perturbation Learning:** Adjust weights slightly, evaluate improvement iteratively.

## Origins of Deep Learning

### Convolutional Neural Networks (CNNs)

* **Use Case:** Primarily image recognition
* **Features:**

  * Extracts local features
  * Weight sharing and pooling for efficiency
  
See more at [[Lecture 13]]

### Recurrent Neural Networks (RNNs)

* **Characteristics:**

  * Feed-forward structure with loop-back connections
  * Designed to process sequential data

* **Challenges:**

  * Issues with vanishing gradients
  * Solution: Long Short-Term Memory (LSTM) units to manage long-range dependencies effectively
