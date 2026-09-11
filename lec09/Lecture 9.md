---

title: "Lecture 9 - Multilayer perceptrons and backpropagation"
type: "note"
date: 2025-06-11
last_modified_at: 2025-06-11
order: 9
tags:
- deep-learning-with-sebastian
- ml
- deep-learning
- generalization
- overfitting
draft: false
toc: true
---

## PyTorch Note

* `F.cross_entropy` already applies **softmax**, so do **not** apply softmax separately in your forward function.

---

## Overfitting and Underfitting

### Bias-Variance Decomposition

$$
\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}
$$

* **Bias**: How far model predictions are from the true function (underfitting).
  $\text{Bias}[\theta] = \mathbb{E}[\hat{\theta}] - \theta$

* **Variance**: Sensitivity to fluctuations in the training data (overfitting).
  $\text{Var}_{\theta}[\hat{\theta}] = \mathbb{E}[\hat{\theta}^2] - \mathbb{E}[\hat{\theta}]^2 = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2]$

* **Consistency** relates to **variance**.

* **How off** the predictions are relates to **bias**.

---

## Parameters vs Hyperparameters

| **Parameters** | **Hyperparameters**                 |
| -------------- | ----------------------------------- |
| Weights        | Mini-batch size                     |
| Biases         | Data normalization schemes          |
|                | Number of epochs                    |
|                | Number of hidden layers             |
|                | Number of hidden units              |
|                | Learning rate                       |
|                | Random seed (for reproducibility)   |
|                | Loss function                       |
|                | Weighting terms                     |
|                | Activation function types           |
|                | Regularization schemes (e.g. L1/L2) |
|                | Weight initialization schemes       |
|                | Optimization algorithm type         |
