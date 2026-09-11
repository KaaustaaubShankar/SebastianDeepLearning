---

title: "Lecture 10 - Regularization to Avoid Overfitting"
type: "note"
date: 2025-06-11
last_modified_at: 2025-06-11
order: 10
tags:
- deep-learning-with-sebastian
- ml
- deep-learning
- overfitting
- regularization
- dropout
- data-augmentation
- generalization
draft: false
toc: true

---

## Techniques for Reducing Overfitting

### Dataset

* Collect more data
* **Data Augmentation**: e.g., rotate images randomly
* **Label Smoothing**
* **Leverage Unlabeled Data**:

  * Semi-Supervised Learning
  * Self-Supervised Learning
* **Leverage Related Data**:

  * Meta-Learning
  * Transfer Learning

### Architecture Setup

* Weight Initialization Strategies
* Activation Functions
* Residual Layers
* Knowledge Distillation

### Normalization

* Input Standardization
* Batch Normalization (and variants)
* Weight Standardization
* Gradient Centralization

### Training Loop

* Adaptive Learning Rates
* Auxiliary Losses
* Gradient Clipping

### Regularization

* L1 or L2 Regularization
* Early Stopping
* Dropout

---

## Early Stopping

1. Split dataset into:

   * Training
   * Validation (for tuning)
   * Test (used once at end)

2. Stop training when training accuracy increases but validation accuracy stagnates or decreases (indicates overfitting).

---

## L1 and L2 Regularization

### L1 (Lasso)

Adds absolute value of weights:

$$
\text{Loss}_{L1} = \text{Loss}_{original} + \lambda \sum_{i} |w_i|
$$

* Promotes sparsity by driving weights to zero.

### L2 (Ridge)

Adds squared weights:

$$
\text{Loss}_{L2} = \text{Loss}_{original} + \lambda \sum_{i} w_i^2
$$

* Shrinks weights gradually.

### Logistic Regression Example

$$
\text{Loss}_{\mathbf{w}, b} = \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(y^{[i]}, \hat{y}^{[i]}) + \frac{2\lambda}{n} \sum_{j=1}^{n} w_j^2
$$

Where $\mathcal{L}$ is binary cross-entropy.

* **High $\lambda$** → High bias
* **Low $\lambda$** → High variance

### Gradient Descent with L2

$$
w_{i,j} = w_{i,j} + \eta \left( \frac{\partial L}{\partial w_{i,j}} + \frac{2\lambda}{n} w_{i,j} \right)
$$

---

## Dropout

* Randomly "drop" neurons using Bernoulli sampling.

$$
\forall i, v_i = \begin{cases}
0 &\text{if } v_i < p \\
1 &\text{otherwise}
\end{cases}, \quad a = a \odot v
$$

* During training: scale activations:
  $a = a \cdot (1 - p)$

* Example: $[4, 5], p = 0.05$

  * Expected value: $[3.8, 4.75]$
  * Formula $a = a \cdot (1 - p)$ simulates this expectation

### Dropout as Ensemble

* Dropout creates a set of different subnetworks.
* During testing, all neurons are used to generate a combined "consensus."

### Dropout in Python

* During training:
  $a_{train} = \frac{a \odot m}{1 - p}$
* Scaling is applied during training, not inference.

| Layer Type | Dropout Probability $p$ |
| ---------- | ----------------------- |
| Fully      | 0.2–0.5                 |
| Conv       | 0.1–0.3                 |

**Placement:**

* After dense layers
* Avoid before softmax or in early layers

**Training Tips:**

* May require more epochs or higher learning rate.
