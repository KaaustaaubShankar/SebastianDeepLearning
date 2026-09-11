---

title: "Lecture 5 - Training Modes and Linear Regression"
type: "note"
date: 2025-06-11
last_modified_at: 2025-06-11
order: 5
tags:
- deep-learning-with-sebastian
- ml
- deep-learning
- gradient-descent
- linear-regression
- optimization
draft: false
toc: true
---

## Training Modes

### Perceptron Learning Algorithm with Bias

Dataset:
$D = \{(\mathbf{x}^{[1]}, y^{[1]}), \dots, (\mathbf{x}^{[n]}, y^{[n]})\}$

1. **Initialization:** Set weights $\mathbf{w} = \mathbf{0}^m$, bias $b = 0$.
2. **Epoch:** Repeat for each training epoch:

   * Predict output: $\hat{y}^{[i]} = \sigma(\mathbf{x}^{[i]T}\mathbf{w} + b)$
   * Compute error: $\text{error} = y^{[i]} - \hat{y}^{[i]}$
   * Update weights and bias:

     $$
     \mathbf{w} \leftarrow \mathbf{w} + \text{error} \times \mathbf{x}^{[i]}, \quad b \leftarrow b + \text{error}
     $$

### Training Modes

#### Online Mode

1. Initialize: $\mathbf{w} = \mathbf{0}^m$, $b = 0$
2. For each epoch:

   * For each training sample $(\mathbf{x}^{[i]}, y^{[i]})$:

     1. Compute output and error
     2. Update weights and bias immediately

#### Batch Mode

1. Initialize: $\mathbf{w} = \mathbf{0}^m$, $b = 0$
2. For each epoch:

   * Initialize accumulators: $\Delta \mathbf{w} = 0, \Delta b = 0$
   * For each training sample:

     1. Compute output and error
     2. Accumulate $\Delta \mathbf{w}, \Delta b$
   * Update after entire dataset:
     $\mathbf{w} \leftarrow \mathbf{w} + \Delta \mathbf{w}, \quad b \leftarrow b + \Delta b$

#### Mini-batch Mode

1. Initialize: $\mathbf{w} = \mathbf{0}^m$, $b = 0$
2. For each epoch:

   * For each mini-batch of size $k$:

     * Initialize accumulators: $\Delta \mathbf{w} = 0, \Delta b = 0$
     * For samples $(\mathbf{x}^{[i]}, y^{[i]}), \dots, (\mathbf{x}^{[i+k]}, y^{[i+k]})$:

       1. Compute output and error
       2. Accumulate $\Delta \mathbf{w}, \Delta b$
     * Update after mini-batch:
       $\mathbf{w} \leftarrow \mathbf{w} + \Delta \mathbf{w}, \quad b \leftarrow b + \Delta b$

## Linear Regression

### Characteristics

* Activation function: Identity function $\sigma(x) = x$
* Output $\hat{y} \in \mathbb{R}$

### Training Methods

#### Least Squares Linear Regression (Brute Force)

* Randomly choose weights, evaluate performance, retain good models.

#### Stochastic Gradient Descent (SGD)

1. Initialize: $\mathbf{w} = \mathbf{0}^m, b = 0$
2. For each epoch:

   * For each training example $(\mathbf{x}^{[i]}, y^{[i]})$:

     1. Compute prediction:
        $\hat{y}^{[i]} = \mathbf{x}^{[i]T}\mathbf{w} + b$
     2. Compute gradients:
        $\nabla_{\mathbf{w}} \mathcal{L} = (\hat{y}^{[i]} - y^{[i]}) \mathbf{x}^{[i]}, \quad \nabla_b \mathcal{L} = (\hat{y}^{[i]} - y^{[i]})$
     3. Update weights and bias:
        $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}}\mathcal{L}, \quad b \leftarrow b - \eta \nabla_b\mathcal{L}$

### Gradient Descent for Linear Regression

* Loss function (convex):

$$
\mathcal{L}(\mathbf{w}, b) = \sum_{i}(\hat{y}^{[i]} - y^{[i]})^2
$$
