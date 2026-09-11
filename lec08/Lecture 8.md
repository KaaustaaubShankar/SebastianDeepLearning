---

title: "Lecture 8 - Logistic Regression and Softmax"
type: "note"
date: 2025-06-11
last_modified_at: 2025-06-11
order: 8
tags:
- deep-learning-with-sebastian
- ml
- deep-learning
- logistic-regression
- softmax
- gradient-descent
- optimization
draft: false
toc: true
---

## Logistic Regression as a Single-Layer Network

### Logistic Sigmoid Function

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Predictive model:

$$
h(\mathbf{x}) = \sigma(\mathbf{w}^{T}\mathbf{x} + b)
$$

Interpretation:

* Probability that input $\mathbf{x}$ belongs to class $y=1$: $h(\mathbf{x})$
* Probability for $y=0$ is $1 - h(\mathbf{x})$

## Negative Log-Likelihood Loss

Maximize:

$$
\prod_{i=1}^{n} \left( \sigma(z^{(i)}) \right)^{y^{(i)}} \left(1 - \sigma(z^{(i)})\right)^{1 - y^{(i)}}
$$

Due to numerical stability, use log transformation:

$$
\log \mathcal{L}(\mathbf{w}) = \sum_{i=1}^{n}\left[ y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)})\log(1 - \sigma(z^{(i)})) \right]
$$

Loss function (negative average log-likelihood):

$$
\mathcal{L}(\mathbf{w}) = -\frac{1}{n} \sum_{i=1}^{n}\left[ y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)})\log(1 - \sigma(z^{(i)})) \right]
$$

## Derivatives for Logistic Regression

* Sigmoid derivative:

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

* Loss derivative w\.r.t. weights:

$$
\frac{\partial \mathcal{L}}{\partial w_j} = (\hat{y} - y)x_j
$$

* Loss derivative w\.r.t. bias:

$$
\frac{\partial \mathcal{L}}{\partial b} = \hat{y} - y
$$

### Stochastic Gradient Descent for Logistic Regression

1. Initialize: $\mathbf{w} = \mathbf{0}^m$, $b = 0$
2. For each epoch:

   * For each training sample $(\mathbf{x}^{[i]}, y^{[i]})$:

     1. Compute prediction:
        $\hat{y}^{[i]} = \sigma(\mathbf{x}^{[i]T}\mathbf{w} + b)$
     2. Compute gradients:
        $\nabla_w \mathcal{L} = (\hat{y}^{[i]} - y^{[i]})\mathbf{x}^{[i]}, \quad \nabla_b \mathcal{L} = (\hat{y}^{[i]} - y^{[i]})$
     3. Update weights and bias:
        $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_w \mathcal{L}, \quad b \leftarrow b - \eta \nabla_b \mathcal{L}$

## Cross-Entropy and Logits

* Binary Cross-Entropy Loss equals Negative Log-Likelihood.

### Multicategory Cross-Entropy

$$
H_a(y) = -\sum_{i=1}^{n}\sum_{k=1}^{K} y_k^{[i]}\log(a_k^{[i]})
$$

## Multinomial Logistic Regression (Softmax Regression)

* **Softmax Activation Function:**

$$
P(y = t \mid \mathbf{z}^{[i]}) = \sigma_{\text{softmax}}(z_t^{[i]}) = \frac{e^{z_t^{[i]}}}{\sum_{j=1}^{h} e^{z_j^{[i]}}}
$$

* **Output:** Probability distribution across $K$ classes.

### One-Hot Encoding

* Each class is represented by a binary vector, with $1$ at the target class index and $0$ elsewhere.

## Gradient Derivatives for Softmax Regression

* Loss w\.r.t. weights (for each class $k$):

$$
\frac{\partial \mathcal{L}}{\partial w_{jk}} = (a_k - y_k)x_j
$$

* Loss w\.r.t. bias:

$$
\frac{\partial \mathcal{L}}{\partial b_k} = a_k - y_k
$$


