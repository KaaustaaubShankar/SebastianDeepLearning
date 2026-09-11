---

title: "Lecture 11 - Input Normalization and Weight Initialization"
type: "note"
date: 2025-06-12
last_modified_at: 2025-06-12
order: 11
tags:
- deep-learning-with-sebastian
- ml
- deep-learning
- overfitting
- initialization
draft: false
toc: true

---

## Input Normalization

When features have very different ranges—like $x_1$ in $[0,1]$ and $x_2$ in $[0,100]$—the cost function becomes stretched or skewed. This makes gradient descent inefficient. The gradients for large-scale features (like $x_2$) are much bigger, which can cause the optimizer to overshoot or zig-zag.

Normalizing the features brings them to a similar scale, which balances the gradients, smooths out the optimization path, and helps the model converge faster.

We normalize by:

$$
x^{'[i]}_j = \frac{x_j^{[i]}-\mu_j}{\sigma_j}
$$

where $\mu$ is the mean and $\sigma_j$ is the standard deviation.

## Batch Normalization

BatchNorm is used to normalize hidden layer inputs, mitigate vanishing/exploding gradients, and improve convergence speed and training stability.

### Steps:

1. **Normalize Net Inputs**

   * $\mu_j = \frac{1}{n} \sum_i z_j^{[i]}$
   * $\sigma_j^{2} = \frac{1}{n} \sum_i(z_j^{[i]} - \mu_j)^2$
   * $z\_j^{'[i]} = \frac{z_j^{[i]}-\mu_j}{\sqrt{\sigma_j^2 + \epsilon}}$

2. **Pre-Activation Scaling**

   * $a_j^{'[i]} = \gamma_j \cdot z_j^{'[i]} + \beta_j$
   * $\gamma$ and $\beta$ are learnable parameters that allow the model to scale and shift the normalized values back if necessary.

```python
# Given batch z: shape (batch_size, num_features)
mu = z.mean(dim=0)
var = z.var(dim=0, unbiased=False)
eps = 1e-5

z_norm = (z - mu) / torch.sqrt(var + eps)
a = gamma * z_norm + beta
output = relu(a)
```

Because $\beta_j$ already provides a shift, we typically omit a separate bias term.

### PyTorch Implementation:

```python
class MultilayerPerceptron(torch.nn.Module):
    def __init__(self, num_features, num_classes, drop_proba,
                 num_hidden_1, num_hidden_2):
        super().__init__()

        self.my_network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(num_features, num_hidden_1, bias=False),
            torch.nn.BatchNorm1d(num_hidden_1),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_1, num_hidden_2, bias=False),
            torch.nn.BatchNorm1d(num_hidden_2),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_2, num_classes)
        )

    def forward(self, x):
        return self.my_network(x)
```

> 💡 **Note:** If using dropout with BatchNorm, apply in the order: **activation → batchnorm → dropout**.

### Why does BatchNorm work?

There is debate over whether it addresses internal covariate shift. What's well-supported is that BatchNorm stabilizes training, allowing for **larger learning rates** and **faster convergence**.

> ✅ **Use BatchNorm with sufficiently large mini-batch sizes** (e.g., 16 or 32).

---

## Weight Initialization

Randomly initializing weights can cause vanishing or exploding gradients. A better approach is to scale weights in proportion to the number of inputs to each layer.

### Xavier Glorot Initialization

<iframe src="../../../assets/plots/sigmoid_relu_plot.html" width="700" height="450" frameborder="0"></iframe>

With tanh and sigmoid, extreme inputs can cause gradients to vanish due to **saturation**. Xavier initialization aims to reduce this by preserving the variance of activations across layers—but it's not foolproof.

#### Method:
1. Initialize weights from either:
   * **Gaussian:**
     $W_{i,j}^{(l)} \sim \mathcal{N}\left(0, \frac{1}{n_{\text{in}}}\right)$
   * **Uniform:**
     $W_{i,j}^{(l)} \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right]$

#### Why does this work?

* Assumes inputs are independent and zero-mean
* Keeps the **variance of activations and gradients** roughly the same across layers
* Helps prevent exploding/vanishing gradients
* Works especially well with **sigmoid** and **tanh** activations

It does this by setting:

$$
\text{Var}(W) = \frac{1}{n_{\text{in}}}
$$

### He Initialization (default in pytorch)
* Designed for Relu and other variants since with Relus zero out half the inputs
#### Method
1. Initalize weights from either
	* Gaussian:$W_{i,j}^{(l)} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$
	* Uniform: $W_{i,j}^{(l)} \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{\text{in}} }}, \sqrt{\frac{6}{n_{\text{in}}}}\right]$
