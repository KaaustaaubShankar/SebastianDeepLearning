---

title: "Lecture 4 - Tensors and Linear Algebra"
type: "note"
date: 2025-06-11
last_modified_at: 2025-06-11
order: 4
tags:
- deep-learning-with-sebastian
- ml
- deep-learning
- linear-algebra
draft: false
toc: true
---

## Tensors in Deep Learning

* **Scalar:** Rank-0 tensor (single number).

* **Vector:** Rank-1 tensor $\mathbf{x} \in \mathbb{R}^{n \times 1}$

  $$
  \mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}
  $$

* **Matrix:** Rank-2 tensor $\mathbf{X} \in \mathbb{R}^{m \times n}$

  * Typically represented as $\mathbf{X} \in \mathbb{R}^{n \times m}$ in AI, where:
  * $x_{m}^{[n]}$ indicates the $m$-th feature of the $n$-th training example.

## Vectors, Matrices, and Broadcasting

### Net Input Calculation

* **Single example:**

  $$
  \mathbf{w}^{T}\mathbf{x} + b = z
  $$

* **Online training:** Update weights after each training example.

### Batch Computation

* Compute multiple outputs simultaneously:

  $$
  \mathbf{Xw} + b = \mathbf{z}
  $$

* Dimensions:

  * $\mathbf{X}$: $n \times m$
  * $\mathbf{w}$: $m \times 1$
  * $\mathbf{z}$: $n \times 1$

## Notational Conventions in Neural Networks

* **Single training example:**

  $$
  \mathbf{x}^{T}\mathbf{w} + b = z
  $$

* **Multiple training examples:**

  $$
  \mathbf{Xw} + b = \mathbf{z}, \quad \text{where } \mathbf{X} \in \mathbb{R}^{n \times m}
  $$

* General matrix multiplication: $(n \times m) \cdot (m \times h) = (n \times h)$

## Fully Connected Layer in PyTorch

### Single Training Example

$$
\sigma(\mathbf{xW}^{T} + b) = \mathbf{a}
$$

* Dimensions:

  * $\mathbf{x}$: $1 \times m$
  * $\mathbf{W}$: $h \times m$
  * $\mathbf{a}, b$: $1 \times h$

### Multiple Training Examples

$$
\sigma(\mathbf{XW}^{T} + b) = \mathbf{A}
$$

* Dimensions:

  * $\mathbf{X}$: $n \times m$
  * $\mathbf{W}$: $h \times m$
  * $\mathbf{A}, b$: $n \times h$

---
