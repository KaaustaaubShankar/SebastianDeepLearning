---
title: "Lecture 1 - Introduction to Deep Learning"
type: "note"
date: 2025-06-11
last_modified_at: 2025-06-11
order: 1
tags:
- ml
- deep-learning-with-sebastian
- deep-learning
- supervised-learning
- unsupervised-learning
- reinforcement-learning
- self-supervised-learning
- semi-supervised-learning
draft: false
toc: true
---

## What is Machine Learning?

* **AI vs Machine Learning:**

  * **AI (Artificial Intelligence):** Involves logic rules and explicitly programmed nested conditional statements.
  * **Machine Learning (ML):** Algorithms learn from data to generalize patterns.

* **Examples:**

  * AI: Logic rules, handcrafted nested if-else statements
  * ML: Generalized linear models, tree-based methods, shallow neural networks, Support Vector Machines (SVM)

## Types of Machine Learning

### Supervised Learning

* **Characteristics:**

  * Uses labeled data
  * Provides direct feedback
  * Predicts future outcomes or events

* **Tasks:**

  * **Regression:** Predict continuous values
  * **Classification:** Categorize data into discrete groups
  * **Ordinal Classification:** Rank data based on order (relative positions)

### Unsupervised Learning

* **Characteristics:**

  * Unlabeled data
  * No direct feedback
  * Finds hidden structures or patterns within the data

* **Tasks:**

  * **Principal Component Analysis (PCA)**: Identifies major axes of variation
  * **Representation Learning / Dimensionality Reduction:** Reduces data dimensionality while preserving important properties
  * **Clustering:** Groups data based on similarity without explicit labels

### Reinforcement Learning

* **Characteristics:**

  * Decision-making process
  * Driven by a reward system
  * Learns sequences of actions to maximize cumulative reward

### Semi-Supervised Learning

* Combines elements of supervised and unsupervised learning
* Partial labeling: some data points have labels, others do not
* Utilizes labeled data to predict and assign labels to unlabeled data

### Self-Supervised Learning

* Generates labels automatically from the data itself (no external annotation)
* Often used in language modeling, computer vision, and representation learning

## Types of Data

* **Structured Data:** Databases, spreadsheets (tabular format)
* **Unstructured Data:** Images (pixels), audio signals, text (natural language sentences)
