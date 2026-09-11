---
title: Lecture 12 - Improving Gradient Descent-based Optimization
type: note
date: 2025-06-14
last_modified_at: 2025-06-14
order: 12
tags:
  - deep-learning-with-sebastian
  - ml
  - deep-learning
  - gradient-descent
  - optimization
draft: false
toc: true
---
## Learning Rate Decay

From [[Lecture 5 - Training Modes and Linear Regression]] , we know that minibatch is used because the gradient is noisier allowing it to escape local minima. But this leads to oscillation in loss over epochs. So how do we combat this? By **decreasing** the learning rate over time

> Note: You should choose the batch size that is as large as your GPU memory and make it proportional to the number of classes in the dataset. Lower batch size may result in better loss but is very bumpy and not as stable as a higher batch size.

There are a couple of ways to go about this:

1. **Exponential Decay**: $\eta_t = \eta_0 \cdot e^{-k\cdot t}$ where $k$ is the decay rate and $t$ is the epoch rate
2. **Cyclical Learning Rates**: learning rate varies between a set of bounds over epochs depending on a policy
	1. **Triangular**:  low → high → low
	2. **Triangular 2**: low → high → low → high/2 → low .....
	3. **Exp Range**: like triangular 2 but we scale down $\eta$ exponentially after each cycle

However, generally higher mini batch size $\leftarrow\rightarrow$  decaying learning rate

## Momentum
Momentum in optimization is kind of like rolling a ball down a hilly landscape, where the two axes represent the model’s parameters and the height represents the loss. Regular gradient descent just follows the steepest path downhill at every step, which can make it wobble back and forth or get stuck in small dips. Momentum fixes this by remembering the direction it's been going and picking up speed in that direction, so it moves more smoothly and doesn't get thrown off by every little bump. This makes it better at finding the lowest point without getting stuck or slowed down too much.

### Method
1. We compute the velocity at time $t$ using the previous velocity, our friction parameter $\alpha$ (more alpha means less friction), and our loss gradient
$$
\triangle w_{i,j}(t) = \alpha \cdot \triangle w_{i,j}(t-1) + \eta\cdot \frac{\partial \mathcal{L}}{\partial w_{i,j}}(t)
$$
2. We update the weight
$$
w_{i,j}(t+1) = w_{i,j}(t)-\triangle w_{i,j}(t)
$$
## Adaptive Learning Rate
* Decrease $\eta$ if gradient changes direction
* Increase $\eta$ if gradient stays consistent

### Simple Method
1. Define a local gain (g) for each weight initialized with $g = 1$ $$
	\triangle w_{i,j} = \eta \cdot g_{i,j}\cdot \frac{\partial \mathcal{L}}{\partial w_{i,j}}
	$$
2. If gradient is consistent, then $g_{i,j}(t) = g_{i,j}(t-1) + \beta$
3. If gradient is not consistent, then $g_{i,j}(t) = g_{i,j}(t-1)\cdot(1-\beta)$

### RMS (Root Mean Squared) Prop
1. Calculate the gradient of loss w.r.t each weight 
$$
g_{i,j}(t) = \frac{\partial \mathcal{L}}{w_{i,j}(t)}
$$
2. Calculate the moving average of the squared gradient for each weight:
$$MS(w_{i,j},t) = \beta \cdot MS(w_{i,j},t-1) + (1-\beta)\left(g_{i,j}(t)\right)^2$$ 
3. Update each weight using the adaptive learning rate
$$
w_{i,j}(t+1) = w_{i,j}(t) - \frac{\eta}{\sqrt{MS(w_{i,j}, t)} + \epsilon} \cdot g_{i,j}(t)
$$
> Typically, $\beta \in [0.9,0.999]$ ($\epsilon$ is there to prevent zero-division)

## ADAM (Adaptive Moment Estimation)
With ADAM, we use adaptive learning rates and momentum

### Methods

1. $m_t,v_t = 0$
2. **Calculate the gradient of loss w.r.t each weight**  
$$
g_{i,j}(t) = \frac{\partial \mathcal{L}}{\partial w_{i,j}(t)}
$$

3. **Update the biased first moment estimate (mean of gradients) according to momentum**  
$$
m_{i,j}(t) = \beta_1 \cdot m_{i,j}(t-1) + (1 - \beta_1) \cdot g_{i,j}(t)
$$

3. **Update the biased second moment estimate (squared gradients) according to RMSProp**  
$$
v_{i,j}(t) = \beta_2 \cdot v_{i,j}(t-1) + (1 - \beta_2) \cdot \left(g_{i,j}(t)\right)^2
$$

4. **Compute bias-corrected moment estimates**  
$$
\hat{m}_{i,j}(t) = \frac{m_{i,j}(t)}{1 - \beta_1^t}
$$  
$$
\hat{v}_{i,j}(t) = \frac{v_{i,j}(t)}{1 - \beta_2^t}
$$

5. **Update the weights**  
$$
w_{i,j}(t+1) = w_{i,j}(t) - \frac{\eta}{\sqrt{\hat{v}_{i,j}(t)} + \epsilon} \cdot \hat{m}_{i,j}(t)
$$
> Note: $\beta_1, \beta_2 \in [0,1)$



### In PyTorch
```python
optimizers = torch.optim.Adam(model.parameters, lr = lr, betas = (0.9,0.999))
```

### How does it fare against others?
Adaptive methods like ADAM have been found to generalize **poorly** compared to SGD.