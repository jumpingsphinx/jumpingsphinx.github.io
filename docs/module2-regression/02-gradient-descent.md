# Lesson 2: Gradient Descent

## Introduction

Gradient descent is **the optimization algorithm that powers modern machine learning**. From linear regression to deep neural networks with billions of parameters, gradient descent is how models learn from data. Understanding this algorithm deeply is essential for any machine learning practitioner.

!!! quote "Why This Matters"
    "If there's one algorithm you must understand completely in machine learning, it's gradient descent. It's not just an optimization technique - it's the fundamental learning mechanism for neural networks, the training engine for most ML models, and the key to understanding how machines actually 'learn'."

In this lesson, you'll learn how gradient descent works from the ground up, implement it from scratch, and gain the intuition needed to debug and optimize machine learning models.

## The Core Problem: Finding the Minimum

### Visual Introduction to Gradient Descent

Before diving into the mathematics, watch this excellent visual explanation of how gradient descent works:

<div style="text-align: center; margin: 20px 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/sDv4f4s2SB8" title="Gradient Descent by 3Blue1Brown" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

Machine learning is fundamentally an optimization problem. We have a **cost function** (also called a **loss function**) that measures how wrong our model's predictions are, and we want to find the model parameters that **minimize** this cost.

### A Simple Analogy

Imagine you're blindfolded on a hilly terrain, and your goal is to reach the lowest point in the valley. You can:

1. Feel the slope of the ground beneath your feet
2. Take a step in the direction that goes downhill
3. Repeat until you can't go any lower

This is exactly what gradient descent does! The "slope" is the **gradient** (derivative) of the cost function, and each "step" updates the model parameters.

### Mathematical Formulation

For a model with parameters $\theta$ (which could be weights $w$ and bias $b$), we want to:

$$\text{minimize}_\theta \quad J(\theta)$$

where $J(\theta)$ is our cost function that measures prediction error.

**Gradient descent update rule:**

$$\theta := \theta - \alpha \nabla J(\theta)$$

where:
- $\theta$ = model parameters (weights and bias)
- $\alpha$ = learning rate (step size)
- $\nabla J(\theta)$ = gradient (partial derivatives) of cost function
- $:=$ means "update to"

Let's break down each component in detail.

## Cost Functions: Measuring Model Error

A cost function (loss function) quantifies how far our model's predictions are from the actual values. Choosing the right cost function is crucial - it defines what "good" means for your model.

### Mean Squared Error (MSE)

The **most common cost function for regression** is Mean Squared Error:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2$$

where:
- $m$ = number of training examples
- $h(x^{(i)}) = wx^{(i)} + b$ = predicted value for example $i$
- $y^{(i)}$ = actual value for example $i$
- The $\frac{1}{2}$ is a convenience factor that cancels when taking derivatives

**Why square the errors?**

1. **Penalizes large errors more**: An error of 10 is penalized 100 times more than an error of 1
2. **Always positive**: $(-5)^2 = 25$, same as $(5)^2$, so errors in both directions count
3. **Differentiable**: Smooth function, easy to compute gradients
4. **Statistical foundation**: Assumes errors are normally distributed

!!! example "MSE Intuition"
    If your model predicts house prices, MSE heavily penalizes being off by $100,000 more than being off by $10,000. This makes sense - we really want to avoid large mistakes!

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# True values vs predictions
y_true = np.array([100, 150, 200, 250, 300])
y_pred_good = np.array([105, 145, 205, 248, 295])  # Close predictions
y_pred_bad = np.array([120, 140, 180, 270, 320])   # Worse predictions

def mse(y_true, y_pred):
    """Calculate Mean Squared Error."""
    m = len(y_true)
    return (1/(2*m)) * np.sum((y_pred - y_true)**2)

# Calculate MSE for both
mse_good = mse(y_true, y_pred_good)
mse_bad = mse(y_true, y_pred_bad)

print("True values:     ", y_true)
print("Good predictions:", y_pred_good)
print("Bad predictions: ", y_pred_bad)
print(f"\nMSE (good model): {mse_good:.2f}")
print(f"MSE (bad model):  {mse_bad:.2f}")
print(f"\nThe bad model has {mse_bad/mse_good:.1f}x higher cost!")

# Visualize individual errors
errors_good = y_pred_good - y_true
errors_bad = y_pred_bad - y_true

print("\nIndividual errors (good):", errors_good)
print("Squared errors (good):   ", errors_good**2)
print(f"Sum: {np.sum(errors_good**2):.2f}")

print("\nIndividual errors (bad): ", errors_bad)
print("Squared errors (bad):    ", errors_bad**2)
print(f"Sum: {np.sum(errors_bad**2):.2f}")
```
</div>

**Key Insight:** MSE is highly sensitive to outliers because errors are squared. A single prediction that's way off can dominate the cost.

### Mean Absolute Error (MAE)

An alternative cost function that's more robust to outliers:

$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} |h(x^{(i)}) - y^{(i)})|$$

**Differences from MSE:**

- Uses absolute value instead of squaring
- Penalizes all errors proportionally (linear penalty)
- More robust to outliers
- Less smooth (has a corner at zero), slightly harder to optimize

<div class="python-interactive" markdown="1">
```python
import numpy as np

def mae(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    m = len(y_true)
    return (1/m) * np.sum(np.abs(y_pred - y_true))

# Example with an outlier
y_true = np.array([100, 150, 200, 250, 300])
y_pred = np.array([105, 145, 205, 248, 500])  # Last prediction is way off!

mse_val = (1/(2*len(y_true))) * np.sum((y_pred - y_true)**2)
mae_val = mae(y_true, y_pred)

print("True values:    ", y_true)
print("Predictions:    ", y_pred)
print("Errors:         ", y_pred - y_true)
print(f"\nMSE: {mse_val:.2f}")
print(f"MAE: {mae_val:.2f}")

# Compare error contributions
errors = y_pred - y_true
print("\nContribution to MSE:", (errors**2) / (2*len(y_true)))
print("Contribution to MAE:", np.abs(errors) / len(y_true))

print("\n🔍 Notice: The outlier (200 error) dominates MSE but not MAE")
```
</div>

### Root Mean Squared Error (RMSE)

Often used for interpretability because it's in the same units as the target variable:

$$RMSE = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})^2}$$

**When to use each:**

- **MSE**: Default choice for regression, emphasizes large errors
- **MAE**: When outliers shouldn't dominate, want equal penalty for all errors
- **RMSE**: For reporting (same units as target), similar properties to MSE

<div class="python-interactive" markdown="1">
```python
import numpy as np

# House price predictions (in thousands of dollars)
y_true = np.array([200, 350, 150, 500, 400])
y_pred = np.array([210, 340, 160, 480, 420])

def mse(y_true, y_pred):
    return (1/(2*len(y_true))) * np.sum((y_pred - y_true)**2)

def rmse(y_true, y_pred):
    return np.sqrt((1/len(y_true)) * np.sum((y_pred - y_true)**2))

def mae(y_true, y_pred):
    return (1/len(y_true)) * np.sum(np.abs(y_pred - y_true))

print("Predicted vs Actual House Prices (in $1000s):")
print(f"Predictions: {y_pred}")
print(f"Actual:      {y_true}")
print(f"Errors:      {y_pred - y_true}")

print(f"\nMSE:  {mse(y_true, y_pred):.2f} (squared thousands)")
print(f"RMSE: {rmse(y_true, y_pred):.2f} (thousands of dollars)")
print(f"MAE:  {mae(y_true, y_pred):.2f} (thousands of dollars)")

print("\n💡 RMSE and MAE are in the same units as the original data,")
print("   making them easier to interpret!")
```
</div>

### Visualizing the Cost Function

For a simple linear model $h(x) = wx + b$, the cost function $J(w, b)$ is a surface in 3D space. Let's visualize this:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Simple dataset: y = 2x + 1 (true relationship)
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5])
y = 2 * X + 1 + np.random.randn(5) * 0.5  # Add small noise

def cost_function(w, b, X, y):
    """Calculate MSE cost for given w and b."""
    m = len(y)
    predictions = w * X + b
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

# Create grid of w and b values
w_vals = np.linspace(-1, 5, 100)
b_vals = np.linspace(-2, 4, 100)
W, B = np.meshgrid(w_vals, b_vals)

# Calculate cost for each (w, b) pair
costs = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        costs[i, j] = cost_function(W[i, j], B[i, j], X, y)

# Plot the cost surface
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 3D surface plot would go here, but we'll use contour plot for browser
contour = ax1.contour(W, B, costs, levels=20, cmap='viridis')
ax1.clabel(contour, inline=True, fontsize=8)
ax1.set_xlabel('Weight (w)')
ax1.set_ylabel('Bias (b)')
ax1.set_title('Cost Function J(w, b) - Contour Plot')
ax1.plot(2, 1, 'r*', markersize=15, label='True minimum')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Cross-section: fix b=1, vary w
b_fixed = 1.0
costs_w = [cost_function(w, b_fixed, X, y) for w in w_vals]
ax2.plot(w_vals, costs_w, 'b-', linewidth=2)
ax2.axvline(x=2, color='r', linestyle='--', label='True w=2')
ax2.set_xlabel('Weight (w)')
ax2.set_ylabel('Cost J(w, b=1)')
ax2.set_title('Cost Function (b fixed at 1)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
fig

print(f"Data: X = {X}")
print(f"      y = {y}")
print(f"\nTrue parameters: w=2, b=1")
print(f"Cost at true params: {cost_function(2, 1, X, y):.4f}")
print(f"Cost at w=0, b=0: {cost_function(0, 0, X, y):.4f}")
print(f"Cost at w=4, b=2: {cost_function(4, 2, X, y):.4f}")
print("\n🎯 The minimum of the cost function is where our model is most accurate!")
```
</div>

**Key Observations:**

1. The cost function is **convex** (bowl-shaped) for linear regression with MSE
2. There's a **global minimum** where the cost is lowest
3. Our goal is to find the parameters $(w, b)$ at this minimum
4. Gradient descent follows the slope downhill to reach this minimum

## The Gradient: Direction of Steepest Ascent

The **gradient** $\nabla J(\theta)$ is a vector of partial derivatives that points in the direction of **steepest increase** of the cost function. To minimize cost, we go in the **opposite direction** (hence the minus sign in the update rule).

### Computing Gradients for Linear Regression

For linear regression with MSE cost:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (wx^{(i)} + b - y^{(i)})^2$$

The gradients (using calculus chain rule) are:

$$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (h(x^{(i)}) - y^{(i)})$$

where $h(x^{(i)}) = wx^{(i)} + b$ is our prediction.

**Intuition:**

- If predictions are too high ($h(x) > y$), gradients are positive → decrease $w$ and $b$
- If predictions are too low ($h(x) < y$), gradients are negative → increase $w$ and $b$
- The gradient magnitude tells us how steep the slope is

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Simple dataset
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])  # y = 2x (perfect linear relationship)

def compute_gradients(w, b, X, y):
    """
    Compute gradients of MSE cost with respect to w and b.

    Returns:
        dw: gradient with respect to weight
        db: gradient with respect to bias
    """
    m = len(y)

    # Predictions
    predictions = w * X + b

    # Errors
    errors = predictions - y

    # Gradients
    dw = (1/m) * np.sum(errors * X)
    db = (1/m) * np.sum(errors)

    return dw, db

# Test with different parameter values
test_cases = [
    (0, 0, "Starting from zero"),
    (1, 0, "w too small"),
    (3, 0, "w too large"),
    (2, 0, "w correct, b correct"),
    (2, 5, "w correct, b too large")
]

for w, b, description in test_cases:
    dw, db = compute_gradients(w, b, X, y)
    predictions = w * X + b
    cost = (1/(2*len(y))) * np.sum((predictions - y)**2)

    print(f"\n{description}: w={w}, b={b}")
    print(f"  Cost: {cost:.4f}")
    print(f"  Gradient dw: {dw:+.4f}")
    print(f"  Gradient db: {db:+.4f}")

    if abs(dw) < 0.01 and abs(db) < 0.01:
        print("  ✓ Near minimum (gradients ≈ 0)!")
    elif dw > 0:
        print("  → Should decrease w")
    else:
        print("  → Should increase w")
```
</div>

**Key Insight:** When gradients are close to zero, we're near a minimum (slope is flat). This is our convergence criterion!

## Gradient Descent Algorithm

Now we can put it all together. Here's the complete gradient descent algorithm:

**Algorithm:**

1. Initialize parameters $w$ and $b$ (usually to small random values or zeros)
2. Repeat until convergence:
   - Compute predictions: $h(x^{(i)}) = wx^{(i)} + b$ for all training examples
   - Compute gradients: $\frac{\partial J}{\partial w}$ and $\frac{\partial J}{\partial b}$
   - Update parameters:
     - $w := w - \alpha \frac{\partial J}{\partial w}$
     - $b := b - \alpha \frac{\partial J}{\partial b}$
   - Optionally: compute and store cost $J(w, b)$ for monitoring

**Convergence:** Stop when gradients are very small, cost stops decreasing, or after a maximum number of iterations.

### Implementation from Scratch

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000):
    """
    Perform gradient descent to learn w and b.

    Parameters:
        X: Training data (features)
        y: Target values
        learning_rate: Step size (alpha)
        num_iterations: Number of iterations to run

    Returns:
        w, b: Learned parameters
        cost_history: Cost at each iteration (for visualization)
    """
    # Initialize parameters
    w = 0.0
    b = 0.0
    m = len(y)

    cost_history = []

    for iteration in range(num_iterations):
        # 1. Make predictions
        predictions = w * X + b

        # 2. Compute cost (for monitoring)
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

        # 3. Compute gradients
        errors = predictions - y
        dw = (1/m) * np.sum(errors * X)
        db = (1/m) * np.sum(errors)

        # 4. Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration:4d}: Cost = {cost:.4f}, w = {w:.4f}, b = {b:.4f}")

    return w, b, cost_history

# Generate dataset: y = 3x + 2 + noise
np.random.seed(42)
X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = 3 * X_train + 2 + np.random.randn(10) * 1.5

# Run gradient descent
print("Training with Gradient Descent...")
print("True parameters: w=3, b=2\n")
w_learned, b_learned, cost_history = gradient_descent(
    X_train, y_train,
    learning_rate=0.01,
    num_iterations=1000
)

print(f"\n✓ Training complete!")
print(f"Learned parameters: w={w_learned:.4f}, b={b_learned:.4f}")
print(f"Final cost: {cost_history[-1]:.4f}")

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Data and learned line
ax1.scatter(X_train, y_train, color='blue', s=50, alpha=0.6, label='Training data')
x_line = np.linspace(0, 11, 100)
y_pred = w_learned * x_line + b_learned
ax1.plot(x_line, y_pred, 'r-', linewidth=2, label=f'Learned: y={w_learned:.2f}x+{b_learned:.2f}')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression with Gradient Descent')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cost history (learning curve)
ax2.plot(cost_history, 'b-', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost J(w, b)')
ax2.set_title('Cost Function During Training')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')  # Log scale to see convergence better

plt.tight_layout()
fig

print("\n📊 The decreasing cost curve shows the model is learning!")
```
</div>

**What's happening here?**

1. We start with $w=0, b=0$ (random initialization)
2. Each iteration, we compute how wrong our predictions are
3. Gradients tell us which direction to adjust $w$ and $b$
4. We take small steps (controlled by learning rate) in that direction
5. Over time, the cost decreases and we converge to good parameters

## Learning Rate: The Most Important Hyperparameter

The learning rate $\alpha$ controls how big each step is. Choosing the right learning rate is **critical**:

- **Too small**: Convergence is very slow, may take millions of iterations
- **Too large**: May overshoot the minimum, oscillate, or even diverge
- **Just right**: Fast convergence to the minimum

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Simple dataset
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5])
y = 2 * X + 1 + np.random.randn(5) * 0.3

def gradient_descent_with_lr(X, y, learning_rate, num_iterations=100):
    """Run gradient descent and return cost history."""
    w, b = 0.0, 0.0
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        predictions = w * X + b
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

        errors = predictions - y
        dw = (1/m) * np.sum(errors * X)
        db = (1/m) * np.sum(errors)

        w = w - learning_rate * dw
        b = b - learning_rate * db

    return cost_history, w, b

# Test different learning rates
learning_rates = [0.001, 0.01, 0.1, 0.5]
colors = ['blue', 'green', 'orange', 'red']

fig, ax = plt.subplots(figsize=(12, 6))

for lr, color in zip(learning_rates, colors):
    cost_history, w_final, b_final = gradient_descent_with_lr(X, y, lr, num_iterations=50)
    ax.plot(cost_history, color=color, linewidth=2, label=f'α={lr} (final cost: {cost_history[-1]:.3f})')
    print(f"Learning rate α={lr:5.3f}: Final w={w_final:.4f}, b={b_final:.4f}, Cost={cost_history[-1]:.4f}")

ax.set_xlabel('Iteration')
ax.set_ylabel('Cost J(w, b)')
ax.set_title('Effect of Learning Rate on Convergence')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
fig

print("\n🔍 Observations:")
print("  • Too small (0.001): Slow convergence")
print("  • Good (0.01, 0.1): Fast, smooth convergence")
print("  • Too large (0.5): May oscillate or diverge")
```
</div>

**How to choose learning rate:**

1. **Start with 0.01 or 0.1** and adjust
2. **Plot the cost curve**: Should decrease smoothly
3. **If oscillating**: Decrease learning rate by 10x
4. **If too slow**: Increase learning rate by 2-3x
5. **Use learning rate schedules**: Start large, decrease over time

### Learning Rate Schedules

Instead of a fixed learning rate, we can decrease it over time:

$$\alpha_t = \frac{\alpha_0}{1 + decay\_rate \cdot t}$$

<div class="python-interactive" markdown="1">
```python
import numpy as np

def learning_rate_schedule(initial_lr, iteration, decay_rate=0.01):
    """Decrease learning rate over time."""
    return initial_lr / (1 + decay_rate * iteration)

# Visualize schedule
initial_lr = 0.1
iterations = np.arange(0, 1000)
lrs = [learning_rate_schedule(initial_lr, t, decay_rate=0.01) for t in iterations]

print("Learning Rate Schedule:")
for i in [0, 100, 500, 999]:
    print(f"  Iteration {i:4d}: α = {lrs[i]:.6f}")

print("\n💡 Starting with larger steps, then taking smaller, more precise steps")
print("   as we get closer to the minimum!")
```
</div>

## Variants of Gradient Descent

There are three main variants based on how much data we use per update:

### 1. Batch Gradient Descent (BGD)

**What we've been doing so far.** Uses **all training examples** to compute gradients.

**Pros:**
- Stable convergence
- Guaranteed to converge to global minimum (for convex functions)
- Exact gradient computation

**Cons:**
- Slow for large datasets (must process all data per update)
- Requires entire dataset in memory
- Can get stuck in local minima (for non-convex functions)

$$\theta := \theta - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta \mathcal{L}(h_\theta(x^{(i)}), y^{(i)})$$

### 2. Stochastic Gradient Descent (SGD)

Uses **one random training example** at a time to compute gradients.

**Pros:**
- Much faster updates (don't wait for entire dataset)
- Can escape shallow local minima (due to noise)
- Enables online learning (update as new data arrives)

**Cons:**
- Noisy updates, cost function fluctuates
- Never truly converges (oscillates around minimum)
- May need learning rate decay

$$\theta := \theta - \alpha \cdot \nabla_\theta \mathcal{L}(h_\theta(x^{(i)}), y^{(i)})$$

for randomly selected example $i$

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Dataset
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = 2 * X + 1 + np.random.randn(10) * 0.5

def batch_gradient_descent(X, y, learning_rate=0.01, num_iterations=100):
    """Standard batch gradient descent."""
    w, b = 0.0, 0.0
    m = len(y)
    cost_history = []

    for iteration in range(num_iterations):
        predictions = w * X + b
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

        errors = predictions - y
        dw = (1/m) * np.sum(errors * X)
        db = (1/m) * np.sum(errors)

        w = w - learning_rate * dw
        b = b - learning_rate * db

    return cost_history

def stochastic_gradient_descent(X, y, learning_rate=0.01, num_epochs=10):
    """Stochastic gradient descent (one example at a time)."""
    w, b = 0.0, 0.0
    m = len(y)
    cost_history = []

    for epoch in range(num_epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(m)

        for i in indices:
            # Single example
            x_i = X[i]
            y_i = y[i]

            # Prediction for this example
            prediction = w * x_i + b

            # Cost on entire dataset (for monitoring)
            all_predictions = w * X + b
            cost = (1/(2*m)) * np.sum((all_predictions - y)**2)
            cost_history.append(cost)

            # Gradient from single example
            error = prediction - y_i
            dw = error * x_i
            db = error

            # Update
            w = w - learning_rate * dw
            b = b - learning_rate * db

    return cost_history

# Compare both
print("Comparing Batch GD vs Stochastic GD...\n")

cost_batch = batch_gradient_descent(X, y, learning_rate=0.01, num_iterations=100)
cost_sgd = stochastic_gradient_descent(X, y, learning_rate=0.01, num_epochs=10)

print(f"Batch GD:      {len(cost_batch)} iterations, final cost: {cost_batch[-1]:.4f}")
print(f"Stochastic GD: {len(cost_sgd)} updates, final cost: {cost_sgd[-1]:.4f}")

# Plot comparison
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(cost_batch, 'b-', linewidth=2, label='Batch GD (smooth)', alpha=0.8)
ax.plot(range(0, len(cost_sgd), 10), cost_sgd[::10], 'r-', linewidth=1,
        label='SGD (noisy, sampled every 10 updates)', alpha=0.6)

ax.set_xlabel('Iteration / Update')
ax.set_ylabel('Cost J(w, b)')
ax.set_title('Batch Gradient Descent vs Stochastic Gradient Descent')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
fig

print("\n🔍 Notice:")
print("  • Batch GD: Smooth convergence")
print("  • SGD: Noisy but makes progress faster (more updates)")
```
</div>

### 3. Mini-Batch Gradient Descent

**The best of both worlds.** Uses a **small batch** of examples (typically 32, 64, 128, or 256).

**Pros:**
- More stable than SGD
- Faster than batch GD
- Efficient use of vectorized operations (GPU acceleration)
- Most commonly used in practice

**Cons:**
- Additional hyperparameter (batch size) to tune

$$\theta := \theta - \alpha \cdot \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \mathcal{L}(h_\theta(x^{(i)}), y^{(i)})$$

where $B$ is a mini-batch of examples

<div class="python-interactive" markdown="1">
```python
import numpy as np

def mini_batch_gradient_descent(X, y, learning_rate=0.01, batch_size=4, num_epochs=20):
    """Mini-batch gradient descent."""
    w, b = 0.0, 0.0
    m = len(y)
    cost_history = []

    for epoch in range(num_epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Process mini-batches
        for i in range(0, m, batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_m = len(X_batch)

            # Predictions on batch
            predictions = w * X_batch + b

            # Gradients from batch
            errors = predictions - y_batch
            dw = (1/batch_m) * np.sum(errors * X_batch)
            db = (1/batch_m) * np.sum(errors)

            # Update
            w = w - learning_rate * dw
            b = b - learning_rate * db

            # Cost on full dataset (for monitoring)
            all_predictions = w * X + b
            cost = (1/(2*m)) * np.sum((all_predictions - y)**2)
            cost_history.append(cost)

    return cost_history, w, b

# Run mini-batch GD
np.random.seed(42)
X = np.linspace(1, 10, 100)
y = 2 * X + 1 + np.random.randn(100) * 1.0

cost_history, w, b = mini_batch_gradient_descent(
    X, y,
    learning_rate=0.01,
    batch_size=32,
    num_epochs=20
)

print(f"Mini-Batch Gradient Descent (batch_size=32)")
print(f"Dataset size: {len(X)} examples")
print(f"Number of batches per epoch: {len(X)//32}")
print(f"Total updates: {len(cost_history)}")
print(f"\nLearned parameters: w={w:.4f}, b={b:.4f}")
print(f"Final cost: {cost_history[-1]:.4f}")
print(f"\n💡 Mini-batch GD is the most common choice for deep learning!")
```
</div>

**Comparison Summary:**

| Method | Data per Update | Speed | Stability | Use Case |
|--------|----------------|-------|-----------|----------|
| **Batch GD** | All examples | Slow | Very stable | Small datasets |
| **SGD** | 1 example | Fast | Noisy | Online learning |
| **Mini-Batch GD** | 32-256 examples | Fast | Good | **Most common** |

## Convergence: When to Stop

How do we know when gradient descent has converged?

### 1. Gradient Magnitude

Stop when gradients become very small (close to zero):

$$\|\nabla J(\theta)\| < \epsilon$$

where $\epsilon$ is a small threshold (e.g., $10^{-6}$)

### 2. Cost Change

Stop when cost stops decreasing significantly:

$$|J^{(t)} - J^{(t-1)}| < \epsilon$$

### 3. Maximum Iterations

Always set a maximum number of iterations to prevent infinite loops.

<div class="python-interactive" markdown="1">
```python
import numpy as np

def gradient_descent_with_convergence(X, y, learning_rate=0.01,
                                     max_iterations=10000,
                                     tolerance=1e-6):
    """
    Gradient descent with multiple convergence criteria.
    """
    w, b = 0.0, 0.0
    m = len(y)
    cost_history = []

    for iteration in range(max_iterations):
        # Predictions and cost
        predictions = w * X + b
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        cost_history.append(cost)

        # Compute gradients
        errors = predictions - y
        dw = (1/m) * np.sum(errors * X)
        db = (1/m) * np.sum(errors)

        # Check convergence: gradient magnitude
        gradient_magnitude = np.sqrt(dw**2 + db**2)
        if gradient_magnitude < tolerance:
            print(f"✓ Converged at iteration {iteration}: gradient magnitude = {gradient_magnitude:.2e}")
            break

        # Check convergence: cost change
        if iteration > 0:
            cost_change = abs(cost_history[-1] - cost_history[-2])
            if cost_change < tolerance:
                print(f"✓ Converged at iteration {iteration}: cost change = {cost_change:.2e}")
                break

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db
    else:
        print(f"⚠ Reached max iterations ({max_iterations})")

    return w, b, cost_history

# Test convergence detection
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5])
y = 2 * X + 1 + np.random.randn(5) * 0.2

w, b, costs = gradient_descent_with_convergence(
    X, y,
    learning_rate=0.1,
    max_iterations=10000,
    tolerance=1e-6
)

print(f"\nFinal parameters: w={w:.6f}, b={b:.6f}")
print(f"Total iterations: {len(costs)}")
print(f"Final cost: {costs[-1]:.8f}")
```
</div>

## Complete Example: Gradient Descent in Action

Let's put everything together with a complete, well-documented implementation:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    """
    Linear Regression using Gradient Descent.

    Parameters:
        learning_rate: Step size for gradient descent
        num_iterations: Maximum number of iterations
        tolerance: Convergence threshold
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.tolerance = tolerance
        self.w = None
        self.b = None
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the model using gradient descent.

        Parameters:
            X: Training features (1D array)
            y: Target values (1D array)
        """
        # Initialize parameters
        self.w = 0.0
        self.b = 0.0
        m = len(y)

        # Gradient descent
        for iteration in range(self.num_iterations):
            # Predictions
            predictions = self.w * X + self.b

            # Cost
            cost = (1/(2*m)) * np.sum((predictions - y)**2)
            self.cost_history.append(cost)

            # Gradients
            errors = predictions - y
            dw = (1/m) * np.sum(errors * X)
            db = (1/m) * np.sum(errors)

            # Check convergence
            if iteration > 0:
                if abs(self.cost_history[-1] - self.cost_history[-2]) < self.tolerance:
                    print(f"Converged at iteration {iteration}")
                    break

            # Update parameters
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

        return self

    def predict(self, X):
        """Make predictions."""
        return self.w * X + self.b

    def score(self, X, y):
        """Compute R² score."""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)

# Generate dataset
np.random.seed(42)
X_train = np.linspace(0, 10, 50)
y_train = 3 * X_train + 7 + np.random.randn(50) * 2

# Train model
model = LinearRegressionGD(learning_rate=0.01, num_iterations=1000)
model.fit(X_train, y_train)

print(f"Learned equation: y = {model.w:.3f}x + {model.b:.3f}")
print(f"R² score: {model.score(X_train, y_train):.4f}")
print(f"Training iterations: {len(model.cost_history)}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Regression line
ax1.scatter(X_train, y_train, alpha=0.6, s=50, label='Training data')
x_line = np.linspace(0, 10, 100)
y_pred = model.predict(x_line)
ax1.plot(x_line, y_pred, 'r-', linewidth=2, label=f'y = {model.w:.2f}x + {model.b:.2f}')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression with Gradient Descent')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Learning curve
ax2.plot(model.cost_history, 'b-', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost J(w, b)')
ax2.set_title('Learning Curve')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
fig

print("\n✓ Model trained successfully!")
```
</div>

## Monitoring and Debugging Gradient Descent

### Common Problems and Solutions

**Problem 1: Cost is increasing**
- **Cause**: Learning rate too high
- **Solution**: Decrease learning rate by 10x

**Problem 2: Cost decreases very slowly**
- **Cause**: Learning rate too small
- **Solution**: Increase learning rate by 2-3x

**Problem 3: Cost oscillates**
- **Cause**: Learning rate too high or features not scaled
- **Solution**: Decrease learning rate, normalize features

**Problem 4: Cost stuck at high value**
- **Cause**: Poor initialization, bad features, or bug in code
- **Solution**: Check implementation, try different initialization, check feature scaling

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate different failure modes
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5])
y = 2 * X + 1

def test_learning_rate(X, y, lr, num_iter=50):
    """Test a specific learning rate."""
    w, b = 0.0, 0.0
    m = len(y)
    costs = []

    for _ in range(num_iter):
        pred = w * X + b
        cost = (1/(2*m)) * np.sum((pred - y)**2)
        costs.append(cost)

        err = pred - y
        dw = (1/m) * np.sum(err * X)
        db = (1/m) * np.sum(err)

        w = w - lr * dw
        b = b - lr * db

        # Stop if cost becomes too large (diverging)
        if cost > 1e10:
            break

    return costs

# Test different learning rates
lrs = [0.001, 0.05, 0.15, 0.3]
labels = ['Too small (0.001)', 'Good (0.05)', 'Borderline (0.15)', 'Too large (0.3)']
colors = ['blue', 'green', 'orange', 'red']

fig, ax = plt.subplots(figsize=(12, 6))

for lr, label, color in zip(lrs, labels, colors):
    costs = test_learning_rate(X, y, lr)
    ax.plot(costs, color=color, linewidth=2, label=label)

ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Effect of Learning Rate on Training')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
fig

print("🔍 Debugging Tips:")
print("  1. Always plot the cost function during training")
print("  2. Cost should decrease monotonically (for batch GD)")
print("  3. If cost increases, learning rate is too high")
print("  4. If cost barely changes, learning rate is too small")
print("  5. Feature scaling helps convergence significantly!")
```
</div>

## Gradient Descent vs Analytical Solution

For linear regression, we can actually solve for optimal parameters directly using the **Normal Equation**:

$$w = (X^T X)^{-1} X^T y$$

**When to use each:**

| Gradient Descent | Normal Equation |
|-----------------|-----------------|
| Works for any model | Only for linear regression |
| Scales to large datasets | Slow for large features ($O(n^3)$) |
| Need to choose learning rate | No hyperparameters |
| Iterative (multiple passes) | One-step solution |
| **Use for:** Large datasets, non-linear models | **Use for:** Small datasets, quick prototyping |

<div class="python-interactive" markdown="1">
```python
import numpy as np
import time

# Generate dataset
np.random.seed(42)
X = np.random.randn(100, 1)
y = 3 * X.flatten() + 2 + np.random.randn(100) * 0.5

# Method 1: Gradient Descent
def gradient_descent_solution(X, y, lr=0.01, iterations=1000):
    w, b = 0.0, 0.0
    m = len(y)
    for _ in range(iterations):
        pred = w * X.flatten() + b
        err = pred - y
        w = w - lr * (1/m) * np.sum(err * X.flatten())
        b = b - lr * (1/m) * np.sum(err)
    return w, b

# Method 2: Normal Equation (closed-form)
def normal_equation_solution(X, y):
    # Add bias term (column of 1s)
    X_with_bias = np.c_[np.ones(len(X)), X]
    # Solve: theta = (X^T X)^-1 X^T y
    theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    b, w = theta[0], theta[1]
    return w, b

# Compare both methods
print("Comparing Gradient Descent vs Normal Equation:\n")

start = time.time()
w_gd, b_gd = gradient_descent_solution(X, y, lr=0.1, iterations=1000)
time_gd = time.time() - start

start = time.time()
w_ne, b_ne = normal_equation_solution(X, y)
time_ne = time.time() - start

print(f"Gradient Descent:  w={w_gd:.6f}, b={b_gd:.6f}  (time: {time_gd*1000:.2f}ms)")
print(f"Normal Equation:   w={w_ne:.6f}, b={b_ne:.6f}  (time: {time_ne*1000:.2f}ms)")
print(f"\nDifference in w: {abs(w_gd - w_ne):.8f}")
print(f"Difference in b: {abs(b_gd - b_ne):.8f}")

print("\n💡 Both methods find essentially the same solution!")
print("   GD is iterative but scales better; Normal Eq is exact but expensive for large features.")
```
</div>

## Key Takeaways

!!! success "Essential Concepts"

    **Cost Functions:**
    - MSE is the standard for regression: $J = \frac{1}{2m}\sum(h(x) - y)^2$
    - MAE is more robust to outliers: $J = \frac{1}{m}\sum|h(x) - y|$
    - RMSE is interpretable (same units as target)

    **Gradient Descent:**
    - Update rule: $\theta := \theta - \alpha \nabla J(\theta)$
    - Follows the negative gradient (steepest descent)
    - Learning rate $\alpha$ controls step size

    **Variants:**
    - Batch GD: Use all data (stable, slow)
    - SGD: Use one example (fast, noisy)
    - Mini-batch GD: Use small batches (best of both)

    **Critical Hyperparameter:**
    - Learning rate too high → divergence
    - Learning rate too low → slow convergence
    - Always monitor the cost function!

    **Convergence:**
    - Stop when gradients ≈ 0 or cost stops decreasing
    - Plot cost vs iteration to verify learning
    - Set maximum iterations to prevent infinite loops

## Practice Exercises

!!! tip "Test Your Understanding"

    1. **Implement gradient descent** for multiple features (use matrix operations)
    2. **Try different learning rates** and observe convergence
    3. **Compare batch vs mini-batch** gradient descent on a dataset
    4. **Add feature normalization** and see how it improves convergence
    5. **Implement learning rate decay** schedule
    6. **Debug a diverging model** by adjusting the learning rate

## Next Steps

Now that you understand how to optimize with gradient descent, you're ready to apply it to classification problems!

[Next: Lesson 3 - Logistic Regression](03-logistic-regression.md){ .md-button .md-button--primary }

[Back: Lesson 1 - Linear Regression](01-linear-regression.md){ .md-button }

---

**Questions or Issues?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues).
