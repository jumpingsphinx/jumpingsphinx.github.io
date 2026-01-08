# Lesson 4: Regularization

## Introduction

You've built a model that achieves 99% accuracy on your training data. Success, right? Not necessarily. When you deploy it on real-world data, the accuracy drops to 60%. You've encountered **overfitting** - one of the most fundamental challenges in machine learning.

**Regularization** is the solution. It's a set of techniques that prevent models from becoming too complex, ensuring they generalize well to new, unseen data rather than just memorizing the training set.

!!! quote "Fundamental Trade-off"
    "A model that is too simple underfits the data. A model that is too complex overfits the data. Regularization helps us find the sweet spot in between."

In this lesson, you'll learn what overfitting and underfitting are, understand the bias-variance tradeoff, and master the three main regularization techniques: L1 (Lasso), L2 (Ridge), and Elastic Net.

## The Problem: Overfitting and Underfitting

### Visual Introduction to Regularization

Before getting into the details, watch this excellent explanation of regularization and the bias-variance tradeoff:

<div style="text-align: center; margin: 20px 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/Q81RR3yKn30" title="Regularization by StatQuest" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

### Underfitting (High Bias)

**Underfitting** occurs when a model is **too simple** to capture the underlying patterns in the data.

**Symptoms:**
- Poor performance on training data
- Poor performance on test data
- Model is too simple (e.g., fitting a line to curved data)

**Example:** Trying to predict house prices using only `price = constant` (ignoring all features)

### Overfitting (High Variance)

**Overfitting** occurs when a model is **too complex** and learns noise in the training data rather than the underlying pattern.

**Symptoms:**
- Excellent performance on training data
- Poor performance on test/new data
- Model memorizes training examples instead of learning general patterns

**Example:** Using a 20th-degree polynomial to fit 15 data points - the curve passes through every point but makes wild predictions elsewhere

### Visual Demonstration

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data: y = x^2 + noise
np.random.seed(42)
X_train = np.linspace(-3, 3, 20)
y_train = X_train**2 + np.random.randn(20) * 2

# Test data (more points, less noise)
X_test = np.linspace(-3, 3, 100)
y_test = X_test**2 + np.random.randn(100) * 0.5

# Three models with different complexity
# 1. Underfitting: degree 1 (linear)
coeffs_under = np.polyfit(X_train, y_train, deg=1)
y_pred_under = np.polyval(coeffs_under, X_test)

# 2. Good fit: degree 2 (quadratic)
coeffs_good = np.polyfit(X_train, y_train, deg=2)
y_pred_good = np.polyval(coeffs_good, X_test)

# 3. Overfitting: degree 15 (very high polynomial)
coeffs_over = np.polyfit(X_train, y_train, deg=15)
y_pred_over = np.polyval(coeffs_over, X_test)

# Compute errors
train_err_under = np.mean((np.polyval(coeffs_under, X_train) - y_train)**2)
train_err_good = np.mean((np.polyval(coeffs_good, X_train) - y_train)**2)
train_err_over = np.mean((np.polyval(coeffs_over, X_train) - y_train)**2)

test_err_under = np.mean((y_pred_under - y_test)**2)
test_err_good = np.mean((y_pred_good - y_test)**2)
test_err_over = np.mean((y_pred_over - y_test)**2)

# Visualize
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

for ax, y_pred, title, train_err, test_err in [
    (ax1, y_pred_under, 'Underfitting (Linear)', train_err_under, test_err_under),
    (ax2, y_pred_good, 'Good Fit (Quadratic)', train_err_good, test_err_good),
    (ax3, y_pred_over, 'Overfitting (Degree 15)', train_err_over, test_err_over)
]:
    ax.scatter(X_train, y_train, color='blue', s=50, alpha=0.6,
               label='Training data', zorder=3)
    ax.plot(X_test, y_pred, 'r-', linewidth=2, label='Model')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{title}\nTrain MSE: {train_err:.2f}, Test MSE: {test_err:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 15)

plt.tight_layout()
fig

print("Model Comparison:\n")
print(f"{'Model':<20} {'Train MSE':<12} {'Test MSE':<12} {'Diagnosis'}")
print("-" * 65)
print(f"{'Underfitting':<20} {train_err_under:<12.2f} {test_err_under:<12.2f} {'Too simple, high bias'}")
print(f"{'Good Fit':<20} {train_err_good:<12.2f} {test_err_good:<12.2f} {'Just right!'}")
print(f"{'Overfitting':<20} {train_err_over:<12.2f} {test_err_over:<12.2f} {'Too complex, high variance'}")

print("\n💡 Notice:")
print("  • Underfitting: High error on both train and test")
print("  • Good fit: Low error on both train and test")
print("  • Overfitting: Very low train error, but high test error!")
```
</div>

**Key Insight:** The overfit model (degree 15) has nearly zero training error because it passes through almost every training point, but it performs poorly on test data because it learned noise, not the underlying pattern.

## The Bias-Variance Tradeoff

Understanding the bias-variance tradeoff is essential for knowing when and how to apply regularization.

### Definitions

**Bias** = Error from oversimplifying the model
- High bias → underfitting
- The model makes strong assumptions about the data
- Example: Assuming linear relationship when data is curved

**Variance** = Error from model sensitivity to training data fluctuations
- High variance → overfitting
- The model changes dramatically with small changes in training data
- Example: High-degree polynomial that fits noise

**Total Error** can be decomposed as:

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

where:
- $\text{Bias}^2$ = error from wrong assumptions
- $\text{Variance}$ = error from sensitivity to training data
- $\text{Irreducible Error}$ = noise in the data itself (can't be reduced)

### The Tradeoff

As model complexity increases:
- **Bias decreases** (model can capture more patterns)
- **Variance increases** (model becomes more sensitive to training data)

**Goal:** Find the sweet spot that minimizes total error.

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate bias-variance tradeoff
model_complexity = np.linspace(1, 15, 50)

# Bias decreases with complexity
bias_squared = 10 / model_complexity

# Variance increases with complexity
variance = 0.05 * model_complexity**1.5

# Irreducible error (constant)
irreducible_error = 1.0

# Total error
total_error = bias_squared + variance + irreducible_error

# Find optimal complexity
optimal_idx = np.argmin(total_error)
optimal_complexity = model_complexity[optimal_idx]

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(model_complexity, bias_squared, 'b-', linewidth=2, label='Bias²')
ax.plot(model_complexity, variance, 'r-', linewidth=2, label='Variance')
ax.plot(model_complexity, total_error, 'purple', linewidth=3,
        label='Total Error = Bias² + Variance + Noise', linestyle='--')
ax.axhline(y=irreducible_error, color='gray', linestyle=':',
           linewidth=1, label='Irreducible Error')

# Mark optimal point
ax.axvline(x=optimal_complexity, color='green', linestyle='--',
           linewidth=2, alpha=0.7)
ax.scatter([optimal_complexity], [total_error[optimal_idx]],
          color='green', s=200, zorder=5, marker='*',
          label=f'Optimal (complexity={optimal_complexity:.1f})')

ax.set_xlabel('Model Complexity')
ax.set_ylabel('Error')
ax.set_title('Bias-Variance Tradeoff')
ax.legend()
ax.grid(True, alpha=0.3)

# Annotate regions
ax.text(3, 8, 'High Bias\n(Underfitting)', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(12, 8, 'High Variance\n(Overfitting)', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
fig

print("Bias-Variance Tradeoff:\n")
print(f"As model complexity increases:")
print(f"  ↓ Bias decreases (can fit more complex patterns)")
print(f"  ↑ Variance increases (becomes sensitive to noise)")
print(f"\nOptimal model complexity: {optimal_complexity:.2f}")
print(f"At this point, total error is minimized!")
print("\n💡 Regularization helps us control this tradeoff!")
```
</div>

## What is Regularization?

**Regularization** adds a penalty term to the cost function that discourages model complexity, typically by penalizing large parameter values (weights).

**Basic idea:**
- Original goal: Minimize prediction error
- With regularization: Minimize (prediction error + complexity penalty)

**General form:**

$$J_{\text{regularized}}(\theta) = J_{\text{original}}(\theta) + \lambda \cdot R(\theta)$$

where:
- $J_{\text{original}}(\theta)$ = original cost function (e.g., MSE or log loss)
- $R(\theta)$ = regularization term (penalty on model complexity)
- $\lambda$ = regularization strength (hyperparameter)

**Effect:**
- Forces weights to be smaller
- Makes model simpler (less likely to overfit)
- Reduces variance, slightly increases bias
- Better generalization to new data

## L2 Regularization (Ridge Regression)

L2 regularization adds the **sum of squared weights** as a penalty term.

### Mathematical Formulation

For linear regression, the Ridge cost function is:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

where:
- First term = MSE (prediction error)
- Second term = L2 penalty on weights
- $\lambda$ = regularization parameter (controls strength)
- Note: We typically **don't regularize the bias** $b$

**In vector notation:**

$$J(w, b) = \frac{1}{2m} \|Xw + b - y\|^2 + \frac{\lambda}{2m} \|w\|^2$$

where $\|w\|^2 = \sum_{j=1}^{n} w_j^2$ is the squared L2 norm.

### Properties of L2 Regularization

1. **Shrinks weights** toward zero (but rarely exactly zero)
2. **Smooth penalty**: Differentiable everywhere
3. **Prefers smaller, distributed weights** rather than few large ones
4. **Closed-form solution exists** (like normal equation)

**Closed-form solution (Ridge Normal Equation):**

$$w = (X^T X + \lambda I)^{-1} X^T y$$

where $I$ is the identity matrix. The $\lambda I$ term ensures the matrix is invertible even when $X^T X$ is singular.

### Effect on Weights

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate how L2 penalty affects weights
# Original cost: MSE = (w1 - 5)^2 + (w2 - 5)^2  (minimum at w1=5, w2=5)
# L2 penalty: lambda * (w1^2 + w2^2)

lambdas = [0, 1, 5, 20]
colors = ['blue', 'green', 'orange', 'red']

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

w1_range = np.linspace(-2, 8, 100)
w2_range = np.linspace(-2, 8, 100)
W1, W2 = np.meshgrid(w1_range, w2_range)

for ax, lam, color in zip(axes, lambdas, colors):
    # Original cost (MSE part)
    mse_cost = (W1 - 5)**2 + (W2 - 5)**2

    # L2 penalty
    l2_penalty = lam * (W1**2 + W2**2)

    # Total cost
    total_cost = mse_cost + l2_penalty

    # Find minimum
    min_idx = np.unravel_index(np.argmin(total_cost), total_cost.shape)
    w1_opt = W1[min_idx]
    w2_opt = W2[min_idx]

    # Plot contours
    contour = ax.contour(W1, W2, total_cost, levels=15, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)

    # Mark optimum
    ax.scatter([w1_opt], [w2_opt], color=color, s=200, marker='*',
              zorder=5, edgecolors='black', linewidths=2,
              label=f'Optimal: ({w1_opt:.2f}, {w2_opt:.2f})')

    # Mark unregularized optimum
    if lam == 0:
        ax.scatter([5], [5], color='black', s=100, marker='x',
                  zorder=5, linewidths=3, label='Unregularized (5, 5)')
    else:
        ax.scatter([5], [5], color='gray', s=100, marker='x',
                  zorder=5, linewidths=2, alpha=0.5)

    ax.set_xlabel('w₁')
    ax.set_ylabel('w₂')
    ax.set_title(f'λ = {lam}\n(Total cost = MSE + {lam}·‖w‖²)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 7)
    ax.set_ylim(-1, 7)

plt.tight_layout()
fig

print("Effect of L2 Regularization on Weights:\n")
print(f"{'λ (lambda)':<15} {'Optimal w₁':<15} {'Optimal w₂':<15} {'‖w‖²'}")
print("-" * 60)

for lam in lambdas:
    mse_cost = (W1 - 5)**2 + (W2 - 5)**2
    total_cost = mse_cost + lam * (W1**2 + W2**2)
    min_idx = np.unravel_index(np.argmin(total_cost), total_cost.shape)
    w1_opt = W1[min_idx]
    w2_opt = W2[min_idx]
    norm_sq = w1_opt**2 + w2_opt**2
    print(f"{lam:<15} {w1_opt:<15.3f} {w2_opt:<15.3f} {norm_sq:.3f}")

print("\n💡 As λ increases, weights shrink toward zero!")
print("   This reduces model complexity and prevents overfitting.")
```
</div>

### Implementation: Ridge Regression

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

class RidgeRegression:
    """
    Ridge Regression (L2 regularization).

    Parameters:
        lambda_: Regularization strength (higher = more regularization)
        learning_rate: Step size for gradient descent
        num_iterations: Number of training iterations
    """

    def __init__(self, lambda_=1.0, learning_rate=0.01, num_iterations=1000):
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.w = None
        self.b = None
        self.cost_history = []

    def fit(self, X, y):
        """Train using gradient descent."""
        m, n = len(y), 1  # m samples, n features (1 for now)

        # Initialize
        self.w = 0.0
        self.b = 0.0

        for iteration in range(self.num_iterations):
            # Predictions
            predictions = self.w * X + self.b

            # Errors
            errors = predictions - y

            # Cost: MSE + L2 penalty
            mse = (1/(2*m)) * np.sum(errors**2)
            l2_penalty = (self.lambda_/(2*m)) * self.w**2
            cost = mse + l2_penalty
            self.cost_history.append(cost)

            # Gradients (with L2 regularization term)
            dw = (1/m) * np.sum(errors * X) + (self.lambda_/m) * self.w
            db = (1/m) * np.sum(errors)

            # Update
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

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

# Generate data with noise (prone to overfitting)
np.random.seed(42)
X_train = np.linspace(0, 10, 15)
y_train = 2 * X_train + 3 + np.random.randn(15) * 3  # y = 2x + 3 + noise

X_test = np.linspace(0, 10, 50)
y_test = 2 * X_test + 3 + np.random.randn(50) * 1

# Train models with different lambda values
lambdas = [0, 0.1, 1.0, 10.0]
models = {}

print("Training Ridge Regression with different λ values:\n")

for lam in lambdas:
    model = RidgeRegression(lambda_=lam, learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train)
    models[lam] = model

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"λ = {lam:5.1f}: w = {model.w:6.3f}, b = {model.b:6.3f}, "
          f"Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Different models
x_line = np.linspace(0, 10, 100)

ax1.scatter(X_train, y_train, color='blue', s=100, alpha=0.6,
           label='Training data', zorder=3)
ax1.scatter(X_test, y_test, color='lightblue', s=30, alpha=0.4,
           label='Test data', zorder=2)

colors = ['red', 'green', 'orange', 'purple']
for lam, color in zip(lambdas, colors):
    y_pred = models[lam].predict(x_line)
    ax1.plot(x_line, y_pred, color=color, linewidth=2,
            label=f'λ={lam} (w={models[lam].w:.2f})')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Ridge Regression: Effect of Regularization')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Regularization path (how weights change with lambda)
lambda_range = np.logspace(-3, 2, 50)
weights = []
biases = []

for lam in lambda_range:
    model = RidgeRegression(lambda_=lam, learning_rate=0.01, num_iterations=1000)
    model.fit(X_train, y_train)
    weights.append(model.w)
    biases.append(model.b)

ax2.semilogx(lambda_range, weights, 'b-', linewidth=2, label='Weight (w)')
ax2.semilogx(lambda_range, biases, 'r-', linewidth=2, label='Bias (b)')
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('λ (Regularization Strength)')
ax2.set_ylabel('Parameter Value')
ax2.set_title('Regularization Path: How Parameters Change with λ')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig

print("\n💡 Notice:")
print("  • As λ increases, weights shrink toward zero")
print("  • Too much regularization (high λ) → underfitting")
print("  • Too little regularization (low λ) → overfitting")
```
</div>

**Key Observations:**

1. When $\lambda = 0$: No regularization, model may overfit
2. Small $\lambda$: Gentle regularization, slight weight reduction
3. Large $\lambda$: Strong regularization, weights shrink significantly
4. Very large $\lambda$: Weights → 0, model becomes constant (underfitting)

## L1 Regularization (Lasso Regression)

L1 regularization adds the **sum of absolute values of weights** as a penalty term.

### Mathematical Formulation

For linear regression, the Lasso cost function is:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{m} \sum_{j=1}^{n} |w_j|$$

where:
- First term = MSE (prediction error)
- Second term = L1 penalty on weights
- $\lambda$ = regularization parameter

**In vector notation:**

$$J(w, b) = \frac{1}{2m} \|Xw + b - y\|^2 + \frac{\lambda}{m} \|w\|_1$$

where $\|w\|_1 = \sum_{j=1}^{n} |w_j|$ is the L1 norm.

### Properties of L1 Regularization

1. **Produces sparse models**: Can set weights to **exactly zero**
2. **Performs automatic feature selection**: Zero weights = features ignored
3. **Non-smooth**: Not differentiable at zero (uses subgradient methods)
4. **No closed-form solution**: Must use iterative optimization

**Why sparsity?** The L1 penalty creates "corners" at zero in the constraint region, making it more likely that the optimal solution has some weights exactly at zero.

### L1 vs L2: Geometric Intuition

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize L1 vs L2 constraint regions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Set up grid
w1 = np.linspace(-2, 2, 400)
w2 = np.linspace(-2, 2, 400)
W1, W2 = np.meshgrid(w1, w2)

# MSE cost contours (ellipses centered at (1, 1))
mse_cost = (W1 - 1)**2 + (W2 - 1)**2

# L2 constraint: w1^2 + w2^2 <= t (circle)
ax1.contour(W1, W2, mse_cost, levels=15, colors='blue', alpha=0.6)
theta = np.linspace(0, 2*np.pi, 100)
r = 1.2  # radius
ax1.fill(r*np.cos(theta), r*np.sin(theta), alpha=0.3, color='red',
        label='L2 constraint: w₁² + w₂² ≤ t')
ax1.scatter([1], [1], color='blue', s=200, marker='*', zorder=5,
           label='Unconstrained minimum')
# Point where contour touches circle (approximately)
ax1.scatter([0.85], [0.85], color='red', s=200, marker='o', zorder=5,
           label='L2 solution (rarely at axis)')
ax1.set_xlabel('w₁', fontsize=12)
ax1.set_ylabel('w₂', fontsize=12)
ax1.set_title('L2 Regularization (Ridge)\nSmooth constraint region', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# L1 constraint: |w1| + |w2| <= t (diamond)
ax2.contour(W1, W2, mse_cost, levels=15, colors='blue', alpha=0.6)
t = 1.5
diamond_w1 = np.array([t, 0, -t, 0, t])
diamond_w2 = np.array([0, t, 0, -t, 0])
ax2.fill(diamond_w1, diamond_w2, alpha=0.3, color='green',
        label='L1 constraint: |w₁| + |w₂| ≤ t')
ax2.scatter([1], [1], color='blue', s=200, marker='*', zorder=5,
           label='Unconstrained minimum')
# Point where contour touches diamond corner
ax2.scatter([0], [1.2], color='green', s=200, marker='o', zorder=5,
           label='L1 solution (often at axis!)')
ax2.annotate('Sparse!\n(w₁=0)', xy=(0, 1.2), xytext=(-0.8, 1.5),
            fontsize=11, color='darkgreen', weight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
ax2.set_xlabel('w₁', fontsize=12)
ax2.set_ylabel('w₂', fontsize=12)
ax2.set_title('L1 Regularization (Lasso)\nDiamond constraint with corners', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
fig

print("L1 vs L2 Regularization:\n")
print("L2 (Ridge):")
print("  • Constraint region: Circle (smooth)")
print("  • Solution: Typically all weights are non-zero")
print("  • Effect: Shrinks weights uniformly")
print("  • Use when: All features are potentially relevant")

print("\nL1 (Lasso):")
print("  • Constraint region: Diamond (has corners at axes)")
print("  • Solution: Often some weights are exactly zero")
print("  • Effect: Performs feature selection")
print("  • Use when: Many irrelevant features (automatic selection)")

print("\n💡 L1's corners make it likely to hit axes → sparse solutions!")
```
</div>

### Implementation: Lasso Regression

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

class LassoRegression:
    """
    Lasso Regression (L1 regularization).

    Uses coordinate descent for optimization (better than gradient descent for L1).

    Parameters:
        lambda_: Regularization strength
        num_iterations: Number of training iterations
        tol: Convergence tolerance
    """

    def __init__(self, lambda_=1.0, num_iterations=1000, tol=1e-4):
        self.lambda_ = lambda_
        self.num_iterations = num_iterations
        self.tol = tol
        self.w = None
        self.b = None
        self.cost_history = []

    def _soft_threshold(self, rho, lambda_):
        """Soft thresholding operator for L1."""
        if rho < -lambda_:
            return rho + lambda_
        elif rho > lambda_:
            return rho - lambda_
        else:
            return 0.0

    def fit(self, X, y):
        """Train using coordinate descent."""
        m = len(y)

        # Initialize
        self.w = 0.0
        self.b = np.mean(y)

        for iteration in range(self.num_iterations):
            w_old = self.w

            # Update bias (no regularization on bias)
            residual = y - self.w * X - self.b
            self.b = np.mean(residual)

            # Update weight using soft thresholding
            residual = y - self.b
            rho = np.dot(X, residual) / m
            lambda_scaled = self.lambda_ / m
            self.w = self._soft_threshold(rho, lambda_scaled) / (np.dot(X, X) / m)

            # Compute cost
            predictions = self.w * X + self.b
            mse = (1/(2*m)) * np.sum((predictions - y)**2)
            l1_penalty = (self.lambda_/m) * np.abs(self.w)
            cost = mse + l1_penalty
            self.cost_history.append(cost)

            # Check convergence
            if np.abs(self.w - w_old) < self.tol:
                break

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

# Generate data
np.random.seed(42)
X_train = np.linspace(0, 10, 20)
y_train = 2 * X_train + 3 + np.random.randn(20) * 3

# Train models with different lambda values
lambdas = [0, 0.5, 2.0, 10.0]
models_lasso = {}

print("Training Lasso Regression with different λ values:\n")

for lam in lambdas:
    model = LassoRegression(lambda_=lam, num_iterations=1000)
    model.fit(X_train, y_train)
    models_lasso[lam] = model

    print(f"λ = {lam:5.1f}: w = {model.w:6.3f}, b = {model.b:6.3f}, "
          f"R² = {model.score(X_train, y_train):.4f}")

    if np.abs(model.w) < 0.01:
        print(f"         → Weight is essentially ZERO! (Feature eliminated)")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Different models
x_line = np.linspace(0, 10, 100)
ax1.scatter(X_train, y_train, color='blue', s=100, alpha=0.6,
           label='Training data', zorder=3)

colors = ['red', 'green', 'orange', 'purple']
for lam, color in zip(lambdas, colors):
    y_pred = models_lasso[lam].predict(x_line)
    label = f'λ={lam} (w={models_lasso[lam].w:.2f})'
    if np.abs(models_lasso[lam].w) < 0.01:
        label += ' ★ SPARSE'
    ax1.plot(x_line, y_pred, color=color, linewidth=2, label=label)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Lasso Regression: Feature Selection via Sparsity')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Regularization path
lambda_range = np.logspace(-2, 2, 50)
weights_lasso = []

for lam in lambda_range:
    model = LassoRegression(lambda_=lam, num_iterations=1000)
    model.fit(X_train, y_train)
    weights_lasso.append(model.w)

ax2.semilogx(lambda_range, weights_lasso, 'b-', linewidth=2, label='Weight (w)')
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2,
           alpha=0.7, label='Zero (feature eliminated)')
ax2.set_xlabel('λ (Regularization Strength)')
ax2.set_ylabel('Weight Value')
ax2.set_title('Lasso Regularization Path: Weights Go to Exactly Zero')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Annotate where weight becomes zero
zero_idx = np.where(np.abs(np.array(weights_lasso)) < 0.01)[0]
if len(zero_idx) > 0:
    first_zero_lambda = lambda_range[zero_idx[0]]
    ax2.axvline(x=first_zero_lambda, color='green', linestyle=':',
               linewidth=2, alpha=0.7)
    ax2.text(first_zero_lambda*1.5, 1.5, f'w→0 at λ≈{first_zero_lambda:.2f}',
            fontsize=10, color='green', weight='bold')

plt.tight_layout()
fig

print("\n💡 Notice:")
print("  • Lasso can set weights to EXACTLY zero")
print("  • This performs automatic feature selection")
print("  • Useful when you have many features but only some are important")
```
</div>

## Elastic Net: Combining L1 and L2

**Elastic Net** combines both L1 and L2 regularization, getting benefits of both.

### Mathematical Formulation

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \left( \alpha \sum_{j=1}^{n} |w_j| + \frac{1-\alpha}{2} \sum_{j=1}^{n} w_j^2 \right)$$

where:
- $\lambda$ = overall regularization strength
- $\alpha \in [0, 1]$ = mixing parameter
  - $\alpha = 0$ → pure L2 (Ridge)
  - $\alpha = 1$ → pure L1 (Lasso)
  - $0 < \alpha < 1$ → combination (Elastic Net)

**Alternative formulation:**

$$J(\theta) = \frac{1}{2m} \|Xw + b - y\|^2 + \lambda_1 \|w\|_1 + \lambda_2 \|w\|^2$$

where $\lambda_1$ controls L1 penalty and $\lambda_2$ controls L2 penalty.

### When to Use Each

| Regularization | Best Use Case | Key Property |
|----------------|---------------|--------------|
| **Ridge (L2)** | All features potentially relevant, correlated features | Shrinks weights, keeps all features |
| **Lasso (L1)** | Many irrelevant features, need feature selection | Sparse solutions (some weights = 0) |
| **Elastic Net** | Many correlated features, need feature selection | Sparse + handles correlations well |

**Example scenarios:**

- **Ridge**: Predicting house prices with 10 features (all relevant: size, bedrooms, location, etc.)
- **Lasso**: Text classification with 10,000 word features (most words irrelevant)
- **Elastic Net**: Gene expression data (thousands of correlated genes, only some relevant)

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data with multiple correlated features
np.random.seed(42)
n_samples = 100

# Two highly correlated features
X1 = np.random.randn(n_samples)
X2 = X1 + np.random.randn(n_samples) * 0.1  # Highly correlated with X1

# True relationship: y = 3*X1 + 3*X2 + noise
# Since X1 ≈ X2, there are many equivalent solutions
y = 3*X1 + 3*X2 + np.random.randn(n_samples) * 2

print("Regularization Comparison with Correlated Features:\n")
print("True model: y = 3·X₁ + 3·X₂ + noise")
print(f"Feature correlation: {np.corrcoef(X1, X2)[0,1]:.3f} (very high!)\n")

# Simulate different regularization approaches
# (In practice, use sklearn; this is illustrative)

# No regularization: Can give unstable results with correlated features
# Ridge: Distributes weight between correlated features
# Lasso: Picks one feature arbitrarily, zeros out others
# Elastic Net: Balances both

results = {
    'None': {'w1': 2.8, 'w2': 3.1, 'description': 'Both features kept'},
    'Ridge (L2)': {'w1': 2.9, 'w2': 2.9, 'description': 'Distributes weight evenly'},
    'Lasso (L1)': {'w1': 5.8, 'w2': 0.0, 'description': 'Picks one, zeros other'},
    'Elastic Net': {'w1': 3.5, 'w2': 2.3, 'description': 'Keeps both, some selection'}
}

print(f"{'Method':<20} {'w₁':<10} {'w₂':<10} {'Behavior'}")
print("-" * 60)
for method, data in results.items():
    print(f"{method:<20} {data['w1']:<10.2f} {data['w2']:<10.2f} {data['description']}")

# Visualize weight distributions
fig, ax = plt.subplots(figsize=(10, 6))

methods = list(results.keys())
w1_vals = [results[m]['w1'] for m in methods]
w2_vals = [results[m]['w2'] for m in methods]

x_pos = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x_pos - width/2, w1_vals, width, label='Weight for X₁',
               color='skyblue', edgecolor='black')
bars2 = ax.bar(x_pos + width/2, w2_vals, width, label='Weight for X₂',
               color='lightcoral', edgecolor='black')

# True values
ax.axhline(y=3, color='green', linestyle='--', linewidth=2,
          label='True weight (3.0)', alpha=0.7)

ax.set_xlabel('Regularization Method', fontsize=12)
ax.set_ylabel('Weight Value', fontsize=12)
ax.set_title('How Different Regularization Methods Handle Correlated Features',
            fontsize=13)
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig

print("\n💡 Key Insights:")
print("  • Ridge: Distributes weight among correlated features")
print("  • Lasso: Arbitrarily picks one, ignores others (unstable)")
print("  • Elastic Net: Gets benefits of both (sparsity + stability)")
print("\n  When features are correlated, Elastic Net is often best!")
```
</div>

## Choosing the Right Regularization Strength (λ)

The hyperparameter $\lambda$ controls the regularization strength. How do we choose it?

### Cross-Validation

The standard approach is **k-fold cross-validation**:

1. Split data into k folds (typically k=5 or k=10)
2. For each value of $\lambda$:
   - Train on k-1 folds, validate on the remaining fold
   - Repeat k times (each fold is validation once)
   - Average the k validation errors
3. Choose $\lambda$ with lowest average validation error

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

class SimpleRidgeCV:
    """Ridge regression with cross-validation for lambda selection."""

    def __init__(self, lambdas, k_folds=5):
        self.lambdas = lambdas
        self.k_folds = k_folds
        self.best_lambda = None
        self.cv_scores = {}

    def _train_ridge(self, X, y, lambda_, lr=0.01, iterations=500):
        """Simple ridge training."""
        m = len(y)
        w, b = 0.0, 0.0

        for _ in range(iterations):
            pred = w * X + b
            err = pred - y
            dw = (1/m) * np.sum(err * X) + (lambda_/m) * w
            db = (1/m) * np.sum(err)
            w -= lr * dw
            b -= lr * db

        return w, b

    def _mse(self, y_true, y_pred):
        """Mean squared error."""
        return np.mean((y_true - y_pred)**2)

    def fit(self, X, y):
        """Find best lambda using k-fold cross-validation."""
        m = len(y)
        fold_size = m // self.k_folds

        for lambda_ in self.lambdas:
            fold_errors = []

            # K-fold cross-validation
            for fold in range(self.k_folds):
                # Split into train and validation
                val_start = fold * fold_size
                val_end = val_start + fold_size

                val_idx = list(range(val_start, val_end))
                train_idx = list(range(0, val_start)) + list(range(val_end, m))

                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]

                # Train on this fold
                w, b = self._train_ridge(X_train_fold, y_train_fold, lambda_)

                # Validate
                y_pred_val = w * X_val_fold + b
                val_error = self._mse(y_val_fold, y_pred_val)
                fold_errors.append(val_error)

            # Average validation error across folds
            avg_error = np.mean(fold_errors)
            self.cv_scores[lambda_] = avg_error

        # Choose best lambda
        self.best_lambda = min(self.cv_scores, key=self.cv_scores.get)

        # Retrain on all data with best lambda
        self.w, self.b = self._train_ridge(X, y, self.best_lambda)

        return self

    def predict(self, X):
        return self.w * X + self.b

# Generate data
np.random.seed(42)
X_train = np.linspace(0, 10, 50)
y_train = 2 * X_train + 3 + np.random.randn(50) * 3

# Cross-validation to find best lambda
lambdas_to_try = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

print("Performing 5-Fold Cross-Validation:\n")

model_cv = SimpleRidgeCV(lambdas_to_try, k_folds=5)
model_cv.fit(X_train, y_train)

print(f"{'λ':<10} {'Avg CV Error':<15}")
print("-" * 30)
for lam in lambdas_to_try:
    marker = " ← BEST" if lam == model_cv.best_lambda else ""
    print(f"{lam:<10.3f} {model_cv.cv_scores[lam]:<15.4f}{marker}")

print(f"\nBest λ found: {model_cv.best_lambda}")
print(f"Learned parameters: w={model_cv.w:.3f}, b={model_cv.b:.3f}")

# Visualize CV results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: CV errors vs lambda
ax1.plot(lambdas_to_try, [model_cv.cv_scores[l] for l in lambdas_to_try],
        'bo-', linewidth=2, markersize=8)
ax1.scatter([model_cv.best_lambda], [model_cv.cv_scores[model_cv.best_lambda]],
           color='red', s=300, marker='*', zorder=5,
           label=f'Best λ={model_cv.best_lambda}')
ax1.set_xlabel('λ (Regularization Strength)')
ax1.set_ylabel('Average Cross-Validation Error')
ax1.set_title('Cross-Validation: Finding Optimal λ')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Final model
x_line = np.linspace(0, 10, 100)
y_pred = model_cv.predict(x_line)

ax2.scatter(X_train, y_train, color='blue', s=50, alpha=0.6,
           label='Training data')
ax2.plot(x_line, y_pred, 'r-', linewidth=2,
        label=f'Best model (λ={model_cv.best_lambda})')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Model Trained with Optimal λ from CV')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig

print("\n💡 Cross-validation helps us choose λ automatically!")
print("   Try different λ values and pick the one with best validation performance.")
```
</div>

### Regularization Path

A **regularization path** shows how model coefficients change as $\lambda$ varies. This helps visualize feature importance and selection.

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate multi-feature scenario (illustrative)
# In reality, you'd use sklearn or implement full multivariate versions

# Generate data with 5 features of varying importance
np.random.seed(42)
n_samples = 100
n_features = 5

X = np.random.randn(n_samples, n_features)
# True coefficients: [5, 3, 1, 0, 0] - last two features are irrelevant
true_coeffs = np.array([5, 3, 1, 0, 0])
y = X @ true_coeffs + np.random.randn(n_samples) * 2

# Simulate regularization paths for Ridge and Lasso
# (These are illustrative - in practice use sklearn)
lambda_range = np.logspace(-2, 2, 50)

# Ridge: All coefficients shrink gradually
ridge_paths = np.array([
    5 * np.exp(-lambda_range * 0.3),   # Feature 1 (most important)
    3 * np.exp(-lambda_range * 0.35),  # Feature 2
    1 * np.exp(-lambda_range * 0.4),   # Feature 3
    0.5 * np.exp(-lambda_range * 0.5), # Feature 4 (noise)
    0.3 * np.exp(-lambda_range * 0.5)  # Feature 5 (noise)
])

# Lasso: Some coefficients go to exactly zero
lasso_paths = np.array([
    np.maximum(5 - lambda_range * 2, 0),      # Feature 1
    np.maximum(3 - lambda_range * 1.5, 0),    # Feature 2
    np.maximum(1 - lambda_range * 0.8, 0),    # Feature 3
    np.maximum(0.5 - lambda_range * 0.3, 0),  # Feature 4 → zero quickly
    np.maximum(0.3 - lambda_range * 0.2, 0)   # Feature 5 → zero quickly
])

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

feature_names = ['Feature 1\n(important)', 'Feature 2\n(important)',
                'Feature 3\n(somewhat)', 'Feature 4\n(noise)', 'Feature 5\n(noise)']
colors = ['red', 'blue', 'green', 'orange', 'purple']

# Ridge path
for i, (path, name, color) in enumerate(zip(ridge_paths, feature_names, colors)):
    ax1.semilogx(lambda_range, path, color=color, linewidth=2,
                label=name, marker='o' if i < 3 else 'x', markersize=3)

ax1.set_xlabel('λ (Regularization Strength)', fontsize=12)
ax1.set_ylabel('Coefficient Value', fontsize=12)
ax1.set_title('Ridge (L2) Regularization Path\nAll features kept, gradually shrunk', fontsize=13)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Lasso path
for i, (path, name, color) in enumerate(zip(lasso_paths, feature_names, colors)):
    ax2.semilogx(lambda_range, path, color=color, linewidth=2,
                label=name, marker='o' if i < 3 else 'x', markersize=3)

ax2.set_xlabel('λ (Regularization Strength)', fontsize=12)
ax2.set_ylabel('Coefficient Value', fontsize=12)
ax2.set_title('Lasso (L1) Regularization Path\nNoise features eliminated (→ 0)', fontsize=13)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

# Annotate sparsity region
ax2.fill_between(lambda_range, -0.5, 5, where=(lambda_range > 1),
                 alpha=0.1, color='green',
                 label='Sparse region\n(some coefficients = 0)')

plt.tight_layout()
fig

print("Regularization Path Analysis:\n")
print("Ridge (L2):")
print("  • All coefficients shrink smoothly toward zero")
print("  • No coefficients reach exactly zero")
print("  • Even irrelevant features have small non-zero weights")

print("\nLasso (L1):")
print("  • Irrelevant features (4, 5) quickly go to zero")
print("  • Important features (1, 2, 3) remain non-zero longer")
print("  • Provides automatic feature selection")

print("\n💡 The regularization path helps you:")
print("   1. Understand which features matter most")
print("   2. Choose an appropriate λ value")
print("   3. See when features get eliminated (Lasso)")
```
</div>

## Practical Example: Comparing All Methods

Let's compare no regularization, Ridge, Lasso, and Elastic Net on a realistic problem.

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate polynomial features for non-linear data
def polynomial_features(X, degree):
    """Create polynomial features up to given degree."""
    features = [X**i for i in range(1, degree + 1)]
    return np.column_stack(features)

# Generate data
np.random.seed(42)
X_train = np.linspace(0, 10, 30)
y_train = 2*np.sin(X_train) + X_train + np.random.randn(30) * 1.5

X_test = np.linspace(0, 10, 100)
y_test = 2*np.sin(X_test) + X_test + np.random.randn(100) * 0.5

# Create polynomial features (degree 10 - prone to overfitting!)
degree = 10
X_train_poly = polynomial_features(X_train, degree)
X_test_poly = polynomial_features(X_test, degree)

# IMPORTANT: Scale features to prevent numerical instability!
# High-degree polynomials have huge values (x^10 can be enormous)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_poly = scaler.fit_transform(X_train_poly)
X_test_poly = scaler.transform(X_test_poly)

print(f"Polynomial Regression with degree {degree}")
print(f"Training samples: {len(X_train)}")
print(f"Number of features: {degree}")
print(f"High-degree polynomial is prone to overfitting!\n")

# Simplified implementations for illustration
def fit_linear(X, y, lambda_l2=0, lambda_l1=0, lr=0.01, iterations=3000):
    """Fit with optional L1 and L2 regularization."""
    m, n = X.shape
    w = np.zeros(n)

    for _ in range(iterations):
        pred = X @ w
        err = pred - y

        # Gradient with regularization
        grad = (X.T @ err) / m
        grad += (lambda_l2 / m) * w  # L2
        grad += (lambda_l1 / m) * np.sign(w)  # L1 (simplified)

        w -= lr * grad

    return w

# Train different models
w_none = fit_linear(X_train_poly, y_train, lambda_l2=0, lambda_l1=0)
w_ridge = fit_linear(X_train_poly, y_train, lambda_l2=5.0, lambda_l1=0)
w_lasso = fit_linear(X_train_poly, y_train, lambda_l2=0, lambda_l1=0.5)
w_elastic = fit_linear(X_train_poly, y_train, lambda_l2=2.0, lambda_l1=0.3)

# Predictions
y_pred_none = X_test_poly @ w_none
y_pred_ridge = X_test_poly @ w_ridge
y_pred_lasso = X_test_poly @ w_lasso
y_pred_elastic = X_test_poly @ w_elastic

# Compute errors
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

print(f"{'Method':<20} {'Train MSE':<12} {'Test MSE':<12} {'# Non-zero'}")
print("-" * 60)

for name, w, y_pred in [
    ('No Regularization', w_none, y_pred_none),
    ('Ridge (L2)', w_ridge, y_pred_ridge),
    ('Lasso (L1)', w_lasso, y_pred_lasso),
    ('Elastic Net', w_elastic, y_pred_elastic)
]:
    train_mse = mse(y_train, X_train_poly @ w)
    test_mse = mse(y_test, y_pred)
    n_nonzero = np.sum(np.abs(w) > 0.01)

    print(f"{name:<20} {train_mse:<12.4f} {test_mse:<12.4f} {n_nonzero}/{degree}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Predictions
ax1.scatter(X_train, y_train, color='black', s=80, alpha=0.6,
           label='Training data', zorder=3)
ax1.plot(X_test, y_test, 'k.', alpha=0.2, label='Test data (true)', markersize=4)

ax1.plot(X_test, y_pred_none, 'r-', linewidth=2, label='No regularization', alpha=0.7)
ax1.plot(X_test, y_pred_ridge, 'b-', linewidth=2, label='Ridge (L2)', alpha=0.7)
ax1.plot(X_test, y_pred_lasso, 'g-', linewidth=2, label='Lasso (L1)', alpha=0.7)
ax1.plot(X_test, y_pred_elastic, 'm-', linewidth=2, label='Elastic Net', alpha=0.7)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title(f'Polynomial Regression (degree {degree})')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-5, 20)

# Plot 2: Coefficient magnitudes
models_data = {
    'No Reg': w_none,
    'Ridge': w_ridge,
    'Lasso': w_lasso,
    'Elastic': w_elastic
}

x_pos = np.arange(degree)
width = 0.2

for i, (name, w) in enumerate(models_data.items()):
    offset = (i - 1.5) * width
    ax2.bar(x_pos + offset, np.abs(w), width, label=name, alpha=0.7)

ax2.set_xlabel('Polynomial Degree')
ax2.set_ylabel('|Coefficient| (absolute value)')
ax2.set_title('Coefficient Magnitudes: Regularization Shrinks Weights')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'x^{i+1}' for i in range(degree)], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig

print("\n💡 Observations:")
print("  • No regularization: Overfits (low train error, high test error)")
print("  • Ridge: Shrinks all coefficients, better generalization")
print("  • Lasso: Sets some coefficients to zero (feature selection)")
print("  • Elastic Net: Combines benefits of both")
```
</div>

## Key Takeaways

!!! success "Essential Concepts"

    **Overfitting vs Underfitting:**
    - Underfitting: Model too simple (high bias)
    - Overfitting: Model too complex (high variance)
    - Goal: Find the right balance

    **Bias-Variance Tradeoff:**
    - Total Error = Bias² + Variance + Irreducible Error
    - As complexity ↑: Bias ↓, Variance ↑
    - Regularization helps control this tradeoff

    **L2 Regularization (Ridge):**
    - Penalty: $\lambda \sum w_j^2$
    - Shrinks all weights toward zero
    - No feature selection (all weights non-zero)
    - Use when: All features potentially relevant

    **L1 Regularization (Lasso):**
    - Penalty: $\lambda \sum |w_j|$
    - Can set weights to exactly zero
    - Automatic feature selection
    - Use when: Many irrelevant features

    **Elastic Net:**
    - Combines L1 and L2
    - Penalty: $\lambda (\alpha |w| + (1-\alpha) w^2)$
    - Gets benefits of both
    - Use when: Many correlated features + need selection

    **Choosing λ:**
    - Use cross-validation
    - Plot regularization path
    - Too small λ → overfitting
    - Too large λ → underfitting

## Regularization in Practice

### When to Use Regularization

**Always consider regularization when:**

1. **More features than samples** ($n > m$)
2. **Features are highly correlated** (multicollinearity)
3. **High-degree polynomials** or complex models
4. **Small dataset** (prone to overfitting)
5. **Poor generalization** (train error ≪ test error)

### Hyperparameter Tuning Checklist

1. **Split data**: Train, validation, test
2. **Try multiple λ values**: Use logarithmic scale (0.001, 0.01, 0.1, 1, 10, 100)
3. **Use cross-validation**: k-fold (typically k=5 or k=10)
4. **Monitor both train and validation error**
5. **Choose λ with best validation performance**
6. **Evaluate on held-out test set**

### Common Mistakes to Avoid

!!! warning "Common Pitfalls"

    - **Don't regularize the bias term** ($b$) - only regularize weights ($w$)
    - **Scale features first** - regularization is sensitive to feature magnitudes
    - **Don't use test set for tuning** - only for final evaluation
    - **Watch for underfitting** - too much regularization is also bad
    - **Consider the problem domain** - which features should be sparse?

## Practice Exercises

!!! tip "Test Your Understanding"

    1. **Implement Ridge regression** with gradient descent and verify against the closed-form solution
    2. **Create a regularization path plot** showing how coefficients change with λ
    3. **Compare Ridge vs Lasso** on a dataset with correlated features
    4. **Implement k-fold cross-validation** from scratch to choose optimal λ
    5. **Visualize the bias-variance tradeoff** by training models of varying complexity
    6. **Apply regularization to logistic regression** (adapt the cost function)

## Next Steps

Congratulations! You've completed Module 2 on Regression Algorithms. You now understand:

- Linear regression and the normal equation
- Gradient descent optimization
- Logistic regression for classification
- Regularization techniques to prevent overfitting

**Ready to apply these concepts?** Check out the module exercises to practice on real datasets!

[View Module 2 Exercises](exercises.md){ .md-button .md-button--primary }

[Back to Module 2 Overview](index.md){ .md-button }

---

**Questions or Issues?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues).
