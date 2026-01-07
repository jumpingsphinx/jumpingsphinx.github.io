# Lesson 3: Logistic Regression

## Introduction

Despite its name, **logistic regression is a classification algorithm**, not a regression algorithm. It's one of the most fundamental and widely-used classification techniques in machine learning, forming the building block for neural networks and deep learning.

!!! quote "Foundation of Deep Learning"
    "Every neuron in a neural network is essentially performing logistic regression. Understanding this algorithm deeply means understanding the core computational unit of deep learning."

In this lesson, you'll learn how to extend linear regression to classification problems, understand probability-based predictions, and implement logistic regression from scratch.

## From Regression to Classification

### The Classification Problem

In regression, we predict **continuous values** (e.g., house prices: $250,000, $300,000, $450,000).

In classification, we predict **discrete categories** (e.g., email type: spam or not spam).

**Examples of classification problems:**

- **Binary classification** (2 classes):
  - Email: Spam vs Not Spam
  - Medical: Disease vs Healthy
  - Finance: Fraud vs Legitimate
  - Customer: Will churn vs Won't churn

- **Multi-class classification** (>2 classes):
  - Image: Cat vs Dog vs Bird
  - Sentiment: Positive vs Neutral vs Negative
  - Digit recognition: 0, 1, 2, ..., 9

### Why Linear Regression Fails for Classification

Let's see what happens if we try to use linear regression for a binary classification problem:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Binary classification dataset: tumor size vs malignant (1) or benign (0)
# Small tumors tend to be benign, large tumors tend to be malignant
X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 0 = benign, 1 = malignant

# Fit linear regression: y = wx + b
X_mean = np.mean(X_train)
y_mean = np.mean(y_train)
w = np.sum((X_train - X_mean) * (y_train - y_mean)) / np.sum((X_train - X_mean)**2)
b = y_mean - w * X_mean

# Make predictions
x_line = np.linspace(0, 12, 100)
y_pred = w * x_line + b

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(X_train, y_train, color='blue', s=100, alpha=0.6,
           label='Training data\n(0=benign, 1=malignant)', zorder=3)
ax.plot(x_line, y_pred, 'r-', linewidth=2, label=f'Linear: y={w:.2f}x+{b:.2f}')
ax.axhline(y=0.5, color='green', linestyle='--', linewidth=1,
           label='Decision boundary (y=0.5)', alpha=0.7)
ax.set_xlabel('Tumor Size')
ax.set_ylabel('Prediction')
ax.set_ylim(-0.5, 1.5)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Linear Regression for Classification - Problems!')

plt.tight_layout()
fig

# Show problems
print("Problems with Linear Regression for Classification:\n")
print(f"1. Predictions can be > 1 or < 0:")
print(f"   At x=0:  prediction = {w*0 + b:.3f}")
print(f"   At x=12: prediction = {w*12 + b:.3f}")

print(f"\n2. Doesn't output probabilities:")
print(f"   What does a prediction of 1.2 mean? 120% malignant?")

print(f"\n3. Sensitive to outliers:")
print(f"   One extreme point can shift the entire line!")

print("\n💡 We need a function that:")
print("   • Always outputs values between 0 and 1 (probabilities)")
print("   • Has a clear decision boundary")
print("   • Gives meaningful probability interpretations")
```
</div>

**Key Problems:**

1. **No probability interpretation**: Linear regression outputs can be any real number
2. **Unbounded predictions**: Values can exceed [0, 1], which doesn't make sense for probabilities
3. **Poor decision boundary**: A single outlier can drastically shift the decision boundary

## The Sigmoid Function: Squashing to Probabilities

The **sigmoid function** (also called the **logistic function**) solves these problems by mapping any real number to a value between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where $z = wx + b$ is our linear combination of inputs.

### Properties of the Sigmoid Function

1. **Range**: $\sigma(z) \in (0, 1)$ for all $z \in \mathbb{R}$
2. **Monotonic**: Always increasing (preserves ordering)
3. **Smooth**: Differentiable everywhere (important for gradient descent)
4. **Symmetric**: $\sigma(-z) = 1 - \sigma(z)$
5. **Interpretable output**: Can be interpreted as probability

**Special values:**

- $\sigma(0) = 0.5$ (neutral point)
- $\sigma(\infty) \approx 1$ (strongly positive)
- $\sigma(-\infty) \approx 0$ (strongly negative)
- $\sigma(z) > 0.5$ when $z > 0$
- $\sigma(z) < 0.5$ when $z < 0$

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """The sigmoid (logistic) function."""
    return 1 / (1 + np.exp(-z))

# Visualize sigmoid function
z_values = np.linspace(-10, 10, 200)
sigma_values = sigmoid(z_values)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Sigmoid function
ax1.plot(z_values, sigma_values, 'b-', linewidth=2, label='σ(z) = 1/(1+e^(-z))')
ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='σ(z)=0.5')
ax1.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='z=0')
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax1.axhline(y=1, color='gray', linestyle='-', linewidth=0.5)
ax1.set_xlabel('z (input)')
ax1.set_ylabel('σ(z) (output)')
ax1.set_title('Sigmoid Function')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Key points
key_z = np.array([-5, -2, -1, 0, 1, 2, 5])
key_sigma = sigmoid(key_z)

ax2.plot(z_values, sigma_values, 'b-', linewidth=2, alpha=0.3)
ax2.scatter(key_z, key_sigma, color='red', s=100, zorder=3)
for z, s in zip(key_z, key_sigma):
    ax2.annotate(f'z={z}\nσ={s:.3f}',
                xy=(z, s), xytext=(z, s+0.15),
                ha='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
ax2.set_xlabel('z')
ax2.set_ylabel('σ(z)')
ax2.set_title('Key Points on Sigmoid')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig

# Demonstrate properties
print("Sigmoid Function Properties:\n")
for z in [-10, -2, -1, 0, 1, 2, 10]:
    s = sigmoid(z)
    print(f"σ({z:3d}) = {s:.6f}")

print(f"\nSymmetry: σ(-2) + σ(2) = {sigmoid(-2):.6f} + {sigmoid(2):.6f} = {sigmoid(-2) + sigmoid(2):.6f}")
print(f"(Always sums to 1!)")

print("\n💡 Sigmoid squashes any input to (0,1), perfect for probabilities!")
```
</div>

### Logistic Regression Model

Combining the linear model with the sigmoid function:

$$h_\theta(x) = \sigma(wx + b) = \frac{1}{1 + e^{-(wx + b)}}$$

**Interpretation:**

- $h_\theta(x)$ = probability that $y = 1$ given input $x$
- $h_\theta(x) \geq 0.5$ → predict class 1 (positive)
- $h_\theta(x) < 0.5$ → predict class 0 (negative)

**Decision boundary:** The point where $wx + b = 0$ (where $\sigma(wx + b) = 0.5$)

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Same tumor dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Hand-tuned logistic regression parameters (we'll learn these later)
w = 1.5
b = -7.5

# Predictions
x_line = np.linspace(0, 12, 100)
z = w * x_line + b
y_pred_prob = sigmoid(z)

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(X, y, color='blue', s=100, alpha=0.6, label='Training data', zorder=3)
ax.plot(x_line, y_pred_prob, 'r-', linewidth=2,
        label=f'Logistic: σ({w}x + {b:.1f})')
ax.axhline(y=0.5, color='green', linestyle='--', linewidth=1,
           label='Decision threshold (p=0.5)', alpha=0.7)

# Decision boundary (where z=0, i.e., wx+b=0)
decision_boundary = -b / w
ax.axvline(x=decision_boundary, color='purple', linestyle='--', linewidth=1,
           label=f'Decision boundary (x={decision_boundary:.1f})', alpha=0.7)

ax.set_xlabel('Tumor Size')
ax.set_ylabel('P(Malignant)')
ax.set_ylim(-0.1, 1.1)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_title('Logistic Regression for Binary Classification')

plt.tight_layout()
fig

# Make predictions
print("Logistic Regression Predictions:\n")
test_sizes = [2, 5, 8, 11]
for x in test_sizes:
    z_val = w * x + b
    prob = sigmoid(z_val)
    prediction = "Malignant (1)" if prob >= 0.5 else "Benign (0)"
    print(f"Tumor size = {x:2d}: z = {z_val:6.2f}, P(malignant) = {prob:.4f} → {prediction}")

print(f"\n✓ All predictions are valid probabilities in [0, 1]!")
print(f"✓ Decision boundary at x = {decision_boundary:.1f}")
```
</div>

## Log Loss: The Cost Function for Classification

For linear regression, we used Mean Squared Error (MSE). For logistic regression, we use **log loss** (also called **binary cross-entropy**).

### Why Not Use MSE?

If we used MSE with the sigmoid function, the cost function becomes **non-convex** (has many local minima), making gradient descent unreliable.

### Binary Cross-Entropy (Log Loss)

The log loss for a single example is:

$$\mathcal{L}(h_\theta(x), y) = \begin{cases}
-\log(h_\theta(x)) & \text{if } y = 1 \\
-\log(1 - h_\theta(x)) & \text{if } y = 0
\end{cases}$$

This can be written compactly as:

$$\mathcal{L}(h_\theta(x), y) = -y \log(h_\theta(x)) - (1-y) \log(1 - h_\theta(x))$$

**For the entire dataset** (averaging over $m$ examples):

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

### Intuition Behind Log Loss

**When $y = 1$ (actual class is positive):**

- If $h_\theta(x) = 1$ (perfect prediction) → $-\log(1) = 0$ (no penalty)
- If $h_\theta(x) = 0.5$ (uncertain) → $-\log(0.5) = 0.69$ (some penalty)
- If $h_\theta(x) \approx 0$ (wrong!) → $-\log(0) \to \infty$ (huge penalty)

**When $y = 0$ (actual class is negative):**

- If $h_\theta(x) = 0$ (perfect prediction) → $-\log(1) = 0$ (no penalty)
- If $h_\theta(x) = 0.5$ (uncertain) → $-\log(0.5) = 0.69$ (some penalty)
- If $h_\theta(x) \approx 1$ (wrong!) → $-\log(0) \to \infty$ (huge penalty)

**Key insight:** Log loss heavily penalizes confident wrong predictions!

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Visualize log loss for different predictions
predictions = np.linspace(0.01, 0.99, 100)  # Avoid log(0)

# Cost when actual y=1 (positive class)
cost_y1 = -np.log(predictions)

# Cost when actual y=0 (negative class)
cost_y0 = -np.log(1 - predictions)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Cost when y=1
ax1.plot(predictions, cost_y1, 'b-', linewidth=2)
ax1.set_xlabel('Predicted Probability h(x)')
ax1.set_ylabel('Cost -log(h(x))')
ax1.set_title('Cost Function When y=1 (Actual: Positive)')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
ax1.legend()
ax1.set_ylim(0, 5)

# Annotations
ax1.annotate('Perfect prediction\nCost = 0', xy=(0.99, -np.log(0.99)),
            xytext=(0.7, 1), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax1.annotate('Wrong prediction\nCost → ∞', xy=(0.01, -np.log(0.01)),
            xytext=(0.2, 3.5), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# Plot 2: Cost when y=0
ax2.plot(predictions, cost_y0, 'r-', linewidth=2)
ax2.set_xlabel('Predicted Probability h(x)')
ax2.set_ylabel('Cost -log(1-h(x))')
ax2.set_title('Cost Function When y=0 (Actual: Negative)')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=0.5, color='blue', linestyle='--', alpha=0.5, label='Threshold')
ax2.legend()
ax2.set_ylim(0, 5)

# Annotations
ax2.annotate('Perfect prediction\nCost = 0', xy=(0.01, -np.log(1-0.01)),
            xytext=(0.3, 1), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax2.annotate('Wrong prediction\nCost → ∞', xy=(0.99, -np.log(1-0.99)),
            xytext=(0.7, 3.5), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
fig

# Examples
print("Log Loss Examples:\n")
examples = [
    (1, 0.9, "Good prediction"),
    (1, 0.5, "Uncertain"),
    (1, 0.1, "Bad prediction"),
    (0, 0.1, "Good prediction"),
    (0, 0.5, "Uncertain"),
    (0, 0.9, "Bad prediction"),
]

for y_true, y_pred, desc in examples:
    if y_true == 1:
        loss = -np.log(y_pred)
    else:
        loss = -np.log(1 - y_pred)
    print(f"y={y_true}, ŷ={y_pred:.1f}: Loss = {loss:.4f}  ({desc})")

print("\n💡 The worse the prediction, the higher the cost!")
```
</div>

### Computing Log Loss

<div class="python-interactive" markdown="1">
```python
import numpy as np

def sigmoid(z):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-z))

def log_loss(y_true, y_pred):
    """
    Compute binary cross-entropy (log loss).

    Parameters:
        y_true: True labels (0 or 1)
        y_pred: Predicted probabilities (between 0 and 1)

    Returns:
        Average log loss across all examples
    """
    m = len(y_true)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Example: compute log loss for different predictions
y_true = np.array([1, 1, 0, 0, 1, 0])

# Good predictions (mostly correct)
y_pred_good = np.array([0.9, 0.85, 0.1, 0.15, 0.95, 0.05])

# Bad predictions (mostly wrong)
y_pred_bad = np.array([0.2, 0.3, 0.8, 0.9, 0.1, 0.95])

# Random predictions
y_pred_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

loss_good = log_loss(y_true, y_pred_good)
loss_bad = log_loss(y_true, y_pred_bad)
loss_random = log_loss(y_true, y_pred_random)

print("True labels:       ", y_true)
print("\nGood predictions:  ", y_pred_good)
print(f"Log loss: {loss_good:.4f}")

print("\nBad predictions:   ", y_pred_bad)
print(f"Log loss: {loss_bad:.4f}")

print("\nRandom predictions:", y_pred_random)
print(f"Log loss: {loss_random:.4f}")

print(f"\n💡 Lower loss = better predictions!")
print(f"   Good model:   {loss_good:.4f}")
print(f"   Random guess: {loss_random:.4f}")
print(f"   Bad model:    {loss_bad:.4f}")
```
</div>

## Gradient Descent for Logistic Regression

Just like linear regression, we use gradient descent to minimize the cost function and learn optimal parameters.

### Computing Gradients

For logistic regression with log loss, the gradients are:

$$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})$$

where $h_\theta(x^{(i)}) = \sigma(wx^{(i)} + b)$

**Remarkably, these have the same form as linear regression!** The difference is that $h_\theta(x)$ now uses the sigmoid function instead of being linear.

### Update Rules

$$w := w - \alpha \frac{\partial J}{\partial w}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

### Implementation from Scratch

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b):
    """
    Compute binary cross-entropy cost.

    Parameters:
        X: Input features
        y: True labels (0 or 1)
        w: Weight parameter
        b: Bias parameter

    Returns:
        cost: Log loss value
    """
    m = len(y)

    # Predictions
    z = w * X + b
    h = sigmoid(z)

    # Avoid log(0)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)

    # Binary cross-entropy
    cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    return cost

def logistic_regression_gradient_descent(X, y, learning_rate=0.1, num_iterations=1000):
    """
    Train logistic regression using gradient descent.

    Parameters:
        X: Training features
        y: Training labels (0 or 1)
        learning_rate: Learning rate (alpha)
        num_iterations: Number of training iterations

    Returns:
        w, b: Learned parameters
        cost_history: Cost at each iteration
    """
    # Initialize parameters
    w = 0.0
    b = 0.0
    m = len(y)

    cost_history = []

    for iteration in range(num_iterations):
        # Forward pass: compute predictions
        z = w * X + b
        h = sigmoid(z)

        # Compute cost
        cost = compute_cost(X, y, w, b)
        cost_history.append(cost)

        # Compute gradients
        errors = h - y
        dw = (1/m) * np.sum(errors * X)
        db = (1/m) * np.sum(errors)

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration:4d}: Cost = {cost:.4f}, w = {w:.4f}, b = {b:.4f}")

    return w, b, cost_history

# Generate binary classification dataset
np.random.seed(42)
# Class 0: small values
X_class0 = np.random.randn(50) * 1.5 + 2
# Class 1: large values
X_class1 = np.random.randn(50) * 1.5 + 8

X_train = np.concatenate([X_class0, X_class1])
y_train = np.concatenate([np.zeros(50), np.ones(50)])

# Shuffle
shuffle_idx = np.random.permutation(len(X_train))
X_train = X_train[shuffle_idx]
y_train = y_train[shuffle_idx]

# Train logistic regression
print("Training Logistic Regression...\n")
w_learned, b_learned, cost_history = logistic_regression_gradient_descent(
    X_train, y_train,
    learning_rate=0.1,
    num_iterations=1000
)

print(f"\n✓ Training complete!")
print(f"Learned parameters: w={w_learned:.4f}, b={b_learned:.4f}")
print(f"Final cost: {cost_history[-1]:.4f}")
print(f"Decision boundary at x = {-b_learned/w_learned:.4f}")

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Decision boundary
x_line = np.linspace(-2, 12, 200)
z_line = w_learned * x_line + b_learned
y_prob = sigmoid(z_line)

# Plot data points
ax1.scatter(X_train[y_train == 0], y_train[y_train == 0],
           color='blue', s=50, alpha=0.6, label='Class 0', zorder=3)
ax1.scatter(X_train[y_train == 1], y_train[y_train == 1],
           color='red', s=50, alpha=0.6, label='Class 1', zorder=3)

# Plot sigmoid curve
ax1.plot(x_line, y_prob, 'g-', linewidth=2, label='P(y=1|x)')

# Decision boundary and threshold
decision_boundary = -b_learned / w_learned
ax1.axvline(x=decision_boundary, color='purple', linestyle='--',
           linewidth=2, label=f'Decision boundary (x={decision_boundary:.2f})')
ax1.axhline(y=0.5, color='orange', linestyle='--',
           linewidth=1, alpha=0.7, label='Threshold (p=0.5)')

ax1.set_xlabel('Feature (x)')
ax1.set_ylabel('Class / Probability')
ax1.set_title('Logistic Regression: Decision Boundary')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Learning curve
ax2.plot(cost_history, 'b-', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost (Log Loss)')
ax2.set_title('Training: Cost Decreases Over Time')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig

print("\n📊 The model learned to separate the two classes!")
```
</div>

## Making Predictions and Evaluating Performance

### Prediction Process

1. **Compute linear combination**: $z = wx + b$
2. **Apply sigmoid**: $h_\theta(x) = \sigma(z)$ (gives probability)
3. **Apply threshold**:
   - If $h_\theta(x) \geq 0.5$ → predict class 1
   - If $h_\theta(x) < 0.5$ → predict class 0

<div class="python-interactive" markdown="1">
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w, b):
    """Predict probabilities."""
    z = w * X + b
    return sigmoid(z)

def predict(X, w, b, threshold=0.5):
    """Predict class labels (0 or 1)."""
    probabilities = predict_proba(X, w, b)
    return (probabilities >= threshold).astype(int)

# Using our trained model from previous example
# w_learned ≈ 1.1, b_learned ≈ -6.0 (approximate values)
w = 1.1
b = -6.0

# Test on new data
X_test = np.array([1, 3, 5, 7, 9, 11])

print("Making Predictions:\n")
print("Feature (x) | z = wx+b | P(y=1) | Prediction")
print("-" * 50)

for x in X_test:
    z = w * x + b
    prob = sigmoid(z)
    pred = 1 if prob >= 0.5 else 0
    print(f"   {x:4.1f}     | {z:7.2f}  | {prob:.4f} |     {pred}")

print("\n💡 Probabilities close to 0 or 1 indicate high confidence!")
print("   Probabilities near 0.5 indicate uncertainty.")
```
</div>

### Evaluation Metrics

For classification, accuracy alone isn't enough. We need multiple metrics:

**1. Accuracy**

$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}}$$

**2. Confusion Matrix**

|                | Predicted 0 | Predicted 1 |
|----------------|-------------|-------------|
| **Actual 0**   | True Negative (TN) | False Positive (FP) |
| **Actual 1**   | False Negative (FN) | True Positive (TP) |

**3. Precision**

$$\text{Precision} = \frac{TP}{TP + FP}$$

Of all positive predictions, how many were correct?

**4. Recall (Sensitivity)**

$$\text{Recall} = \frac{TP}{TP + FN}$$

Of all actual positives, how many did we catch?

**5. F1 Score**

$$F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

Harmonic mean of precision and recall.

<div class="python-interactive" markdown="1">
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def evaluate_classification(y_true, y_pred):
    """
    Compute classification metrics.

    Parameters:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Metrics
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    }

# Example evaluation
np.random.seed(42)

# Generate test data
X_test = np.concatenate([
    np.random.randn(30) * 1.5 + 2,   # Class 0
    np.random.randn(30) * 1.5 + 8    # Class 1
])
y_test = np.concatenate([np.zeros(30), np.ones(30)])

# Make predictions with learned model
w, b = 1.1, -6.0
y_pred_prob = sigmoid(w * X_test + b)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Evaluate
metrics = evaluate_classification(y_test, y_pred)

print("Model Evaluation:\n")
print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1 Score:  {metrics['f1_score']:.4f}")

print("\nConfusion Matrix:")
cm = metrics['confusion_matrix']
print(f"                Predicted 0  Predicted 1")
print(f"Actual 0 (neg)      {cm['TN']:3d}          {cm['FP']:3d}       (FP = False Positive)")
print(f"Actual 1 (pos)      {cm['FN']:3d}          {cm['TP']:3d}       (FN = False Negative)")

print(f"\nTrue Positives:  {cm['TP']} (correctly identified positives)")
print(f"True Negatives:  {cm['TN']} (correctly identified negatives)")
print(f"False Positives: {cm['FP']} (false alarms)")
print(f"False Negatives: {cm['FN']} (missed positives)")

print("\n💡 Different metrics matter for different applications:")
print("   • Medical diagnosis: High recall (don't miss sick patients)")
print("   • Spam filter: High precision (don't mark important emails as spam)")
print("   • Balanced: Use F1 score")
```
</div>

## Decision Boundaries and Visualization

The **decision boundary** is where the model switches from predicting class 0 to class 1. Mathematically, it's where:

$$wx + b = 0 \quad \Rightarrow \quad x = -\frac{b}{w}$$

For multi-dimensional data (multiple features), the decision boundary becomes a hyperplane.

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Create datasets with different separability
np.random.seed(42)

# Dataset 1: Well-separated
X1_c0 = np.random.randn(50) * 1.0 + 2
X1_c1 = np.random.randn(50) * 1.0 + 8
X1 = np.concatenate([X1_c0, X1_c1])
y1 = np.concatenate([np.zeros(50), np.ones(50)])
w1, b1 = 1.2, -6.0

# Dataset 2: Overlapping
X2_c0 = np.random.randn(50) * 2.0 + 4
X2_c1 = np.random.randn(50) * 2.0 + 7
X2 = np.concatenate([X2_c0, X2_c1])
y2 = np.concatenate([np.zeros(50), np.ones(50)])
w2, b2 = 0.8, -4.5

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x_line = np.linspace(-2, 12, 200)

# Plot 1: Well-separated
for ax, X, y, w, b, title in [
    (ax1, X1, y1, w1, b1, 'Well-Separated Classes'),
    (ax2, X2, y2, w2, b2, 'Overlapping Classes')
]:
    # Data points
    ax.scatter(X[y == 0], y[y == 0], color='blue', s=50, alpha=0.6, label='Class 0')
    ax.scatter(X[y == 1], y[y == 1], color='red', s=50, alpha=0.6, label='Class 1')

    # Sigmoid curve
    y_prob = sigmoid(w * x_line + b)
    ax.plot(x_line, y_prob, 'g-', linewidth=2, label='P(y=1|x)')

    # Decision boundary
    decision_boundary = -b / w
    ax.axvline(x=decision_boundary, color='purple', linestyle='--',
              linewidth=2, label=f'Boundary (x={decision_boundary:.2f})')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)

    # Shade regions
    ax.fill_between(x_line, 0, 1, where=(x_line < decision_boundary),
                    alpha=0.1, color='blue', label='Predict 0')
    ax.fill_between(x_line, 0, 1, where=(x_line >= decision_boundary),
                    alpha=0.1, color='red', label='Predict 1')

    ax.set_xlabel('Feature (x)')
    ax.set_ylabel('Class / Probability')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig

print("Decision Boundaries:\n")
print(f"Well-separated: x = {-b1/w1:.2f} (clear separation)")
print(f"Overlapping:    x = {-b2/w2:.2f} (some misclassifications inevitable)")
print("\n💡 When classes overlap, perfect classification is impossible!")
print("   The model learns the best probabilistic boundary.")
```
</div>

## Multi-Class Classification

Logistic regression can be extended to multi-class problems (>2 classes) using two approaches:

### 1. One-vs-Rest (OvR)

Train **one binary classifier per class**:

- Classifier 1: Class 1 vs (Class 2 or Class 3 or ...)
- Classifier 2: Class 2 vs (Class 1 or Class 3 or ...)
- Classifier 3: Class 3 vs (Class 1 or Class 2 or ...)

**Prediction:** Choose the class with the highest probability.

<div class="python-interactive" markdown="1">
```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Simulate One-vs-Rest for 3 classes
# Imagine we trained 3 binary classifiers

# Test example
x_test = 5.0

# Each classifier outputs probability that the example belongs to its class
# Classifier 1 (Class 0 vs rest)
w1, b1 = -0.8, 2.0
prob_class0 = sigmoid(w1 * x_test + b1)

# Classifier 2 (Class 1 vs rest)
w2, b2 = 0.2, -1.0
prob_class1 = sigmoid(w2 * x_test + b1)

# Classifier 3 (Class 2 vs rest)
w3, b3 = 1.0, -5.0
prob_class2 = sigmoid(w3 * x_test + b3)

print("One-vs-Rest Multi-Class Classification:\n")
print(f"Input: x = {x_test}")
print(f"\nClassifier outputs:")
print(f"  P(Class 0) = {prob_class0:.4f}")
print(f"  P(Class 1) = {prob_class1:.4f}")
print(f"  P(Class 2) = {prob_class2:.4f}")

predicted_class = np.argmax([prob_class0, prob_class1, prob_class2])
max_prob = np.max([prob_class0, prob_class1, prob_class2])

print(f"\nPrediction: Class {predicted_class} (probability: {max_prob:.4f})")
print("\n💡 Choose the class with highest probability!")
```
</div>

### 2. Softmax Regression (Multinomial Logistic Regression)

For $K$ classes, use the **softmax function** to compute probabilities:

$$P(y = k | x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

where $z_k = w_k^T x + b_k$ for class $k$.

**Properties:**

- Outputs sum to 1: $\sum_{k=1}^{K} P(y=k|x) = 1$
- Generalizes sigmoid (for $K=2$, softmax = sigmoid)
- Used as final layer in neural networks for classification

<div class="python-interactive" markdown="1">
```python
import numpy as np

def softmax(z):
    """
    Softmax function for multi-class classification.

    Parameters:
        z: Array of logits (one per class)

    Returns:
        probabilities: Array that sums to 1
    """
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

# Example: 4 classes
x = 3.0

# Different weights and biases for each class
w = np.array([0.5, 1.0, -0.5, 0.2])
b = np.array([0.1, -1.0, 2.0, 0.5])

# Compute logits (linear scores)
z = w * x + b

# Apply softmax to get probabilities
probabilities = softmax(z)

print("Softmax Multi-Class Classification:\n")
print(f"Input: x = {x}")
print(f"\nLogits (z = wx + b):")
for i, logit in enumerate(z):
    print(f"  Class {i}: z = {logit:.4f}")

print(f"\nSoftmax probabilities:")
for i, prob in enumerate(probabilities):
    print(f"  P(Class {i}) = {prob:.4f}")

print(f"\nSum of probabilities: {np.sum(probabilities):.6f} (should be 1.0)")

predicted_class = np.argmax(probabilities)
print(f"\nPredicted class: {predicted_class} (highest probability)")

print("\n💡 Softmax ensures probabilities are valid and sum to 1!")
```
</div>

## Complete Implementation: Logistic Regression Class

Let's build a complete, reusable logistic regression implementation:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    Logistic Regression classifier using gradient descent.

    Parameters:
        learning_rate: Step size for gradient descent
        num_iterations: Number of training iterations
        verbose: Whether to print training progress
    """

    def __init__(self, learning_rate=0.1, num_iterations=1000, verbose=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.w = None
        self.b = None
        self.cost_history = []

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, X, y):
        """Compute binary cross-entropy cost."""
        m = len(y)
        z = self.w * X + self.b
        h = self._sigmoid(z)

        # Avoid log(0)
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)

        cost = -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def fit(self, X, y):
        """
        Train the logistic regression model.

        Parameters:
            X: Training features (1D array)
            y: Training labels (0 or 1)
        """
        # Initialize parameters
        self.w = 0.0
        self.b = 0.0
        m = len(y)

        # Gradient descent
        for iteration in range(self.num_iterations):
            # Forward pass
            z = self.w * X + self.b
            h = self._sigmoid(z)

            # Compute cost
            cost = self._compute_cost(X, y)
            self.cost_history.append(cost)

            # Compute gradients
            errors = h - y
            dw = (1/m) * np.sum(errors * X)
            db = (1/m) * np.sum(errors)

            # Update parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # Print progress
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.4f}")

        if self.verbose:
            print(f"\n✓ Training complete! Final cost: {cost:.4f}")

        return self

    def predict_proba(self, X):
        """Predict probabilities."""
        z = self.w * X + self.b
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        """Predict class labels."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def score(self, X, y):
        """Compute accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_decision_boundary(self):
        """Return x-coordinate of decision boundary."""
        return -self.b / self.w if self.w != 0 else None

# Generate dataset
np.random.seed(42)
X_train = np.concatenate([
    np.random.randn(60) * 1.5 + 2,
    np.random.randn(60) * 1.5 + 8
])
y_train = np.concatenate([np.zeros(60), np.ones(60)])

# Shuffle
idx = np.random.permutation(len(X_train))
X_train, y_train = X_train[idx], y_train[idx]

# Train model
print("Training Logistic Regression Model:\n")
model = LogisticRegression(learning_rate=0.1, num_iterations=500, verbose=True)
model.fit(X_train, y_train)

# Evaluate
train_accuracy = model.score(X_train, y_train)
print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
print(f"Learned parameters: w={model.w:.4f}, b={model.b:.4f}")
print(f"Decision boundary: x={model.get_decision_boundary():.4f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Decision boundary
x_line = np.linspace(-2, 12, 200)
y_prob = model.predict_proba(x_line)

ax1.scatter(X_train[y_train == 0], y_train[y_train == 0],
           color='blue', s=50, alpha=0.6, label='Class 0')
ax1.scatter(X_train[y_train == 1], y_train[y_train == 1],
           color='red', s=50, alpha=0.6, label='Class 1')
ax1.plot(x_line, y_prob, 'g-', linewidth=2, label='P(y=1|x)')
ax1.axvline(x=model.get_decision_boundary(), color='purple',
           linestyle='--', linewidth=2, label='Decision boundary')
ax1.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Feature')
ax1.set_ylabel('Class / Probability')
ax1.set_title('Trained Logistic Regression Model')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Learning curve
ax2.plot(model.cost_history, 'b-', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost (Log Loss)')
ax2.set_title('Learning Curve')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig

print("\n✓ Model training visualization complete!")
```
</div>

## Key Takeaways

!!! success "Essential Concepts"

    **Logistic Regression Basics:**
    - Classification algorithm despite the name "regression"
    - Uses sigmoid function: $\sigma(z) = \frac{1}{1 + e^{-z}}$
    - Model: $h_\theta(x) = \sigma(wx + b)$
    - Output is probability: $h_\theta(x) \in (0, 1)$

    **Cost Function:**
    - Binary Cross-Entropy (Log Loss)
    - $J(\theta) = -\frac{1}{m}\sum[y\log(h) + (1-y)\log(1-h)]$
    - Penalizes confident wrong predictions heavily
    - Convex for gradient descent

    **Training:**
    - Use gradient descent (same form as linear regression!)
    - $\frac{\partial J}{\partial w} = \frac{1}{m}\sum(h(x) - y) \cdot x$
    - Decision boundary at $wx + b = 0$

    **Prediction:**
    - Compute probability: $P(y=1|x) = h_\theta(x)$
    - Threshold (typically 0.5) to get class label
    - Predict class 1 if $h_\theta(x) \geq 0.5$

    **Evaluation:**
    - Accuracy: overall correctness
    - Precision: quality of positive predictions
    - Recall: coverage of actual positives
    - F1 Score: harmonic mean of precision and recall

    **Multi-Class:**
    - One-vs-Rest: Train K binary classifiers
    - Softmax: Generalization of sigmoid for K classes

## Common Applications

Logistic regression is widely used in:

- **Medical Diagnosis**: Disease present or absent
- **Credit Scoring**: Loan default prediction
- **Email Filtering**: Spam detection
- **Marketing**: Customer conversion prediction
- **Image Classification**: Simple object recognition
- **Natural Language Processing**: Sentiment analysis, text classification

## Practice Exercises

!!! tip "Test Your Understanding"

    1. **Implement logistic regression** for a dataset with multiple features (use matrix operations)
    2. **Experiment with different thresholds** (not just 0.5) and observe effect on precision/recall
    3. **Visualize decision boundaries** for 2D data
    4. **Implement One-vs-Rest** for a 3-class problem
    5. **Compare with sklearn's LogisticRegression** - do you get similar results?
    6. **Handle imbalanced classes** - what happens when 90% of data is one class?

## Next Steps

Now that you understand logistic regression, you're ready to learn about regularization techniques that prevent overfitting!

[Next: Lesson 4 - Regularization](04-regularization.md){ .md-button .md-button--primary }

[Back: Lesson 2 - Gradient Descent](02-gradient-descent.md){ .md-button }

---

**Questions or Issues?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues).
