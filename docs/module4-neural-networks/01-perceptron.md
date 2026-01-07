# Lesson 1: The Perceptron

## Introduction

Welcome to the foundation of modern deep learning! The perceptron, invented by Frank Rosenblatt in 1957, is the simplest form of an artificial neural network. While basic, understanding the perceptron is essential because:

- **Foundation of Deep Learning**: Every modern neural network is built from perceptron-like units
- **Historical Significance**: Sparked the first wave of AI research
- **Conceptual Clarity**: Simple enough to understand completely, yet powerful enough to be useful
- **Building Block**: Understanding one neuron makes understanding millions easier

In this lesson, you'll learn how a single artificial neuron processes information, makes decisions, and learns from data. This is where your journey into neural networks begins!

!!! info "Video: Neural Networks Explained"
    For an excellent visual introduction to neural networks, watch this video:

    <iframe width="560" height="315" src="https://www.youtube.com/embed/aircAruvnKk" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

    *Neural Networks by 3Blue1Brown*

## Learning Objectives

By the end of this lesson, you will:

- ✅ Understand the biological inspiration behind artificial neurons
- ✅ Implement the perceptron model from scratch
- ✅ Master activation functions: sigmoid, tanh, and ReLU
- ✅ Train a perceptron using the perceptron learning algorithm
- ✅ Recognize the limitations of single-layer perceptrons
- ✅ Understand linear separability and the XOR problem

**Estimated Time**: 45 minutes

## From Biology to Mathematics

### The Biological Neuron

Artificial neural networks are inspired by the human brain. A biological neuron:

1. **Receives signals** through dendrites from thousands of other neurons
2. **Integrates** these signals in the cell body (soma)
3. **Fires** an electrical impulse down the axon if the combined signal exceeds a threshold
4. **Transmits** this signal to other neurons through synapses

**Key Insight**: The neuron performs a simple computation: sum weighted inputs, and fire if the sum exceeds a threshold. This is exactly what a perceptron does!

### The Mathematical Model

An artificial neuron (perceptron) mimics this behavior:

$$
\text{output} = f\left(\sum_{i=1}^{n} w_i x_i + b\right)
$$

Where:
- $x_i$ are **inputs** (like signals from dendrites)
- $w_i$ are **weights** (like synaptic strengths)
- $b$ is the **bias** (like the firing threshold)
- $f$ is the **activation function** (like the firing behavior)

Let's visualize this:

```
     x₁ ───w₁─┐
     x₂ ───w₂─┤
     x₃ ───w₃─┼──→ Σ ──→ f(·) ──→ output
      ⋮    ⋮  │      ↑
     xₙ ───wₙ─┘      b (bias)
```

The perceptron computes: **weighted sum + bias**, then **applies activation function**.

## The Perceptron Model

### Mathematical Formulation

Let's be more precise. Given an input vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$:

**Step 1: Compute weighted sum**

$$
z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^{n} w_i x_i + b
$$

**Step 2: Apply activation function**

$$
\hat{y} = f(z)
$$

The complete perceptron function is:

$$
\hat{y} = f(\mathbf{w}^T \mathbf{x} + b)
$$

Where:
- $\mathbf{x} \in \mathbb{R}^n$ is the input vector
- $\mathbf{w} \in \mathbb{R}^n$ are the weights (parameters to learn)
- $b \in \mathbb{R}$ is the bias term (parameter to learn)
- $f: \mathbb{R} \to \mathbb{R}$ is the activation function
- $\hat{y}$ is the predicted output

### Geometric Interpretation

In 2D, the perceptron defines a **decision boundary**:

$$
w_1 x_1 + w_2 x_2 + b = 0
$$

This is a **line**! The perceptron classifies points based on which side of this line they fall:

- If $\mathbf{w}^T \mathbf{x} + b > 0$: predict one class
- If $\mathbf{w}^T \mathbf{x} + b < 0$: predict the other class

In higher dimensions, this becomes a **hyperplane** that divides the space.

**Critical Insight**: A single perceptron can only learn linearly separable patterns. This is both its power and its limitation!

## Activation Functions

The activation function $f(z)$ determines how the perceptron responds to its input. Different functions give different behaviors.

### Why Do We Need Activation Functions?

Without activation functions, neural networks would just be linear transformations:

$$
\text{output} = \mathbf{w}^T \mathbf{x} + b
$$

No matter how many layers you stack, this remains linear! Activation functions introduce **non-linearity**, allowing neural networks to learn complex patterns.

### Common Activation Functions

#### 1. Step Function (Original Perceptron)

$$
f(z) = \begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
$$

**Characteristics**:
- Output is binary: 0 or 1
- Simple decision boundary
- **Not differentiable** at $z=0$ (problem for gradient descent!)
- Historically important but rarely used today

#### 2. Sigmoid Function

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Characteristics**:
- Smooth S-shaped curve
- Output range: $(0, 1)$ - interpretable as probability!
- Differentiable everywhere: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- **Vanishing gradients** for large $|z|$ (problem in deep networks)
- Common in output layer for binary classification

#### 3. Hyperbolic Tangent (tanh)

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

**Characteristics**:
- Smooth S-shaped curve
- Output range: $(-1, 1)$ - zero-centered (better than sigmoid)
- Differentiable: $\tanh'(z) = 1 - \tanh^2(z)$
- Still suffers from vanishing gradients
- Often preferred over sigmoid in hidden layers

#### 4. Rectified Linear Unit (ReLU)

$$
\text{ReLU}(z) = \max(0, z) = \begin{cases}
z & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

**Characteristics**:
- Dead simple: just threshold at zero
- Output range: $[0, \infty)$
- Derivative: $f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$
- **No vanishing gradient** for positive values
- **Most popular** in modern deep learning
- Can suffer from "dying ReLU" problem (neurons stuck at 0)

### Visualizing Activation Functions

Let's see how these functions behave:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def step(z):
    return (z >= 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

# Create input range
z = np.linspace(-5, 5, 200)

# Plot all activation functions
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Step function
axes[0, 0].plot(z, step(z), 'b-', linewidth=2)
axes[0, 0].set_title('Step Function', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('z')
axes[0, 0].set_ylabel('f(z)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
axes[0, 0].axvline(x=0, color='k', linewidth=0.5)

# Sigmoid function
axes[0, 1].plot(z, sigmoid(z), 'r-', linewidth=2)
axes[0, 1].set_title('Sigmoid Function', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('z')
axes[0, 1].set_ylabel('σ(z)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
axes[0, 1].axvline(x=0, color='k', linewidth=0.5)

# Tanh function
axes[1, 0].plot(z, tanh(z), 'g-', linewidth=2)
axes[1, 0].set_title('Tanh Function', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('z')
axes[1, 0].set_ylabel('tanh(z)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
axes[1, 0].axvline(x=0, color='k', linewidth=0.5)

# ReLU function
axes[1, 1].plot(z, relu(z), 'm-', linewidth=2)
axes[1, 1].set_title('ReLU Function', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('z')
axes[1, 1].set_ylabel('ReLU(z)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
axes[1, 1].axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

print("Activation Function Properties:")
print("=" * 60)
print(f"{'Function':<12} {'Range':<20} {'Differentiable':<15} {'Zero-Centered'}")
print("-" * 60)
print(f"{'Step':<12} {'[0, 1]':<20} {'No':<15} {'No'}")
print(f"{'Sigmoid':<12} {'(0, 1)':<20} {'Yes':<15} {'No'}")
print(f"{'Tanh':<12} {'(-1, 1)':<20} {'Yes':<15} {'Yes'}")
print(f"{'ReLU':<12} {'[0, ∞)':<20} {'Almost':<15} {'No'}")
```
</div>

**Observations**:
- Sigmoid and tanh saturate (flatten out) for large $|z|$
- ReLU is linear for positive values - simple and effective!
- Step function has a discontinuous jump - not smooth

### Derivatives of Activation Functions

For backpropagation (which we'll learn later), we need derivatives:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions and their derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh_derivative(z):
    return 1 - np.tanh(z)**2

def relu_derivative(z):
    return (z > 0).astype(float)

# Create input range
z = np.linspace(-5, 5, 200)

# Plot derivatives
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Sigmoid derivative
axes[0].plot(z, sigmoid_derivative(z), 'r-', linewidth=2)
axes[0].set_title("Sigmoid Derivative: σ'(z) = σ(z)(1-σ(z))", fontsize=12)
axes[0].set_xlabel('z')
axes[0].set_ylabel("σ'(z)")
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0, color='k', linewidth=0.5)
axes[0].axvline(x=0, color='k', linewidth=0.5)

# Tanh derivative
axes[1].plot(z, tanh_derivative(z), 'g-', linewidth=2)
axes[1].set_title("Tanh Derivative: tanh'(z) = 1 - tanh²(z)", fontsize=12)
axes[1].set_xlabel('z')
axes[1].set_ylabel("tanh'(z)")
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='k', linewidth=0.5)
axes[1].axvline(x=0, color='k', linewidth=0.5)

# ReLU derivative
axes[2].plot(z, relu_derivative(z), 'm-', linewidth=2)
axes[2].set_title("ReLU Derivative: ReLU'(z) = 1 if z>0 else 0", fontsize=12)
axes[2].set_xlabel('z')
axes[2].set_ylabel("ReLU'(z)")
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=0, color='k', linewidth=0.5)
axes[2].axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.show()

print("\nKey Observations:")
print("• Sigmoid derivative peaks at 0.25 when z=0")
print("• Tanh derivative peaks at 1.0 when z=0")
print("• ReLU derivative is constant 1 for z>0 (no saturation!)")
print("\nVanishing Gradient Problem:")
print("• Sigmoid and tanh derivatives → 0 as |z| → ∞")
print("• This causes gradients to vanish in deep networks")
print("• ReLU doesn't have this problem for positive values")
```
</div>

!!! warning "Vanishing Gradient Problem"
    Notice how sigmoid and tanh derivatives approach zero for large $|z|$. In deep networks, this causes gradients to become extremely small during backpropagation, making learning very slow. This is why ReLU became popular - it doesn't saturate for positive values!

## Implementing a Simple Perceptron

Let's build a perceptron from scratch for binary classification.

### Perceptron Class

<div class="python-interactive" markdown="1">
```python
import numpy as np

class Perceptron:
    """
    Simple perceptron for binary classification.

    Parameters:
    -----------
    n_inputs : int
        Number of input features
    activation : str
        Activation function: 'step', 'sigmoid', 'tanh', or 'relu'
    learning_rate : float
        Learning rate for weight updates
    """

    def __init__(self, n_inputs, activation='sigmoid', learning_rate=0.01):
        # Initialize weights randomly (small values)
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0
        self.learning_rate = learning_rate
        self.activation_name = activation

        # Set activation function
        if activation == 'step':
            self.activation = lambda z: (z >= 0).astype(float)
        elif activation == 'sigmoid':
            self.activation = lambda z: 1 / (1 + np.exp(-z))
        elif activation == 'tanh':
            self.activation = np.tanh
        elif activation == 'relu':
            self.activation = lambda z: np.maximum(0, z)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def predict(self, X):
        """
        Make predictions for input data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted values
        """
        # Compute weighted sum: z = w^T x + b
        z = np.dot(X, self.weights) + self.bias
        # Apply activation function
        return self.activation(z)

    def fit(self, X, y, epochs=100):
        """
        Train the perceptron using the perceptron learning rule.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (0 or 1)
        epochs : int
            Number of training epochs
        """
        for epoch in range(epochs):
            # Make predictions
            predictions = self.predict(X)

            # Compute errors
            errors = y - predictions

            # Update weights: w = w + η * (y - ŷ) * x
            self.weights += self.learning_rate * np.dot(X.T, errors)

            # Update bias: b = b + η * (y - ŷ)
            self.bias += self.learning_rate * np.sum(errors)

            # Compute accuracy every 10 epochs
            if (epoch + 1) % 10 == 0:
                accuracy = np.mean((predictions > 0.5) == y)
                print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.4f}")

# Example usage
print("Perceptron Classifier")
print("=" * 60)
print(f"Weights shape: {Perceptron(n_inputs=2).weights.shape}")
print(f"Bias: {Perceptron(n_inputs=2).bias}")
```
</div>

### Example: Learning the AND Gate

The AND gate is a classic example of a linearly separable problem:

| $x_1$ | $x_2$ | AND |
|-------|-------|-----|
| 0     | 0     | 0   |
| 0     | 1     | 0   |
| 1     | 0     | 0   |
| 1     | 1     | 1   |

Let's train a perceptron to learn this:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Create AND gate dataset
X_and = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Train perceptron
perceptron = Perceptron(n_inputs=2, activation='sigmoid', learning_rate=0.1)
print("Training Perceptron on AND Gate:")
print("-" * 60)
perceptron.fit(X_and, y_and, epochs=100)

# Make predictions
print("\nFinal Predictions:")
print("-" * 60)
predictions = perceptron.predict(X_and)
for i in range(len(X_and)):
    print(f"Input: {X_and[i]}, True: {y_and[i]}, "
          f"Predicted: {predictions[i]:.4f}, "
          f"Class: {int(predictions[i] > 0.5)}")

# Visualize decision boundary
def plot_decision_boundary(perceptron, X, y, title):
    # Create mesh
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict on mesh
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(label='Prediction')

    # Plot data points
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', s=100,
                edgecolors='k', label='Class 0', marker='o')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', s=100,
                edgecolors='k', label='Class 1', marker='s')

    # Plot decision boundary (where prediction = 0.5)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

    plt.xlabel('$x_1$', fontsize=12)
    plt.ylabel('$x_2$', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

plot_decision_boundary(perceptron, X_and, y_and,
                      'Perceptron Decision Boundary for AND Gate')

print(f"\nLearned Parameters:")
print(f"Weights: {perceptron.weights}")
print(f"Bias: {perceptron.bias:.4f}")
```
</div>

**Analysis**: The perceptron successfully learned the AND gate! The decision boundary (black line) correctly separates the two classes. This works because AND is **linearly separable**.

### Example: Learning the OR Gate

Let's try another linearly separable problem:

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Create OR gate dataset
X_or = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
y_or = np.array([0, 1, 1, 1])

# Train perceptron
perceptron_or = Perceptron(n_inputs=2, activation='sigmoid', learning_rate=0.1)
print("Training Perceptron on OR Gate:")
print("-" * 60)
perceptron_or.fit(X_or, y_or, epochs=100)

# Make predictions
print("\nFinal Predictions:")
print("-" * 60)
predictions = perceptron_or.predict(X_or)
for i in range(len(X_or)):
    print(f"Input: {X_or[i]}, True: {y_or[i]}, "
          f"Predicted: {predictions[i]:.4f}, "
          f"Class: {int(predictions[i] > 0.5)}")

plot_decision_boundary(perceptron_or, X_or, y_or,
                      'Perceptron Decision Boundary for OR Gate')
```
</div>

**Success again!** The OR gate is also linearly separable, so the perceptron learns it easily.

## The XOR Problem: Perceptron's Limitation

Now let's try the XOR (exclusive OR) gate:

| $x_1$ | $x_2$ | XOR |
|-------|-------|-----|
| 0     | 0     | 0   |
| 0     | 1     | 1   |
| 1     | 0     | 1   |
| 1     | 1     | 0   |

This is **not linearly separable**! You cannot draw a single straight line to separate the classes.

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Create XOR gate dataset
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_xor = np.array([0, 1, 1, 0])

# Visualize the XOR problem
plt.figure(figsize=(8, 6))
plt.scatter(X_xor[y_xor==0, 0], X_xor[y_xor==0, 1], c='blue', s=200,
            edgecolors='k', label='Class 0', marker='o')
plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1], c='red', s=200,
            edgecolors='k', label='Class 1', marker='s')
plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$x_2$', fontsize=12)
plt.title('XOR Problem: Not Linearly Separable!', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.show()

print("Can you draw a single straight line to separate blue from red?")
print("NO! This is the famous XOR problem.")
print("\nTry to draw any line - it will always misclassify at least one point.")
```
</div>

Let's see what happens when we train a perceptron on XOR:

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Train perceptron on XOR (this will fail!)
perceptron_xor = Perceptron(n_inputs=2, activation='sigmoid', learning_rate=0.1)
print("Training Perceptron on XOR Gate (this will struggle!):")
print("-" * 60)
perceptron_xor.fit(X_xor, y_xor, epochs=200)

# Make predictions
print("\nFinal Predictions:")
print("-" * 60)
predictions = perceptron_xor.predict(X_xor)
for i in range(len(X_xor)):
    print(f"Input: {X_xor[i]}, True: {y_xor[i]}, "
          f"Predicted: {predictions[i]:.4f}, "
          f"Class: {int(predictions[i] > 0.5)}")

accuracy = np.mean((predictions > 0.5) == y_xor)
print(f"\nFinal Accuracy: {accuracy:.4f}")
print("Notice: The perceptron cannot achieve 100% accuracy!")

plot_decision_boundary(perceptron_xor, X_xor, y_xor,
                      'Perceptron Fails on XOR: Decision Boundary')
```
</div>

!!! warning "Linear Separability"
    The perceptron **cannot** learn XOR because XOR is not linearly separable! No matter how long you train, a single perceptron will always misclassify at least one point.

    This limitation led to the "AI Winter" of the 1970s. The solution? **Multi-layer networks** (which we'll learn in Lesson 2)!

### Understanding Linear Separability

A dataset is **linearly separable** if you can draw a hyperplane that perfectly separates the classes.

**Examples**:
- ✅ AND gate: Linearly separable
- ✅ OR gate: Linearly separable
- ❌ XOR gate: NOT linearly separable

**In general**:
- 2D: Need a line
- 3D: Need a plane
- nD: Need a hyperplane (n-1 dimensional)

**Why XOR fails**: Notice that the two blue points are diagonal from each other, as are the red points. No single line can separate them!

## The Perceptron Learning Algorithm

Let's understand how the perceptron learns. The algorithm is beautifully simple:

### Algorithm Steps

For each training example $(\mathbf{x}^{(i)}, y^{(i)})$:

1. **Predict**: $\hat{y}^{(i)} = f(\mathbf{w}^T \mathbf{x}^{(i)} + b)$

2. **Compute Error**: $e^{(i)} = y^{(i)} - \hat{y}^{(i)}$

3. **Update Weights**: $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot e^{(i)} \cdot \mathbf{x}^{(i)}$

4. **Update Bias**: $b \leftarrow b + \eta \cdot e^{(i)}$

Where $\eta$ is the **learning rate** (step size).

### Intuition Behind the Update Rule

The weight update $\mathbf{w} \leftarrow \mathbf{w} + \eta \cdot e \cdot \mathbf{x}$ is brilliant:

- If **prediction is correct** ($e = 0$): weights don't change ✅
- If **prediction too low** ($e > 0$): weights increase in direction of $\mathbf{x}$ ⬆️
- If **prediction too high** ($e < 0$): weights decrease in direction of $\mathbf{x}$ ⬇️

The magnitude of change is proportional to:
- **Error size** (larger error → larger update)
- **Input magnitude** (features with larger values get larger updates)
- **Learning rate** (controls step size)

### Convergence Theorem

**Perceptron Convergence Theorem** (Rosenblatt, 1958):

If the data is linearly separable, the perceptron learning algorithm is **guaranteed to converge** to a solution in a finite number of steps!

However, if the data is NOT linearly separable (like XOR), it will never converge to perfect accuracy.

## Visualizing Learning Process

Let's watch the perceptron learn in real-time:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_progress(X, y, epochs=20):
    """
    Visualize how the decision boundary evolves during training.
    """
    perceptron = Perceptron(n_inputs=2, activation='sigmoid', learning_rate=0.1)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Track epochs to visualize
    epochs_to_plot = [0, 5, 10, 20, 50, 100]

    for idx, target_epoch in enumerate(epochs_to_plot):
        # Train up to target epoch
        if target_epoch > 0:
            perceptron.fit(X, y, epochs=target_epoch if idx == 0 else
                          target_epoch - epochs_to_plot[idx-1])

        # Create mesh for decision boundary
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot
        axes[idx].contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
        axes[idx].scatter(X[y==0, 0], X[y==0, 1], c='blue', s=100,
                         edgecolors='k', marker='o')
        axes[idx].scatter(X[y==1, 0], X[y==1, 1], c='red', s=100,
                         edgecolors='k', marker='s')
        axes[idx].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)

        # Compute accuracy
        predictions = perceptron.predict(X)
        accuracy = np.mean((predictions > 0.5) == y)

        axes[idx].set_title(f'Epoch {target_epoch}: Accuracy = {accuracy:.2f}',
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('$x_1$')
        axes[idx].set_ylabel('$x_2$')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Visualize learning on AND gate
print("Watch the perceptron learn the AND gate:")
plot_learning_progress(X_and, y_and)

print("\nNotice how the decision boundary (black line) adjusts over time!")
print("Initially random, it quickly finds the correct separation.")
```
</div>

**Observations**:
- Early epochs: Decision boundary is poorly positioned
- Middle epochs: Boundary moves toward correct position
- Later epochs: Boundary stabilizes once all points are correctly classified

## Practical Considerations

### Choosing Activation Functions

| Use Case | Recommended Activation |
|----------|------------------------|
| **Binary Classification** (output layer) | Sigmoid (outputs probability) |
| **Hidden Layers** (modern networks) | ReLU (fast, no vanishing gradient) |
| **When outputs need to be zero-centered** | Tanh (range: -1 to 1) |
| **Historical/theoretical study** | Step function |

### Setting the Learning Rate

The learning rate $\eta$ controls how quickly the model learns:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Compare different learning rates
learning_rates = [0.001, 0.01, 0.1, 1.0]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, lr in enumerate(learning_rates):
    perceptron = Perceptron(n_inputs=2, activation='sigmoid', learning_rate=lr)

    # Track accuracy over epochs
    accuracies = []
    for epoch in range(100):
        perceptron.fit(X_and, y_and, epochs=1)
        predictions = perceptron.predict(X_and)
        accuracy = np.mean((predictions > 0.5) == y_and)
        accuracies.append(accuracy)

    # Plot learning curve
    axes[idx].plot(accuracies, linewidth=2)
    axes[idx].set_xlabel('Epoch', fontsize=12)
    axes[idx].set_ylabel('Accuracy', fontsize=12)
    axes[idx].set_title(f'Learning Rate = {lr}', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim([0, 1.05])

plt.tight_layout()
plt.show()

print("Learning Rate Effects:")
print("=" * 60)
print("Too small (0.001): Learns slowly, may not converge")
print("Just right (0.01-0.1): Learns steadily and converges")
print("Too large (1.0): May oscillate or diverge")
```
</div>

!!! tip "Learning Rate Guidelines"
    - **Too small**: Learning is very slow
    - **Too large**: May overshoot optimal weights, causing oscillation
    - **Good range**: Typically 0.001 to 0.1
    - **Adaptive**: Modern optimizers like Adam automatically adjust learning rates

## Real-World Application: Iris Classification

Let's apply our perceptron to a real dataset: classifying Iris flowers!

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Iris dataset (we'll use only 2 classes for binary classification)
iris = load_iris()
# Use only setosa (0) and versicolor (1)
mask = iris.target != 2
X = iris.data[mask, :2]  # Use only first 2 features for visualization
y = iris.target[mask]

print("Iris Dataset:")
print("=" * 60)
print(f"Number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Classes: {np.unique(y)}")
print(f"Feature names: {iris.feature_names[:2]}")

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features (important for gradient-based learning!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train perceptron
perceptron = Perceptron(n_inputs=2, activation='sigmoid', learning_rate=0.1)
print("\nTraining Perceptron:")
print("-" * 60)
perceptron.fit(X_train_scaled, y_train, epochs=100)

# Evaluate on test set
test_predictions = perceptron.predict(X_test_scaled)
test_accuracy = np.mean((test_predictions > 0.5) == y_test)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Samples Correct: {int(test_accuracy * len(y_test))}/{len(y_test)}")

# Visualize decision boundary on test set
def plot_iris_decision_boundary(perceptron, X_train, y_train, X_test, y_test, scaler):
    # Create mesh in original scale
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Scale mesh points and predict
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    Z = perceptron.predict(mesh_points_scaled)
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    plt.colorbar(label='Prediction')

    # Plot training data
    plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
                c='blue', s=80, edgecolors='k', label='Train: Setosa',
                marker='o', alpha=0.7)
    plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
                c='red', s=80, edgecolors='k', label='Train: Versicolor',
                marker='s', alpha=0.7)

    # Plot test data with different markers
    plt.scatter(X_test[y_test==0, 0], X_test[y_test==0, 1],
                c='blue', s=120, edgecolors='yellow', linewidths=2,
                label='Test: Setosa', marker='o')
    plt.scatter(X_test[y_test==1, 0], X_test[y_test==1, 1],
                c='red', s=120, edgecolors='yellow', linewidths=2,
                label='Test: Versicolor', marker='s')

    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5)

    plt.xlabel(f'{iris.feature_names[0]}', fontsize=12)
    plt.ylabel(f'{iris.feature_names[1]}', fontsize=12)
    plt.title('Perceptron: Iris Classification (Setosa vs Versicolor)',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.show()

plot_iris_decision_boundary(perceptron, X_train, y_train, X_test, y_test, scaler)
```
</div>

**Success!** The perceptron can classify these two Iris species because they're linearly separable in this 2D space.

!!! tip "Feature Scaling"
    Notice we used `StandardScaler` to normalize the features. This is crucial for perceptron learning because:
    - Features with larger scales dominate the dot product
    - Standardization makes all features contribute equally
    - Learning converges faster with scaled features

## Summary

### Key Takeaways

!!! success "What You Learned"
    **1. The Perceptron Model**
    - Artificial neuron inspired by biology
    - Computes: $\hat{y} = f(\mathbf{w}^T \mathbf{x} + b)$
    - Weights and bias are learned from data

    **2. Activation Functions**
    - Step: Binary output (original perceptron)
    - Sigmoid: Smooth, outputs probabilities
    - Tanh: Zero-centered, range (-1, 1)
    - ReLU: Modern favorite, no vanishing gradient

    **3. Learning Algorithm**
    - Update rule: $\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \mathbf{x}$
    - Guaranteed convergence for linearly separable data
    - Learning rate controls step size

    **4. Linear Separability**
    - Perceptron can only learn linearly separable patterns
    - Works: AND, OR gates
    - Fails: XOR gate
    - Solution: Multi-layer networks! (next lesson)

### Limitations

The single-layer perceptron has fundamental limitations:

- ❌ Cannot learn XOR or any non-linearly separable function
- ❌ Limited to linear decision boundaries
- ❌ Cannot capture complex patterns

These limitations motivated the development of **multi-layer perceptrons** (neural networks), which we'll explore in the next lesson!

### When to Use Perceptrons

Despite limitations, perceptrons are still useful:

- ✅ Simple binary classification with linear boundary
- ✅ Fast training and prediction
- ✅ Interpretable weights
- ✅ Good baseline model
- ✅ Building block for understanding neural networks

## Next Steps

Congratulations! You now understand the fundamental building block of neural networks. But we've seen the perceptron has serious limitations - it can only learn linear patterns.

**In Lesson 2**, we'll overcome this limitation by stacking perceptrons into **multi-layer networks**. You'll learn:
- How combining simple units creates powerful models
- The architecture of feedforward neural networks
- How depth enables learning complex, non-linear patterns
- The universal approximation theorem

**Ready to build your first real neural network?**

[Continue to Lesson 2: Feedforward Networks](02-feedforward-networks.md){ .md-button .md-button--primary }

[Return to Module 4 Overview](index.md){ .md-button }

---

**Questions or feedback?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues).
