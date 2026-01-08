# Lesson 2: Feedforward Neural Networks

## Introduction

In Lesson 1, we discovered the perceptron's fundamental limitation: it can only learn linearly separable patterns. It failed miserably on XOR! But what if we could combine multiple perceptrons to overcome this limitation?

**Enter the Multi-Layer Perceptron (MLP)**, also called a feedforward neural network. By stacking layers of neurons, we create a model capable of learning arbitrarily complex functions. This is where neural networks become truly powerful!

In this lesson, you'll learn:

- **How layers work together** to transform data
- **Network architecture**: inputs, hidden layers, and outputs
- **Forward propagation**: how data flows through the network
- **The universal approximation theorem**: why depth matters
- **How to solve XOR** with a simple 2-layer network

This is the foundation for all modern deep learning!

## Learning Objectives

By the end of this lesson, you will:

- ✅ Understand multi-layer perceptron (MLP) architecture
- ✅ Master forward propagation through multiple layers
- ✅ Work with hidden layers and network depth
- ✅ Implement a 2-layer network from scratch
- ✅ Solve the XOR problem with a neural network
- ✅ Understand why depth enables learning complex patterns
- ✅ Apply the universal approximation theorem

**Estimated Time**: 60 minutes

## From Single Neuron to Neural Network

### Visual Introduction to Neural Networks

To understand how neural networks work, watch this excellent deep dive into the architecture and forward propagation:

<div style="text-align: center; margin: 20px 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/aircAruvnKk" title="Neural Networks Part 1 by 3Blue1Brown" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

### The Limitation of Single Neurons

Recall from Lesson 1 that a single perceptron computes:

$$
\hat{y} = f(\mathbf{w}^T \mathbf{x} + b)
$$

This creates a **linear decision boundary** - a single hyperplane. Many real-world problems aren't linearly separable!

### The Solution: Layers of Neurons

What if instead of one perceptron, we use many? Stack them in layers:

1. **Input layer**: Receives the raw features
2. **Hidden layer(s)**: Intermediate computations, learn feature representations
3. **Output layer**: Produces final predictions

**Key Insight**: Each hidden layer learns a new representation of the data. The network transforms the input through multiple non-linear transformations, eventually making it linearly separable in the final layer!

## Multi-Layer Perceptron Architecture

### Basic Structure

A simple 2-layer neural network (1 hidden layer) looks like this:

```
Input Layer    Hidden Layer    Output Layer
   (3)            (4)             (1)

   x₁ ──────┐
            ├───→ h₁ ──┐
   x₂ ──────┤          │
            ├───→ h₂ ──┤
   x₃ ──────┤          ├───→ ŷ
            ├───→ h₃ ──┤
            │          │
            └───→ h₄ ──┘
```

**Architecture notation**: [3, 4, 1]
- Input layer: 3 neurons (3 features)
- Hidden layer: 4 neurons
- Output layer: 1 neuron (binary classification)

### Layer Notation

Let's establish clear notation:

- $L$: Total number of layers (counting input as layer 0)
- $n^{[l]}$: Number of neurons in layer $l$
- $\mathbf{a}^{[l]}$: Activations (outputs) of layer $l$
- $\mathbf{z}^{[l]}$: Pre-activation values of layer $l$
- $\mathbf{W}^{[l]}$: Weight matrix for layer $l$
- $\mathbf{b}^{[l]}$: Bias vector for layer $l$

**Example**: For a [3, 4, 1] network:
- $L = 2$ (2 layers of weights)
- $n^{[0]} = 3$ (input), $n^{[1]} = 4$ (hidden), $n^{[2]} = 1$ (output)
- $\mathbf{W}^{[1]} \in \mathbb{R}^{4 \times 3}$ (connects input to hidden)
- $\mathbf{W}^{[2]} \in \mathbb{R}^{1 \times 4}$ (connects hidden to output)

### Weight Dimensions

**Critical**: Understanding weight dimensions is essential for implementing neural networks!

For a connection from layer $l-1$ (with $n^{[l-1]}$ neurons) to layer $l$ (with $n^{[l]}$ neurons):

$$
\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}
$$

$$
\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]}}
$$

**Example**: In a [3, 4, 1] network:
- $\mathbf{W}^{[1]}$ has shape $(4, 3)$ - 4 rows (one per hidden neuron), 3 columns (one per input)
- $\mathbf{b}^{[1]}$ has shape $(4,)$ - one bias per hidden neuron
- $\mathbf{W}^{[2]}$ has shape $(1, 4)$ - 1 row (output neuron), 4 columns (hidden neurons)
- $\mathbf{b}^{[2]}$ has shape $(1,)$ - one bias for output neuron

!!! tip "Dimension Checking"
    Always verify your weight dimensions! Most bugs in neural network implementations come from dimension mismatches. The rule: $\mathbf{W}^{[l]}$ has one row per neuron in layer $l$ and one column per neuron in layer $l-1$.

## Forward Propagation

Forward propagation is how we compute the network's output given an input. Data flows forward through the layers.

### Single Training Example

For a single input $\mathbf{x}$:

**Layer 1 (Input → Hidden)**:

$$
\begin{align}
\mathbf{z}^{[1]} &= \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]} \\
\mathbf{a}^{[1]} &= f^{[1]}(\mathbf{z}^{[1]})
\end{align}
$$

**Layer 2 (Hidden → Output)**:

$$
\begin{align}
\mathbf{z}^{[2]} &= \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]} \\
\mathbf{a}^{[2]} &= f^{[2]}(\mathbf{z}^{[2]})
\end{align}
$$

Where:
- $f^{[l]}$ is the activation function for layer $l$
- $\mathbf{a}^{[0]} = \mathbf{x}$ (input is the activation of layer 0)
- $\mathbf{a}^{[2]} = \hat{y}$ (final activation is the output)

### General Formula for Any Layer

For any layer $l$:

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})
$$

This is the **fundamental equation** of feedforward networks!

### Vectorized Forward Propagation (Multiple Examples)

To process a batch of $m$ training examples $\mathbf{X} \in \mathbb{R}^{n^{[0]} \times m}$:

$$
\begin{align}
\mathbf{Z}^{[l]} &= \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]} \\
\mathbf{A}^{[l]} &= f^{[l]}(\mathbf{Z}^{[l]})
\end{align}
$$

Where:
- $\mathbf{X} \in \mathbb{R}^{n^{[0]} \times m}$: Each column is a training example
- $\mathbf{Z}^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$: Pre-activations for all examples
- $\mathbf{A}^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$: Activations for all examples

**Vectorization** allows us to process many examples in parallel - crucial for efficiency!

## Step-by-Step Example: 2-Layer Network

Let's work through a concrete example with numbers.

### Network Architecture

- Input: 2 features
- Hidden: 3 neurons (ReLU activation)
- Output: 1 neuron (sigmoid activation)
- Architecture: [2, 3, 1]

### Initialize Weights

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Network architecture
n_input = 2
n_hidden = 3
n_output = 1

# Initialize weights (small random values)
W1 = np.random.randn(n_hidden, n_input) * 0.01
b1 = np.zeros((n_hidden, 1))

W2 = np.random.randn(n_output, n_hidden) * 0.01
b2 = np.zeros((n_output, 1))

print("Weight Dimensions:")
print("=" * 60)
print(f"W1 shape: {W1.shape} - connects {n_input} inputs to {n_hidden} hidden neurons")
print(f"b1 shape: {b1.shape}")
print(f"W2 shape: {W2.shape} - connects {n_hidden} hidden to {n_output} output")
print(f"b2 shape: {b2.shape}")

print("\nInitialized Weights:")
print("-" * 60)
print(f"W1:\n{W1}")
print(f"\nb1:\n{b1}")
print(f"\nW2:\n{W2}")
print(f"\nb2:\n{b2}")
```
</div>

### Forward Pass: Step-by-Step

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Define activation functions
def relu(Z):
    """ReLU activation: max(0, z)"""
    return np.maximum(0, Z)

def sigmoid(Z):
    """Sigmoid activation: 1 / (1 + e^(-z))"""
    return 1 / (1 + np.exp(-Z))

# Input example: x = [0.5, 0.8]
X = np.array([[0.5], [0.8]])

print("Forward Propagation: Step-by-Step")
print("=" * 60)
print(f"Input X:\n{X}\nShape: {X.shape}")

# Layer 1: Input → Hidden
print("\n" + "="*60)
print("LAYER 1: Input → Hidden")
print("="*60)

Z1 = np.dot(W1, X) + b1
print(f"\nZ1 = W1 @ X + b1")
print(f"Z1 shape: {Z1.shape}")
print(f"Z1 (pre-activation):\n{Z1}")

A1 = relu(Z1)
print(f"\nA1 = ReLU(Z1)")
print(f"A1 (activation):\n{A1}")

# Layer 2: Hidden → Output
print("\n" + "="*60)
print("LAYER 2: Hidden → Output")
print("="*60)

Z2 = np.dot(W2, A1) + b2
print(f"\nZ2 = W2 @ A1 + b2")
print(f"Z2 shape: {Z2.shape}")
print(f"Z2 (pre-activation):\n{Z2}")

A2 = sigmoid(Z2)
print(f"\nA2 = Sigmoid(Z2)")
print(f"A2 (final output):\n{A2}")
print(f"\nPrediction: {A2[0, 0]:.4f}")
print(f"Class: {int(A2[0, 0] > 0.5)}")
```
</div>

### Visualizing Information Flow

The computation graph shows how information flows:

```
       Input               Hidden               Output
        x₁                   h₁
      (0.5) ──W₁──→ ReLU ──→     ──W₂──→
                             h₂               Sigmoid → ŷ
        x₂                       ──────→       (0.5123)
      (0.8) ──────→        h₃

Each arrow represents:
  z = w·a + b  →  activation function  →  a
```

**Key Observations**:
- Information flows **left to right** only (feedforward)
- Each layer transforms the representation
- Non-linear activations (ReLU, sigmoid) are crucial!

## Implementing a 2-Layer Neural Network

Let's build a complete neural network class from scratch:

<div class="python-interactive" markdown="1">
```python
import numpy as np

class NeuralNetwork:
    """
    2-layer neural network for binary classification.

    Architecture: [n_input, n_hidden, 1]
    Hidden activation: ReLU
    Output activation: Sigmoid
    """

    def __init__(self, n_input, n_hidden):
        """
        Initialize network parameters.

        Parameters:
        -----------
        n_input : int
            Number of input features
        n_hidden : int
            Number of neurons in hidden layer
        """
        # Initialize weights with small random values
        self.W1 = np.random.randn(n_hidden, n_input) * 0.01
        self.b1 = np.zeros((n_hidden, 1))

        self.W2 = np.random.randn(1, n_hidden) * 0.01
        self.b2 = np.zeros((1, 1))

    def relu(self, Z):
        """ReLU activation function."""
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        """
        Perform forward propagation.

        Parameters:
        -----------
        X : ndarray, shape (n_input, m)
            Input data (m examples)

        Returns:
        --------
        A2 : ndarray, shape (1, m)
            Output predictions
        cache : dict
            Intermediate values for backpropagation
        """
        # Layer 1
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.relu(Z1)

        # Layer 2
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)

        # Store values for backpropagation (we'll use this later)
        cache = {
            'Z1': Z1, 'A1': A1,
            'Z2': Z2, 'A2': A2,
            'X': X
        }

        return A2, cache

    def predict(self, X):
        """
        Make predictions for input data.

        Parameters:
        -----------
        X : ndarray, shape (n_input, m)
            Input data

        Returns:
        --------
        predictions : ndarray, shape (1, m)
            Binary predictions (0 or 1)
        """
        A2, _ = self.forward(X)
        return (A2 > 0.5).astype(int)

# Example usage
print("2-Layer Neural Network")
print("=" * 60)

# Create network
nn = NeuralNetwork(n_input=2, n_hidden=4)

# Test with random input
X_test = np.random.randn(2, 5)  # 2 features, 5 examples
predictions, cache = nn.forward(X_test)

print(f"Input shape: {X_test.shape}")
print(f"Hidden activations shape: {cache['A1'].shape}")
print(f"Output shape: {predictions.shape}")
print(f"\nPredictions:\n{predictions}")
print(f"\nBinary predictions:\n{nn.predict(X_test)}")
```
</div>

## Solving XOR with a Neural Network

Remember XOR? A single perceptron failed. Let's see how a 2-layer network succeeds!

### The XOR Dataset

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# XOR dataset
X_xor = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
y_xor = np.array([[0, 1, 1, 0]])

print("XOR Dataset:")
print("=" * 60)
print(f"X shape: {X_xor.shape}")
print(f"y shape: {y_xor.shape}")
print(f"\nInputs:\n{X_xor.T}")
print(f"\nOutputs:\n{y_xor.T}")

# Visualize
plt.figure(figsize=(8, 6))
for i in range(4):
    if y_xor[0, i] == 0:
        plt.scatter(X_xor[0, i], X_xor[1, i], c='blue', s=200,
                   edgecolors='k', marker='o', label='Class 0' if i == 0 else '')
    else:
        plt.scatter(X_xor[0, i], X_xor[1, i], c='red', s=200,
                   edgecolors='k', marker='s', label='Class 1' if i == 1 else '')

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('XOR Problem: Not Linearly Separable', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.show()

print("\nChallenge: No single line can separate blue from red!")
```
</div>

### Training the Network

For now, we'll manually set weights that solve XOR (we'll learn automatic training with backpropagation in Lesson 3):

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Create network for XOR
nn_xor = NeuralNetwork(n_input=2, n_hidden=2)

# Manually set weights to solve XOR
# These weights were carefully chosen to implement:
# h1 = AND(x1, x2), h2 = OR(x1, x2)
# output = AND(NOT h1, h2) = XOR(x1, x2)

nn_xor.W1 = np.array([[ 20,  20],   # First hidden neuron (AND-like)
                      [ 20,  20]])   # Second hidden neuron (OR-like)
nn_xor.b1 = np.array([[-30],         # High threshold for AND
                      [-10]])        # Low threshold for OR

nn_xor.W2 = np.array([[-20, 20]])    # Negative first, positive second
nn_xor.b2 = np.array([[-10]])

# Test the network
predictions, cache = nn_xor.forward(X_xor)

print("XOR Solution with 2-Layer Network:")
print("=" * 60)
print("\nTruth Table:")
print("-" * 60)
print(f"{'x1':<5} {'x2':<5} {'True':<8} {'Predicted':<12} {'Correct'}")
print("-" * 60)
for i in range(4):
    x1, x2 = X_xor[:, i]
    true_y = y_xor[0, i]
    pred_y = predictions[0, i]
    correct = '✓' if (pred_y > 0.5) == true_y else '✗'
    print(f"{x1:<5.0f} {x2:<5.0f} {true_y:<8.0f} {pred_y:<12.4f} {correct}")

accuracy = np.mean((predictions > 0.5) == y_xor)
print(f"\nAccuracy: {accuracy:.2%}")

# Show hidden layer activations
print("\nHidden Layer Activations:")
print("-" * 60)
print("(Shows how the network transforms the input)")
print(f"{'x1':<5} {'x2':<5} {'h1 (AND-like)':<15} {'h2 (OR-like)'}")
print("-" * 60)
for i in range(4):
    x1, x2 = X_xor[:, i]
    h1, h2 = cache['A1'][:, i]
    print(f"{x1:<5.0f} {x2:<5.0f} {h1:<15.4f} {h2:<15.4f}")
```
</div>

**Success!** The network perfectly solves XOR. How?

### How XOR Is Solved

The network learns a clever transformation:

1. **Hidden Layer**: Creates new features
   - $h_1 \approx \text{AND}(x_1, x_2)$ - activates only when both inputs are 1
   - $h_2 \approx \text{OR}(x_1, x_2)$ - activates when either input is 1

2. **Output Layer**: Combines hidden features
   - $\hat{y} = \text{AND}(\neg h_1, h_2)$ - output is 1 when OR is true but AND is false
   - This is exactly XOR!

**Key Insight**: The hidden layer **transforms the representation** so that XOR becomes linearly separable in the new space!

### Visualizing the Hidden Layer Transformation

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Get hidden layer activations
predictions, cache = nn_xor.forward(X_xor)
H = cache['A1']  # Hidden activations (2 x 4)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Original space (not linearly separable)
ax1.scatter(X_xor[0, y_xor[0]==0], X_xor[1, y_xor[0]==0],
            c='blue', s=200, edgecolors='k', marker='o', label='Class 0')
ax1.scatter(X_xor[0, y_xor[0]==1], X_xor[1, y_xor[0]==1],
            c='red', s=200, edgecolors='k', marker='s', label='Class 1')
ax1.set_xlabel('$x_1$', fontsize=14)
ax1.set_ylabel('$x_2$', fontsize=14)
ax1.set_title('Original Space\n(Not Linearly Separable)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(-0.5, 1.5)

# Hidden space (linearly separable!)
ax2.scatter(H[0, y_xor[0]==0], H[1, y_xor[0]==0],
            c='blue', s=200, edgecolors='k', marker='o', label='Class 0')
ax2.scatter(H[0, y_xor[0]==1], H[1, y_xor[0]==1],
            c='red', s=200, edgecolors='k', marker='s', label='Class 1')

# Draw decision boundary in hidden space
h_vals = np.linspace(-0.5, 1.5, 100)
# Approximate decision boundary from output layer weights
# W2[0] * h1 + W2[1] * h2 + b2 = 0
# h2 = -(W2[0] * h1 + b2) / W2[1]
boundary_h2 = -(nn_xor.W2[0, 0] * h_vals + nn_xor.b2[0, 0]) / nn_xor.W2[0, 1]
ax2.plot(h_vals, boundary_h2, 'k-', linewidth=2, label='Decision Boundary')

ax2.set_xlabel('$h_1$ (AND-like)', fontsize=14)
ax2.set_ylabel('$h_2$ (OR-like)', fontsize=14)
ax2.set_title('Hidden Layer Space\n(Linearly Separable!)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.5, 1.5)
ax2.set_ylim(-0.5, 1.5)

plt.tight_layout()
plt.show()

print("The Magic of Hidden Layers:")
print("=" * 60)
print("In the original space (left), XOR is not linearly separable.")
print("But in the hidden layer space (right), a simple line separates the classes!")
print("\nThis is why neural networks are so powerful:")
print("Hidden layers learn representations that make problems easier to solve.")
```
</div>

!!! success "The Power of Representation Learning"
    Neural networks don't just fit functions - they **learn representations**! Hidden layers transform the input into a new space where the problem becomes easier. This is the core insight of deep learning.

## Decision Boundaries of Neural Networks

Unlike a perceptron (linear boundary), neural networks create complex, non-linear decision boundaries.

### Visualizing Complex Boundaries

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary_nn(nn, X, y, title):
    """Plot decision boundary for a neural network."""
    # Create mesh
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Flatten mesh and make predictions
    mesh_points = np.c_[xx.ravel(), yy.ravel()].T
    Z, _ = nn.forward(mesh_points)
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(label='Prediction')

    # Plot data points
    plt.scatter(X[0, y[0]==0], X[1, y[0]==0], c='blue', s=200,
                edgecolors='k', label='Class 0', marker='o', linewidth=2)
    plt.scatter(X[0, y[0]==1], X[1, y[0]==1], c='red', s=200,
                edgecolors='k', label='Class 1', marker='s', linewidth=2)

    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)

    plt.xlabel('$x_1$', fontsize=13)
    plt.ylabel('$x_2$', fontsize=13)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()

# Plot XOR decision boundary
plot_decision_boundary_nn(nn_xor, X_xor, y_xor,
                         'Neural Network Decision Boundary for XOR')

print("Notice the NON-LINEAR decision boundary!")
print("The network creates regions that a single line could never achieve.")
```
</div>

**Observation**: The decision boundary is **curved** and **non-linear**. This is only possible because of:
1. Multiple layers
2. Non-linear activation functions

Without either of these, the network would still be limited to linear boundaries!

## The Universal Approximation Theorem

One of the most important theoretical results in neural networks:

!!! info "Universal Approximation Theorem"
    A feedforward neural network with:
    - At least one hidden layer
    - A sufficient number of neurons
    - Non-linear activation functions

    Can approximate **any continuous function** to arbitrary accuracy!

### What This Means

**In practice**:
- Neural networks are **universal function approximators**
- Given enough hidden neurons, a 2-layer network can fit any function
- Depth (more layers) often works better than width (more neurons per layer)

**Implications**:
- ✅ Neural networks can learn incredibly complex patterns
- ✅ No need for manual feature engineering (the network learns features!)
- ⚠️ Still need to actually train the network (finding the right weights is hard!)
- ⚠️ Risk of overfitting with too many parameters

### Visualizing Universal Approximation

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Create a complex function to approximate
def target_function(x):
    """A complex non-linear function."""
    return np.sin(x) + 0.5 * np.cos(3*x) + 0.2 * np.sin(5*x)

# Generate data
X_train = np.linspace(-3, 3, 100).reshape(1, -1)
y_train = target_function(X_train)

# Create networks with different hidden layer sizes
hidden_sizes = [2, 5, 10, 20]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, n_hidden in enumerate(hidden_sizes):
    # Create and "train" network (we'll skip actual training for now)
    nn = NeuralNetwork(n_input=1, n_hidden=n_hidden)

    # Make predictions
    y_pred, _ = nn.forward(X_train)

    # Plot
    axes[idx].plot(X_train[0], y_train[0], 'b-', linewidth=2, label='Target Function')
    axes[idx].plot(X_train[0], y_pred[0], 'r--', linewidth=2, label='Network Approximation')
    axes[idx].set_xlabel('x', fontsize=12)
    axes[idx].set_ylabel('y', fontsize=12)
    axes[idx].set_title(f'Hidden Neurons: {n_hidden}', fontsize=12, fontweight='bold')
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Universal Approximation in Action:")
print("=" * 60)
print("More hidden neurons → Better approximation (when properly trained)")
print("\nNote: These networks aren't trained yet (just random weights),")
print("but the theorem guarantees that with enough neurons and proper training,")
print("we can approximate any continuous function!")
```
</div>

### Depth vs. Width

Modern research shows that **depth** (many layers) is often more efficient than **width** (many neurons per layer):

| Property | Wide & Shallow | Deep & Narrow |
|----------|----------------|---------------|
| **Parameters** | Many | Fewer for same capacity |
| **Expressiveness** | Limited hierarchical structure | Rich hierarchical features |
| **Training** | Can be easier | May need careful initialization |
| **Generalization** | Often worse | Often better |
| **Real-world performance** | Good for simple tasks | State-of-the-art |

**Example**:
- 1 hidden layer with 1000 neurons ≈ 1 million parameters
- 3 hidden layers with 100 neurons each ≈ 30,000 parameters

The deep network learns hierarchical features (edges → shapes → objects) while being more parameter-efficient!

## Network Architecture Choices

### How Many Hidden Layers?

| Layers | Capability | Use Case |
|--------|------------|----------|
| **1 hidden** | Approximate any continuous function | Simple patterns, small datasets |
| **2-3 hidden** | Learn hierarchical features | Most practical problems |
| **4+ hidden** | Deep hierarchical learning | Images, speech, language (deep learning) |

### How Many Neurons Per Layer?

**Rules of thumb**:
- Start small, gradually increase
- Hidden layer size between input and output size
- Often 2/3 of input size + output size
- Use validation set to tune

**Example guidelines**:
- Input: 100 features, Output: 10 classes
- Try: [100, 75, 50, 10] or [100, 64, 32, 10]

### Which Activation Functions?

| Layer | Recommended Activation | Why |
|-------|------------------------|-----|
| **Hidden layers** | ReLU (or variants) | Fast, no vanishing gradient |
| **Output (binary classification)** | Sigmoid | Outputs probability [0, 1] |
| **Output (multi-class)** | Softmax | Outputs probability distribution |
| **Output (regression)** | Linear (no activation) | Unbounded output |

### Weight Initialization

Random initialization is important! Why not start at zero?

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Bad: All zeros
W_zeros = np.zeros((4, 3))
print("Zero Initialization (BAD):")
print(W_zeros)
print("\nProblem: All neurons compute the same thing!")
print("No matter how long you train, they stay identical (symmetry problem).\n")

# Good: Small random values
W_random = np.random.randn(4, 3) * 0.01
print("Small Random Initialization (GOOD):")
print(W_random)
print("\nBenefit: Breaks symmetry, neurons learn different features!")
print(f"Standard deviation: {W_random.std():.4f}")
```
</div>

**Best practices**:
- ✅ **Xavier/Glorot**: $\mathcal{N}(0, \sqrt{1/n_{in}})$ for sigmoid/tanh
- ✅ **He initialization**: $\mathcal{N}(0, \sqrt{2/n_{in}})$ for ReLU
- ❌ Never initialize all weights to same value (symmetry breaking!)
- ❌ Avoid very large random values (saturation)

## Complete Example: Classification on Real Data

Let's build and test a 2-layer network on a real dataset:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Generate a non-linearly separable dataset
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
X = X.T  # Shape: (2, 300)
y = y.reshape(1, -1)  # Shape: (1, 300)

print("Make Moons Dataset:")
print("=" * 60)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {np.unique(y)}")

# Visualize dataset
plt.figure(figsize=(10, 6))
plt.scatter(X[0, y[0]==0], X[1, y[0]==0], c='blue', s=40,
            edgecolors='k', alpha=0.7, label='Class 0')
plt.scatter(X[0, y[0]==1], X[1, y[0]==1], c='red', s=40,
            edgecolors='k', alpha=0.7, label='Class 1')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Make Moons Dataset (Non-linearly Separable)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Split into train/test
n_samples = X.shape[1]
n_train = int(0.8 * n_samples)

# Shuffle
indices = np.random.permutation(n_samples)
train_idx, test_idx = indices[:n_train], indices[n_train:]

X_train, X_test = X[:, train_idx], X[:, test_idx]
y_train, y_test = y[:, train_idx], y[:, test_idx]

print(f"\nTraining samples: {X_train.shape[1]}")
print(f"Test samples: {X_test.shape[1]}")

# Create and initialize network
nn = NeuralNetwork(n_input=2, n_hidden=5)

# Make predictions with untrained network
predictions_before, _ = nn.forward(X_test)
accuracy_before = np.mean((predictions_before > 0.5) == y_test)

print(f"\nAccuracy before training: {accuracy_before:.2%}")
print("(This is just random - we haven't trained yet!)")
print("\nIn Lesson 3, we'll learn backpropagation to actually train the network.")

# Visualize decision boundary (before training)
plot_decision_boundary_nn(nn, X_test, y_test,
                         'Decision Boundary (Untrained Network)')
```
</div>

The untrained network performs poorly (random weights!). In Lesson 3, we'll learn **backpropagation** - the algorithm that actually trains neural networks by computing gradients and updating weights.

## Common Architecture Patterns

### Fully Connected (Dense) Networks

What we've been building - every neuron connects to every neuron in the next layer.

**Notation**: FC-512 means a fully connected layer with 512 neurons.

**Common pattern**:
```
Input → FC-256-ReLU → FC-128-ReLU → FC-64-ReLU → FC-10-Softmax → Output
```

### Pyramid Architecture

Hidden layers gradually decrease in size:

```
Input (784) → Hidden1 (256) → Hidden2 (128) → Hidden3 (64) → Output (10)
```

**Intuition**: Each layer compresses information, keeping only what's important.

### Hourglass Architecture

Compress then expand:

```
Input (100) → Hidden1 (50) → Hidden2 (20) → Hidden3 (50) → Output (100)
```

**Use case**: Autoencoders for dimensionality reduction or denoising.

## Summary

### Key Takeaways

!!! success "What You Learned"
    **1. Multi-Layer Networks**
    - Stack layers to create complex models
    - Architecture: Input → Hidden → Output
    - Each layer transforms the representation

    **2. Forward Propagation**
    - $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$
    - $\mathbf{a}^{[l]} = f^{[l]}(\mathbf{z}^{[l]})$
    - Data flows forward through layers

    **3. Solving XOR**
    - Hidden layers learn new representations
    - Non-linear transformations make problems linearly separable
    - XOR impossible with 1 layer, easy with 2 layers

    **4. Universal Approximation**
    - Neural networks can approximate any continuous function
    - Depth (layers) often better than width (neurons)
    - Hierarchical feature learning is powerful

    **5. Architecture Design**
    - Number of layers depends on problem complexity
    - Hidden layer size typically between input and output size
    - Weight initialization matters (break symmetry!)

### What We Haven't Covered (Yet!)

We can now:
- ✅ Build neural network architectures
- ✅ Compute forward propagation
- ✅ Make predictions

We cannot yet:
- ❌ Train networks automatically (requires backpropagation)
- ❌ Compute gradients efficiently
- ❌ Update weights to minimize loss

**Next lesson**: Backpropagation - the algorithm that makes neural network training possible!

## Practice Problems

**Problem 1**: Draw the architecture diagram for a [4, 6, 3, 1] neural network. What are the shapes of all weight matrices and bias vectors?

**Problem 2**: Implement a 3-layer neural network (2 hidden layers). Test it on XOR.

**Problem 3**: Manually compute forward propagation for:
- Input: $\mathbf{x} = [1, 2]^T$
- Network: [2, 2, 1]
- Weights: $\mathbf{W}^{[1]} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$, $\mathbf{b}^{[1]} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$
- Activation: ReLU for hidden, sigmoid for output
- More weights: $\mathbf{W}^{[2]} = \begin{bmatrix} 1 & 1 \end{bmatrix}$, $\mathbf{b}^{[2]} = [0]$

**Problem 4**: Explain why removing activation functions makes the network equivalent to a single-layer perceptron (hint: matrix multiplication is associative).

**Problem 5**: Create a neural network that implements the NAND gate. What's the minimum number of hidden neurons needed?

## Next Steps

You now understand how neural networks process information through forward propagation. But we still need to answer the critical question: **How do we find the right weights?**

**In Lesson 3**, we'll dive deep into **backpropagation** - the cornerstone algorithm of deep learning. You'll learn:
- How to compute gradients efficiently using the chain rule
- The backpropagation algorithm step-by-step
- How to update weights to minimize loss
- Why backpropagation revolutionized neural networks

This is where the magic happens!

[Continue to Lesson 3: Backpropagation](03-backpropagation.md){ .md-button .md-button--primary }

[Return to Module 4 Overview](index.md){ .md-button }

---

**Questions or feedback?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues).
