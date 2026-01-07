# Lesson 3: Backpropagation

## Introduction

You've built neural networks and computed forward propagation. But there's a crucial missing piece: **How do we train them?** How do we find the weights that actually solve our problem?

The answer is **backpropagation** - one of the most important algorithms in machine learning. Backpropagation efficiently computes gradients of the loss function with respect to every weight in the network, enabling gradient descent optimization.

**Why is backpropagation revolutionary?**

- **Efficient**: Computes all gradients in one backward pass
- **Elegant**: Uses the chain rule recursively
- **Universal**: Works for any network architecture
- **Foundational**: Powers all modern deep learning

In this lesson, we'll build backpropagation from first principles, understand the mathematics deeply, and implement it from scratch.

!!! quote "Historical Note"
    Backpropagation was popularized by Rumelhart, Hinton, and Williams in 1986, though variants existed earlier. It transformed neural networks from theoretical curiosities to practical tools, sparking the modern deep learning revolution.

## Learning Objectives

By the end of this lesson, you will:

- ✅ Understand the chain rule and computational graphs
- ✅ Derive backpropagation equations step-by-step
- ✅ Compute gradients layer by layer (backward pass)
- ✅ Implement backpropagation from scratch
- ✅ Understand gradient descent weight updates
- ✅ Debug gradient computation with gradient checking
- ✅ Train a neural network end-to-end

**Estimated Time**: 75 minutes

## The Big Picture: Training a Neural Network

### The Training Loop

Training a neural network follows this pattern:

```
1. Initialize weights randomly
2. Repeat until convergence:
   a. Forward Pass: Compute predictions
   b. Compute Loss: How wrong are predictions?
   c. Backward Pass: Compute gradients (backpropagation!)
   d. Update Weights: w ← w - η·∇w (gradient descent)
```

We already know steps 1, 2a, and 2b. **This lesson focuses on 2c and 2d**.

### What Are We Optimizing?

Given training data $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^m$, we want to find weights that minimize the **loss function**:

$$
J(\mathbf{W}, \mathbf{b}) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(hat{y}^{(i)}, y^{(i)})
$$

Where $\mathcal{L}$ is a loss function (e.g., binary cross-entropy for classification).

**Gradient descent** requires computing:

$$
\frac{\partial J}{\partial \mathbf{W}^{[l]}}, \quad \frac{\partial J}{\partial \mathbf{b}^{[l]}}
$$

For **every layer** $l$. This seems computationally expensive! Backpropagation makes it efficient.

## The Chain Rule: Foundation of Backpropagation

### Single Variable Chain Rule

Recall from calculus: if $y = f(g(x))$, then:

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}
$$

**Example**: $y = (2x + 3)^2$

Let $g = 2x + 3$, so $y = g^2$:

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = 2g \cdot 2 = 4(2x + 3)
$$

### Multivariable Chain Rule

For a function $z = f(x, y)$ where $x = x(t)$ and $y = y(t)$:

$$
\frac{dz}{dt} = \frac{\partial z}{\partial x} \frac{dx}{dt} + \frac{\partial z}{\partial y} \frac{dy}{dt}
$$

**Key insight**: The gradient flows back through all paths!

### Computational Graphs

We can visualize function composition as a graph:

```
Forward Pass:        x → [×2] → [+3] → [²] → y
Backward Pass:       x ← [×2] ← [+3] ← [²] ← ∂y/∂y = 1
```

Each node computes:
- **Forward**: Output given input
- **Backward**: Gradient with respect to its input

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Example: Compute y = (2x + 3)^2 and its derivative
def forward_example(x):
    """Forward pass through computation graph."""
    # Node 1: a = 2x
    a = 2 * x
    # Node 2: b = a + 3
    b = a + 3
    # Node 3: y = b^2
    y = b ** 2
    return y, (a, b)  # Return output and intermediate values

def backward_example(x, y, cache):
    """Backward pass: compute dy/dx."""
    a, b = cache

    # Start with dy/dy = 1
    dy_dy = 1

    # Node 3: y = b^2, so dy/db = 2b
    dy_db = dy_dy * (2 * b)

    # Node 2: b = a + 3, so db/da = 1
    dy_da = dy_db * 1

    # Node 1: a = 2x, so da/dx = 2
    dy_dx = dy_da * 2

    return dy_dx

# Test
x = 5.0
y, cache = forward_example(x)
gradient = backward_example(x, y, cache)

print("Chain Rule Example:")
print("=" * 60)
print(f"x = {x}")
print(f"y = (2x + 3)^2 = {y}")
print(f"dy/dx = {gradient}")
print(f"\nVerification: dy/dx = 4(2x + 3) = {4 * (2*x + 3)}")
print(f"Match: {np.isclose(gradient, 4 * (2*x + 3))}")
```
</div>

**Key Insight**: We can compute the gradient by storing intermediate values during the forward pass, then multiplying local gradients backward through the graph!

## Backpropagation for a 2-Layer Network

Let's derive backpropagation for a simple network step-by-step.

### Network Setup

**Architecture**: [2, 3, 1]
- Input: 2 features ($x_1, x_2$)
- Hidden: 3 neurons (ReLU)
- Output: 1 neuron (sigmoid)

**Forward Pass Equations**:

$$
\begin{align}
\mathbf{z}^{[1]} &= \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]} \quad &\text{(pre-activation, hidden)} \\
\mathbf{a}^{[1]} &= \text{ReLU}(\mathbf{z}^{[1]}) \quad &\text{(activation, hidden)} \\
\mathbf{z}^{[2]} &= \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]} \quad &\text{(pre-activation, output)} \\
\mathbf{a}^{[2]} &= \sigma(\mathbf{z}^{[2]}) \quad &\text{(activation, output)} \\
\hat{y} &= \mathbf{a}^{[2]} \quad &\text{(prediction)}
\end{align}
$$

**Loss Function** (binary cross-entropy):

$$
\mathcal{L}(\hat{y}, y) = -y \log(\hat{y}) - (1-y) \log(1-\hat{y})
$$

### Goal: Compute Gradients

We need:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[2]}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[2]}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}}, \quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[1]}}
$$

### Step 1: Output Layer Gradients

**Compute** $\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[2]}}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[2]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}
$$

For binary cross-entropy with sigmoid, this simplifies beautifully:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[2]}} = \frac{\mathbf{a}^{[2]} - y}{\mathbf{a}^{[2]}(1 - \mathbf{a}^{[2]})}
$$

**Compute** $\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}}$:

Using chain rule: $\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[2]}} \cdot \frac{\partial \mathbf{a}^{[2]}}{\partial \mathbf{z}^{[2]}}$

Since $\mathbf{a}^{[2]} = \sigma(\mathbf{z}^{[2]})$ and $\sigma'(z) = \sigma(z)(1-\sigma(z))$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}} = \frac{\mathbf{a}^{[2]} - y}{\mathbf{a}^{[2]}(1 - \mathbf{a}^{[2]})} \cdot \mathbf{a}^{[2]}(1 - \mathbf{a}^{[2]}) = \mathbf{a}^{[2]} - y
$$

**Amazing simplification!** Let's call this $\mathbf{d}\mathbf{z}^{[2]} = \mathbf{a}^{[2]} - y$.

**Compute** $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[2]}}$:

Since $\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[2]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[2]}} \cdot \frac{\partial \mathbf{z}^{[2]}}{\partial \mathbf{W}^{[2]}} = \mathbf{d}\mathbf{z}^{[2]} \cdot (\mathbf{a}^{[1]})^T
$$

**Compute** $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[2]}}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[2]}} = \mathbf{d}\mathbf{z}^{[2]}
$$

### Step 2: Hidden Layer Gradients

**Compute** $\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}}$:

Chain rule through $\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} = (\mathbf{W}^{[2]})^T \mathbf{d}\mathbf{z}^{[2]}
$$

**Compute** $\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}}$:

Since $\mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]})$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} \odot \text{ReLU}'(\mathbf{z}^{[1]})
$$

Where $\text{ReLU}'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}$

So: $\mathbf{d}\mathbf{z}^{[1]} = (\mathbf{W}^{[2]})^T \mathbf{d}\mathbf{z}^{[2]} \odot \text{ReLU}'(\mathbf{z}^{[1]})$

**Compute** $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}}$ and $\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[1]}}$:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} = \mathbf{d}\mathbf{z}^{[1]} \cdot \mathbf{x}^T
$$

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[1]}} = \mathbf{d}\mathbf{z}^{[1]}
$$

### Summary: Backpropagation Equations

For a 2-layer network:

**Layer 2 (output)**:
$$
\begin{align}
\mathbf{d}\mathbf{z}^{[2]} &= \mathbf{a}^{[2]} - y \\
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[2]}} &= \mathbf{d}\mathbf{z}^{[2]} \cdot (\mathbf{a}^{[1]})^T \\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[2]}} &= \mathbf{d}\mathbf{z}^{[2]}}
\end{align}
$$

**Layer 1 (hidden)**:
$$
\begin{align}
\mathbf{d}\mathbf{z}^{[1]} &= (\mathbf{W}^{[2]})^T \mathbf{d}\mathbf{z}^{[2]} \odot \text{ReLU}'(\mathbf{z}^{[1]}) \\
\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[1]}} &= \mathbf{d}\mathbf{z}^{[1]} \cdot \mathbf{x}^T \\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[1]}} &= \mathbf{d}\mathbf{z}^{[1]}}
\end{align}
$$

**Pattern**: Gradients flow **backward** through the network!

## Vectorized Backpropagation (Multiple Examples)

For a batch of $m$ examples, we compute the **average** gradient:

$$
\frac{\partial J}{\partial \mathbf{W}^{[l]}} = \frac{1}{m} \mathbf{d}\mathbf{Z}^{[l]} \cdot (\mathbf{A}^{[l-1]})^T
$$

$$
\frac{\partial J}{\partial \mathbf{b}^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{d}\mathbf{z}^{[l](i)}
$$

Where:
- $\mathbf{Z}^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$: pre-activations for all $m$ examples
- $\mathbf{d}\mathbf{Z}^{[l]} \in \mathbb{R}^{n^{[l]} \times m}$: gradients for all examples

## Implementing Backpropagation from Scratch

Let's implement backpropagation step-by-step:

<div class="python-interactive" markdown="1">
```python
import numpy as np

def sigmoid(Z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    """ReLU activation function."""
    return np.maximum(0, Z)

def relu_derivative(Z):
    """Derivative of ReLU."""
    return (Z > 0).astype(float)

def binary_cross_entropy(A, Y):
    """
    Binary cross-entropy loss.

    Parameters:
    -----------
    A : ndarray, shape (1, m)
        Predictions
    Y : ndarray, shape (1, m)
        True labels
    """
    m = Y.shape[1]
    epsilon = 1e-8  # Prevent log(0)
    loss = -np.mean(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
    return loss

def forward_propagation(X, W1, b1, W2, b2):
    """
    Forward propagation through 2-layer network.

    Returns:
    --------
    A2 : ndarray
        Output predictions
    cache : dict
        Values needed for backpropagation
    """
    # Layer 1
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    # Layer 2
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {
        'X': X,
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }

    return A2, cache

def backward_propagation(Y, cache, W1, b1, W2, b2):
    """
    Backward propagation to compute gradients.

    Parameters:
    -----------
    Y : ndarray, shape (1, m)
        True labels
    cache : dict
        Values from forward propagation
    W1, b1, W2, b2 : ndarrays
        Current weights and biases

    Returns:
    --------
    grads : dict
        Gradients for all parameters
    """
    m = Y.shape[1]
    X = cache['X']
    Z1 = cache['Z1']
    A1 = cache['A1']
    A2 = cache['A2']

    # Layer 2 gradients (output layer)
    dZ2 = A2 - Y  # Simplified for binary cross-entropy + sigmoid
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # Layer 1 gradients (hidden layer)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    return grads

# Example usage
print("Backpropagation Implementation")
print("=" * 60)

# Create toy dataset
np.random.seed(42)
X = np.random.randn(2, 4)  # 2 features, 4 examples
Y = np.array([[0, 1, 1, 0]])  # Labels

# Initialize weights
W1 = np.random.randn(3, 2) * 0.01
b1 = np.zeros((3, 1))
W2 = np.random.randn(1, 3) * 0.01
b2 = np.zeros((1, 1))

# Forward pass
A2, cache = forward_propagation(X, W1, b1, W2, b2)
loss = binary_cross_entropy(A2, Y)

print(f"Predictions: {A2}")
print(f"True labels: {Y}")
print(f"Loss: {loss:.4f}")

# Backward pass
grads = backward_propagation(Y, cache, W1, b1, W2, b2)

print("\nComputed Gradients:")
print("-" * 60)
for param, grad in grads.items():
    print(f"{param} shape: {grad.shape}")
    print(f"{param}:\n{grad}\n")
```
</div>

**Success!** We've computed all gradients. Now we can use them to update weights.

## Gradient Descent: Updating Weights

With gradients computed, we update weights to minimize the loss:

$$
\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \alpha \frac{\partial J}{\partial \mathbf{W}^{[l]}}
$$

$$
\mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \alpha \frac{\partial J}{\partial \mathbf{b}^{[l]}}
$$

Where $\alpha$ is the **learning rate** (step size).

<div class="python-interactive" markdown="1">
```python
import numpy as np

def update_parameters(W1, b1, W2, b2, grads, learning_rate):
    """
    Update parameters using gradient descent.

    Parameters:
    -----------
    W1, b1, W2, b2 : ndarrays
        Current parameters
    grads : dict
        Gradients from backpropagation
    learning_rate : float
        Learning rate (alpha)

    Returns:
    --------
    Updated parameters
    """
    W1 = W1 - learning_rate * grads['dW1']
    b1 = b1 - learning_rate * grads['db1']
    W2 = W2 - learning_rate * grads['dW2']
    b2 = b2 - learning_rate * grads['db2']

    return W1, b1, W2, b2

# Update weights
learning_rate = 0.1
W1_new, b1_new, W2_new, b2_new = update_parameters(
    W1, b1, W2, b2, grads, learning_rate
)

print("Parameter Updates:")
print("=" * 60)
print(f"W1 change: {np.linalg.norm(W1_new - W1):.6f}")
print(f"W2 change: {np.linalg.norm(W2_new - W2):.6f}")

# Check if loss decreased
A2_new, cache_new = forward_propagation(X, W1_new, b1_new, W2_new, b2_new)
loss_new = binary_cross_entropy(A2_new, Y)

print(f"\nLoss before update: {loss:.6f}")
print(f"Loss after update:  {loss_new:.6f}")
print(f"Improvement: {loss - loss_new:.6f}")

if loss_new < loss:
    print("✓ Loss decreased! Gradient descent is working.")
else:
    print("✗ Loss increased. Try smaller learning rate.")
```
</div>

**Key Observation**: After one gradient descent step, the loss should decrease (if learning rate is appropriate). This confirms backpropagation is working correctly!

## Training Loop: Putting It All Together

Now let's train a network for multiple iterations:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

def train_network(X, Y, n_hidden, learning_rate=0.1, epochs=1000, print_every=100):
    """
    Train a 2-layer neural network.

    Parameters:
    -----------
    X : ndarray, shape (n_features, m)
        Training data
    Y : ndarray, shape (1, m)
        Labels
    n_hidden : int
        Number of hidden neurons
    learning_rate : float
        Learning rate
    epochs : int
        Number of training iterations
    print_every : int
        Print loss every N epochs

    Returns:
    --------
    parameters : dict
        Trained weights and biases
    losses : list
        Loss at each epoch
    """
    # Initialize parameters
    n_input = X.shape[0]
    W1 = np.random.randn(n_hidden, n_input) * 0.01
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(1, n_hidden) * 0.01
    b2 = np.zeros((1, 1))

    losses = []

    for epoch in range(epochs):
        # Forward propagation
        A2, cache = forward_propagation(X, W1, b1, W2, b2)

        # Compute loss
        loss = binary_cross_entropy(A2, Y)
        losses.append(loss)

        # Backward propagation
        grads = backward_propagation(Y, cache, W1, b1, W2, b2)

        # Update parameters
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, grads, learning_rate)

        # Print progress
        if (epoch + 1) % print_every == 0:
            accuracy = np.mean((A2 > 0.5) == Y)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parameters, losses

# Train on XOR problem
X_xor = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]])
y_xor = np.array([[0, 1, 1, 0]])

print("Training on XOR Problem:")
print("=" * 60)
parameters, losses = train_network(X_xor, y_xor, n_hidden=4,
                                  learning_rate=0.5, epochs=2000)

# Test trained network
A2_trained, _ = forward_propagation(X_xor, parameters['W1'], parameters['b1'],
                                   parameters['W2'], parameters['b2'])

print("\nFinal Results:")
print("-" * 60)
print(f"{'x1':<5} {'x2':<5} {'True':<8} {'Predicted':<12} {'Correct'}")
print("-" * 60)
for i in range(4):
    x1, x2 = X_xor[:, i]
    true_y = y_xor[0, i]
    pred_y = A2_trained[0, i]
    correct = '✓' if (pred_y > 0.5) == true_y else '✗'
    print(f"{x1:<5.0f} {x2:<5.0f} {true_y:<8.0f} {pred_y:<12.4f} {correct}")

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Learning Curve: XOR Problem', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()

print("\n✓ Successfully trained network to solve XOR using backpropagation!")
```
</div>

**Amazing!** The network learned to solve XOR from scratch using only backpropagation and gradient descent.

## Gradient Checking: Verifying Backpropagation

Backpropagation is complex - how do we know our implementation is correct? **Gradient checking** compares analytical gradients (from backprop) with numerical gradients (approximated using finite differences).

### Numerical Gradient Approximation

The derivative at a point can be approximated:

$$
\frac{\partial J}{\partial \theta} \approx \frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2\epsilon}
$$

This is slow (requires 2 forward passes per parameter!) but very accurate.

<div class="python-interactive" markdown="1">
```python
import numpy as np

def numerical_gradient(X, Y, parameters, param_name, epsilon=1e-7):
    """
    Compute numerical gradient for a specific parameter.

    Parameters:
    -----------
    X, Y : ndarrays
        Data and labels
    parameters : dict
        Current parameters
    param_name : str
        Name of parameter to check (e.g., 'W1')
    epsilon : float
        Small value for finite difference

    Returns:
    --------
    grad_numerical : ndarray
        Numerical gradient
    """
    # Get parameter
    param = parameters[param_name]
    grad_numerical = np.zeros_like(param)

    # Iterate over all elements
    it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        old_value = param[idx]

        # Compute J(theta + epsilon)
        param[idx] = old_value + epsilon
        W1, b1, W2, b2 = parameters['W1'], parameters['b1'], parameters['W2'], parameters['b2']
        A2_plus, _ = forward_propagation(X, W1, b1, W2, b2)
        loss_plus = binary_cross_entropy(A2_plus, Y)

        # Compute J(theta - epsilon)
        param[idx] = old_value - epsilon
        A2_minus, _ = forward_propagation(X, W1, b1, W2, b2)
        loss_minus = binary_cross_entropy(A2_minus, Y)

        # Numerical gradient
        grad_numerical[idx] = (loss_plus - loss_minus) / (2 * epsilon)

        # Restore value
        param[idx] = old_value
        it.iternext()

    return grad_numerical

def gradient_check(X, Y, parameters, grads, epsilon=1e-7):
    """
    Check if backpropagation gradients are correct.

    Returns:
    --------
    difference : float
        Relative difference between gradients
    """
    # Convert gradients to vectors
    params_vector = []
    grads_vector = []

    for param_name in ['W1', 'b1', 'W2', 'b2']:
        params_vector.append(parameters[param_name].ravel())
        grad_name = 'd' + param_name
        grads_vector.append(grads[grad_name].ravel())

    params_vector = np.concatenate(params_vector)
    grads_vector = np.concatenate(grads_vector)

    # Compute numerical gradient
    num_grads_vector = []
    for param_name in ['W1', 'b1', 'W2', 'b2']:
        num_grad = numerical_gradient(X, Y, parameters, param_name, epsilon)
        num_grads_vector.append(num_grad.ravel())

    num_grads_vector = np.concatenate(num_grads_vector)

    # Compute relative difference
    numerator = np.linalg.norm(grads_vector - num_grads_vector)
    denominator = np.linalg.norm(grads_vector) + np.linalg.norm(num_grads_vector)
    difference = numerator / denominator

    return difference

# Perform gradient check
print("Gradient Checking:")
print("=" * 60)
print("Comparing backpropagation gradients with numerical gradients...")
print("(This may take a few seconds)\n")

difference = gradient_check(X_xor, y_xor, parameters, grads)

print(f"Relative difference: {difference:.2e}")
print()

if difference < 1e-7:
    print("✓ Gradient check passed! Backpropagation is correct.")
elif difference < 1e-5:
    print("⚠ Gradient check OK (small difference, likely due to numerical precision).")
else:
    print("✗ Gradient check failed! Check backpropagation implementation.")

print("\nRule of thumb:")
print("  < 1e-7: Excellent")
print("  < 1e-5: Good")
print("  < 1e-3: OK (check for bugs)")
print("  > 1e-3: Likely incorrect")
```
</div>

!!! warning "Gradient Checking in Practice"
    - **Use gradient checking during development** to verify backpropagation
    - **Disable it during training** (too slow!)
    - If gradient check fails, debug layer-by-layer
    - Common bugs: dimension mismatches, wrong derivatives, missing transposes

## General Backpropagation Algorithm

For an $L$-layer network, backpropagation follows this pattern:

### Forward Pass

For $l = 1, 2, \ldots, L$:

$$
\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})
$$

Where $\mathbf{A}^{[0]} = \mathbf{X}$ and $g^{[l]}$ is the activation function.

### Backward Pass

For $l = L, L-1, \ldots, 1$:

**Compute gradient w.r.t. pre-activation**:

$$
\mathbf{d}\mathbf{Z}^{[l]} = \begin{cases}
\mathbf{A}^{[L]} - \mathbf{Y} & \text{if } l = L \text{ (output layer)} \\
(\mathbf{W}^{[l+1]})^T \mathbf{d}\mathbf{Z}^{[l+1]} \odot g'^{[l]}(\mathbf{Z}^{[l]}) & \text{if } l < L \text{ (hidden layer)}
\end{cases}
$$

**Compute parameter gradients**:

$$
\frac{\partial J}{\partial \mathbf{W}^{[l]}} = \frac{1}{m} \mathbf{d}\mathbf{Z}^{[l]} \cdot (\mathbf{A}^{[l-1]})^T
$$

$$
\frac{\partial J}{\partial \mathbf{b}^{[l]}} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{d}\mathbf{Z}^{[l](i)}
$$

**Key pattern**: Gradients flow backward, and we reuse intermediate values from the forward pass!

## Activation Function Derivatives

Quick reference for common activation derivatives:

| Activation | $g(z)$ | $g'(z)$ |
|------------|--------|---------|
| **Sigmoid** | $\frac{1}{1+e^{-z}}$ | $g(z)(1-g(z))$ |
| **Tanh** | $\tanh(z)$ | $1 - g(z)^2$ |
| **ReLU** | $\max(0, z)$ | $\begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}$ |
| **Leaky ReLU** | $\max(0.01z, z)$ | $\begin{cases} 1 & z > 0 \\ 0.01 & z \leq 0 \end{cases}$ |

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Activation function derivatives
def sigmoid_derivative(Z):
    s = sigmoid(Z)
    return s * (1 - s)

def tanh_derivative(Z):
    return 1 - np.tanh(Z)**2

def relu_derivative(Z):
    return (Z > 0).astype(float)

def leaky_relu_derivative(Z, alpha=0.01):
    return np.where(Z > 0, 1, alpha)

# Test derivatives
Z_test = np.array([-2, -1, 0, 1, 2])

print("Activation Function Derivatives:")
print("=" * 60)
print(f"{'Z':<10} {'σ\'(Z)':<12} {'tanh\'(Z)':<12} {'ReLU\'(Z)':<12}")
print("-" * 60)
for z in Z_test:
    sig_deriv = sigmoid_derivative(z)
    tanh_deriv = tanh_derivative(z)
    relu_deriv = relu_derivative(z)
    print(f"{z:<10.1f} {sig_deriv:<12.4f} {tanh_deriv:<12.4f} {relu_deriv:<12.0f}")
```
</div>

## Common Backpropagation Bugs and How to Fix Them

### 1. Dimension Mismatches

**Problem**: Matrix dimensions don't align.

**Fix**: Always check shapes! Use assertions:

```python
assert dW1.shape == W1.shape, f"Shape mismatch: {dW1.shape} != {W1.shape}"
```

### 2. Wrong Transpose

**Problem**: Forgot to transpose in gradient computation.

**Fix**: Remember: $\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \mathbf{d}\mathbf{Z}^{[l]} \cdot (\mathbf{A}^{[l-1]})^T$

### 3. Forgot Element-wise Multiplication

**Problem**: Used matrix multiplication instead of element-wise (`*` vs `@`).

**Fix**: For activation derivatives, use `*` or `np.multiply()` (element-wise).

### 4. Didn't Average Over Batch

**Problem**: Forgot to divide by $m$ (number of examples).

**Fix**: Always include `(1/m)` in gradient computations.

### 5. Wrong Activation Derivative

**Problem**: Used wrong derivative formula.

**Fix**: Double-check derivative formulas. Use gradient checking!

## Visualizing Backpropagation

Let's visualize how gradients flow backward:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Train network and track gradient magnitudes
def track_gradients(X, Y, epochs=100):
    """Train network and record gradient magnitudes."""
    n_hidden = 4
    n_input = X.shape[0]

    # Initialize
    W1 = np.random.randn(n_hidden, n_input) * 0.01
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(1, n_hidden) * 0.01
    b2 = np.zeros((1, 1))

    grad_W1_norms = []
    grad_W2_norms = []

    for epoch in range(epochs):
        # Forward + backward
        A2, cache = forward_propagation(X, W1, b1, W2, b2)
        grads = backward_propagation(Y, cache, W1, b1, W2, b2)

        # Track gradient magnitudes
        grad_W1_norms.append(np.linalg.norm(grads['dW1']))
        grad_W2_norms.append(np.linalg.norm(grads['dW2']))

        # Update
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, grads, learning_rate=0.5)

    return grad_W1_norms, grad_W2_norms

# Track gradients during training
grad_W1_norms, grad_W2_norms = track_gradients(X_xor, y_xor, epochs=200)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(grad_W1_norms, label='Layer 1 (Hidden)', linewidth=2)
ax1.plot(grad_W2_norms, label='Layer 2 (Output)', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Gradient Magnitude', fontsize=12)
ax1.set_title('Gradient Magnitudes During Training', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.semilogy(grad_W1_norms, label='Layer 1 (Hidden)', linewidth=2)
ax2.semilogy(grad_W2_norms, label='Layer 2 (Output)', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Gradient Magnitude (log scale)', fontsize=12)
ax2.set_title('Gradient Magnitudes (Log Scale)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Gradient Flow Observations:")
print("=" * 60)
print("• Gradients are larger in early training (network is far from optimum)")
print("• Gradients decrease as network converges")
print("• Layer 2 gradients often larger (closer to loss function)")
print("• If gradients vanish (→0) or explode (→∞), training fails!")
```
</div>

## Mini-Batch Gradient Descent

For large datasets, computing gradients over all examples is slow. **Mini-batch gradient descent** processes small batches:

```python
for epoch in range(num_epochs):
    # Shuffle data
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]

    # Process mini-batches
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[:, i:i+batch_size]
        Y_batch = Y_shuffled[:, i:i+batch_size]

        # Forward + backward + update on batch
        A2, cache = forward_propagation(X_batch, W1, b1, W2, b2)
        grads = backward_propagation(Y_batch, cache, W1, b1, W2, b2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, grads, learning_rate)
```

**Batch size trade-offs**:
- **Small batches (16-32)**: Noisy gradients, faster iterations, better generalization
- **Large batches (256-1024)**: Smoother gradients, slower iterations, may overfit
- **Full batch**: Most accurate gradient, very slow for large datasets

## Summary

### Key Takeaways

!!! success "What You Learned"
    **1. Backpropagation Algorithm**
    - Efficiently computes gradients using the chain rule
    - Flows backward through the network
    - Reuses forward pass intermediate values

    **2. Gradient Computation**
    - $\mathbf{d}\mathbf{Z}^{[L]} = \mathbf{A}^{[L]} - \mathbf{Y}$ (output layer)
    - $\mathbf{d}\mathbf{Z}^{[l]} = (\mathbf{W}^{[l+1]})^T \mathbf{d}\mathbf{Z}^{[l+1]} \odot g'^{[l]}(\mathbf{Z}^{[l]})$ (hidden layers)
    - $\frac{\partial J}{\partial \mathbf{W}^{[l]}} = \frac{1}{m} \mathbf{d}\mathbf{Z}^{[l]} \cdot (\mathbf{A}^{[l-1]})^T$

    **3. Training Process**
    - Forward pass: compute predictions
    - Compute loss: measure error
    - Backward pass: compute gradients (backpropagation)
    - Update weights: gradient descent

    **4. Verification**
    - Use gradient checking to verify backpropagation
    - Compare with numerical gradients
    - Disable during actual training (too slow)

    **5. Practical Tips**
    - Always check dimensions
    - Use mini-batches for efficiency
    - Monitor gradient magnitudes
    - Start with small learning rates

### Why Backpropagation Matters

Backpropagation is **THE** algorithm that makes deep learning possible:

- **Scalable**: Works efficiently for networks with millions of parameters
- **Automatic**: Modern frameworks (PyTorch, TensorFlow) do it for you
- **Universal**: Same algorithm for any architecture
- **Foundation**: Understanding backprop helps debug and design better models

## Practice Problems

**Problem 1**: Derive the backpropagation equations for a network with tanh activation in the hidden layer instead of ReLU.

**Problem 2**: Implement backpropagation for a 3-layer network (2 hidden layers).

**Problem 3**: What happens to gradients if you use very large weight initializations? Very small? Try it!

**Problem 4**: Implement momentum gradient descent:
$$\mathbf{v}^{[l]} = \beta \mathbf{v}^{[l]} + (1-\beta) \mathbf{d}\mathbf{W}^{[l]}$$
$$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \alpha \mathbf{v}^{[l]}$$

**Problem 5**: Visualize the computational graph for your network. Label each node with its forward and backward computation.

## Next Steps

You now understand the core algorithm of neural network training! But implementing everything from scratch for complex problems is tedious.

**In Lesson 4**, we'll build a **production-quality NumPy implementation** with:
- Modular design (Layer classes, Activation classes)
- Complete training pipeline
- Multiple datasets (MNIST digit classification!)
- Proper evaluation and visualization
- Best practices and debugging

Then in Lessons 5-6, we'll learn **PyTorch** - the modern framework that handles backpropagation automatically!

[Continue to Lesson 4: NumPy Implementation](04-numpy-implementation.md){ .md-button .md-button--primary }

[Return to Module 4 Overview](index.md){ .md-button }

---

**Questions or feedback?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues).
