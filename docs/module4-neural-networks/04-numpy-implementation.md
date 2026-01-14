# Lesson 4: Complete NumPy Implementation

## Introduction

In the previous lessons, we learned the theory: perceptrons, feedforward networks, and backpropagation. Now comes the exciting part—building a **complete neural network library from scratch** using only NumPy!

This lesson represents the culmination of everything we've learned so far. We'll transform mathematical equations into working code, building a production-quality neural network framework that mirrors how modern libraries like PyTorch and TensorFlow work internally.

### Why Build from Scratch?

You might wonder: "Why build a neural network from scratch when libraries like PyTorch already exist?" Here's why this exercise is invaluable:

1. **Deep Understanding**: You'll understand exactly what happens when you call `model.fit()` or `loss.backward()`
2. **Better Debugging**: When your PyTorch model doesn't converge, you'll know where to look
3. **Framework Mastery**: You'll recognize patterns across all ML frameworks
4. **Interview Prep**: This is a common interview question for ML engineering roles
5. **Customization**: Sometimes you need to implement custom layers or loss functions
6. **Appreciation**: You'll appreciate the elegance and efficiency of modern frameworks

### What We'll Build

Our implementation will be:

- **Modular**: Clean class-based design (Layer, Activation, Loss classes)
- **Flexible**: Supports arbitrary network architectures
- **Complete**: Full training pipeline with mini-batch gradient descent
- **Practical**: Test on real problems (XOR, spirals, MNIST)
- **Professional**: Well-documented, debuggable, extensible
- **Educational**: Every line explained in detail

By the end, you'll have code you can actually use for simple problems, and more importantly, deep insight into how neural networks work internally.

## Learning Objectives

By the end of this lesson, you will:

- ✅ Design a modular, object-oriented neural network architecture
- ✅ Implement Layer classes with forward and backward propagation
- ✅ Implement Activation functions (ReLU, Sigmoid, Tanh, Softmax)
- ✅ Implement Loss functions (Binary/Categorical Cross-Entropy, MSE)
- ✅ Build a complete NeuralNetwork class with training loop
- ✅ Implement mini-batch gradient descent
- ✅ Train networks on classic problems (XOR, spirals)
- ✅ Understand weight initialization strategies (He, Xavier)
- ✅ Visualize learning curves and decision boundaries
- ✅ Debug neural networks systematically
- ✅ Apply best practices for network design

**Estimated Time**: 120-150 minutes

!!! warning "Prerequisites"
    This lesson assumes you understand:

    - Perceptrons and their limitations (Lesson 1)
    - Feedforward network architecture (Lesson 2)
    - Backpropagation algorithm (Lesson 3)
    - Matrix multiplication and vectorization
    - NumPy basics

    If any of these are unclear, review the previous lessons first!

## Design Principles

Before writing any code, we need to design our architecture thoughtfully. Good software design makes the difference between code that works once and code that's maintainable, extensible, and professional.

### Object-Oriented Architecture

We'll use a **class-based design** that mirrors modern frameworks like PyTorch and Keras. This approach provides:

1. **Encapsulation**: Each component manages its own state
2. **Abstraction**: Clean interfaces hide implementation details
3. **Modularity**: Components can be developed and tested independently
4. **Extensibility**: Easy to add new layer types, activations, or optimizers

Here's our high-level architecture:

```
NeuralNetwork
├── Layers (list of Layer objects)
│   ├── DenseLayer (fully connected)
│   ├── (extensible to Conv2D, LSTM, Dropout, etc.)
│   └── Each layer has:
│       ├── Parameters (W, b)
│       ├── Gradients (dW, db)
│       ├── Cached values (for backprop)
│       └── Activation function
│
├── Loss Function
│   ├── BinaryCrossEntropy (binary classification)
│   ├── CategoricalCrossEntropy (multi-class)
│   └── MeanSquaredError (regression)
│
└── Training State
    ├── History (loss, accuracy over epochs)
    └── Hyperparameters (learning rate, batch size)
```

**Why This Design?**

- **Separation of Concerns**: Layers handle transformation, activations handle non-linearity, loss computes error
- **Single Responsibility**: Each class has one job
- **Easy Testing**: Test each component independently
- **Extensibility**: Add new components without modifying existing code
- **Readability**: Clear structure that matches the mathematics

### Interface Design

Each component implements a **standard interface**. This is the secret to modularity—as long as components respect the interface, they're interchangeable.

#### Layer Interface

Every layer must implement:

```python
class Layer:
    def forward(self, A_prev):
        """
        Forward propagation.

        Parameters:
        -----------
        A_prev : ndarray, shape (n_inputs, m)
            Activations from previous layer (or input data)

        Returns:
        --------
        A : ndarray, shape (n_outputs, m)
            Activations of this layer
        """
        pass

    def backward(self, dA):
        """
        Backward propagation.

        Parameters:
        -----------
        dA : ndarray, shape (n_outputs, m)
            Gradient of loss w.r.t. this layer's activations

        Returns:
        --------
        dA_prev : ndarray, shape (n_inputs, m)
            Gradient of loss w.r.t. previous layer's activations
        """
        pass

    def update(self, learning_rate):
        """
        Update parameters using computed gradients.

        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        """
        pass
```

**Key Insights**:

- `forward` caches intermediate values needed for `backward`
- `backward` computes gradients and passes them to the previous layer
- `update` modifies parameters based on gradients

#### Activation Interface

Activations are stateless functions (no parameters to learn):

```python
class Activation:
    def forward(self, Z):
        """
        Apply activation function element-wise.

        Parameters:
        -----------
        Z : ndarray, shape (n, m)
            Pre-activation values

        Returns:
        --------
        A : ndarray, shape (n, m)
            Post-activation values
        """
        pass

    def backward(self, dA):
        """
        Compute gradient of activation function.

        Parameters:
        -----------
        dA : ndarray, shape (n, m)
            Gradient w.r.t. activations

        Returns:
        --------
        dZ : ndarray, shape (n, m)
            Gradient w.r.t. pre-activations
        """
        pass
```

**Key Insight**: Activation derivatives depend only on cached forward pass values.

#### Loss Interface

Loss functions measure prediction error:

```python
class Loss:
    def compute(self, Y_pred, Y_true):
        """
        Calculate loss value.

        Parameters:
        -----------
        Y_pred : ndarray, shape (n_classes, m)
            Predicted outputs
        Y_true : ndarray, shape (n_classes, m)
            True labels (one-hot encoded)

        Returns:
        --------
        loss : float
            Average loss across all samples
        """
        pass

    def gradient(self, Y_pred, Y_true):
        """
        Compute gradient of loss w.r.t. predictions.

        Parameters:
        -----------
        Y_pred : ndarray
            Predicted outputs
        Y_true : ndarray
            True labels

        Returns:
        --------
        dL : ndarray, same shape as Y_pred
            Gradient of loss w.r.t. predictions
        """
        pass
```

**Key Insight**: Combined loss+activation gradients often simplify (e.g., softmax+cross-entropy).

### Data Flow: The Big Picture

Let's trace data through a complete forward-backward pass:

```
FORWARD PASS (left to right):
─────────────────────────────
X → [Layer 1] → A1 → [Layer 2] → A2 → ... → Y_pred
    (W1, b1)        (W2, b2)                    ↓
                                            Loss(Y_pred, Y_true)
                                                ↓
BACKWARD PASS (right to left):                Loss value
─────────────────────────────                  ↓
    ← dW1, db1  ←  dA1  ← dW2, db2  ← dA2  ←  dL
                                                ↑
UPDATE STEP:                              Compute ∂L/∂Y_pred
────────────
W1 := W1 - α·dW1
b1 := b1 - α·db1
W2 := W2 - α·dW2
b2 := b2 - α·db2
```

**Forward Pass**: Data flows forward, each layer computes outputs
**Backward Pass**: Gradients flow backward, each layer computes parameter gradients
**Update Step**: Parameters adjusted to reduce loss

This pattern repeats for every training iteration until the network converges.

### Why Modular Design Matters

Consider what happens when you want to:

**Add a new activation function** (e.g., Swish):
```python
class Swish(Activation):
    def forward(self, Z):
        self.Z = Z
        return Z * sigmoid(Z)

    def backward(self, dA):
        sig = sigmoid(self.Z)
        return dA * (sig + self.Z * sig * (1 - sig))
```

That's it! No need to modify any existing code. The layer will automatically use it correctly.

**Add a new layer type** (e.g., Dropout):
```python
class Dropout(Layer):
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob

    def forward(self, A_prev):
        self.mask = (np.random.rand(*A_prev.shape) < self.keep_prob)
        return A_prev * self.mask / self.keep_prob

    def backward(self, dA):
        return dA * self.mask / self.keep_prob

    def update(self, learning_rate):
        pass  # No parameters to update
```

Drop it into your network and it just works!

This is the power of good design—it makes complex systems manageable and extensible.

## Building Blocks: Activation Functions

Activation functions are the **source of non-linearity** in neural networks. Without them, stacking layers would be pointless—multiple linear transformations compose to just another linear transformation.

### Why Activation Functions Matter

Recall from Lesson 1 that a perceptron without activation (or with linear activation) can only learn linear decision boundaries. Activation functions enable networks to:

1. **Learn non-linear patterns**: Curves, circles, complex decision boundaries
2. **Introduce gradients**: Enable backpropagation to work
3. **Normalize outputs**: Keep values in reasonable ranges
4. **Enable deep learning**: Without them, depth doesn't add expressiveness

### Activation Function Requirements

A good activation function should:

- **Be non-linear**: Otherwise network collapses to linear model
- **Be differentiable**: Need gradients for backpropagation (almost everywhere is OK)
- **Preserve information**: Avoid saturation where gradients vanish
- **Be computationally efficient**: Will be called millions of times

Let's implement each activation function as a class, following our interface design.

### Implementation: Base Class

<div class="python-interactive" markdown="1">
```python
import numpy as np

class Activation:
    """
    Base class for activation functions.

    All activation functions should inherit from this class
    and implement the forward() and backward() methods.
    """

    def forward(self, Z):
        """
        Apply activation function element-wise.

        Parameters:
        -----------
        Z : ndarray
            Pre-activation values

        Returns:
        --------
        ndarray
            Activated values
        """
        raise NotImplementedError("Subclass must implement forward()")

    def backward(self, dA):
        """
        Compute gradient of activation function.

        Parameters:
        -----------
        dA : ndarray
            Gradient of loss w.r.t. activations

        Returns:
        --------
        ndarray
            Gradient of loss w.r.t. pre-activations
        """
        raise NotImplementedError("Subclass must implement backward()")

class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.

    Formula: f(z) = max(0, z)

    Derivative: f'(z) = 1 if z > 0, else 0

    Properties:
    -----------
    - Most popular activation for hidden layers
    - Computationally efficient (just thresholding)
    - Solves vanishing gradient problem for positive inputs
    - Can suffer from "dying ReLU" problem (neurons stuck at 0)

    When to use:
    ------------
    - Default choice for hidden layers
    - Works well with He initialization
    - Good for most deep networks
    """

    def forward(self, Z):
        """
        Forward pass: f(z) = max(0, z)

        Intuition: "Turn off" negative neurons, pass positive ones unchanged.
        This creates a sparse representation (many zeros).

        Parameters:
        -----------
        Z : ndarray, shape (n, m)
            Pre-activation values (can be any real numbers)

        Returns:
        --------
        A : ndarray, shape (n, m)
            Activations (all non-negative)
        """
        # Cache input for backward pass
        # We need this to know which neurons were active
        self.Z = Z

        # Element-wise maximum with 0
        # Equivalent to: A[i,j] = Z[i,j] if Z[i,j] > 0 else 0
        return np.maximum(0, Z)

    def backward(self, dA):
        """
        Backward pass: f'(z) = 1 if z > 0 else 0

        Intuition: Gradient flows through only if neuron was active (z > 0).
        This is the chain rule: dL/dZ = dL/dA * dA/dZ

        Parameters:
        -----------
        dA : ndarray, shape (n, m)
            Gradient w.r.t. activation (from next layer)

        Returns:
        --------
        dZ : ndarray, shape (n, m)
            Gradient w.r.t. pre-activation (to previous layer)
        """
        # Derivative of ReLU is indicator function:
        # - 1 where Z > 0 (neuron was active)
        # - 0 where Z <= 0 (neuron was inactive)
        #
        # Note: Technically undefined at Z=0, but we define it as 0
        dZ = dA * (self.Z > 0).astype(float)
        return dZ

class Sigmoid(Activation):
    """
    Sigmoid (Logistic) activation function.

    Formula: σ(z) = 1 / (1 + e^(-z))

    Derivative: σ'(z) = σ(z) · (1 - σ(z))

    Properties:
    -----------
    - Outputs in range (0, 1) - interpreted as probabilities
    - Smooth S-shaped curve
    - Saturates at both ends (gradients → 0)
    - Not zero-centered (all outputs positive)

    When to use:
    ------------
    - Output layer for binary classification
    - Gates in LSTM/GRU cells
    - Avoid for hidden layers (use tanh or ReLU instead)
    """

    def forward(self, Z):
        """
        Forward pass: σ(z) = 1 / (1 + e^(-z))

        Intuition: Squashes any real number to (0, 1).
        - Large positive z → output ≈ 1
        - Large negative z → output ≈ 0
        - z = 0 → output = 0.5

        Parameters:
        -----------
        Z : ndarray, shape (n, m)
            Pre-activation values (any real numbers)

        Returns:
        --------
        A : ndarray, shape (n, m)
            Activations in range (0, 1)
        """
        # Cache for backward pass
        # We'll use the output A, not the input Z
        self.A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # Clip to prevent overflow
        return self.A

    def backward(self, dA):
        """
        Backward pass: σ'(z) = σ(z)(1 - σ(z))

        Intuition: Derivative is largest at z=0 (value=0.25),
        and approaches 0 as |z| increases (saturation).

        Beautiful property: derivative depends only on forward output!
        σ'(z) = σ(z)(1 - σ(z)) where σ(z) was computed in forward pass.

        Parameters:
        -----------
        dA : ndarray, shape (n, m)
            Gradient w.r.t. activation

        Returns:
        --------
        dZ : ndarray, shape (n, m)
            Gradient w.r.t. pre-activation
        """
        # σ'(z) = σ(z) * (1 - σ(z))
        # We already have σ(z) cached as self.A
        dZ = dA * self.A * (1 - self.A)
        return dZ

class Tanh(Activation):
    """
    Hyperbolic Tangent activation function.

    Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))

    Derivative: tanh'(z) = 1 - tanh²(z)

    Properties:
    -----------
    - Outputs in range (-1, 1)
    - Zero-centered (unlike sigmoid)
    - Saturates at both ends
    - Stronger gradients than sigmoid

    When to use:
    ------------
    - Hidden layers (better than sigmoid, but ReLU usually better)
    - RNN/LSTM cells
    - When you need zero-centered outputs
    """

    def forward(self, Z):
        """
        Forward pass: tanh(z)

        Intuition: Like sigmoid, but centered at 0.
        - Large positive z → output ≈ 1
        - Large negative z → output ≈ -1
        - z = 0 → output = 0

        Being zero-centered helps gradients flow better.

        Parameters:
        -----------
        Z : ndarray, shape (n, m)
            Pre-activation values

        Returns:
        --------
        A : ndarray, shape (n, m)
            Activations in range (-1, 1)
        """
        # NumPy has optimized tanh
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):
        """
        Backward pass: tanh'(z) = 1 - tanh²(z)

        Intuition: Maximum derivative at z=0 (value=1),
        approaches 0 as |z| increases.

        Parameters:
        -----------
        dA : ndarray, shape (n, m)
            Gradient w.r.t. activation

        Returns:
        --------
        dZ : ndarray, shape (n, m)
            Gradient w.r.t. pre-activation
        """
        # Derivative: 1 - tanh²(z)
        # We have tanh(z) cached as self.A
        dZ = dA * (1 - self.A ** 2)
        return dZ

class Softmax(Activation):
    """
    Softmax activation for multi-class classification.

    Formula: softmax(z)_i = e^(z_i) / Σ_j e^(z_j)

    Properties:
    -----------
    - Outputs sum to 1 (probability distribution)
    - All outputs in range (0, 1)
    - Differentiable everywhere
    - Amplifies differences (largest input gets largest output)

    When to use:
    ------------
    - Output layer for multi-class classification (ONLY)
    - Never use in hidden layers
    - Almost always paired with categorical cross-entropy loss
    """

    def forward(self, Z):
        """
        Forward pass: softmax(z)_i = e^(z_i) / Σ_j e^(z_j)

        Intuition: Convert arbitrary real numbers (logits) into
        a probability distribution. Largest input gets largest probability.

        Numerically stable implementation: subtract max before exponentiating
        to prevent overflow.

        Parameters:
        -----------
        Z : ndarray, shape (n_classes, m)
            Pre-activation values (logits)

        Returns:
        --------
        A : ndarray, shape (n_classes, m)
            Class probabilities (sum to 1 along axis 0)
        """
        # Numerical stability: subtract max
        # softmax(z) = softmax(z - c) for any constant c
        # Choosing c = max(z) prevents exp(large number) overflow
        Z_shifted = Z - np.max(Z, axis=0, keepdims=True)

        # Compute exponentials
        exp_Z = np.exp(Z_shifted)

        # Normalize to get probabilities
        self.A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        return self.A

    def backward(self, dA):
        """
        Backward pass for softmax.

        WARNING: This is complex! The Jacobian is a matrix, not a scalar.

        For softmax_i: ∂softmax_i/∂z_j = softmax_i * (δ_ij - softmax_j)
        where δ_ij is Kronecker delta.

        However, when combined with categorical cross-entropy loss,
        the gradient simplifies beautifully to: dL/dZ = A - Y

        This implementation assumes softmax is used with cross-entropy.

        Parameters:
        -----------
        dA : ndarray
            Gradient w.r.t. activation (usually from loss function)

        Returns:
        --------
        dZ : ndarray
            Gradient w.r.t. pre-activation
        """
        # For numerical stability and simplicity, we assume softmax
        # is combined with cross-entropy loss, which simplifies the gradient.
        # The loss function will handle computing the correct gradient.
        return dA

# Test activations
print("Activation Functions Test")
print("=" * 60)

Z_test = np.array([[-2, -1, 0, 1, 2]])

relu = ReLU()
A_relu = relu.forward(Z_test)
print(f"ReLU({Z_test[0]}) = {A_relu[0]}")

sigmoid = Sigmoid()
A_sigmoid = sigmoid.forward(Z_test)
print(f"Sigmoid({Z_test[0]}) = {A_sigmoid[0]}")

tanh = Tanh()
A_tanh = tanh.forward(Z_test)
print(f"Tanh({Z_test[0]}) = {A_tanh[0]}")

# Test softmax
Z_softmax = np.array([[1, 2, 3]]).T
softmax = Softmax()
A_softmax = softmax.forward(Z_softmax)
print(f"\nSoftmax([1, 2, 3]) = {A_softmax.T[0]}")
print(f"Sum: {np.sum(A_softmax):.6f} (should be 1.0)")
```
</div>

!!! tip "Comparing Activation Functions"
    **ReLU**: Default choice for hidden layers. Fast, effective, but can "die"

    **Sigmoid**: Output layer for binary classification. Outputs in [0,1]. Saturates easily.

    **Tanh**: Better than sigmoid for hidden layers (zero-centered). Still saturates.

    **Softmax**: Output layer for multi-class classification. Outputs sum to 1 (probabilities).

    **Modern variants**: Leaky ReLU, ELU, SELU, Swish, GELU solve dying ReLU problem

## Weight Initialization: The Hidden Critical Detail

Before we implement layers, we need to understand **weight initialization**—one of the most important (and often overlooked) aspects of training neural networks.

### Why Initialization Matters

**Bad initialization can doom your network before training even starts!**

Consider what happens with different initializations:

**All zeros**: Every neuron computes the same output → No gradient flow → No learning

**Too small**: Activations shrink as they pass through layers → **Vanishing gradients**

**Too large**: Activations explode as they pass through layers → **Exploding gradients**

**Just right**: Activations and gradients flow through the network in a balanced way ✨

### The Vanishing/Exploding Gradient Problem

Let's understand why random initialization matters. Consider a 10-layer network where each layer multiplies by weight matrix $\mathbf{W}$.

**If weights are too small** (e.g., $W \sim 0.01$):
$$\text{Activation}_{10} \approx (0.01)^{10} \times \text{Input} = 10^{-20} \times \text{Input}$$

Activations vanish to zero! Gradients (which flow backward) also vanish.

**If weights are too large** (e.g., $W \sim 10$):
$$\text{Activation}_{10} \approx 10^{10} \times \text{Input}$$

Activations explode to infinity! Gradients also explode (or cause NaN).

### The Solution: Variance-Preserving Initialization

We want to initialize weights so that:

1. **Forward pass**: Variance of activations stays roughly constant across layers
2. **Backward pass**: Variance of gradients stays roughly constant across layers

This ensures signals (both activations and gradients) flow through the entire network.

### Xavier (Glorot) Initialization

**For tanh and sigmoid activations**, use **Xavier initialization**:

$$W_{ij} \sim \mathcal{N}\left(0, \frac{1}{n_{in}}\right) \quad \text{or} \quad W_{ij} \sim \text{Uniform}\left(-\sqrt{\frac{3}{n_{in}}}, \sqrt{\frac{3}{n_{in}}}\right)$$

Where $n_{in}$ is the number of input neurons to the layer.

**Intuition**: Scale weights inversely with fan-in. More inputs → smaller weights (to prevent variance explosion).

**Derivation** (simplified):
- Each neuron computes $z = \sum_{i=1}^{n_{in}} w_i x_i$
- If $x_i$ have variance $\sigma_x^2$ and $w_i$ have variance $\sigma_w^2$
- Then $\text{Var}(z) = n_{in} \cdot \sigma_w^2 \cdot \sigma_x^2$
- To keep $\text{Var}(z) = \text{Var}(x)$, we need $\sigma_w^2 = 1/n_{in}$

### He Initialization

**For ReLU activations**, use **He initialization**:

$$W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

**Why different from Xavier?** ReLU kills half the neurons (sets them to 0), effectively halving the variance. The factor of 2 compensates for this.

**When to use what?**

| Activation | Initialization | Variance |
|------------|----------------|----------|
| Sigmoid | Xavier | $1/n_{in}$ |
| Tanh | Xavier | $1/n_{in}$ |
| ReLU | He | $2/n_{in}$ |
| Leaky ReLU | He | $2/n_{in}$ |

### Visualizing the Impact

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_forward_pass(n_layers, n_neurons, init_scale, activation='relu'):
    """
    Simulate how activation variances evolve through layers.
    """
    # Initialize input
    X = np.random.randn(n_neurons, 1000)  # 1000 samples
    variances = [np.var(X)]

    for layer in range(n_layers):
        # Initialize weights
        W = np.random.randn(n_neurons, n_neurons) * init_scale

        # Linear transformation
        Z = W @ X

        # Apply activation
        if activation == 'relu':
            X = np.maximum(0, Z)
        elif activation == 'tanh':
            X = np.tanh(Z)

        variances.append(np.var(X))

    return variances

# Test different initializations
n_layers = 10
n_neurons = 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ReLU with different initializations
ax1 = axes[0]
var_small = simulate_forward_pass(n_layers, n_neurons, 0.01, 'relu')
var_bad = simulate_forward_pass(n_layers, n_neurons, 1.0, 'relu')
var_he = simulate_forward_pass(n_layers, n_neurons, np.sqrt(2.0/n_neurons), 'relu')

ax1.plot(var_small, 'r-', label='Too small (0.01)', linewidth=2)
ax1.plot(var_bad, 'b-', label='Standard (1.0)', linewidth=2)
ax1.plot(var_he, 'g-', label='He initialization', linewidth=2)
ax1.set_xlabel('Layer')
ax1.set_ylabel('Activation Variance')
ax1.set_title('ReLU: Activation Variance Across Layers')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Tanh with different initializations
ax2 = axes[1]
var_small_tanh = simulate_forward_pass(n_layers, n_neurons, 0.01, 'tanh')
var_bad_tanh = simulate_forward_pass(n_layers, n_neurons, 1.0, 'tanh')
var_xavier_tanh = simulate_forward_pass(n_layers, n_neurons, np.sqrt(1.0/n_neurons), 'tanh')

ax2.plot(var_small_tanh, 'r-', label='Too small (0.01)', linewidth=2)
ax2.plot(var_bad_tanh, 'b-', label='Standard (1.0)', linewidth=2)
ax2.plot(var_xavier_tanh, 'g-', label='Xavier initialization', linewidth=2)
ax2.set_xlabel('Layer')
ax2.set_ylabel('Activation Variance')
ax2.set_title('Tanh: Activation Variance Across Layers')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()

print("✓ Proper initialization maintains stable variance across layers")
print("✗ Poor initialization causes vanishing/exploding activations")
```
</div>

!!! warning "Common Initialization Mistakes"
    ❌ **All zeros**: Breaks symmetry—all neurons learn the same thing

    ❌ **All ones**: No symmetry breaking, poor gradient flow

    ❌ **Too large**: Exploding gradients, saturated activations

    ❌ **Too small**: Vanishing gradients, slow/no learning

    ❌ **Wrong scale for activation**: He for sigmoid, Xavier for ReLU

    ✅ **Use He for ReLU, Xavier for tanh/sigmoid**

## Dense Layer Implementation

Now let's implement a fully connected (dense) layer with proper initialization:

<div class="python-interactive" markdown="1">
```python
import numpy as np

class DenseLayer:
    """
    Fully connected (dense) layer.

    Performs: Z = W·A_prev + b
    """

    def __init__(self, n_inputs, n_neurons, activation='relu'):
        """
        Initialize layer parameters.

        Parameters:
        -----------
        n_inputs : int
            Number of input features
        n_neurons : int
            Number of neurons in this layer
        activation : str or Activation
            Activation function ('relu', 'sigmoid', 'tanh', 'softmax')
        """
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

        # Initialize weights: He initialization for ReLU
        if activation == 'relu':
            self.W = np.random.randn(n_neurons, n_inputs) * np.sqrt(2.0 / n_inputs)
        else:
            # Xavier initialization for sigmoid/tanh
            self.W = np.random.randn(n_neurons, n_inputs) * np.sqrt(1.0 / n_inputs)

        self.b = np.zeros((n_neurons, 1))

        # Set activation
        if isinstance(activation, str):
            activation_map = {
                'relu': ReLU(),
                'sigmoid': Sigmoid(),
                'tanh': Tanh(),
                'softmax': Softmax()
            }
            self.activation = activation_map[activation.lower()]
        else:
            self.activation = activation

        # For gradient accumulation
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        """
        Forward propagation.

        Parameters:
        -----------
        A_prev : ndarray, shape (n_inputs, m)
            Activations from previous layer

        Returns:
        --------
        A : ndarray, shape (n_neurons, m)
            Activations of this layer
        """
        self.A_prev = A_prev  # Cache for backward pass

        # Linear transformation
        self.Z = np.dot(self.W, A_prev) + self.b

        # Apply activation
        self.A = self.activation.forward(self.Z)

        return self.A

    def backward(self, dA):
        """
        Backward propagation.

        Parameters:
        -----------
        dA : ndarray, shape (n_neurons, m)
            Gradient w.r.t. activations

        Returns:
        --------
        dA_prev : ndarray, shape (n_inputs, m)
            Gradient w.r.t. previous layer activations
        """
        m = dA.shape[1]

        # Gradient through activation
        dZ = self.activation.backward(dA)

        # Gradient w.r.t. weights and biases
        self.dW = (1/m) * np.dot(dZ, self.A_prev.T)
        self.db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

        # Gradient w.r.t. previous layer
        dA_prev = np.dot(self.W.T, dZ)

        return dA_prev

    def update(self, learning_rate):
        """
        Update parameters using gradient descent.

        Parameters:
        -----------
        learning_rate : float
            Learning rate
        """
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

# Test DenseLayer
print("DenseLayer Test")
print("=" * 60)

# Create layer
layer = DenseLayer(n_inputs=3, n_neurons=4, activation='relu')

# Forward pass
X = np.random.randn(3, 5)  # 3 features, 5 examples
A = layer.forward(X)

print(f"Input shape: {X.shape}")
print(f"Output shape: {A.shape}")
print(f"W shape: {layer.W.shape}")
print(f"b shape: {layer.b.shape}")

# Backward pass
dA = np.random.randn(4, 5)
dA_prev = layer.backward(dA)

print(f"\nGradient shapes:")
print(f"dA shape: {dA.shape}")
print(f"dW shape: {layer.dW.shape}")
print(f"db shape: {layer.db.shape}")
print(f"dA_prev shape: {dA_prev.shape}")
```
</div>

## Loss Functions

Implement loss functions for training:

<div class="python-interactive" markdown="1">
```python
import numpy as np

class Loss:
    """Base class for loss functions."""

    def compute(self, Y_pred, Y_true):
        """Compute loss."""
        raise NotImplementedError

    def gradient(self, Y_pred, Y_true):
        """Compute gradient of loss w.r.t. predictions."""
        raise NotImplementedError

class BinaryCrossEntropy(Loss):
    """Binary cross-entropy loss for binary classification."""

    def compute(self, Y_pred, Y_true):
        """
        L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

        Parameters:
        -----------
        Y_pred : ndarray, shape (1, m)
            Predicted probabilities
        Y_true : ndarray, shape (1, m)
            True labels (0 or 1)

        Returns:
        --------
        loss : float
            Average loss
        """
        m = Y_true.shape[1]
        epsilon = 1e-8  # Prevent log(0)

        loss = -np.mean(
            Y_true * np.log(Y_pred + epsilon) +
            (1 - Y_true) * np.log(1 - Y_pred + epsilon)
        )

        return loss

    def gradient(self, Y_pred, Y_true):
        """
        Compute gradient of loss w.r.t. predictions.

        dL/dA = (A - Y) / [A(1-A)]
        """
        # We need to return dL/dA, because the Sigmoid activation's backward()
        # will multiply by dA/dZ (which is A*(1-A)) to get dL/dZ = A - Y.
        #
        # For numerical stability, we clip values to avoid division by zero.
        epsilon = 1e-15
        Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
        return (Y_pred - Y_true) / (Y_pred * (1 - Y_pred))

class CategoricalCrossEntropy(Loss):
    """Categorical cross-entropy for multi-class classification."""

    def compute(self, Y_pred, Y_true):
        """
        L = -Σ y_i·log(ŷ_i)

        Parameters:
        -----------
        Y_pred : ndarray, shape (n_classes, m)
            Predicted probabilities
        Y_true : ndarray, shape (n_classes, m)
            True labels (one-hot encoded)

        Returns:
        --------
        loss : float
            Average loss
        """
        m = Y_true.shape[1]
        epsilon = 1e-8

        loss = -np.mean(np.sum(Y_true * np.log(Y_pred + epsilon), axis=0))

        return loss

    def gradient(self, Y_pred, Y_true):
        """
        For categorical cross-entropy with softmax:
        Gradient simplifies to: A - Y
        """
        return Y_pred - Y_true

class MeanSquaredError(Loss):
    """Mean squared error loss for regression."""

    def compute(self, Y_pred, Y_true):
        """L = (1/2m)·Σ(ŷ - y)²"""
        m = Y_true.shape[1]
        loss = (1/(2*m)) * np.sum((Y_pred - Y_true)**2)
        return loss

    def gradient(self, Y_pred, Y_true):
        """dL/dŷ = (ŷ - y)"""
        return Y_pred - Y_true

# Test loss functions
print("Loss Functions Test")
print("=" * 60)

# Binary cross-entropy
Y_pred_binary = np.array([[0.7, 0.2, 0.9, 0.1]])
Y_true_binary = np.array([[1, 0, 1, 0]])

bce = BinaryCrossEntropy()
loss = bce.compute(Y_pred_binary, Y_true_binary)
grad = bce.gradient(Y_pred_binary, Y_true_binary)

print(f"Binary Cross-Entropy Loss: {loss:.4f}")
print(f"Gradient shape: {grad.shape}")

# Categorical cross-entropy
Y_pred_cat = np.array([[0.7, 0.2, 0.1],
                       [0.2, 0.7, 0.2],
                       [0.1, 0.1, 0.7]])
Y_true_cat = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

cce = CategoricalCrossEntropy()
loss = cce.compute(Y_pred_cat, Y_true_cat)

print(f"\nCategorical Cross-Entropy Loss: {loss:.4f}")
```
</div>

## Complete Neural Network Class

Now let's build the complete neural network that uses our modular components:

<div class="python-interactive" markdown="1">
```python
import numpy as np

class NeuralNetwork:
    """Modular neural network with arbitrary architecture."""

    def __init__(self, loss='binary_crossentropy'):
        """Initialize neural network."""
        self.layers = []

        # Set loss function
        if isinstance(loss, str):
            loss_map = {
                'binary_crossentropy': BinaryCrossEntropy(),
                'categorical_crossentropy': CategoricalCrossEntropy(),
                'mse': MeanSquaredError()
            }
            self.loss_fn = loss_map[loss.lower()]
        else:
            self.loss_fn = loss

        # Training history
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)

    def forward(self, X):
        """Forward propagation through all layers."""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, Y_true):
        """Backward propagation through all layers."""
        Y_pred = self.layers[-1].A
        dA = self.loss_fn.gradient(Y_pred, Y_true)

        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update(self, learning_rate):
        """Update all layer parameters."""
        for layer in self.layers:
            layer.update(learning_rate)

    def train_step(self, X, Y, learning_rate):
        """Single training step."""
        Y_pred = self.forward(X)
        loss = self.loss_fn.compute(Y_pred, Y)
        self.backward(Y)
        self.update(learning_rate)
        return loss

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, epochs=100,
            learning_rate=0.01, batch_size=32, verbose=True, print_every=10):
        """Train the neural network."""
        m = X_train.shape[1]

        for epoch in range(epochs):
            # Shuffle training data
            permutation = np.random.permutation(m)
            X_shuffled = X_train[:, permutation]
            Y_shuffled = Y_train[:, permutation]

            # Mini-batch training
            epoch_loss = 0
            n_batches = max(1, m // batch_size)

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                Y_batch = Y_shuffled[:, i:i+batch_size]
                batch_loss = self.train_step(X_batch, Y_batch, learning_rate)
                epoch_loss += batch_loss

            # Average loss
            avg_loss = epoch_loss / n_batches
            self.history['loss'].append(avg_loss)

            # Training accuracy
            Y_pred_train = self.predict(X_train)
            train_acc = self.accuracy(Y_pred_train, Y_train)
            self.history['accuracy'].append(train_acc)

            # Validation metrics
            if X_val is not None and Y_val is not None:
                Y_pred_val = self.forward(X_val)
                val_loss = self.loss_fn.compute(Y_pred_val, Y_val)
                self.history['val_loss'].append(val_loss)

                Y_pred_val_class = self.predict(X_val)
                val_acc = self.accuracy(Y_pred_val_class, Y_val)
                self.history['val_accuracy'].append(val_acc)

            # Print progress
            if verbose and (epoch + 1) % print_every == 0:
                if X_val is not None:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}, "
                          f"acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}, acc: {train_acc:.4f}")

    def predict(self, X):
        """Make predictions."""
        Y_pred = self.forward(X)

        if Y_pred.shape[0] == 1:
            return (Y_pred > 0.5).astype(int)
        else:
            return np.argmax(Y_pred, axis=0)

    def accuracy(self, Y_pred, Y_true):
        """Compute classification accuracy."""
        if Y_true.shape[0] > 1:
            Y_true = np.argmax(Y_true, axis=0)
        return np.mean(Y_pred == Y_true.flatten())

print("✓ Complete NeuralNetwork class implemented!")
```
</div>

## Training on XOR

Let's test our complete implementation:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# XOR dataset
X_xor = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=float)
Y_xor = np.array([[0, 1, 1, 0]], dtype=float)

# Create network
nn_xor = NeuralNetwork(loss='binary_crossentropy')
nn_xor.add(DenseLayer(2, 4, activation='relu'))
nn_xor.add(DenseLayer(4, 1, activation='sigmoid'))

# Train
nn_xor.fit(X_xor, Y_xor, epochs=2000, learning_rate=0.5, batch_size=4, print_every=200)

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(nn_xor.history['loss'], linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('XOR Training: Loss over Time')
plt.grid(True, alpha=0.3)
plt.show()

# Test predictions
predictions = nn_xor.predict(X_xor)
print(f"\n✓ XOR solved with {100*nn_xor.accuracy(predictions, Y_xor):.0f}% accuracy!")
```
</div>

## Understanding Mini-Batch Gradient Descent

Our `fit()` method implements **mini-batch gradient descent**—let's understand why this is important and how it differs from other variants.

### Three Flavors of Gradient Descent

**1. Batch Gradient Descent** (batch_size = m, all training data):
```python
# Use ALL samples to compute gradient
dW = (1/m) * X @ (Y_pred - Y_true).T
W = W - learning_rate * dW
```

**Pros**: Stable, smooth convergence
**Cons**: Slow for large datasets, can't fit in memory, stuck in local minima

**2. Stochastic Gradient Descent (SGD)** (batch_size = 1):
```python
# Use ONE sample at a time
for each sample (x_i, y_i):
    dW = x_i * (y_pred_i - y_true_i)
    W = W - learning_rate * dW
```

**Pros**: Fast, can escape local minima, online learning
**Cons**: Noisy, erratic convergence, inefficient (no vectorization)

**3. Mini-Batch Gradient Descent** (1 < batch_size < m):
```python
# Use SMALL batches
for each batch (X_batch, Y_batch):
    dW = (1/batch_size) * X_batch @ (Y_pred - Y_true).T
    W = W - learning_rate * dW
```

**Pros**: Best of both worlds—fast, stable, vectorized
**Cons**: One more hyperparameter to tune

### Choosing Batch Size

| Batch Size | Speed | Memory | Convergence | Generalization |
|------------|-------|--------|-------------|----------------|
| 1 (SGD) | Slow | Low | Noisy | Better |
| 32 | Fast | Low | Good | Good |
| 64 | Fast | Medium | Good | Good |
| 256 | Faster | High | Smooth | Worse |
| Full (BGD) | Slowest | Very High | Very Smooth | Worst |

!!! tip "Practical Guidelines"
    - **Start with 32**: Good default for most problems
    - **Use powers of 2**: 16, 32, 64, 128 (efficient for GPUs)
    - **Larger batches**: More stable but may overfit
    - **Smaller batches**: More noise, better generalization
    - **Limited memory**: Reduce batch size
    - **Large datasets**: 128-256 typical

### Visualizing Batch Size Effects

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate spiral dataset
np.random.seed(42)
N = 100  # points per class
K = 3    # classes
X = np.zeros((N*K, 2))
y = np.zeros(N*K, dtype='uint8')

for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

# Convert to network format
X = X.T  # (2, 300)
Y = np.eye(K)[y].T  # One-hot encode (3, 300)

# Train with different batch sizes
batch_sizes = [1, 10, 50, 300]
histories = {}

for bs in batch_sizes:
    print(f"\nTraining with batch_size={bs}")
    nn = NeuralNetwork(loss='categorical_crossentropy')
    nn.add(DenseLayer(2, 100, activation='relu'))
    nn.add(DenseLayer(100, K, activation='softmax'))

    nn.fit(X, Y, epochs=200, learning_rate=0.1, batch_size=bs,
           verbose=False)
    histories[bs] = nn.history['loss']

# Plot learning curves
plt.figure(figsize=(12, 5))

# Subplot 1: All curves together
plt.subplot(1, 2, 1)
for bs in batch_sizes:
    label = f'Batch={bs}' if bs < 300 else 'Full Batch'
    plt.plot(histories[bs], label=label, linewidth=2, alpha=0.7)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Learning Curves: Effect of Batch Size', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Last 50 epochs (zoomed)
plt.subplot(1, 2, 2)
for bs in batch_sizes:
    label = f'Batch={bs}' if bs < 300 else 'Full Batch'
    plt.plot(histories[bs][-50:], label=label, linewidth=2, alpha=0.7)
plt.xlabel('Epoch (last 50)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Convergence Behavior (Zoomed)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nObservations:")
print("  • SGD (batch=1): Noisy but explores more")
print("  • Mini-batch (10-50): Good balance")
print("  • Full batch (300): Smooth but may overfit")
```
</div>

## Debugging Neural Networks: A Practical Guide

Training neural networks is as much art as science. Here's how to systematically debug when things go wrong.

### Common Problems and Solutions

#### Problem 1: Loss is NaN or Infinity

**Symptoms**: Loss becomes NaN after a few iterations

**Causes**:
- Learning rate too high → exploding gradients
- Poor weight initialization
- Numerical instability in loss function

**Solutions**:
```python
# 1. Reduce learning rate
learning_rate = 0.001  # Start small

# 2. Check for NaN in data
assert not np.isnan(X).any(), "NaN in input data!"
assert not np.isinf(X).any(), "Inf in input data!"

# 3. Add gradient clipping
def clip_gradients(layer, max_norm=5.0):
    grad_norm = np.sqrt(np.sum(layer.dW**2) + np.sum(layer.db**2))
    if grad_norm > max_norm:
        layer.dW *= max_norm / grad_norm
        layer.db *= max_norm / grad_norm

# 4. Use proper initialization (He/Xavier)
# 5. Add epsilon to log in loss functions
```

#### Problem 2: Loss Not Decreasing

**Symptoms**: Loss stays constant or increases

**Causes**:
- Learning rate too small
- Wrong loss function for the task
- Data not preprocessed
- Bug in backpropagation

**Debugging steps**:
```python
# 1. Overfit on a single batch (sanity check)
X_tiny = X_train[:, :10]  # Just 10 samples
Y_tiny = Y_train[:, :10]

nn.fit(X_tiny, Y_tiny, epochs=1000, learning_rate=0.1)
# Should reach ~0 loss if implementation is correct!

# 2. Check gradients numerically
def numerical_gradient(f, x, h=1e-5):
    """
    Compute gradient numerically for debugging.
    Compare against backprop gradients.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        old_value = x[idx]

        x[idx] = old_value + h
        f_plus = f()

        x[idx] = old_value - h
        f_minus = f()

        grad[idx] = (f_plus - f_minus) / (2 * h)
        x[idx] = old_value
        it.iternext()

    return grad

# 3. Visualize activations
for i, layer in enumerate(nn.layers):
    print(f"Layer {i} activation mean: {np.mean(layer.A):.4f}")
    print(f"Layer {i} activation std:  {np.std(layer.A):.4f}")
    # Healthy: mean near 0, std around 1
```

#### Problem 3: Training Accuracy High, Validation Low

**Symptoms**: Overfitting—memorizing training data

**Solutions**:
```python
# 1. Add dropout
class Dropout(Layer):
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob
        self.training = True

    def forward(self, A_prev):
        if self.training:
            self.mask = np.random.rand(*A_prev.shape) < self.keep_prob
            return A_prev * self.mask / self.keep_prob
        else:
            return A_prev

# 2. Add L2 regularization
def compute_loss_with_l2(Y_pred, Y_true, layers, lambda_reg=0.01):
    loss = base_loss(Y_pred, Y_true)
    l2_penalty = sum(np.sum(layer.W**2) for layer in layers)
    return loss + (lambda_reg / (2*m)) * l2_penalty

# 3. Reduce network size
# 4. Increase training data (augmentation)
# 5. Early stopping on validation loss
```

#### Problem 4: Learning is Very Slow

**Symptoms**: Loss decreases very slowly

**Solutions**:
```python
# 1. Increase learning rate
learning_rate = 0.1  # Try larger

# 2. Use better optimizer (momentum, Adam)
class MomentumOptimizer:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.velocities = {}

    def update(self, layer, layer_id):
        if layer_id not in self.velocities:
            self.velocities[layer_id] = {
                'dW': np.zeros_like(layer.W),
                'db': np.zeros_like(layer.b)
            }

        v = self.velocities[layer_id]
        v['dW'] = self.beta * v['dW'] + (1 - self.beta) * layer.dW
        v['db'] = self.beta * v['db'] + (1 - self.beta) * layer.db

        layer.W -= self.lr * v['dW']
        layer.b -= self.lr * v['db']

# 3. Normalize input features
X_normalized = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

# 4. Use batch normalization
```

### Gradient Checking: The Ultimate Sanity Test

<div class="python-interactive" markdown="1">
```python
import numpy as np

def gradient_check(nn, X, Y, epsilon=1e-7):
    """
    Numerically verify backpropagation gradients.

    Returns relative error - should be < 1e-7 for correct implementation.
    """
    # Forward and backward pass
    nn.forward(X)
    nn.backward(Y)

    # Store computed gradients
    computed_grads = []
    numerical_grads = []

    for layer in nn.layers:
        if hasattr(layer, 'W'):
            # Flatten gradients
            computed_grads.append(layer.dW.ravel())
            computed_grads.append(layer.db.ravel())

            # Numerical gradient for W
            num_grad_W = np.zeros_like(layer.W)
            it = np.nditer(layer.W, flags=['multi_index'])

            while not it.finished:
                idx = it.multi_index
                old_val = layer.W[idx]

                layer.W[idx] = old_val + epsilon
                loss_plus = nn.loss_fn.compute(nn.forward(X), Y)

                layer.W[idx] = old_val - epsilon
                loss_minus = nn.loss_fn.compute(nn.forward(X), Y)

                num_grad_W[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                layer.W[idx] = old_val
                it.iternext()

            numerical_grads.append(num_grad_W.ravel())

            # Numerical gradient for b
            num_grad_b = np.zeros_like(layer.b)
            it = np.nditer(layer.b, flags=['multi_index'])

            while not it.finished:
                idx = it.multi_index
                old_val = layer.b[idx]

                layer.b[idx] = old_val + epsilon
                loss_plus = nn.loss_fn.compute(nn.forward(X), Y)

                layer.b[idx] = old_val - epsilon
                loss_minus = nn.loss_fn.compute(nn.forward(X), Y)

                num_grad_b[idx] = (loss_plus - loss_minus) / (2 * epsilon)
                layer.b[idx] = old_val
                it.iternext()

            numerical_grads.append(num_grad_b.ravel())

    # Concatenate all gradients
    computed = np.concatenate(computed_grads)
    numerical = np.concatenate(numerical_grads)

    # Compute relative error
    numerator = np.linalg.norm(computed - numerical)
    denominator = np.linalg.norm(computed) + np.linalg.norm(numerical)
    relative_error = numerator / denominator

    print(f"Relative error: {relative_error:.2e}")

    if relative_error < 1e-7:
        print("✓ Gradient check passed! Backprop is correct.")
    elif relative_error < 1e-5:
        print("⚠ Gradient check OK, but could be better.")
    else:
        print("✗ Gradient check FAILED! Bug in backprop.")

    return relative_error

# Test on small network
nn_test = NeuralNetwork(loss='binary_crossentropy')
nn_test.add(DenseLayer(2, 3, activation='relu'))
nn_test.add(DenseLayer(3, 1, activation='sigmoid'))

X_test = np.random.randn(2, 5)
Y_test = np.random.rand(1, 5)

gradient_check(nn_test, X_test, Y_test)
```
</div>

## Visualizing Decision Boundaries

One of the best ways to understand what your network has learned is to visualize its decision boundaries.

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(nn, X, Y, resolution=0.01):
    """
    Plot 2D decision boundary for a trained neural network.

    Parameters:
    -----------
    nn : NeuralNetwork
        Trained network
    X : ndarray, shape (2, m)
        2D input data
    Y : ndarray
        Labels
    resolution : float
        Grid resolution
    """
    # Create mesh grid
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    # Predict on mesh grid
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.colorbar(label='Predicted Class')

    # Plot training data
    if Y.shape[0] == 1:
        # Binary classification
        plt.scatter(X[0, Y[0]==0], X[1, Y[0]==0],
                   c='blue', marker='o', s=50, edgecolors='k',
                   label='Class 0')
        plt.scatter(X[0, Y[0]==1], X[1, Y[0]==1],
                   c='red', marker='s', s=50, edgecolors='k',
                   label='Class 1')
    else:
        # Multi-class
        Y_labels = np.argmax(Y, axis=0)
        for k in range(Y.shape[0]):
            mask = Y_labels == k
            plt.scatter(X[0, mask], X[1, mask],
                       marker='o', s=50, edgecolors='k',
                       label=f'Class {k}')

    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('Decision Boundary Visualization', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Example: Train and visualize XOR
X_xor = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=float)
Y_xor = np.array([[0, 1, 1, 0]], dtype=float)

nn_vis = NeuralNetwork(loss='binary_crossentropy')
nn_vis.add(DenseLayer(2, 8, activation='relu'))
nn_vis.add(DenseLayer(8, 1, activation='sigmoid'))

nn_vis.fit(X_xor, Y_xor, epochs=1000, learning_rate=0.5,
           batch_size=4, verbose=False)

plot_decision_boundary(nn_vis, X_xor, Y_xor, resolution=0.01)

print(f"XOR solved with {100*nn_vis.accuracy(nn_vis.predict(X_xor), Y_xor):.0f}% accuracy")
```
</div>

## Performance Optimization Tips

Our implementation prioritizes clarity over speed, but here are ways to make it faster:

### 1. Vectorization (Already Done!)

```python
# Slow: Loop over samples
for i in range(m):
    output[i] = np.dot(W, X[:, i])

# Fast: Vectorized
output = np.dot(W, X)  # All samples at once
```

### 2. In-Place Operations

```python
# Creates new array
A = A + 1

# Modifies in-place (faster, less memory)
A += 1

# Similarly for weights update
self.W -= learning_rate * self.dW  # In-place
```

### 3. Avoid Unnecessary Copies

```python
# Bad: Creates copy
Z_shifted = Z - np.max(Z)

# Good: Uses view when possible
Z -= np.max(Z)  # Modifies in-place
```

### 4. Profile Your Code

```python
import time

start = time.time()
nn.fit(X_train, Y_train, epochs=100)
end = time.time()

print(f"Training took {end - start:.2f} seconds")

# Or use line_profiler for detailed analysis
%load_ext line_profiler
%lprun -f nn.fit nn.fit(X_train, Y_train, epochs=100)
```

## Summary

### Key Takeaways

!!! success "What You Built"
    **Complete Neural Network Library**
    - Modular design with Layer, Activation, and Loss classes
    - Supports arbitrary architectures
    - Full training pipeline with mini-batch gradient descent
    - Professional code quality

    **Best Practices Applied**
    - Proper weight initialization (He/Xavier)
    - Clean object-oriented design
    - Efficient vectorized operations
    - Comprehensive documentation

## Practice Problems

**Problem 1**: Extend the library to support L2 regularization in the DenseLayer class.

**Problem 2**: Implement dropout regularization during training.

**Problem 3**: Add momentum or Adam optimizer instead of plain gradient descent.

**Problem 4**: Create a learning rate scheduler that decays the learning rate over time.

**Problem 5**: Implement early stopping based on validation loss.

## Next Steps

You've built a complete neural network library from scratch! You now understand exactly how neural networks work internally.

**In Lesson 5**, we'll learn **PyTorch** - see how a modern framework handles all this automatically with automatic differentiation, GPU acceleration, and pre-built components!

[Continue to Lesson 5: PyTorch Basics](05-pytorch-basics.md){ .md-button .md-button--primary }

[Practice with Exercise 4: NumPy Implementation](exercises.md#exercise-4-numpy-neural-network-implementation){ .md-button }

[Return to Module 4 Overview](index.md){ .md-button }

---

**Questions or feedback?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues).