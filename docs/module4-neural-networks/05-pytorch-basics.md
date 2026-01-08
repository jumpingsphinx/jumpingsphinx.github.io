# Lesson 5: PyTorch Basics

## Introduction

In the previous lesson, we built a complete neural network library from scratch using only NumPy. You now understand exactly how neural networks work internally—forward propagation, backpropagation, gradient descent, and all the intricate details.

Now it's time to see how **PyTorch**—one of the most popular deep learning frameworks—makes all of this dramatically easier while maintaining flexibility and performance.

### Why PyTorch?

After building everything from scratch, you might wonder: "Why use a framework at all?" Here's why PyTorch is transformative:

1. **Automatic Differentiation**: No more manual backprop! PyTorch computes gradients automatically
2. **GPU Acceleration**: Train models 10-100x faster on GPUs
3. **Production Ready**: Battle-tested code used by researchers and companies worldwide
4. **Rich Ecosystem**: Pre-trained models, datasets, utilities
5. **Pythonic Design**: Feels natural if you know Python and NumPy
6. **Dynamic Computation Graphs**: Debug like normal Python code
7. **Active Community**: Extensive tutorials, forums, and support

### What You'll Learn

By the end of this lesson, you will:

- ✅ Understand PyTorch tensors and their relationship to NumPy arrays
- ✅ Perform tensor operations on CPU and GPU
- ✅ Build neural networks using `nn.Module`
- ✅ Understand automatic differentiation with `autograd`
- ✅ Train models using PyTorch's built-in optimizers
- ✅ Load and preprocess data with `DataLoader`
- ✅ Save and load trained models
- ✅ Compare PyTorch code to your NumPy implementation
- ✅ Appreciate the power and elegance of modern frameworks

**Estimated Time**: 90-120 minutes

!!! tip "Installing PyTorch"
    Before starting, install PyTorch:

    ```bash
    # CPU only
    pip install torch torchvision

    # With CUDA (GPU support)
    # Visit https://pytorch.org/get-started/locally/ for specific instructions
    ```

## PyTorch Tensors: The Foundation

### Visual Introduction to PyTorch

Before diving into the code, watch this practical introduction to PyTorch:

<div style="text-align: center; margin: 20px 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/c36lUUr864M" title="PyTorch in 100 Seconds" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

Tensors are PyTorch's fundamental data structure—think of them as NumPy arrays on steroids, with GPU support and automatic differentiation.

### Creating Tensors

<div class="python-interactive" markdown="1">
```python
import torch
import numpy as np

print("=== Creating Tensors ===\n")

# From Python lists
t1 = torch.tensor([1, 2, 3, 4])
print(f"From list: {t1}")
print(f"Shape: {t1.shape}, dtype: {t1.dtype}\n")

# From NumPy array (shares memory!)
np_array = np.array([[1, 2], [3, 4]])
t2 = torch.from_numpy(np_array)
print(f"From NumPy:\n{t2}")
print(f"Type: {t2.dtype}\n")

# Initialized tensors
t3 = torch.zeros(3, 4)
print(f"Zeros:\n{t3}\n")

t4 = torch.ones(2, 3)
print(f"Ones:\n{t4}\n")

t5 = torch.randn(2, 3)  # Standard normal distribution
print(f"Random (normal):\n{t5}\n")

t6 = torch.rand(2, 3)  # Uniform [0, 1)
print(f"Random (uniform):\n{t6}\n")

# Specific ranges
t7 = torch.arange(0, 10, 2)  # Like Python range
print(f"Arange: {t7}\n")

t8 = torch.linspace(0, 1, 5)  # 5 evenly spaced points
print(f"Linspace: {t8}\n")

# Like another tensor
t9 = torch.zeros_like(t2)
print(f"Zeros like:\n{t9}")
```
</div>

### Tensor Operations

PyTorch operations are nearly identical to NumPy:

<div class="python-interactive" markdown="1">
```python
import torch

print("=== Tensor Operations ===\n")

# Create tensors
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print("A:")
print(a)
print("\nB:")
print(b)

# Element-wise operations
print("\n--- Element-wise ---")
print(f"A + B:\n{a + b}\n")
print(f"A * B (element-wise):\n{a * b}\n")
print(f"A ** 2:\n{a ** 2}\n")

# Matrix operations
print("--- Matrix Operations ---")
print(f"Matrix multiply (A @ B):\n{a @ b}\n")
print(f"Or torch.mm(A, B):\n{torch.mm(a, b)}\n")

# Reductions
print("--- Reductions ---")
print(f"Sum: {a.sum()}")
print(f"Sum along axis 0: {a.sum(dim=0)}")
print(f"Sum along axis 1: {a.sum(dim=1)}")
print(f"Mean: {a.mean()}")
print(f"Max: {a.max()}")
print(f"Min: {a.min()}\n")

# Reshaping
print("--- Reshaping ---")
c = torch.arange(12)
print(f"Original: {c}")
print(f"Reshape to (3, 4):\n{c.reshape(3, 4)}")
print(f"Reshape to (2, 6):\n{c.reshape(2, 6)}")
print(f"Flatten: {c.reshape(3, 4).flatten()}\n")

# Indexing (just like NumPy)
print("--- Indexing ---")
d = torch.arange(20).reshape(4, 5)
print(f"Tensor:\n{d}")
print(f"First row: {d[0]}")
print(f"First column: {d[:, 0]}")
print(f"Slice: d[1:3, 2:4]:\n{d[1:3, 2:4]}")
```
</div>

### NumPy vs PyTorch: Side by Side

<div class="python-interactive" markdown="1">
```python
import numpy as np
import torch

print("=== NumPy vs PyTorch ===\n")

# Create arrays
np_arr = np.array([[1, 2], [3, 4]])
torch_tensor = torch.tensor([[1, 2], [3, 4]])

print("NumPy:")
print(f"  Shape: {np_arr.shape}")
print(f"  Dtype: {np_arr.dtype}")
print(f"  Sum: {np_arr.sum()}\n")

print("PyTorch:")
print(f"  Shape: {torch_tensor.shape}")
print(f"  Dtype: {torch_tensor.dtype}")
print(f"  Sum: {torch_tensor.sum()}\n")

# Conversion between NumPy and PyTorch
print("--- Conversion ---")
torch_from_np = torch.from_numpy(np_arr)
print(f"Torch from NumPy: {torch_from_np}")

np_from_torch = torch_tensor.numpy()
print(f"NumPy from Torch: {np_from_torch}\n")

# Key differences
print("--- Key Differences ---")
print("NumPy:")
print(f"  arr.T (transpose): \n{np_arr.T}")
print(f"  arr.reshape(-1): {np_arr.reshape(-1)}\n")

print("PyTorch:")
print(f"  tensor.T (transpose): \n{torch_tensor.T}")
print(f"  tensor.reshape(-1): {torch_tensor.reshape(-1)}")
print(f"  OR tensor.view(-1): {torch_tensor.view(-1)}")  # Like reshape but stricter
```
</div>

!!! info "NumPy vs PyTorch Quick Reference"
    | Operation | NumPy | PyTorch |
    |-----------|-------|---------|
    | Create zeros | `np.zeros((3, 4))` | `torch.zeros(3, 4)` |
    | Random normal | `np.random.randn(3, 4)` | `torch.randn(3, 4)` |
    | Matrix multiply | `A @ B` or `np.dot(A, B)` | `A @ B` or `torch.mm(A, B)` |
    | Sum | `arr.sum()` | `tensor.sum()` |
    | Reshape | `arr.reshape(3, 4)` | `tensor.reshape(3, 4)` or `tensor.view(3, 4)` |
    | Transpose | `arr.T` | `tensor.T` |
    | To device | N/A | `tensor.to('cuda')` |

## GPU Acceleration

One of PyTorch's superpowers is seamless GPU acceleration:

<div class="python-interactive" markdown="1">
```python
import torch

print("=== GPU Acceleration ===\n")

# Check if CUDA (NVIDIA GPU) is available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}\n")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Create tensor on CPU
cpu_tensor = torch.randn(1000, 1000)
print(f"CPU tensor device: {cpu_tensor.device}")

# Move to GPU
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to(device)
    print(f"GPU tensor device: {gpu_tensor.device}")

    # Or create directly on GPU
    gpu_tensor2 = torch.randn(1000, 1000, device=device)
    print(f"Created on GPU: {gpu_tensor2.device}")

    # Benchmark: Matrix multiply on CPU vs GPU
    import time

    # CPU
    start = time.time()
    result_cpu = cpu_tensor @ cpu_tensor
    cpu_time = time.time() - start

    # GPU
    torch.cuda.synchronize()  # Wait for GPU to finish
    start = time.time()
    result_gpu = gpu_tensor @ gpu_tensor
    torch.cuda.synchronize()
    gpu_time = time.time() - start

    print(f"\nMatrix multiplication (1000x1000):")
    print(f"  CPU time: {cpu_time*1000:.2f} ms")
    print(f"  GPU time: {gpu_time*1000:.2f} ms")
    print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
else:
    print("No GPU available - install CUDA version of PyTorch for GPU support")
```
</div>

!!! warning "GPU Memory Management"
    - GPU memory is limited—watch your batch sizes!
    - Use `tensor.cpu()` to move back to CPU
    - Use `torch.cuda.empty_cache()` to free unused memory
    - Monitor GPU usage with `nvidia-smi` command

## Automatic Differentiation with Autograd

This is where PyTorch truly shines—**automatic differentiation**. No more manual backpropagation!

### The Basics

<div class="python-interactive" markdown="1">
```python
import torch

print("=== Autograd Basics ===\n")

# Create tensor with gradient tracking enabled
x = torch.tensor(2.0, requires_grad=True)
print(f"x = {x}")
print(f"requires_grad: {x.requires_grad}\n")

# Define a function
y = x ** 2 + 2 * x + 1
print(f"y = x² + 2x + 1 = {y}\n")

# Compute gradient dy/dx
y.backward()  # Magic happens here!

print(f"dy/dx = {x.grad}")
print(f"Analytical: dy/dx = 2x + 2 = 2({x.item()}) + 2 = {2*x.item() + 2}")
```
</div>

### How Autograd Works

<div class="python-interactive" markdown="1">
```python
import torch

print("=== Understanding Autograd ===\n")

# Create input
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"x: {x}\n")

# Build computation graph
y = x * 2  # y = 2x
print(f"y = 2x: {y}")
print(f"y.grad_fn: {y.grad_fn}\n")  # Tracks how y was computed

z = y ** 2  # z = (2x)² = 4x²
print(f"z = y²: {z}")
print(f"z.grad_fn: {z.grad_fn}\n")

out = z.mean()  # Average: (4x₁² + 4x₂² + 4x₃²) / 3
print(f"out = mean(z): {out}")
print(f"out.grad_fn: {out.grad_fn}\n")

# Backpropagate
print("Computing gradients...")
out.backward()

print(f"x.grad (∂out/∂x): {x.grad}")
print("\nAnalytical:")
print("  out = (4x₁² + 4x₂² + 4x₃²) / 3")
print("  ∂out/∂x_i = 8x_i / 3")
print(f"  For x=[1,2,3]: {8*x/3}")
```
</div>

### Comparing: Manual vs Autograd

<div class="python-interactive" markdown="1">
```python
import torch

print("=== Manual vs Autograd ===\n")

# Example: Simple function f(x) = x² + 3x
x_val = 5.0

# Manual gradient
def f_manual(x):
    return x**2 + 3*x

def df_manual(x):
    return 2*x + 3  # Derivative we computed by hand

print("Manual Differentiation:")
print(f"  f({x_val}) = {f_manual(x_val)}")
print(f"  f'({x_val}) = {df_manual(x_val)}\n")

# Autograd
x = torch.tensor(x_val, requires_grad=True)
f = x**2 + 3*x
f.backward()

print("Automatic Differentiation:")
print(f"  f({x_val}) = {f.item()}")
print(f"  f'({x_val}) = {x.grad.item()}\n")

print("✓ They match perfectly!")
```
</div>

### Gradient Accumulation

<div class="python-interactive" markdown="1">
```python
import torch

print("=== Gradient Accumulation ===\n")

x = torch.tensor(3.0, requires_grad=True)

# First forward-backward
y1 = x ** 2
y1.backward()
print(f"After first backward, x.grad = {x.grad}")

# Second forward-backward WITHOUT zeroing gradient
y2 = x + 1
y2.backward()
print(f"After second backward, x.grad = {x.grad} (accumulated!)\n")

# This is why we need to zero gradients in training loops!
print("Proper way:")
x = torch.tensor(3.0, requires_grad=True)
y1 = x ** 2
y1.backward()
print(f"First: x.grad = {x.grad}")

x.grad.zero_()  # Zero the gradient
y2 = x + 1
y2.backward()
print(f"Second (after zero): x.grad = {x.grad}")
```
</div>

!!! tip "Key Autograd Concepts"
    - `requires_grad=True`: Track operations for gradient computation
    - `.backward()`: Compute gradients via backpropagation
    - `.grad`: Access computed gradients
    - `.grad.zero_()`: Reset gradients (MUST do before each training step)
    - `torch.no_grad()`: Disable gradient tracking (e.g., during inference)

## Building Neural Networks with nn.Module

PyTorch provides `torch.nn` for building neural networks. Let's recreate our NumPy implementation in PyTorch:

### Your First PyTorch Network

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn

print("=== Simple Neural Network ===\n")

# Define network architecture
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Define forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Create network
net = SimpleNet(input_size=2, hidden_size=4, output_size=1)
print(net)
print()

# Check parameters
print("Parameters:")
for name, param in net.named_parameters():
    print(f"  {name}: shape {param.shape}")
print()

# Count parameters
total_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {total_params}")
print()

# Forward pass
X = torch.randn(5, 2)  # 5 samples, 2 features
print(f"Input shape: {X.shape}")

output = net(X)
print(f"Output shape: {output.shape}")
print(f"Output:\n{output}")
```
</div>

### Sequential API (Simpler Alternative)

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn

print("=== Sequential API ===\n")

# Same network, simpler syntax
net = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

print(net)
print()

# Forward pass
X = torch.randn(5, 2)
output = net(X)
print(f"Output: {output.squeeze()}")
```
</div>

!!! tip "nn.Module vs nn.Sequential"
    **Use nn.Module when:**
    - You need complex forward logic (branching, loops)
    - You want to store intermediate activations
    - You need custom behavior

    **Use nn.Sequential when:**
    - Simple linear stack of layers
    - No branching or custom logic
    - Cleaner, more concise code

### Available Layers and Activations

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn

print("=== Common Layers ===\n")

# Fully connected layers
fc = nn.Linear(in_features=10, out_features=5)
print(f"Linear: {fc}")
print(f"  Weight shape: {fc.weight.shape}")
print(f"  Bias shape: {fc.bias.shape}\n")

# Activation functions
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=1)  # Along which dimension
print("Activations: ReLU, Sigmoid, Tanh, Softmax\n")

# Dropout (regularization)
dropout = nn.Dropout(p=0.5)  # Drop 50% of neurons
print(f"Dropout: {dropout}\n")

# Batch Normalization
bn = nn.BatchNorm1d(num_features=5)
print(f"BatchNorm: {bn}\n")

# Convolutional layers (for images)
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
print(f"Conv2d: {conv}")
print(f"  Weight shape: {conv.weight.shape}\n")

# Pooling
maxpool = nn.MaxPool2d(kernel_size=2)
print(f"MaxPool2d: {maxpool}\n")

# Recurrent layers
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
print(f"LSTM: {lstm}")
```
</div>

## Training a Model: Complete Example

Let's train a neural network on the XOR problem, comparing it to our NumPy implementation:

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

print("=== Training on XOR ===\n")

# XOR dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

print("Dataset:")
print(f"X:\n{X}")
print(f"y: {y.T}\n")

# Define model
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

print("Model architecture:")
print(model)
print()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Training loop
epochs = 2000
losses = []

for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward pass
    optimizer.zero_grad()  # Zero gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update weights

    # Record loss
    losses.append(loss.item())

    # Print progress
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print()

# Test predictions
with torch.no_grad():  # Don't track gradients during inference
    predictions = model(X)
    predictions_binary = (predictions > 0.5).float()

print("Final Predictions:")
print(f"Input | True | Predicted | Binary")
print("-" * 40)
for i in range(len(X)):
    print(f"{X[i].numpy()} | {y[i].item():.0f}    | {predictions[i].item():.4f}     | {predictions_binary[i].item():.0f}")

accuracy = (predictions_binary == y).float().mean()
print(f"\n✓ Accuracy: {accuracy.item()*100:.0f}%")

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('XOR Training: Loss over Time')
plt.grid(True, alpha=0.3)
plt.show()
```
</div>

### Comparison: NumPy vs PyTorch

Let's see how our PyTorch code compares to the NumPy implementation:

<div class="python-interactive" markdown="1">
```python
print("=== NumPy vs PyTorch Comparison ===\n")

print("NUMPY IMPLEMENTATION:")
print("""
# Define network
nn = NeuralNetwork(loss='binary_crossentropy')
nn.add(DenseLayer(2, 4, activation='relu'))
nn.add(DenseLayer(4, 1, activation='sigmoid'))

# Train
nn.fit(X_xor, Y_xor, epochs=2000, learning_rate=0.5)

# Predict
predictions = nn.predict(X_xor)
""")

print("\nPYTORCH IMPLEMENTATION:")
print("""
# Define model
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# Training loop
for epoch in range(2000):
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict
with torch.no_grad():
    predictions = model(X)
""")

print("\nKey Differences:")
print("  ✓ PyTorch: Automatic differentiation (no manual backprop!)")
print("  ✓ PyTorch: Built-in optimizers (SGD, Adam, etc.)")
print("  ✓ PyTorch: GPU support out of the box")
print("  ✓ NumPy: You understand exactly what's happening")
print("  ✓ Both: Achieve same results!")
```
</div>

## Optimizers: Beyond Vanilla SGD

PyTorch provides many sophisticated optimizers:

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Create dummy model
model = nn.Linear(10, 1)

print("=== PyTorch Optimizers ===\n")

# 1. Stochastic Gradient Descent (SGD)
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)
print("SGD:")
print("  Basic gradient descent")
print("  W := W - lr * dW\n")

# 2. SGD with Momentum
optimizer_momentum = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print("SGD with Momentum:")
print("  Accelerates learning in consistent directions")
print("  v := β*v + dW")
print("  W := W - lr * v\n")

# 3. Adam (Adaptive Moment Estimation)
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
print("Adam:")
print("  Combines momentum + adaptive learning rates")
print("  Most popular choice for deep learning")
print("  Usually works well with default settings\n")

# 4. RMSprop
optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=0.01)
print("RMSprop:")
print("  Adaptive learning rate per parameter")
print("  Good for RNNs\n")

# 5. AdamW (Adam with Weight Decay)
optimizer_adamw = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
print("AdamW:")
print("  Adam with proper weight decay (L2 regularization)")
print("  Often better than regular Adam\n")

print("Rule of thumb:")
print("  • Start with Adam (lr=0.001)")
print("  • Try AdamW if you need regularization")
print("  • Use SGD with momentum for final fine-tuning")
```
</div>

### Comparing Optimizers

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Generate non-convex optimization problem (Rosenbrock function)
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# Train with different optimizers
def optimize_with(optimizer_class, lr, **kwargs):
    x = torch.tensor([-1.0, 1.0], requires_grad=True)
    optimizer = optimizer_class([x], lr=lr, **kwargs)

    path = [x.detach().clone().numpy()]
    losses = []

    for _ in range(100):
        optimizer.zero_grad()
        loss = rosenbrock(x[0], x[1])
        loss.backward()
        optimizer.step()

        path.append(x.detach().clone().numpy())
        losses.append(loss.item())

    return np.array(path), losses

# Compare optimizers
optimizers_config = [
    (optim.SGD, 0.001, {}, 'SGD'),
    (optim.SGD, 0.001, {'momentum': 0.9}, 'SGD + Momentum'),
    (optim.Adam, 0.01, {}, 'Adam'),
    (optim.RMSprop, 0.01, {}, 'RMSprop')
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot loss curves
ax1 = axes[0]
for opt_class, lr, kwargs, label in optimizers_config:
    _, losses = optimize_with(opt_class, lr, **kwargs)
    ax1.plot(losses, label=label, linewidth=2)

ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.set_title('Optimizer Comparison: Loss over Time')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot optimization paths
ax2 = axes[1]
x = np.linspace(-1.5, 1.5, 100)
y = np.linspace(-0.5, 1.5, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)
ax2.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.6)

for opt_class, lr, kwargs, label in optimizers_config:
    path, _ = optimize_with(opt_class, lr, **kwargs)
    ax2.plot(path[:, 0], path[:, 1], 'o-', label=label, markersize=3, linewidth=1.5)

ax2.plot(1, 1, 'r*', markersize=20, label='Optimum')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Optimization Paths')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Observations:")
print("  • SGD: Slow, but stable")
print("  • Momentum: Faster convergence")
print("  • Adam: Fast and adaptive")
print("  • RMSprop: Good for non-stationary problems")
```
</div>

## Data Loading with DataLoader

For real projects, use PyTorch's DataLoader for efficient batching and shuffling:

<div class="python-interactive" markdown="1">
```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

print("=== DataLoader Example ===\n")

# Custom Dataset
class XORDataset(Dataset):
    def __init__(self):
        self.X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        self.y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset and dataloader
dataset = XORDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"Dataset size: {len(dataset)}")
print(f"Batch size: 2")
print(f"Number of batches: {len(dataloader)}\n")

# Iterate through batches
print("Sample batches:")
for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    print(f"  X: {X_batch.numpy()}")
    print(f"  y: {y_batch.T.numpy()}")
```
</div>

### Training with DataLoader

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Dataset
class SpiralDataset(Dataset):
    def __init__(self, n_samples=100):
        # Generate spiral data
        theta = torch.linspace(0, 4*np.pi, n_samples)
        r = torch.linspace(0, 1, n_samples)

        # Class 0: outer spiral
        X0 = torch.stack([r*torch.cos(theta), r*torch.sin(theta)], dim=1)
        y0 = torch.zeros(n_samples, 1)

        # Class 1: inner spiral (shifted)
        X1 = torch.stack([r*torch.cos(theta+np.pi), r*torch.sin(theta+np.pi)], dim=1)
        y1 = torch.ones(n_samples, 1)

        self.X = torch.cat([X0, X1])
        self.y = torch.cat([y0, y1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset and dataloader
dataset = SpiralDataset(n_samples=200)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
model = nn.Sequential(
    nn.Linear(2, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop with DataLoader
epochs = 50
for epoch in range(epochs):
    epoch_loss = 0
    for X_batch, y_batch in dataloader:
        # Forward
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

print("\n✓ Training complete with DataLoader!")
```
</div>

## Saving and Loading Models

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn

print("=== Saving and Loading Models ===\n")

# Create and train a model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

print("Original model:")
print(model)
print(f"First layer weight (first 3 values): {model[0].weight[0, :3].data}\n")

# Method 1: Save entire model (not recommended)
torch.save(model, 'model.pth')
print("✓ Saved entire model to 'model.pth'")

loaded_model = torch.load('model.pth')
print("✓ Loaded entire model")
print(f"Loaded first layer weight: {loaded_model[0].weight[0, :3].data}\n")

# Method 2: Save state dict (recommended)
torch.save(model.state_dict(), 'model_state.pth')
print("✓ Saved state dict to 'model_state.pth'")

# Load state dict
new_model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)
new_model.load_state_dict(torch.load('model_state.pth'))
print("✓ Loaded state dict into new model")
print(f"New model first layer weight: {new_model[0].weight[0, :3].data}\n")

# Method 3: Save checkpoint (with optimizer state)
optimizer = torch.optim.Adam(model.parameters())
epoch = 42

checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}

torch.save(checkpoint, 'checkpoint.pth')
print("✓ Saved checkpoint with model + optimizer state\n")

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

print(f"✓ Loaded checkpoint from epoch {epoch}")

print("\nBest Practice:")
print("  • Use state_dict() for model weights")
print("  • Save checkpoints during training")
print("  • Include epoch, optimizer state, and loss")
```
</div>

## Putting It All Together: Complete Training Pipeline

Here's a production-quality training pipeline:

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt

# 1. Define Dataset
class SimpleDataset(Dataset):
    def __init__(self, n_samples=1000):
        X = torch.randn(n_samples, 10)
        y = (X.sum(dim=1, keepdim=True) > 0).float()  # Simple classification
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2. Define Model
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 3. Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()  # Set to training mode
    total_loss = 0

    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# 4. Validation function
def validate(model, dataloader, criterion, device):
    model.eval()  # Set to evaluation mode
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # Don't track gradients
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

            # Calculate accuracy
            predicted = (y_pred > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

# 5. Main training loop
def train_model(model, train_loader, val_loader, criterion, optimizer,
                epochs, device, patience=10):
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    return history

# 6. Run everything
print("=== Complete Training Pipeline ===\n")

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Data
dataset = SimpleDataset(n_samples=1000)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}\n")

# Model
model = Classifier(input_size=10, hidden_sizes=[64, 32], output_size=1)
model = model.to(device)
print(f"Model:\n{model}\n")

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
print("Training...\n")
history = train_model(model, train_loader, val_loader, criterion, optimizer,
                     epochs=100, device=device, patience=10)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2, color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Training complete!")
print(f"Final validation accuracy: {history['val_acc'][-1]*100:.2f}%")
```
</div>

## Summary

### Key Takeaways

!!! success "What You Learned"
    **PyTorch Fundamentals**
    - Tensors: NumPy-like arrays with GPU support
    - Autograd: Automatic differentiation eliminates manual backprop
    - nn.Module: Clean way to define neural networks
    - Optimizers: SGD, Adam, and many more built-in
    - DataLoader: Efficient batching and data loading

    **Advantages Over NumPy**
    - ✅ Automatic gradients via autograd
    - ✅ GPU acceleration (10-100x speedup)
    - ✅ Production-ready, battle-tested code
    - ✅ Rich ecosystem of tools and pre-trained models
    - ✅ Active community and extensive documentation

    **When to Use What**
    - **NumPy**: Learning, understanding internals, small experiments
    - **PyTorch**: Research, production, anything serious
    - **Both**: Use NumPy knowledge to understand PyTorch!

### PyTorch vs NumPy Implementation Comparison

| Aspect | NumPy (Our Implementation) | PyTorch |
|--------|---------------------------|---------|
| **Forward Pass** | Manual matrix operations | `model(X)` |
| **Backward Pass** | Manual gradient computation | `loss.backward()` |
| **Weight Update** | `W -= lr * dW` | `optimizer.step()` |
| **Activation Functions** | Implement from scratch | `nn.ReLU()`, `nn.Sigmoid()`, etc. |
| **Loss Functions** | Implement from scratch | `nn.BCELoss()`, `nn.CrossEntropyLoss()`, etc. |
| **GPU Support** | None | `tensor.to('cuda')` |
| **Speed** | Slow (CPU only) | Fast (CPU) to Very Fast (GPU) |
| **Learning Value** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Production Use** | ❌ | ✅ |

## Practice Problems

**Problem 1**: Modify the XOR network to use different activation functions (Tanh, LeakyReLU). Compare performance.

**Problem 2**: Implement a multi-class classifier for the spiral dataset we created. Use CrossEntropyLoss.

**Problem 3**: Add dropout and batch normalization to a network. Compare training with and without regularization.

**Problem 4**: Implement a custom loss function (e.g., Focal Loss for imbalanced data).

**Problem 5**: Create a custom Dataset class for loading images from a folder.

## Next Steps

You now have a solid foundation in PyTorch! In the next lesson, we'll explore **advanced PyTorch topics**:

- Custom layers and models
- Transfer learning with pre-trained models
- Advanced training techniques
- Model debugging and visualization
- Deploying PyTorch models

[Continue to Lesson 6: PyTorch Advanced](06-pytorch-advanced.md){ .md-button .md-button--primary }

[Complete the Exercises](exercises.md){ .md-button }

[Back to Module Overview](index.md){ .md-button }

---

**Questions or feedback?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues).
