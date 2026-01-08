# Lesson 6: PyTorch Advanced

## Introduction

In the previous lesson, you learned PyTorch fundamentals—tensors, autograd, nn.Module, and basic training loops. You can now build and train neural networks efficiently.

This lesson takes you to the next level. We'll explore **advanced PyTorch techniques** used in production systems and research. You'll learn how to build custom components, use pre-trained models, implement advanced training strategies, and deploy your models.

### What You'll Learn

By the end of this lesson, you will:

- ✅ Implement custom layers and loss functions
- ✅ Use transfer learning with pre-trained models
- ✅ Implement advanced architectures (ResNet, Attention)
- ✅ Master learning rate schedulers and advanced optimizers
- ✅ Use mixed precision training for faster computation
- ✅ Implement gradient accumulation for large batch sizes
- ✅ Debug and profile PyTorch models
- ✅ Use TensorBoard for visualization
- ✅ Export models for deployment (ONNX, TorchScript)
- ✅ Implement data augmentation strategies

**Estimated Time**: 120-150 minutes

!!! warning "Prerequisites"
    This lesson assumes you've completed Lesson 5 (PyTorch Basics) and are comfortable with:

    - PyTorch tensors and operations
    - Building models with nn.Module
    - Basic training loops
    - Autograd and backpropagation

## Custom Layers

Sometimes PyTorch's built-in layers aren't enough. Here's how to create your own:

### Basic Custom Layer

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn

class CustomLinear(nn.Module):
    """
    Custom implementation of a linear layer.

    This is educational—use nn.Linear in practice!
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize weights properly
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x):
        """
        Forward pass: y = xW^T + b

        Parameters:
        -----------
        x : tensor, shape (batch_size, in_features)

        Returns:
        --------
        y : tensor, shape (batch_size, out_features)
        """
        output = torch.matmul(x, self.weight.t())

        if self.bias is not None:
            output += self.bias

        return output

# Test custom layer
print("=== Custom Linear Layer ===\n")

layer = CustomLinear(10, 5)
print(f"Layer: {layer}")
print(f"Weight shape: {layer.weight.shape}")
print(f"Bias shape: {layer.bias.shape}\n")

# Compare with nn.Linear
x = torch.randn(3, 10)
output_custom = layer(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output_custom.shape}")
print(f"Output:\n{output_custom}")
```
</div>

### Advanced Custom Layer: Self-Attention

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Self-Attention mechanism (simplified version of what's in Transformers).

    Computes attention weights and weighted sum of values.
    """

    def __init__(self, embed_dim, num_heads=1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Parameters:
        -----------
        x : tensor, shape (batch_size, seq_len, embed_dim)

        Returns:
        --------
        output : tensor, shape (batch_size, seq_len, embed_dim)
        attention_weights : tensor, shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Linear projections
        Q = self.query(x)  # (batch, seq_len, embed_dim)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now shape: (batch, num_heads, seq_len, head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Shape: (batch, num_heads, seq_len, seq_len)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        attended = torch.matmul(attention_weights, V)
        # Shape: (batch, num_heads, seq_len, head_dim)

        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Output projection
        output = self.out(attended)

        return output, attention_weights.mean(dim=1)  # Average over heads

# Test
print("=== Self-Attention Layer ===\n")

attn = SelfAttention(embed_dim=64, num_heads=4)
x = torch.randn(2, 10, 64)  # (batch=2, seq_len=10, embed_dim=64)

output, weights = attn(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"\nAttention weights (first sample):\n{weights[0]}")
```
</div>

### Custom Activation Function

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn

class Swish(nn.Module):
    """
    Swish activation: f(x) = x * sigmoid(βx)

    When β=1, this is also called SiLU (Sigmoid Linear Unit).
    Often performs better than ReLU in deep networks.
    """

    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class Mish(nn.Module):
    """
    Mish activation: f(x) = x * tanh(softplus(x))

    where softplus(x) = ln(1 + e^x)

    Smooth, non-monotonic, often better than ReLU.
    """

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Compare activations
print("=== Custom Activation Functions ===\n")

x = torch.linspace(-3, 3, 100)

relu = nn.ReLU()
swish = Swish()
mish = Mish()

y_relu = relu(x)
y_swish = swish(x)
y_mish = mish(x)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(x.numpy(), y_relu.detach().numpy(), label='ReLU', linewidth=2)
plt.plot(x.numpy(), y_swish.detach().numpy(), label='Swish', linewidth=2)
plt.plot(x.numpy(), y_mish.detach().numpy(), label='Mish', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Comparing Activation Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Key differences:")
print("  • ReLU: Simple, fast, can die (zero gradient for x<0)")
print("  • Swish: Smooth, learnable β, better gradients")
print("  • Mish: Smooth, non-monotonic, state-of-the-art for some tasks")
```
</div>

## Custom Loss Functions

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)

    where p_t is the model's estimated probability for the correct class.

    Parameters:
    -----------
    alpha : float
        Weighting factor for positive class
    gamma : float
        Focusing parameter (higher = focus more on hard examples)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Parameters:
        -----------
        inputs : tensor, shape (batch_size, num_classes)
            Raw logits (before softmax)
        targets : tensor, shape (batch_size,)
            Class labels (integers)
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)

        # Get probability of correct class
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        return focal_loss.mean()

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.

    Measures overlap between predicted and ground truth masks.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Parameters:
        -----------
        inputs : tensor, shape (batch_size, ...)
            Predicted probabilities [0, 1]
        targets : tensor, shape (batch_size, ...)
            Ground truth binary mask {0, 1}
        """
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute intersection and union
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

# Test Focal Loss
print("=== Custom Loss Functions ===\n")

# Simulate class imbalance scenario
inputs = torch.randn(100, 3)  # 100 samples, 3 classes
targets = torch.cat([
    torch.zeros(90, dtype=torch.long),  # 90 samples of class 0
    torch.ones(8, dtype=torch.long),     # 8 samples of class 1
    torch.full((2,), 2, dtype=torch.long)  # 2 samples of class 2
])

# Standard Cross-Entropy
ce_loss = F.cross_entropy(inputs, targets)
print(f"Cross-Entropy Loss: {ce_loss.item():.4f}")

# Focal Loss (focuses on hard/rare examples)
focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
focal_loss = focal_loss_fn(inputs, targets)
print(f"Focal Loss: {focal_loss.item():.4f}")

print("\nFocal Loss benefits:")
print("  • Down-weights easy examples (high confidence)")
print("  • Focuses on hard examples (low confidence)")
print("  • Helps with class imbalance")
```
</div>

## Transfer Learning

Transfer learning is **one of the most powerful techniques** in deep learning. Use knowledge from pre-trained models!

### Loading Pre-trained Models

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torchvision.models as models

print("=== Transfer Learning with Pre-trained Models ===\n")

# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)

print("Pre-trained ResNet18 architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# See the final layer
print(f"\nOriginal final layer (for ImageNet 1000 classes):")
print(model.fc)

# Modify for your task (e.g., 10 classes)
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

print(f"\nModified final layer (for {num_classes} classes):")
print(model.fc)

print("\n✓ Model ready for transfer learning!")
```
</div>

### Fine-tuning Strategies

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torchvision.models as models

print("=== Fine-tuning Strategies ===\n")

model = models.resnet18(pretrained=True)
num_classes = 10

# Strategy 1: Freeze all layers, train only final layer
print("Strategy 1: Feature Extraction")
print("  • Freeze all pre-trained layers")
print("  • Train only the new final layer")
print("  • Fast, works well with small datasets\n")

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Replace final layer (this will be trainable)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Check which parameters are trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable_params:,} / {total_params:,} parameters\n")

# Strategy 2: Fine-tune last few layers
print("Strategy 2: Partial Fine-tuning")
print("  • Freeze early layers (low-level features)")
print("  • Fine-tune later layers (high-level features)")
print("  • Train new final layer")
print("  • Medium speed, works well with medium datasets\n")

model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer4 (last residual block)
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable_params:,} / {total_params:,} parameters\n")

# Strategy 3: Fine-tune entire network
print("Strategy 3: Full Fine-tuning")
print("  • All layers trainable")
print("  • Use lower learning rate for pre-trained layers")
print("  • Use higher learning rate for new layers")
print("  • Slow, needs large dataset\n")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# All parameters trainable
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable_params:,} / {total_params:,} parameters\n")

# Use differential learning rates
optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-5},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])

print("Differential learning rates:")
print("  • Early layers: 1e-5 (small changes)")
print("  • Middle layers: 1e-4")
print("  • Final layer: 1e-3 (large changes)")
```
</div>

### Complete Transfer Learning Example

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Dummy dataset for demonstration
class DummyImageDataset(Dataset):
    def __init__(self, n_samples=100, n_classes=10):
        self.n_samples = n_samples
        self.n_classes = n_classes

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Random 224x224 RGB image
        image = torch.randn(3, 224, 224)
        label = torch.randint(0, self.n_classes, (1,)).item()
        return image, label

print("=== Complete Transfer Learning Pipeline ===\n")

# 1. Load pre-trained model
model = models.resnet18(pretrained=True)

# 2. Modify for our task
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 3. Choose fine-tuning strategy (freeze early layers)
for name, param in model.named_parameters():
    if 'fc' not in name:  # Freeze all except final layer
        param.requires_grad = False

print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 4. Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# 5. Create data loaders
train_dataset = DummyImageDataset(n_samples=100, n_classes=10)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 6. Training loop
print("\nTraining...")
model.train()
for epoch in range(3):
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/3, Loss: {avg_loss:.4f}")

print("\n✓ Transfer learning complete!")
print("\nKey takeaways:")
print("  • Start with pre-trained weights (ImageNet, etc.)")
print("  • Freeze early layers for small datasets")
print("  • Use lower learning rates for fine-tuning")
print("  • Can achieve great results with little data!")
```
</div>

## Learning Rate Schedulers

Learning rate is often the **most important hyperparameter**. Schedulers adjust it during training:

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
import matplotlib.pyplot as plt

print("=== Learning Rate Schedulers ===\n")

# Dummy model
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Test different schedulers
schedulers = {
    'StepLR': StepLR(optimizer, step_size=30, gamma=0.1),
    'ExponentialLR': ExponentialLR(optimizer, gamma=0.95),
    'CosineAnnealingLR': CosineAnnealingLR(optimizer, T_max=100),
    'ReduceLROnPlateau': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
}

# Simulate training
epochs = 100
lr_history = {name: [] for name in schedulers.keys()}

for name, scheduler in schedulers.items():
    # Reset optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    if name == 'StepLR':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif name == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=0.95)
    elif name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
    elif name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    for epoch in range(epochs):
        lr_history[name].append(optimizer.param_groups[0]['lr'])

        # Simulate training step
        if name == 'ReduceLROnPlateau':
            # Simulate validation loss
            val_loss = 1.0 / (epoch + 1) + np.random.rand() * 0.1
            scheduler.step(val_loss)
        else:
            scheduler.step()

# Plot learning rate schedules
plt.figure(figsize=(12, 6))
for name, lrs in lr_history.items():
    plt.plot(lrs, label=name, linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedules')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()

print("Scheduler descriptions:\n")
print("1. StepLR:")
print("   • Reduce LR by factor every N epochs")
print("   • Simple, predictable")
print("   • lr_new = lr * gamma every step_size epochs\n")

print("2. ExponentialLR:")
print("   • Exponential decay")
print("   • lr_new = lr * gamma every epoch\n")

print("3. CosineAnnealingLR:")
print("   • Cosine decay from initial to 0")
print("   • Smooth, gradual reduction")
print("   • Often works well with warm restarts\n")

print("4. ReduceLROnPlateau:")
print("   • Reduce when validation metric plateaus")
print("   • Adaptive, responds to training")
print("   • Reduce by factor when no improvement for patience epochs\n")

print("Best practices:")
print("  • Start with CosineAnnealingLR or ReduceLROnPlateau")
print("  • Combine with warmup (increase LR at start)")
print("  • Tune based on validation loss, not training loss")
```
</div>

### Learning Rate Warmup

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def warmup_lr_scheduler(optimizer, warmup_epochs, total_epochs, max_lr=0.1):
    """
    Learning rate scheduler with warmup.

    Linearly increases LR from 0 to max_lr over warmup_epochs,
    then decreases with cosine annealing.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup: linear increase
            return epoch / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Demo
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

scheduler = warmup_lr_scheduler(optimizer, warmup_epochs=10, total_epochs=100)

lrs = []
for epoch in range(100):
    lrs.append(optimizer.param_groups[0]['lr'])
    scheduler.step()

plt.figure(figsize=(10, 6))
plt.plot(lrs, linewidth=2)
plt.axvline(x=10, color='r', linestyle='--', label='Warmup end')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule with Warmup')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Learning Rate Warmup:")
print("  • Prevents early instability")
print("  • Gradually increases LR from 0 to max")
print("  • Common in Transformers (BERT, GPT)")
print("  • Typically 5-10% of total epochs")
```
</div>

## Mixed Precision Training

Train faster and use less memory with automatic mixed precision (AMP):

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import time

print("=== Mixed Precision Training ===\n")

# Create model and data
model = nn.Sequential(
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Dummy data
X = torch.randn(128, 1000).to(device)
y = torch.randint(0, 10, (128,)).to(device)

# Traditional training (FP32)
print("Traditional Training (FP32)...")
start = time.time()

for _ in range(10):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

fp32_time = time.time() - start
print(f"FP32 time: {fp32_time:.3f}s\n")

# Mixed precision training (FP16 + FP32)
if torch.cuda.is_available():
    print("Mixed Precision Training (FP16 + FP32)...")

    model = nn.Sequential(
        nn.Linear(1000, 500),
        nn.ReLU(),
        nn.Linear(500, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    start = time.time()

    for _ in range(10):
        optimizer.zero_grad()

        # Automatic mixed precision
        with autocast():
            output = model(X)
            loss = criterion(output, y)

        # Scale loss and backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    amp_time = time.time() - start
    print(f"AMP time: {amp_time:.3f}s")
    print(f"Speedup: {fp32_time/amp_time:.2f}x\n")

    print("How it works:")
    print("  • Most operations in FP16 (faster, less memory)")
    print("  • Loss scaling prevents underflow")
    print("  • Critical operations (loss, normalization) in FP32")
    print("  • Automatic! Just wrap forward pass with autocast()")
else:
    print("Mixed precision requires CUDA")

print("\nBenefits:")
print("  • 2-3x speedup on modern GPUs")
print("  • 50% less memory usage")
print("  • Minimal code changes")
print("  • No loss in accuracy (usually)")
```
</div>

## Gradient Accumulation

Simulate large batch sizes on limited GPU memory:

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torch.optim as optim

print("=== Gradient Accumulation ===\n")

model = nn.Linear(100, 10)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Problem: Want batch_size=128, but only have memory for batch_size=32
# Solution: Accumulate gradients over 4 mini-batches

accumulation_steps = 4
effective_batch_size = 32 * accumulation_steps  # 128

print(f"Effective batch size: {effective_batch_size}")
print(f"Accumulation steps: {accumulation_steps}\n")

# Training loop with gradient accumulation
model.train()
optimizer.zero_grad()

for step in range(12):  # 12 mini-batches
    # Mini-batch data
    X_batch = torch.randn(32, 100)
    y_batch = torch.randint(0, 10, (32,))

    # Forward pass
    output = model(X_batch)
    loss = criterion(output, y_batch)

    # Scale loss by accumulation steps
    loss = loss / accumulation_steps

    # Backward pass (accumulate gradients)
    loss.backward()

    # Update weights every accumulation_steps
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        print(f"Step {step+1}: Updated weights (accumulated {accumulation_steps} batches)")

print("\nGradient Accumulation:")
print("  • Simulates large batch sizes")
print("  • Saves GPU memory")
print("  • Divide loss by accumulation_steps")
print("  • Update weights after N mini-batches")
print("  • Gradient = sum of mini-batch gradients")
```
</div>

## Model Debugging and Profiling

### Debugging Tools

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn

print("=== Model Debugging ===\n")

class DebugModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        print(f"Input shape: {x.shape}")

        x = self.fc1(x)
        print(f"After fc1: {x.shape}, mean={x.mean().item():.3f}, std={x.std().item():.3f}")

        x = torch.relu(x)
        print(f"After ReLU: {x.shape}, mean={x.mean().item():.3f}, std={x.std().item():.3f}")

        x = self.fc2(x)
        print(f"After fc2: {x.shape}, mean={x.mean().item():.3f}, std={x.std().item():.3f}")

        return x

model = DebugModel()
X = torch.randn(5, 10)

print("Forward pass with debug prints:\n")
output = model(X)

print("\n" + "="*60)
print("\nDebugging Checklist:")
print("  ✓ Check tensor shapes at each layer")
print("  ✓ Monitor activation statistics (mean, std)")
print("  ✓ Look for NaN or Inf values")
print("  ✓ Verify gradients are flowing (check .grad)")
print("  ✓ Use torch.autograd.set_detect_anomaly(True)")
```
</div>

### Gradient Checking

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

print("=== Gradient Checking ===\n")

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

X = torch.randn(3, 10, requires_grad=True)
y = torch.randn(3, 1)

# Forward and backward
output = model(X)
loss = nn.MSELoss()(output, y)
loss.backward()

# Check gradients
print("Gradients:")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_mean = param.grad.mean().item()
        grad_std = param.grad.std().item()
        grad_max = param.grad.abs().max().item()

        print(f"{name}:")
        print(f"  Mean: {grad_mean:.6f}")
        print(f"  Std: {grad_std:.6f}")
        print(f"  Max: {grad_max:.6f}")

        # Check for issues
        if torch.isnan(param.grad).any():
            print(f"  ⚠️ WARNING: NaN gradients!")
        if torch.isinf(param.grad).any():
            print(f"  ⚠️ WARNING: Inf gradients!")
        if grad_max > 100:
            print(f"  ⚠️ WARNING: Very large gradients (exploding)!")
        if grad_max < 1e-7:
            print(f"  ⚠️ WARNING: Very small gradients (vanishing)!")

print("\nGradient health indicators:")
print("  • Good: gradients in range [1e-4, 10]")
print("  • Exploding: gradients > 100")
print("  • Vanishing: gradients < 1e-7")
print("  • Use gradient clipping if exploding")
```
</div>

## TensorBoard Integration

Visualize training with TensorBoard:

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

print("=== TensorBoard Integration ===\n")

# Create TensorBoard writer
writer = SummaryWriter('runs/experiment_1')

# Simple model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
print("Training...")
for epoch in range(100):
    # Dummy data
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)

    # Forward
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)

    # Backward
    loss.backward()
    optimizer.step()

    # Log to TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)

    # Log learning rate
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

    # Log gradients and weights every 10 epochs
    if epoch % 10 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param.data, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

# Log model graph
writer.add_graph(model, X)

# Close writer
writer.close()

print("✓ Training complete!")
print("\nTo view TensorBoard:")
print("  1. Run: tensorboard --logdir=runs")
print("  2. Open browser to: http://localhost:6006")
print("\nLogged:")
print("  • Training loss")
print("  • Learning rate")
print("  • Weight histograms")
print("  • Gradient histograms")
print("  • Model graph")
```
</div>

## Model Export and Deployment

### TorchScript (JIT Compilation)

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn

print("=== TorchScript Export ===\n")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return torch.relu(self.fc(x))

# Create model
model = SimpleModel()
model.eval()

# Method 1: Tracing (records operations)
print("1. Tracing (trace operations on example input):")
example_input = torch.randn(1, 10)
traced_model = torch.jit.trace(model, example_input)

# Save
torch.jit.save(traced_model, 'model_traced.pt')
print("   ✓ Saved traced model\n")

# Method 2: Scripting (converts Python to TorchScript)
print("2. Scripting (converts Python code directly):")
scripted_model = torch.jit.script(model)

# Save
torch.jit.save(scripted_model, 'model_scripted.pt')
print("   ✓ Saved scripted model\n")

# Load and use
loaded_model = torch.jit.load('model_traced.pt')
output = loaded_model(example_input)
print(f"Loaded model output: {output.shape}\n")

print("Tracing vs Scripting:")
print("  • Tracing: Faster, simpler, but can't handle control flow")
print("  • Scripting: Handles if/else/loops, more complex")
print("\nBenefits:")
print("  • No Python dependency")
print("  • Optimized for inference")
print("  • Deploy in C++")
```
</div>

### ONNX Export

<div class="python-interactive" markdown="1">
```python
import torch
import torch.nn as nn

print("=== ONNX Export ===\n")

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
model.eval()

# Export to ONNX
dummy_input = torch.randn(1, 10)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✓ Exported to ONNX format\n")

print("ONNX Benefits:")
print("  • Framework-agnostic (run in TensorFlow, Caffe, etc.)")
print("  • Optimize for different hardware")
print("  • Deploy on edge devices")
print("  • Use with ONNX Runtime for fast inference")
```
</div>

## Data Augmentation

<div class="python-interactive" markdown="1">
```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

print("=== Data Augmentation ===\n")

# Create transform pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

print("Training Augmentation:")
print("  • RandomResizedCrop: Random crop and resize")
print("  • RandomHorizontalFlip: Flip 50% of images")
print("  • RandomRotation: Rotate up to 15°")
print("  • ColorJitter: Vary brightness, contrast, saturation")
print("  • Normalize: ImageNet statistics\n")

print("Validation/Test (No Augmentation):")
print("  • Resize to 256x256")
print("  • CenterCrop to 224x224")
print("  • Normalize: Same statistics\n")

print("Why augment?")
print("  • Increases effective dataset size")
print("  • Improves generalization")
print("  • Reduces overfitting")
print("  • Makes model robust to variations")
```
</div>

## Summary

### Key Takeaways

!!! success "What You Learned"
    **Advanced Techniques**
    - Custom layers, activations, and loss functions
    - Transfer learning with pre-trained models
    - Learning rate scheduling and warmup
    - Mixed precision training (2-3x speedup!)
    - Gradient accumulation for large batches

    **Production Skills**
    - Model debugging and profiling
    - TensorBoard visualization
    - Model export (TorchScript, ONNX)
    - Data augmentation strategies

    **Best Practices**
    - Always use transfer learning when possible
    - Schedule your learning rate
    - Use mixed precision on modern GPUs
    - Monitor gradients during training
    - Augment your data

### Advanced PyTorch Patterns

| Pattern | When to Use |
|---------|-------------|
| **Transfer Learning** | Always! (unless very novel task) |
| **Custom Layers** | Implementing new architectures (Transformers, etc.) |
| **Custom Loss** | Domain-specific objectives (imbalance, segmentation) |
| **LR Scheduling** | All serious training |
| **Mixed Precision** | Training on modern GPUs (V100, A100) |
| **Gradient Accumulation** | Large models, limited GPU memory |
| **TensorBoard** | Understanding training dynamics |
| **Model Export** | Deploying to production |

## Practice Problems

**Problem 1**: Implement a Residual Block (ResNet building block) as a custom layer.

**Problem 2**: Fine-tune a pre-trained ResNet50 on a custom dataset. Compare feature extraction vs full fine-tuning.

**Problem 3**: Implement Cutout data augmentation (randomly mask parts of images).

**Problem 4**: Create a custom loss function combining cross-entropy and center loss for face recognition.

**Problem 5**: Use TensorBoard to debug why a model isn't learning. Log activations, gradients, and learning rate.

**Problem 6**: Export a model to ONNX and run inference with ONNX Runtime.

## Next Steps

Congratulations! You've completed Module 4: Neural Networks. You now understand:

- ✅ The theory: perceptrons, feedforward networks, backpropagation
- ✅ The practice: NumPy implementation from scratch
- ✅ Modern tools: PyTorch basics and advanced techniques
- ✅ Production skills: debugging, visualization, deployment

**Continue your journey:**

- **Module 5**: Convolutional Neural Networks (CNNs) for computer vision
- **Module 6**: Recurrent Neural Networks (RNNs) for sequences
- **Module 7**: Transformers and Attention Mechanisms
- **Advanced Topics**: GANs, VAEs, Reinforcement Learning

[Complete the Exercises](exercises.md){ .md-button .md-button--primary }

[Back to Module Overview](index.md){ .md-button }

---

**Questions or feedback?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues).
