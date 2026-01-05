# Module 4: Neural Networks

## Overview

Neural networks are the foundation of modern deep learning. In this module, you'll build neural networks from scratch using NumPy to understand how they work, then use PyTorch to implement production-ready deep learning models.

## Why Neural Networks Matter

Deep learning powers today's AI revolution:
- **Computer Vision**: Image classification, object detection, segmentation
- **Natural Language Processing**: Translation, chatbots, text generation
- **Speech Recognition**: Voice assistants, transcription
- **Generative AI**: DALL-E, ChatGPT, Stable Diffusion
- **Reinforcement Learning**: Game AI, robotics, autonomous systems

## Learning Objectives

- ✅ Understand the perceptron and activation functions
- ✅ Build feedforward neural networks
- ✅ Master backpropagation algorithm
- ✅ Implement neural networks from scratch with NumPy
- ✅ Use PyTorch for modern deep learning
- ✅ Apply regularization and optimization techniques
- ✅ Train and evaluate neural network models

## Prerequisites

- Modules 1-3 completed
- Strong understanding of matrix operations
- Gradient descent and backpropagation concepts
- Python and NumPy proficiency

## Module Structure

### Lesson 1: The Perceptron (45 min)
- Biological neuron inspiration
- Perceptron model and algorithm
- Activation functions: sigmoid, tanh, ReLU
- Linear separability and limitations

[Start Lesson 1](01-perceptron.md){ .md-button .md-button--primary }

### Lesson 2: Feedforward Networks (60 min)
- Multi-layer perceptron (MLP) architecture
- Hidden layers and network depth
- Forward propagation mathematics
- Universal approximation theorem

[Start Lesson 2](02-feedforward-networks.md){ .md-button }

### Lesson 3: Backpropagation (75 min)
- Chain rule and gradient computation
- Backpropagation algorithm step-by-step
- Computing gradients layer by layer
- Weight updates and learning

[Start Lesson 3](03-backpropagation.md){ .md-button }

### Lesson 4: NumPy Implementation (90 min)
- Complete neural network from scratch
- Modular design: layers, activations, losses
- Training loop and mini-batch processing
- Debugging neural networks

[Start Lesson 4](04-numpy-implementation.md){ .md-button }

### Lesson 5: PyTorch Basics (75 min)
- Introduction to PyTorch
- Tensors and autograd
- Building models with nn.Module
- Optimizers and loss functions
- Training and evaluation loops

[Start Lesson 5](05-pytorch-basics.md){ .md-button }

### Lesson 6: PyTorch Advanced (90 min)
- Custom datasets and DataLoaders
- Model checkpointing and saving
- Learning rate scheduling
- Regularization: dropout, batch normalization
- Transfer learning basics

[Start Lesson 6](06-pytorch-advanced.md){ .md-button }

### Exercises (8-10 hours)
- Perceptron implementation from scratch
- Multi-layer network with NumPy
- PyTorch neural network for MNIST
- Classification with regularization
- Advanced PyTorch techniques

[View Exercises](exercises.md){ .md-button }

## Learning Path

```
1. Understand single neuron (Perceptron)
   ↓
2. Build multi-layer networks
   ↓
3. Learn backpropagation math
   ↓
4. Implement from scratch (NumPy)
   ↓
5. Use modern framework (PyTorch)
   ↓
6. Apply advanced techniques
```

## Key Concepts

| Concept | Purpose | Implementation |
|---------|---------|----------------|
| **Activation Functions** | Introduce non-linearity | ReLU, sigmoid, tanh |
| **Backpropagation** | Compute gradients | Chain rule recursively |
| **Optimization** | Update weights | SGD, Adam, RMSprop |
| **Regularization** | Prevent overfitting | Dropout, batch norm, L2 |

## Datasets

- **MNIST**: Handwritten digit recognition (10 classes, 28x28 images)
- **Fashion-MNIST**: Clothing classification (10 classes)
- **CIFAR-10**: Natural images (10 classes, 32x32 color)
- **Custom datasets**: Build your own data loaders

## Tools

- **NumPy**: From-scratch implementation
- **PyTorch**: Modern deep learning framework
- **Matplotlib**: Visualize learning and predictions
- **torchvision**: Pre-built datasets and models

## Two-Part Approach

### Part 1: NumPy Implementation (Deep Understanding)
Build everything from scratch to understand:
- How forward propagation works
- How backpropagation computes gradients
- How weights are updated
- Why deep learning works

### Part 2: PyTorch (Practical Skills)
Use industry-standard tools for:
- Automatic differentiation
- GPU acceleration
- Scalable training
- Production deployment

## Tips for Success

!!! tip "Start Simple"
    Begin with small networks and simple datasets. MNIST before CIFAR-10!

!!! tip "Check Gradients"
    Use gradient checking to verify your backpropagation implementation.

!!! tip "Monitor Training"
    Always plot training and validation loss. Watch for overfitting!

!!! tip "Use GPU"
    PyTorch makes GPU usage easy. Even free Colab GPUs help!

!!! warning "Common Mistakes"
    - Forgetting to zero gradients in PyTorch
    - Wrong tensor dimensions
    - Not shuffling data
    - Learning rate too high/low

## Estimated Time: 16-20 hours

## Real-World Applications

After this module:
- **Image Classification**: Build custom classifiers
- **Time Series**: Stock prediction, forecasting
- **NLP Tasks**: Sentiment analysis, text classification
- **Anomaly Detection**: Fraud detection, monitoring
- **Embeddings**: Feature learning for other tasks

## Beyond This Module

Paths to explore further:
- **Convolutional Neural Networks (CNNs)**: Computer vision
- **Recurrent Neural Networks (RNNs)**: Sequential data
- **Transformers**: State-of-the-art NLP
- **GANs**: Generative models
- **Reinforcement Learning**: Decision making

## Ready to Start?

Let's begin by understanding the perceptron - the building block of neural networks!

[Start Lesson 1: The Perceptron](01-perceptron.md){ .md-button .md-button--primary }

[Or review Module 3 first](../module3-trees/index.md){ .md-button }

---

**Questions?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues).
