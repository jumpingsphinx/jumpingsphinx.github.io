# Module 4 Exercises: Neural Networks

## Overview

Time to build neural networks from the ground up! These exercises will guide you through implementing perceptrons, feedforward networks, and backpropagation from scratch, then transition to using PyTorch for modern deep learning. By the end, you'll understand both the fundamentals and the practical tools used in production.

## Before You Start

### Setup

1. Ensure you have the required packages:
   ```bash
   pip install numpy matplotlib scikit-learn torch torchvision
   ```

2. Open Jupyter Lab:
   ```bash
   jupyter lab
   ```

3. Navigate to `notebooks/module4-neural-networks/`

4. Start with `exercise1-perceptron.ipynb`

### Exercise Format

Each exercise includes:
- **Learning objectives**: What you'll practice
- **Background**: Quick concept review
- **Tasks**: Step-by-step implementation
- **Hints**: Help when you're stuck
- **Visualizations**: See your networks in action
- **Validation**: Test your implementation
- **Reflection questions**: Deepen your understanding

### Tips for Success

!!! tip "Best Practices"
    - **Implement from scratch first**: Understanding fundamentals is crucial
    - **Visualize decision boundaries**: See what your network learns
    - **Check gradients numerically**: Validate your backprop implementation
    - **Start with simple problems**: XOR before MNIST
    - **Debug with small networks**: 2-3 neurons before hundreds
    - **Print shapes everywhere**: Dimension mismatches are common

## Exercise 1: The Perceptron

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module4-neural-networks/exercise1-perceptron.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module4-neural-networks/solutions/solution_exercise1-perceptron.ipynb" target="_blank">Open Solution in Colab</a>

**Time:** 1.5-2 hours

### What You'll Learn

- Implement a single perceptron from scratch
- Understand activation functions (sigmoid, tanh, ReLU)
- Train with the perceptron learning algorithm
- Visualize decision boundaries
- Recognize linear separability limitations
- Understand the XOR problem

### Topics Covered

- Perceptron forward pass: weighted sum + activation
- Binary classification with perceptrons
- Perceptron learning rule
- Activation functions and their derivatives
- Decision boundary visualization
- Linear separability
- The XOR problem (why perceptrons fail)

### Key Concepts

```python
# Forward pass: y = σ(w^T x + b)
# Update rule: w = w + η(y_true - y_pred)x
# Activation functions: sigmoid, tanh, ReLU
```

### Datasets Used

- Linearly separable synthetic data (2D)
- AND, OR logic gates
- XOR (to show limitation)

### What You'll Build

1. **Perceptron class**: Complete implementation from scratch
2. **Activation function library**: Sigmoid, tanh, ReLU with derivatives
3. **Decision boundary visualizer**: Plot classification regions
4. **XOR demonstrator**: Show perceptron's fundamental limitation

---

## Exercise 2: Feedforward Neural Networks

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module4-neural-networks/exercise2-feedforward-networks.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module4-neural-networks/solutions/solution_exercise2-feedforward-networks.ipynb" target="_blank">Open Solution in Colab</a>

**Time:** 2-3 hours

### What You'll Learn

- Build multi-layer perceptrons (MLPs)
- Implement forward propagation through multiple layers
- Understand hidden layers and network depth
- Solve XOR with a 2-layer network
- Work with different architectures
- Visualize what hidden layers learn

### Topics Covered

- Multi-layer perceptron architecture
- Layer-by-layer forward propagation
- Matrix formulation of forward pass
- Hidden layer representations
- Network depth and width
- Universal approximation theorem
- XOR solution with neural networks

### Key Concepts

```python
# Layer 1: z[1] = W[1]x + b[1], a[1] = σ(z[1])
# Layer 2: z[2] = W[2]a[1] + b[2], a[2] = σ(z[2])
# Output: ŷ = a[L]
```

### What You'll Build

1. **Layer class**: Modular layer implementation
2. **Network class**: Multi-layer neural network
3. **XOR solver**: Solve XOR with a 2-layer network
4. **Hidden layer visualizer**: See what hidden neurons learn
5. **Architecture explorer**: Test different network shapes

---

## Exercise 3: Backpropagation

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module4-neural-networks/exercise3-backpropagation.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module4-neural-networks/solutions/solution_exercise3-backpropagation.ipynb" target="_blank">Open Solution in Colab</a>

**Time:** 3-4 hours

### What You'll Learn

- Derive backpropagation equations from first principles
- Implement backward pass layer-by-layer
- Compute gradients using the chain rule
- Perform gradient checking to validate implementation
- Update weights with gradient descent
- Train a neural network end-to-end

### Topics Covered

- Chain rule for gradient computation
- Backward pass through layers
- Gradient computation for weights and biases
- Loss function gradients (MSE, cross-entropy)
- Gradient descent weight updates
- Gradient checking for debugging
- Training loop implementation

### Key Concepts

```python
# Backward pass:
# dL/dW[l] = dL/da[l] * da[l]/dz[l] * dz[l]/dW[l]
# Chain rule: dL/dW[l] = δ[l] * a[l-1]^T
# Update: W[l] = W[l] - η * dL/dW[l]
```

### What You'll Build

1. **Backpropagation engine**: Complete backward pass implementation
2. **Gradient checker**: Numerical gradient validation
3. **Training loop**: Full training pipeline
4. **Loss plotter**: Visualize convergence
5. **Complete neural network library**: Forward + backward pass

---

## Exercise 4: NumPy Neural Network Implementation

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module4-neural-networks/exercise4-numpy-implementation.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module4-neural-networks/solutions/solution_exercise4-numpy-implementation.ipynb" target="_blank">Open Solution in Colab</a>

**Time:** 3-4 hours

### What You'll Learn

- Build a complete, production-quality neural network library
- Implement multiple activation functions
- Support multiple loss functions
- Add mini-batch training
- Implement different optimizers (SGD, momentum)
- Train on real datasets (MNIST, fashion MNIST)
- Evaluate and visualize results

### Topics Covered

- Modular neural network architecture
- Mini-batch gradient descent
- Data loading and preprocessing
- Batch normalization concepts
- Dropout for regularization
- Learning rate schedules
- Early stopping
- Model evaluation and validation

### Datasets Used

- MNIST (handwritten digits)
- Fashion MNIST (clothing classification)
- Iris dataset (simple classification)

### What You'll Build

1. **Complete NN library**: Production-ready NumPy implementation
2. **MNIST classifier**: Digit recognition neural network
3. **Hyperparameter tuner**: Grid search for best parameters
4. **Performance analyzer**: Confusion matrix, accuracy curves
5. **Visualization suite**: Weights, activations, gradients

---

## Exercise 5: PyTorch Basics

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module4-neural-networks/exercise5-pytorch-basics.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module4-neural-networks/solutions/solution_exercise5-pytorch-basics.ipynb" target="_blank">Open Solution in Colab</a>

**Time:** 2-3 hours

### What You'll Learn

- Create and manipulate PyTorch tensors
- Build neural networks with `nn.Module`
- Use automatic differentiation with `autograd`
- Train models with PyTorch optimizers
- Load data with `DataLoader`
- Transfer models to GPU
- Save and load trained models

### Topics Covered

- PyTorch tensor operations
- Building models with `nn.Module` and `nn.Sequential`
- Loss functions: `nn.MSELoss`, `nn.CrossEntropyLoss`
- Optimizers: `optim.SGD`, `optim.Adam`
- Data loading with `DataLoader`
- GPU acceleration with `.to(device)`
- Model saving and loading
- Comparison with NumPy implementation

### What You'll Build

1. **PyTorch neural network**: Reimplement your NumPy network
2. **MNIST classifier (PyTorch version)**: Train on MNIST
3. **GPU accelerator**: Train on GPU if available
4. **Model persistence**: Save/load trained models
5. **Benchmark comparison**: NumPy vs PyTorch speed

---

## Exercise 6: PyTorch Advanced Topics

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module4-neural-networks/exercise6-pytorch-advanced.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<a href="https://colab.research.google.com/github/jumpingsphinx/jumpingsphinx.github.io/blob/main/notebooks/module4-neural-networks/solutions/solution_exercise6-pytorch-advanced.ipynb" target="_blank">Open Solution in Colab</a>

**Time:** 3-4 hours

### What You'll Learn

- Implement custom loss functions
- Create custom layers and modules
- Use learning rate schedulers
- Apply data augmentation
- Implement callbacks (early stopping, checkpointing)
- Use TensorBoard for visualization
- Fine-tune pre-trained models

### Topics Covered

- Custom `nn.Module` layers
- Custom loss functions
- Learning rate scheduling
- Data augmentation techniques
- Model checkpointing
- TensorBoard integration
- Transfer learning basics
- Advanced optimization techniques

### What You'll Build

1. **Custom layers**: Implement your own layer types
2. **Advanced MNIST classifier**: With all the bells and whistles
3. **Transfer learning model**: Use pre-trained networks
4. **Training dashboard**: TensorBoard visualization
5. **Experiment tracker**: Compare multiple model configurations

---

## Solutions

After completing each exercise, review the solutions to:
- Compare your approach with the reference implementation
- Learn alternative methods
- Understand best practices
- Debug any issues

!!! tip "Learning with Solutions"
    Compare your approach with the reference solution. Did you implement it differently? Is one way more efficient or readable?
    
    If you get stuck, look at the solution for a specific part, then try to continue without looking.

## Assessment Questions

After completing all exercises, test your understanding:

### Conceptual

1. **Why can't a single perceptron solve XOR?**
   - Think about linear separability and decision boundaries

2. **What does each layer in a neural network learn?**
   - Consider feature hierarchy and abstraction

3. **Why is backpropagation efficient?**
   - Compare to computing gradients individually

4. **What's the vanishing gradient problem?**
   - How does it relate to activation functions and depth?

5. **Why use ReLU instead of sigmoid?**
   - Consider gradient flow and computational efficiency

### Practical

1. **How do you choose network architecture?**
   - Number of layers, neurons per layer, activation functions

2. **What batch size should you use?**
   - Consider memory, convergence, and generalization

3. **How do you diagnose overfitting in neural networks?**
   - What metrics and techniques help?

4. **When should you use dropout vs L2 regularization?**
   - What are the tradeoffs?

5. **How do you choose a learning rate?**
   - What strategies and schedules work best?

### Debugging

1. **Loss is NaN. What's wrong?**
   - Likely exploding gradients or numerical instability

2. **Training accuracy is perfect but test accuracy is poor. Why?**
   - Classic overfitting - need regularization

3. **Gradients are all zero. What happened?**
   - Dying ReLU or saturated sigmoid/tanh

4. **Network predicts same class for everything. Why?**
   - Class imbalance, bad initialization, or learning rate issues

## Reflection Questions

Think deeply about what you've learned:

1. How does understanding backpropagation help you debug neural networks?
2. What surprised you most about implementing neural networks from scratch?
3. How does PyTorch make deep learning easier? What does it abstract away?
4. When would you implement something from scratch vs use PyTorch?
5. How do neural networks relate to the regression models you learned earlier?

## Common Mistakes to Avoid

!!! warning "Watch Out For"
    - **Wrong matrix dimensions**: Always check shapes in forward/backward pass
    - **Forgetting bias terms**: Every layer needs a bias
    - **Not checking gradients**: Use gradient checking to validate backprop
    - **Bad weight initialization**: Random initialization matters!
    - **Wrong loss function**: MSE for regression, cross-entropy for classification
    - **Not normalizing inputs**: Neural networks are sensitive to input scale
    - **Learning rate too high/low**: Start with 0.01 and adjust
    - **Training on unnormalized data**: Always normalize features

## Going Further

### Challenge Exercises

Want more practice? Try these:

1. **Implement batch normalization**: Add BatchNorm layers from scratch
2. **Build a convolutional layer**: Understand CNNs by implementing convolution
3. **Create an LSTM**: Implement recurrent neural networks
4. **Implement Adam optimizer**: Modern optimization from scratch
5. **Build an autoencoder**: Unsupervised learning with neural networks

### Real-World Projects

Apply your skills:

1. **Image classification**: Train a classifier on CIFAR-10 or ImageNet subset
2. **Text sentiment analysis**: Use neural networks for NLP
3. **Time series prediction**: Stock prices or weather forecasting
4. **Anomaly detection**: Autoencoder-based outlier detection
5. **Style transfer**: Neural artistic style transfer
6. **Generative models**: Simple GAN or VAE

## Next Steps

Congratulations on completing Module 4! You now understand the foundations of neural networks and modern deep learning frameworks.

### Continue Learning

- **Computer Vision**: Convolutional Neural Networks (CNNs)
- **Natural Language Processing**: Recurrent Neural Networks (RNNs), Transformers
- **Generative Models**: GANs, VAEs, Diffusion Models
- **Reinforcement Learning**: Deep Q-Networks, Policy Gradients
- **Advanced PyTorch**: Distributed training, mixed precision, custom CUDA kernels

### Recommended Resources

- **Books**:
  - "Deep Learning" by Goodfellow, Bengio, Courville
  - "Neural Networks and Deep Learning" by Michael Nielsen (free online)
- **Courses**:
  - Fast.ai Practical Deep Learning
  - Stanford CS231n (Computer Vision)
  - Stanford CS224n (NLP)
- **Practice**:
  - Kaggle competitions
  - Papers with Code
  - Build your own projects!

---

## Help and Support

**Stuck on an exercise?**

1. Re-read the relevant lesson
2. Check the hints in the notebook
3. Use gradient checking to debug backprop
4. Print shapes and intermediate values
5. Search the error message online
6. Look at the solution for that specific part
7. Open an issue on [GitHub](https://github.com/jumpingsphinx/jumpingsphinx.github.io/issues)

**Found a bug or have a suggestion?**

Please [open an issue](https://github.com/jumpingsphinx/jumpingsphinx.github.io/issues) or submit a pull request!

---

Good luck with the exercises! Remember: **understanding comes from implementation**. Build it yourself before relying on libraries!
