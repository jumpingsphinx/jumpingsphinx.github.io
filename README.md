# ML101 - Machine Learning Fundamentals

A comprehensive, hands-on course for learning machine learning from the ground up. This repository combines polished documentation with interactive Jupyter notebooks to guide you through essential ML concepts.

## Documentation

Visit our [documentation site](https://jumpingsphinx.github.io/ML101/) for the complete learning experience.

## Course Structure

### Module 1: Linear Algebra Basics
Learn the foundational mathematics behind machine learning:
- Vectors and vector operations
- Matrices and matrix operations
- Eigenvalues and eigenvectors
- Principal Component Analysis (PCA)

### Module 2: Regression Algorithms
Master regression techniques and optimization:
- Linear regression
- Gradient descent optimization
- Logistic regression for classification
- L1/L2 regularization techniques

### Module 3: Tree-Based Algorithms
Understand decision trees and ensemble methods:
- Decision tree fundamentals
- Random Forest
- Gradient boosting
- XGBoost

### Module 4: Neural Networks
Build neural networks from scratch and with PyTorch:
- Perceptron and activation functions
- Feedforward networks
- Backpropagation
- NumPy implementation from scratch
- PyTorch for modern deep learning

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/jumpingsphinx/ML101.git
   cd ML101
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Lab**
   ```bash
   jupyter lab
   ```

5. **Start learning!**
   - Read the lessons on our [documentation site](https://jumpingsphinx.github.io/ML101/)
   - Complete the exercises in the `notebooks/` directory
   - Check your solutions against the provided solution notebooks

## Repository Structure

```
ML101/
├── docs/              # Documentation source (MkDocs)
├── notebooks/         # Jupyter exercise notebooks
│   ├── module1-linear-algebra/
│   ├── module2-regression/
│   ├── module3-trees/
│   └── module4-neural-networks/
├── src/               # Utility modules
└── tests/             # Unit tests
```

## Learning Path

1. Start with Module 1 to build mathematical foundations
2. Progress through Module 2 to understand regression and optimization
3. Explore Module 3 for tree-based algorithms
4. Complete Module 4 to master neural networks

Each module includes:
- Detailed lesson pages with theory and examples
- Hands-on Jupyter notebook exercises
- Complete solutions with explanations

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) for documentation
- [Jupyter](https://jupyter.org/) for interactive notebooks
- [NumPy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/), [PyTorch](https://pytorch.org/) for ML implementations