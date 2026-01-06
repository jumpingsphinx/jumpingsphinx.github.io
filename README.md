# ML101 - Machine Learning Fundamentals

> **Learn machine learning entirely in your browser - no installation required!**

[![Documentation](https://img.shields.io/badge/docs-live-brightgreen)](https://jumpingsphinx.github.io/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jumpingsphinx/ML101/blob/main/notebooks/module1-linear-algebra/exercise1-vectors.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ML101 is a comprehensive, **fully interactive** course designed to teach you machine learning from scratch. Whether you're a complete beginner or looking to solidify your fundamentals, this course offers a unique browser-based learning experience.

## ✨ What Makes This Course Different?

- **🚀 Interactive Code Examples** - Run Python code directly in your browser with zero setup
- **☁️ Cloud-Based Exercises** - One-click access to Google Colab notebooks
- **💻 Works Anywhere** - No installation needed - works on any device, even Chromebooks
- **📚 Complete Curriculum** - 4 modules covering linear algebra to neural networks
- **🎓 Learn by Doing** - Build algorithms from scratch, then use industry tools

## 🎯 Quick Start

**Want to start immediately?** Choose your path:

### Path 1: Browser-Only (Recommended for Beginners)
**No installation. Start learning in 30 seconds.**

1. Visit the [**interactive lessons**](https://jumpingsphinx.github.io/)
2. Click "▶ Run Code" buttons to execute Python in your browser
3. Click "Open in Colab" badges to complete exercises in the cloud

[**Start Learning Now →**](https://jumpingsphinx.github.io/)

### Path 2: Local Development (For Advanced Users)

```bash
# Clone the repository
git clone https://github.com/jumpingsphinx/ML101.git
cd ML101

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

[**Full Setup Guide →**](https://jumpingsphinx.github.io/getting-started/)

## 📖 Course Modules

### [Module 1: Linear Algebra Basics](https://jumpingsphinx.github.io/module1-linear-algebra/)
**Foundation for Understanding ML** • ⏱️ 4-6 hours

Learn the mathematical foundations of machine learning:
- ✅ Vectors and vector operations
- ✅ Matrices and transformations
- ✅ Eigenvalues and eigenvectors
- ✅ Principal Component Analysis (PCA)

**Exercises:** [Vector Operations](https://colab.research.google.com/github/jumpingsphinx/ML101/blob/main/notebooks/module1-linear-algebra/exercise1-vectors.ipynb) | Matrix Manipulations (coming soon) | PCA Implementation (coming soon)

---

### [Module 2: Regression Algorithms](https://jumpingsphinx.github.io/module2-regression/)
**Predicting Continuous Values** • ⏱️ 6-8 hours • 🚧 Coming Soon

Master regression techniques and optimization:
- Linear regression from scratch
- Gradient descent optimization
- Logistic regression for classification
- L1/L2 regularization techniques

---

### [Module 3: Tree-Based Algorithms](https://jumpingsphinx.github.io/module3-trees/)
**Decision Trees and Ensemble Methods** • ⏱️ 6-8 hours • 🚧 Coming Soon

Understand decision trees and powerful ensemble methods:
- Decision tree fundamentals
- Random Forest for robust predictions
- Gradient boosting concepts
- XGBoost for high-performance ML

---

### [Module 4: Neural Networks](https://jumpingsphinx.github.io/module4-neural-networks/)
**Deep Learning Fundamentals** • ⏱️ 8-10 hours • 🚧 Coming Soon

Build neural networks from scratch and with PyTorch:
- Perceptron and activation functions
- Feedforward networks and backpropagation
- NumPy implementation from scratch
- PyTorch for modern deep learning

## 🎓 Learning Philosophy

### 1. Interactive First
Every concept includes runnable code examples. Click "▶ Run Code" and see Python execute in your browser - powered by [Pyodide](https://pyodide.org/).

### 2. Build from Scratch
Implement algorithms using NumPy before using libraries. Understanding the internals makes you a better practitioner.

### 3. Modern Tools
After mastering fundamentals, learn industry-standard libraries: scikit-learn, XGBoost, and PyTorch.

### 4. Progressive Complexity
Start with heavy guidance, progress to open-ended challenges. Real-world ML requires both.

## 🏃 Try It Now

**Don't want to read? Jump right in:**

1. **[Start Module 1 →](https://jumpingsphinx.github.io/module1-linear-algebra/)** - Begin with interactive linear algebra lessons
2. **[Try an Exercise →](https://colab.research.google.com/github/jumpingsphinx/ML101/blob/main/notebooks/module1-linear-algebra/exercise1-vectors.ipynb)** - Open a Colab notebook and start coding
3. **[See the Demo →](https://jumpingsphinx.github.io/)** - Run Python code directly on the homepage

## 📁 Repository Structure

```
ML101/
├── docs/                          # MkDocs documentation source
│   ├── index.md                   # Landing page with live code demo
│   ├── getting-started.md         # Three learning paths explained
│   ├── module1-linear-algebra/    # Complete Module 1 lessons
│   ├── module2-regression/        # Module 2 (coming soon)
│   ├── module3-trees/             # Module 3 (coming soon)
│   ├── module4-neural-networks/   # Module 4 (coming soon)
│   └── resources/                 # Math primer, Python refresher, datasets
│
├── notebooks/                     # Jupyter exercise notebooks
│   └── module1-linear-algebra/
│       └── exercise1-vectors.ipynb
│
├── requirements.txt               # Python dependencies
├── requirements-dev.txt           # Development dependencies (MkDocs, etc.)
└── mkdocs.yml                     # Documentation configuration
```

## 🛠️ Tech Stack

**Course Content:**
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) - Beautiful, responsive documentation
- [Pyodide](https://pyodide.org/) - Python in WebAssembly for browser execution
- [Google Colab](https://colab.research.google.com/) - Free cloud notebooks with GPU

**ML Libraries:**
- [NumPy](https://numpy.org/) - Numerical computing
- [scikit-learn](https://scikit-learn.org/) - Machine learning algorithms
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - Visualization

## 🤝 Contributing

Contributions are welcome! Whether you want to:
- 🐛 Fix a bug or typo
- 📝 Improve documentation
- 💡 Add new content or exercises
- 🎨 Enhance visualizations

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Feel free to use, share, and adapt these materials for educational purposes.

## 🙏 Acknowledgments

This course was built with the help of:
- The amazing open-source ML community
- Contributors and issue reporters
- [Pyodide](https://pyodide.org/) team for making Python-in-browser possible
- [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) for the beautiful theme

## 📬 Questions or Feedback?

- 💬 [Open an issue](https://github.com/jumpingsphinx/ML101/issues) for bugs or questions
- ⭐ Star this repo if you find it helpful!
- 🔗 Share with others learning ML

---

<div align="center">

**Ready to start your machine learning journey?**

[**🚀 Start Learning →**](https://jumpingsphinx.github.io/)

</div>
