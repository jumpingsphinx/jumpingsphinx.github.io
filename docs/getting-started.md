# Getting Started

This guide will help you set up your environment and get ready to start learning machine learning fundamentals.

## System Requirements

### Minimum Requirements
- **Operating System:** Windows 10/11, macOS 10.14+, or Linux
- **Python:** Version 3.9 or higher
- **RAM:** 4GB minimum (8GB recommended)
- **Disk Space:** 2GB for dependencies and datasets

### Recommended Setup
- **RAM:** 8GB or more for faster computations
- **Python:** Version 3.10 or 3.11 for best compatibility
- **GPU:** Not required, but helpful for Module 4 (Neural Networks)

## Installation Steps

### Step 1: Verify Python Installation

First, check if Python is installed and which version you have:

```bash
python --version
```

or

```bash
python3 --version
```

You should see Python 3.9 or higher. If not, [download Python](https://www.python.org/downloads/).

!!! warning "Python Version"
    Make sure you have Python 3.9 or higher. Older versions may not be compatible with all dependencies.

### Step 2: Clone the Repository

Clone the ML101 repository to your local machine:

```bash
git clone https://github.com/jumpingsphinx/ML101.git
cd ML101
```

If you don't have Git installed, you can [download it here](https://git-scm.com/downloads) or download the repository as a ZIP file from GitHub.

### Step 3: Create a Virtual Environment

Creating a virtual environment keeps your ML101 dependencies isolated from other Python projects.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the beginning of your command prompt, indicating the virtual environment is active.

!!! tip "Virtual Environments"
    Always activate your virtual environment before working on ML101 projects. This ensures you're using the correct package versions.

### Step 4: Install Dependencies

With your virtual environment activated, install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- NumPy, SciPy, and Pandas for numerical computing
- scikit-learn for machine learning algorithms
- PyTorch for deep learning (Module 4)
- Matplotlib and Seaborn for visualization
- Jupyter Lab for interactive notebooks

The installation may take 5-10 minutes depending on your internet connection.

### Step 5: Verify Installation

Let's verify everything is installed correctly:

```python
python -c "import numpy, pandas, sklearn, torch, matplotlib; print('All packages installed successfully!')"
```

If you see "All packages installed successfully!" you're ready to go!

### Step 6: Launch Jupyter Lab

Start Jupyter Lab to access the interactive notebooks:

```bash
jupyter lab
```

This will open Jupyter Lab in your web browser (usually at `http://localhost:8888`).

!!! info "First Time Using Jupyter?"
    Jupyter Lab is an interactive development environment for notebooks. You can write code, see results immediately, and mix code with explanatory text.

## Navigating the Repository

Once Jupyter Lab is open, you'll see the file browser. Here's what each folder contains:

```
ML101/
├── notebooks/              ← Your exercises are here!
│   ├── module1-linear-algebra/
│   ├── module2-regression/
│   ├── module3-trees/
│   └── module4-neural-networks/
├── docs/                   ← Documentation source files
├── src/                    ← Utility functions (optional)
└── tests/                  ← Tests for utilities
```

## Your Learning Workflow

Here's the recommended way to work through each module:

1. **Read the Lesson:** Visit the documentation page for the lesson you're working on (this site!)

2. **Open the Exercise Notebook:**
   - Navigate to `notebooks/module-X/` in Jupyter Lab
   - Open the exercise notebook (e.g., `exercise1-vectors.ipynb`)

3. **Complete the Exercises:**
   - Read the instructions carefully
   - Write code in the provided cells
   - Run cells to see results (Shift + Enter)

4. **Check Your Work:**
   - Try to complete exercises without looking at solutions
   - Use the hints if you get stuck
   - Verify your output matches expected results

5. **Review Solutions:**
   - Navigate to the `solutions/` subfolder
   - Compare your approach with the solution
   - Understand alternative methods

6. **Experiment:**
   - Modify parameters and see what happens
   - Try the code on different inputs
   - Answer the reflection questions

## Jupyter Lab Quick Reference

### Essential Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Run cell and advance | `Shift + Enter` |
| Run cell in place | `Ctrl + Enter` |
| Insert cell above | `A` (in command mode) |
| Insert cell below | `B` (in command mode) |
| Delete cell | `DD` (press D twice, in command mode) |
| Undo cell deletion | `Z` (in command mode) |
| Change to markdown | `M` (in command mode) |
| Change to code | `Y` (in command mode) |

### Command Mode vs Edit Mode

- **Edit Mode:** Press `Enter` to edit a cell's content
- **Command Mode:** Press `Esc` to navigate between cells

## Troubleshooting

### Import Errors

If you get "ModuleNotFoundError":
```bash
# Make sure virtual environment is activated
# Re-run pip install
pip install -r requirements.txt
```

### Jupyter Lab Won't Start

Try:
```bash
pip install --upgrade jupyterlab
jupyter lab --port=8889  # Try a different port
```

### PyTorch Installation Issues

If PyTorch doesn't install properly:

**CPU-only version (smaller, faster download):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Check [PyTorch website](https://pytorch.org/get-started/locally/) for your specific platform.**

### Memory Issues

If you encounter memory errors:
- Close other applications
- Restart Jupyter Lab
- Work with smaller datasets initially
- Consider using Google Colab (free cloud notebooks)

## Alternative: Google Colab

If you can't install locally, use Google Colab (free cloud notebooks with GPU access):

1. Visit [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook
3. Upload exercise notebooks from the ML101 repository
4. Install dependencies in a code cell:
   ```python
   !pip install xgboost
   ```

!!! info "Colab Note"
    Most dependencies (NumPy, pandas, scikit-learn) are pre-installed in Colab.

## Development Dependencies (Optional)

If you want to build the documentation locally or contribute to the project:

```bash
pip install -r requirements-dev.txt
```

This installs MkDocs and development tools. You can then build the documentation:

```bash
mkdocs serve
```

Visit `http://127.0.0.1:8000` to see the documentation site locally.

## Next Steps

Now that you're set up, it's time to start learning!

1. Review the [Learning Path](learning-path.md) for tips on how to approach the course
2. Start with [Module 1: Linear Algebra Basics](module1-linear-algebra/index.md)
3. Open your first exercise notebook in Jupyter Lab

[View Learning Path](learning-path.md){ .md-button .md-button--primary }
[Start Module 1](module1-linear-algebra/index.md){ .md-button }

---

**Having issues?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues) and we'll help you out!
