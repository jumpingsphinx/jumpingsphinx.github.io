# Contributing to ML101

Thank you for your interest in contributing to ML101! This document provides guidelines and instructions for contributing.

## Ways to Contribute

### 1. Report Issues

Found a bug or have a suggestion?
- Check if the issue already exists
- Open a new issue with a clear title and description
- Include code examples or screenshots if relevant
- Tag appropriately (bug, enhancement, documentation, etc.)

### 2. Fix Typos and Improve Documentation

- Fix spelling/grammar mistakes
- Clarify confusing explanations
- Add missing information
- Improve code examples

### 3. Add Code Examples

- Provide alternative implementations
- Add visualization examples
- Create helper functions
- Write utility scripts

### 4. Create New Content

- Write additional exercises
- Add lesson content for placeholder files
- Create supplementary materials
- Develop advanced topics

### 5. Improve Existing Content

- Enhance explanations
- Add more examples
- Improve visualizations
- Optimize code

## Getting Started

### 1. Fork the Repository

Click the "Fork" button on GitHub to create your own copy.

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR-USERNAME/ML101.git
cd ML101
```

### 3. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## Development Guidelines

### Code Style

**Python Code:**
- Follow PEP 8 style guide
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small
- Use type hints when helpful

```python
def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels

    Returns:
    --------
    float
        Accuracy score between 0 and 1
    """
    return np.mean(y_true == y_pred)
```

**Format Code:**
```bash
# Format with black
black your_file.py

# Check with flake8
flake8 your_file.py
```

### Documentation

**Markdown Files:**
- Use clear headings (H1, H2, H3)
- Include code examples in fenced blocks with language
- Add admonitions for tips, warnings, notes
- Keep line length reasonable (wrap at 80-100 chars for readability)

**Code Comments:**
- Explain *why*, not *what*
- Use comments for complex logic
- Keep comments up-to-date with code

### Jupyter Notebooks

**Before Committing:**
- Clear all outputs: Cell → All Output → Clear
- Test notebooks run top-to-bottom without errors
- Remove any personal data or API keys
- Keep cells focused and well-organized

**Notebook Structure:**
```markdown
# Module X - Exercise Y: Topic

## Learning Objectives
- Objective 1
- Objective 2

## Prerequisites
- Prerequisite knowledge

## Setup
import numpy as np
# ...

## Part 1: Section Name
### Background
Brief explanation

### Exercise 1.1
**Task:** What to implement
**Hints:** Helpful guidance

# Your code here
```

## Content Guidelines

### Writing Style

- **Be clear and concise**: Avoid jargon when possible
- **Use examples**: Show, don't just tell
- **Build progressively**: Start simple, add complexity
- **Connect to ML**: Explain why concepts matter
- **Be encouraging**: Assume best intentions from learners

### Mathematical Notation

Use LaTeX for math:
```markdown
Inline: $y = mx + b$

Block:
$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2
$$
```

### Code Examples

- **Always test code**: Ensure examples run correctly
- **Show output**: Include expected results
- **Be explicit**: Don't assume too much background knowledge
- **Use real data**: Prefer actual datasets over toy examples

## Testing

### Documentation

Test that documentation builds:
```bash
mkdocs build --strict
```

Preview locally:
```bash
mkdocs serve
```

Visit http://127.0.0.1:8000 to view.

### Code

Run tests (if applicable):
```bash
pytest tests/
```

### Notebooks

Ensure notebooks execute without errors:
```bash
# Install nbconvert if needed
pip install nbconvert

# Test execution
jupyter nbconvert --to notebook --execute your_notebook.ipynb
```

## Commit Guidelines

### Commit Messages

Write clear, descriptive commit messages:

```
Add gradient descent visualization to Module 2

- Created interactive plot showing optimization path
- Added learning rate comparison
- Included code examples for batch/SGD/mini-batch
```

**Format:**
- First line: Brief summary (50 chars or less)
- Blank line
- Detailed description (wrap at 72 chars)
- Use present tense ("Add" not "Added")
- Be specific about what changed

### Commits to Avoid

- Don't commit personal config files
- Don't commit large binary files
- Don't commit API keys or credentials
- Don't commit notebook outputs (clear first)

## Pull Request Process

### 1. Update Your Fork

```bash
# Add upstream remote (one time)
git remote add upstream https://github.com/jumpingsphinx/ML101.git

# Fetch and merge changes
git fetch upstream
git merge upstream/main
```

### 2. Push Your Changes

```bash
git push origin feature/your-feature-name
```

### 3. Open Pull Request

- Go to your fork on GitHub
- Click "New Pull Request"
- Select your feature branch
- Fill out the PR template

### 4. PR Description

Include:
- **What**: What does this PR do?
- **Why**: Why is this change needed?
- **How**: How does it work?
- **Testing**: How did you test it?
- **Screenshots**: If relevant

Example:
```markdown
## Description
Adds comprehensive linear regression lesson to Module 2.

## Changes
- Created 01-linear-regression.md with theory and examples
- Added NumPy implementation from scratch
- Included visualizations of regression line
- Added comparison with sklearn implementation

## Testing
- Built documentation locally with `mkdocs build --strict`
- Tested all code examples in notebook
- Verified mathematical formulas render correctly

## Screenshots
![Regression visualization](path/to/image.png)
```

### 5. Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Make changes in your branch and push
- PR will be merged when approved

## Content Areas Needing Help

Current priorities:

### High Priority
- [ ] Jupyter notebook exercises for all modules
- [ ] Solution notebooks with detailed explanations
- [ ] Lesson content for Module 2-4 (currently placeholders)

### Medium Priority
- [ ] More visualizations and diagrams
- [ ] Additional practice problems
- [ ] Video content or screencasts
- [ ] Translation to other languages

### Lower Priority
- [ ] Advanced topics (beyond core curriculum)
- [ ] Alternative implementations (e.g., TensorFlow)
- [ ] Interactive widgets in notebooks

## Style Guide Summary

**Files and Folders:**
- Use lowercase with hyphens: `module1-linear-algebra`
- Be descriptive: `exercise1-vectors.ipynb` not `ex1.ipynb`

**Code:**
- Python: PEP 8
- Variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`

**Documentation:**
- Headings: Title Case for H1, Sentence case for H2+
- Lists: Capitalize first word, end with period if full sentence
- Links: Descriptive text, not "click here"

## Questions?

- Open an issue for discussion
- Tag with `question` label
- Be specific about what you need help with

## Code of Conduct

Be respectful and constructive:
- Welcome newcomers
- Provide helpful feedback
- Assume good intentions
- Focus on ideas, not people
- No harassment or discrimination

## Recognition

Contributors will be recognized:
- In commit history
- In release notes (for significant contributions)
- On project README (for major contributions)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ML101! Your help makes machine learning education more accessible to everyone. 🎓
