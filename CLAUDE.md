# CLAUDE.md - ML101 Project Guidelines

## Project Overview

ML101 is a comprehensive, interactive machine learning course with browser-based code execution. The project uses MkDocs Material for documentation, Pyodide for in-browser Python execution, and Jupyter notebooks for exercises.

## Repository Structure

```
ML101/
├── docs/                          # MkDocs documentation source
│   ├── index.md                   # Landing page
│   ├── getting-started.md         # Setup instructions
│   ├── learning-path.md           # Course navigation
│   ├── module1-linear-algebra/    # Module 1 lessons
│   ├── module2-regression/        # Module 2 lessons
│   ├── module3-trees/             # Module 3 lessons
│   ├── module4-neural-networks/   # Module 4 lessons
│   ├── resources/                 # Supplementary materials
│   └── assets/                    # CSS, JS, images
├── notebooks/                     # Jupyter exercise notebooks
│   └── module1-linear-algebra/
│       └── exercise1-vectors.ipynb
├── src/                           # Utility modules (Python)
├── tests/                         # Test files
├── mkdocs.yml                     # MkDocs configuration
├── requirements.txt               # Python dependencies
└── requirements-dev.txt           # Development dependencies
```

## Key Technologies

- **Documentation**: MkDocs Material (`mkdocs.yml`)
- **Browser Python**: Pyodide (WebAssembly Python)
- **Exercises**: Jupyter Notebooks with Google Colab integration
- **Math Rendering**: MathJax (LaTeX syntax)
- **Styling**: Custom CSS in `docs/assets/stylesheets/`
- **Interactivity**: Custom JS in `docs/assets/javascripts/`

## Development Commands

### Documentation

```bash
# Build documentation (check for errors)
mkdocs build --strict

# Serve locally with hot reload
mkdocs serve
# Visit http://127.0.0.1:8000

# Deploy to GitHub Pages (automated via CI)
mkdocs gh-deploy
```

### Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

### Jupyter Notebooks

```bash
# Launch Jupyter Lab
jupyter lab

# Clear notebook outputs before committing
jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb

# Test notebook execution
jupyter nbconvert --to notebook --execute notebook.ipynb
```

## Content Guidelines

### Markdown Files (docs/)

- Use MkDocs Material admonitions: `!!! tip`, `!!! warning`, `!!! info`
- Use fenced code blocks with language identifiers
- LaTeX math: inline `$formula$`, block `$$formula$$`
- Interactive Python blocks use this wrapper:
  ```markdown
  <div class="python-interactive" markdown="1">
  ```python
  # Your code here
  ```
  </div>
  ```

### Code Style

- **Python**: Follow PEP 8, use meaningful variable names
- **Docstrings**: NumPy style for functions
- **Comments**: Explain *why*, not *what*
- **Line length**: 80-100 characters for readability

### Jupyter Notebooks

- Clear all outputs before committing
- Include "Open in Colab" badge at top
- Structure: Learning Objectives → Setup → Exercises → Solutions → Reflection
- Test that notebooks run top-to-bottom without errors

## File Naming Conventions

- **Lessons**: `01-topic-name.md`, `02-topic-name.md`
- **Exercises**: `exercise1-topic.ipynb`
- **Solutions**: `solutions/solution1-topic.ipynb`
- **Use lowercase with hyphens**, not underscores or spaces

## Adding New Content

### New Lesson

1. Create `docs/moduleX-name/XX-lesson-name.md`
2. Add entry to `mkdocs.yml` under `nav:`
3. Include learning objectives, examples, and exercises link
4. Add interactive code blocks where appropriate

### New Exercise Notebook

1. Create `notebooks/moduleX-name/exerciseX-topic.ipynb`
2. Follow the existing exercise template structure
3. Add Colab badge with correct GitHub path
4. Create corresponding solution notebook

### Placeholder Content

For "Coming Soon" pages, use this template:
```markdown
# Under Construction

!!! info "Coming Soon"
    This lesson is currently being developed.
```

## Testing Checklist

Before committing changes:

- [ ] `mkdocs build --strict` passes without errors
- [ ] Documentation renders correctly locally (`mkdocs serve`)
- [ ] All links work (internal and external)
- [ ] Code examples are syntactically correct
- [ ] Math formulas render properly
- [ ] Interactive code blocks have the `python-interactive` wrapper
- [ ] Notebook outputs are cleared
- [ ] No hardcoded local paths

## Git Workflow

### Commit Messages

Use clear, descriptive commit messages:

```
<type>: <short description>

<optional body with details>
```

Do NOT include contributors, generated with claude code, or anything like that. 

**Types:**
- `feat`: New feature or content
- `fix`: Bug fix or correction
- `docs`: Documentation only changes
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat: Add gradient descent lesson to Module 2

- Created comprehensive gradient descent explanation
- Added interactive visualization code
- Included batch/SGD/mini-batch comparison

fix: Correct matrix multiplication formula in Lesson 2

docs: Update README with new Colab badges
```

### Branching Strategy

```bash
# Create feature branch
git checkout -b feature/module2-regression

# Create fix branch
git checkout -b fix/typo-module1
```

### Committing and Pushing Changes

**After completing any work session:**

```bash
# 1. Check what changed
git status
git diff

# 2. Stage changes
git add .
# Or stage specific files:
git add docs/module2-regression/01-linear-regression.md

# 3. Commit with descriptive message
git commit -m "feat: Add linear regression lesson with interactive examples"

# 4. Push to remote
git push origin main
# Or if on a branch:
git push origin feature/module2-regression

# 5. Verify CI/CD passes (GitHub Actions)
```

**For significant changes, create a PR:**

```bash
git checkout -b feature/new-content
# ... make changes ...
git add .
git commit -m "feat: Description of changes"
git push origin feature/new-content
# Then create PR on GitHub
```

### Important: Always Commit and Push

**Before ending any session:**

1. **Stage all changes**: `git add .`
2. **Commit with clear message**: `git commit -m "type: description"`
3. **Push to remote**: `git push origin <branch>`
4. **Verify**: Check GitHub to confirm changes are pushed

This ensures:
- Work is not lost
- Changes are versioned and recoverable
- CI/CD can validate the build
- Collaborators can see updates

## CI/CD Pipeline

The project uses GitHub Actions (`.github/workflows/deploy-docs.yml`):

- **On Push to main**: Build and deploy docs to GitHub Pages
- **On PR**: Build docs to verify no errors

## Common Issues & Solutions

### MkDocs Build Fails

```bash
# Check for broken links or missing files
mkdocs build --strict 2>&1 | grep -i error
```

### Pyodide Code Doesn't Run

- Ensure code block has `python` language identifier
- Verify parent div has `python-interactive` class
- Check browser console for JavaScript errors

### Math Not Rendering

- Use `$...$` for inline, `$$...$$` for block
- Escape special characters in code blocks
- Verify MathJax is loading (check network tab)

### Colab Badge Not Working

- Verify GitHub path is correct
- Format: `https://colab.research.google.com/github/USER/REPO/blob/BRANCH/path/to/notebook.ipynb`

## Quick Reference

| Task | Command |
|------|---------|
| Build docs | `mkdocs build --strict` |
| Serve locally | `mkdocs serve` |
| Run tests | `pytest tests/` |
| Format code | `black .` |
| Check style | `flake8 .` |
| Clear notebook outputs | `jupyter nbconvert --clear-output --inplace *.ipynb` |
| Commit changes | `git add . && git commit -m "message"` |
| Push changes | `git push origin main` |

## Contact & Resources

- **Repository**: https://github.com/jumpingsphinx/jumpingsphinx.github.io
- **Live Site**: https://jumpingsphinx.github.io/
- **Issues**: https://github.com/jumpingsphinx/jumpingsphinx.github.io/issues

---

**Remember**: Always commit and push your changes before ending a work session to maintain proper version control!