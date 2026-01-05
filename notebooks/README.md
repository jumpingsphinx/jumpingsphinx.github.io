# ML101 Notebooks

This directory contains all the interactive Jupyter notebook exercises for the ML101 course.

## Structure

```
notebooks/
├── module1-linear-algebra/
│   ├── exercise1-vectors.ipynb
│   ├── exercise2-matrices.ipynb
│   ├── exercise3-pca.ipynb
│   └── solutions/
│       ├── solution1-vectors.ipynb
│       ├── solution2-matrices.ipynb
│       └── solution3-pca.ipynb
├── module2-regression/
│   ├── exercise1-linear-regression.ipynb
│   ├── exercise2-gradient-descent.ipynb
│   ├── exercise3-logistic-regression.ipynb
│   ├── exercise4-regularization.ipynb
│   └── solutions/
├── module3-trees/
│   ├── exercise1-decision-trees.ipynb
│   ├── exercise2-random-forest.ipynb
│   ├── exercise3-xgboost.ipynb
│   ├── exercise4-comparison.ipynb
│   └── solutions/
└── module4-neural-networks/
    ├── exercise1-perceptron-numpy.ipynb
    ├── exercise2-mlp-numpy.ipynb
    ├── exercise3-pytorch-basics.ipynb
    ├── exercise4-pytorch-classification.ipynb
    ├── exercise5-pytorch-advanced.ipynb
    └── solutions/
```

## How to Use

### 1. Launch Jupyter Lab

```bash
# Make sure you're in the ML101 directory with venv activated
jupyter lab
```

### 2. Navigate to the Module

In Jupyter Lab's file browser, open:
- `notebooks/module1-linear-algebra/` for Module 1
- `notebooks/module2-regression/` for Module 2
- And so on...

### 3. Open an Exercise Notebook

Start with `exercise1-vectors.ipynb` and work through sequentially.

### 4. Work Through the Exercise

- Read the instructions carefully
- Write code in the provided cells
- Run cells with `Shift + Enter`
- Check your output against expected results

### 5. Review Solutions

After completing an exercise:
- Navigate to the `solutions/` subfolder
- Open the corresponding solution notebook
- Compare your approach with the reference solution
- Learn alternative methods

## Notebook Guidelines

### Running Cells

- **Run current cell**: `Shift + Enter`
- **Run cell in place**: `Ctrl + Enter`
- **Insert cell above**: Press `A` in command mode
- **Insert cell below**: Press `B` in command mode
- **Delete cell**: Press `DD` in command mode

### Saving Work

- Notebooks auto-save every few minutes
- Manually save: `Ctrl + S` or click the save icon
- **Important**: Don't rely only on auto-save!

### Restarting Kernel

If something goes wrong:
1. Kernel → Restart Kernel
2. Re-run all cells from the top

### Best Practices

!!! tip "Notebook Best Practices"
    - **Run cells in order**: Don't jump around
    - **Clear output before committing**: Cell → All Output → Clear
    - **Comment your code**: Explain your reasoning
    - **Print intermediate results**: Use `print()` to debug
    - **Check shapes**: Use `.shape` to verify dimensions
    - **Don't modify solution notebooks**: Make a copy if you want to experiment

## Exercise Format

Each exercise notebook follows this structure:

```python
# Module X - Exercise Y: Topic Name

## Learning Objectives
- Objective 1
- Objective 2

## Prerequisites
- Required knowledge

## Setup
# Import statements and configuration

## Part 1: Section Name
### Background
Brief explanation

### Exercise 1.1
**Task:** What to implement
**Hints:**
- Hint 1
- Hint 2

# Your code here


## Part 2: Next Section
...

## Challenge (Optional)
More difficult problems

## Reflection Questions
Conceptual questions to deepen understanding
```

## Common Issues

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Make sure venv is activated
pip install -r requirements.txt
```

### Kernel Issues

If kernel won't start:
```bash
# Reinstall kernel
pip install --upgrade ipykernel
python -m ipykernel install --user
```

### Cell Won't Run

- Check for syntax errors
- Restart kernel and run cells from top
- Clear output and try again

## Tips for Success

### Time Management

- Budget 1-2 hours per exercise
- Take breaks between exercises
- Don't rush - understanding > speed

### Getting Unstuck

1. Re-read the lesson on the documentation site
2. Use the hints provided in the notebook
3. Search online (Stack Overflow, NumPy docs)
4. Check one section of the solution
5. Ask for help (GitHub issues)

### Maximizing Learning

- **Type code, don't copy-paste**: Builds muscle memory
- **Experiment**: Change parameters and see what happens
- **Visualize**: Use matplotlib liberally
- **Explain aloud**: Teach concepts to rubber duck
- **Take notes**: Document insights and "aha!" moments

## Solution Guidelines

### When to Look at Solutions

✅ **Good reasons:**
- You've spent 30+ minutes stuck
- You want to verify your working solution
- You want to learn alternative approaches
- You're reviewing after completing the exercise

❌ **Avoid:**
- Looking before attempting the exercise
- Copy-pasting without understanding
- Skipping directly to solutions

### How to Use Solutions

1. **Compare approaches**: Is your solution different? Why?
2. **Check efficiency**: Is the solution more efficient?
3. **Learn idioms**: Note NumPy/Python best practices
4. **Understand trade-offs**: Why was this approach chosen?

## Experiment Ideas

After completing exercises, try:

1. **Different parameters**: Change values and observe effects
2. **Different datasets**: Apply to other data
3. **Variations**: Modify the problem slightly
4. **Optimizations**: Can you make it faster?
5. **Visualizations**: Create plots to understand better

## Creating Your Own Notebooks

Want to experiment more?

1. **Duplicate an exercise**: Right-click → Duplicate
2. **Create new notebook**: File → New → Notebook
3. **Import ML101 utilities**:
   ```python
   import sys
   sys.path.append('..')
   from src.utils import plotting
   ```

## Notebook Maintenance

### Clearing Outputs

Before committing to Git:
```bash
# Clear all outputs
jupyter nbconvert --clear-output --inplace notebook.ipynb
```

Or in Jupyter Lab: Cell → All Output → Clear

### Formatting Code

Use black to format code cells:
```bash
pip install black[jupyter]
black notebook.ipynb
```

## Resources

### Jupyter Documentation
- [Jupyter Lab User Guide](https://jupyterlab.readthedocs.io/)
- [Jupyter Notebook Keyboard Shortcuts](https://towardsdatascience.com/jypyter-notebook-shortcuts-bf0101a98330)

### NumPy Documentation
- [NumPy User Guide](https://numpy.org/doc/stable/user/)
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)

### Visualization
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)

## Getting Help

**Issues with notebooks?**
- Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues)
- Tag it with `notebooks` label
- Include error messages and environment details

**Contributing improvements?**
- Fork the repository
- Make your changes
- Submit a pull request

---

Happy learning! 🚀
