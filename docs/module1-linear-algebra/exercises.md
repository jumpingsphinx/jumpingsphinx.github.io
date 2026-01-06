# Module 1 Exercises: Linear Algebra

## Overview

Time to put your linear algebra knowledge into practice! These exercises will help you build intuition and coding skills with vectors, matrices, and eigenvalue decomposition.

## Before You Start

### Setup

1. Open Jupyter Lab:
   ```bash
   jupyter lab
   ```

2. Navigate to `notebooks/module1-linear-algebra/`

3. Start with `exercise1-vectors.ipynb`

### Exercise Format

Each exercise includes:
- **Learning objectives**: What you'll practice
- **Background**: Quick concept review
- **Tasks**: Step-by-step implementation
- **Hints**: Help when you're stuck
- **Expected output**: What your results should look like
- **Reflection questions**: Deepen your understanding

### Tips for Success

!!! tip "Best Practices"
    - **Try first, then look at hints**: Struggling helps learning
    - **Run cells frequently**: Test as you go
    - **Print shapes**: Use `.shape` to debug dimension mismatches
    - **Visualize**: Use matplotlib to see what's happening
    - **Check solutions only after trying**: Compare your approach

## Exercise 1: Vector Operations with NumPy

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jumpingsphinx/ML101/blob/main/notebooks/module1-linear-algebra/exercise1-vectors.ipynb)

**Time:** 1-1.5 hours

### What You'll Learn

- Create and manipulate vectors with NumPy
- Implement vector operations (addition, scaling, dot product)
- Compute vector norms (L1, L2, infinity)
- Calculate distances and similarities between vectors
- Visualize vectors in 2D

### Topics Covered

- Vector creation and indexing
- Vector arithmetic operations
- Dot product and its applications
- Vector norms and normalization
- Euclidean distance
- Cosine similarity
- Vector visualization

### Key Functions

```python
np.array(), np.dot(), np.linalg.norm(),
np.zeros(), np.ones(), np.random.rand()
```

### What You'll Build

1. **Vector calculator**: Functions for basic vector operations
2. **Similarity finder**: Find most similar vectors in a dataset
3. **Visualizer**: Plot vectors and their operations

---

## Exercise 2: Matrix Manipulations and Transformations

!!! info "Coming Soon"
    This exercise is currently being developed. Check back soon!

**Time:** 1.5-2 hours

### What You'll Learn

- Create and manipulate matrices with NumPy
- Perform matrix operations (transpose, multiplication)
- Apply matrix transformations to vectors
- Work with real datasets as matrices
- Solve systems of linear equations

### Topics Covered

- Matrix creation and indexing
- Matrix transpose and properties
- Matrix-vector multiplication
- Matrix-matrix multiplication
- Matrix transformations (rotation, scaling, shear)
- Dataset operations (centering, normalizing)
- Solving linear systems

### Key Functions

```python
np.array(), A.T, A @ B, np.linalg.inv(),
np.linalg.solve(), np.linalg.det()
```

### What You'll Build

1. **Transformation visualizer**: See how matrices transform 2D space
2. **Dataset processor**: Center and normalize real data
3. **Linear system solver**: Solve $Ax = b$ problems

---

## Exercise 3: PCA Implementation from Scratch

!!! info "Coming Soon"
    This exercise is currently being developed. Check back soon!

**Time:** 2-3 hours

### What You'll Learn

- Implement PCA from scratch using NumPy
- Compute and interpret eigenvalues and eigenvectors
- Reduce dimensionality of real datasets
- Visualize principal components
- Compare with sklearn's PCA implementation

### Topics Covered

- Data centering and standardization
- Covariance matrix computation
- Eigenvalue decomposition
- Selecting principal components
- Data projection onto PCs
- Explained variance analysis
- PCA visualization (2D and 3D)

### Key Functions

```python
np.cov(), np.linalg.eig(), np.linalg.svd(),
sklearn.decomposition.PCA
```

### What You'll Build

1. **PCA from scratch**: Complete implementation without sklearn
2. **Dimensionality reducer**: Reduce high-dimensional data to 2D/3D
3. **Variance analyzer**: Determine optimal number of components
4. **Comparison tool**: Validate against sklearn's PCA

### Datasets Used

- Iris dataset (4D → 2D)
- Digits dataset (64D → 2D for visualization)
- Custom synthetic data

---

## Solutions

After completing each exercise, review the solutions to:
- Compare your approach with the reference implementation
- Learn alternative methods
- Understand best practices
- Debug any issues

!!! info "Solutions Coming Soon"
    Solution notebooks will be added as exercises are completed. Exercise 1 solution is currently being developed.

!!! warning "Use Solutions Wisely"
    Try to complete exercises independently first. Looking at solutions too early prevents deep learning. Use them for verification and learning alternative approaches.

## Assessment Questions

After completing all exercises, test your understanding:

1. **Conceptual**
   - When would you use L1 norm vs L2 norm?
   - Why is matrix multiplication not commutative?
   - What does it mean if two eigenvectors are orthogonal?
   - When should you use PCA in a real project?

2. **Practical**
   - How do you check if a matrix is invertible?
   - What's the fastest way to compute pairwise distances in NumPy?
   - How do you choose the number of principal components?
   - What happens if you don't center data before PCA?

3. **Debugging**
   - You get "ValueError: shapes not aligned". What's wrong?
   - Your PCA shows negative eigenvalues. Why?
   - Matrix inverse fails. What could be the cause?

## Reflection Questions

Think deeply about what you've learned:

1. How does understanding linear algebra help you understand machine learning better?
2. What was the most surprising thing you learned?
3. Which concept was hardest to grasp? How did you overcome it?
4. How would you explain PCA to someone with no ML background?
5. Where could you apply these techniques in your work or projects?

## Common Mistakes to Avoid

!!! warning "Watch Out For"
    - **Not checking dimensions**: Always print `.shape` before operations
    - **Confusing dot product with element-wise**: Use `@` for dot, `*` for element-wise
    - **Forgetting to center data**: PCA requires zero-mean data
    - **Wrong axis in numpy operations**: `axis=0` is rows, `axis=1` is columns
    - **Not normalizing features**: Features with different scales can dominate PCA

## Going Further

### Challenge Exercises

Want more practice? Try these:

1. **Implement SVD-based PCA**: Use SVD instead of eigendecomposition
2. **Incremental PCA**: Handle datasets that don't fit in memory
3. **Kernel PCA**: Extend PCA to capture non-linear relationships
4. **t-SNE**: Implement t-SNE for visualization (uses eigenvalues!)
5. **Image compression**: Use SVD to compress images

### Real-World Projects

Apply your skills:

1. **Face recognition**: Use PCA for eigenfaces
2. **Recommender system**: Matrix factorization for collaborative filtering
3. **Data visualization**: Reduce high-dimensional data for plotting
4. **Feature engineering**: Use PCA for feature extraction

## Next Module

Once you're comfortable with linear algebra, you're ready for regression!

[Continue to Module 2: Regression Algorithms](../module2-regression/index.md){ .md-button .md-button--primary }

---

## Help and Support

**Stuck on an exercise?**

1. Re-read the relevant lesson
2. Check the hints in the notebook
3. Search the error message online
4. Look at the solution for that specific part
5. Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues)

**Found a bug or have a suggestion?**

Please [open an issue](https://github.com/jumpingsphinx/ML101/issues) or submit a pull request!

---

Good luck with the exercises! Remember: **struggle is part of learning**. Don't give up!
