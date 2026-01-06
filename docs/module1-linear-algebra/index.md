# Module 1: Linear Algebra Basics

## Overview

Linear algebra is the mathematical foundation of machine learning. Understanding vectors, matrices, and their operations is essential for grasping how ML algorithms work under the hood. This module will give you the mathematical tools you need for the rest of the course.

## Why Linear Algebra Matters in Machine Learning

In machine learning:
- **Data is represented as vectors and matrices**: Each data sample is a vector, and datasets are matrices
- **Models perform transformations**: Linear algebra describes how inputs become outputs
- **Optimization uses gradients**: Derivatives are computed using matrix operations
- **Dimensionality reduction**: Techniques like PCA rely on eigenvalue decomposition
- **Neural networks**: Weights are matrices, and forward/backward propagation uses matrix multiplication

!!! quote "Why Study This?"
    "You can't truly understand machine learning without understanding linear algebra. Every ML algorithm performs operations on vectors and matrices."

## Learning Objectives

By the end of this module, you will be able to:

- ✅ Understand vectors geometrically and algebraically
- ✅ Perform vector operations (addition, scaling, dot product)
- ✅ Work with matrices and matrix operations
- ✅ Compute matrix-vector and matrix-matrix products
- ✅ Understand eigenvalues and eigenvectors
- ✅ Apply PCA for dimensionality reduction
- ✅ Implement these concepts using NumPy

## Prerequisites

- **Python basics**: Variables, functions, loops
- **High school math**: Basic algebra and geometry
- **NumPy installation**: Required for all exercises

## Module Structure

### Lesson 1: Vectors
**Time: 45 minutes**

Learn the fundamentals of vectors - the building blocks of linear algebra.

- What are vectors?
- Vector operations: addition, scaling, dot product
- Vector norms (L1, L2, infinity)
- Geometric interpretation
- Implementation with NumPy

[Start Lesson 1](01-vectors.md){ .md-button .md-button--primary }

---

### Lesson 2: Matrices
**Time: 45 minutes**

Understand matrices and how they represent data and transformations.

- Matrix fundamentals and notation
- Types of matrices (identity, diagonal, symmetric)
- Matrix operations (addition, multiplication, transpose)
- Matrix-vector multiplication
- Implementation with NumPy

[Start Lesson 2](02-matrices.md){ .md-button }

---

### Lesson 3: Advanced Matrix Operations
**Time: 60 minutes**

Dive deeper into matrix operations used in ML algorithms.

- Matrix inverse and pseudo-inverse
- Determinants and their meaning
- Rank and linear independence
- Matrix decompositions (LU, QR)
- Solving systems of equations

[Start Lesson 3](03-matrix-operations.md){ .md-button }

---

### Lesson 4: Eigenvalues and Eigenvectors
**Time: 60 minutes**

Master eigenvalues and eigenvectors, crucial for PCA and many ML algorithms.

- Eigenvalue and eigenvector intuition
- Computing eigenvalues
- Eigendecomposition
- Singular Value Decomposition (SVD)
- Principal Component Analysis (PCA)

[Start Lesson 4](04-eigenvalues.md){ .md-button }

---

### Exercises
**Time: 3-4 hours**

Apply what you've learned through hands-on Jupyter notebook exercises.

- Exercise 1: Vector operations with NumPy
- Exercise 2: Matrix manipulations and transformations
- Exercise 3: PCA implementation from scratch

[View Exercises](exercises.md){ .md-button }

## Key Concepts

Throughout this module, you'll encounter these fundamental concepts:

| Concept | Description | ML Application |
|---------|-------------|----------------|
| **Vector** | Ordered list of numbers | Data samples, features |
| **Matrix** | 2D array of numbers | Datasets, weight matrices |
| **Dot Product** | Sum of element-wise products | Similarity measures |
| **Matrix Multiplication** | Combine transformations | Neural network layers |
| **Eigenvalues** | Scale factors for eigenvectors | PCA, stability analysis |
| **Eigenvectors** | Special directions preserved by transformation | Principal components |

## What You'll Build

By the end of this module, you'll implement:

1. **Vector Operations Library**: Functions for vector math
2. **Matrix Transformation Visualizer**: See how matrices transform space
3. **PCA from Scratch**: Dimensionality reduction without sklearn

## Tips for Success

!!! tip "Visualization is Key"
    Linear algebra becomes much clearer when you visualize it. Use matplotlib to plot vectors, transformations, and data projections.

!!! tip "Start Small"
    Begin with 2D examples before moving to higher dimensions. It's easier to visualize and debug.

!!! tip "NumPy Broadcasting"
    Learn how NumPy broadcasting works early. It will save you from writing many loops.

!!! warning "Common Pitfall"
    Matrix multiplication is not commutative: `A @ B ≠ B @ A`. Always pay attention to order and dimensions.

## Estimated Time

- **Reading lessons:** 3-4 hours
- **Completing exercises:** 3-4 hours
- **Total:** 6-8 hours

Take your time! Understanding this material deeply will make the rest of the course much easier.

## Resources

### Recommended Videos
- [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) - Excellent visual explanations
- [Khan Academy - Linear Algebra](https://www.khanacademy.org/math/linear-algebra) - Comprehensive tutorials

### Additional Reading
- [Linear Algebra Review (Stanford CS229)](http://cs229.stanford.edu/section/cs229-linalg.pdf)
- [NumPy for MATLAB Users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)

## Ready to Start?

Let's begin with the fundamentals of vectors!

[Start Lesson 1: Vectors](01-vectors.md){ .md-button .md-button--primary }

---

**Questions?** Open an issue on [GitHub](https://github.com/jumpingsphinx/jumpingsphinx.github.io/issues).
