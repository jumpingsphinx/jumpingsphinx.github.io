# Lesson 3: Advanced Matrix Operations

## Introduction

This lesson covers advanced matrix operations essential for machine learning: inverses, determinants, rank, and matrix decompositions.

## Matrix Inverse

The inverse of a matrix $A$ is denoted $A^{-1}$ and satisfies:

$$AA^{-1} = A^{-1}A = I$$

### Computing Inverse in NumPy

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
A_inv = np.linalg.inv(A)

# Verify: A @ A_inv should be identity
print(A @ A_inv)
# [[1, 0],
#  [0, 1]]
```

**ML Application:** Solving normal equations in linear regression:

$$\hat{w} = (X^TX)^{-1}X^Ty$$

## Determinant

The determinant measures how much a matrix scales space:

```python
A = np.array([[1, 2],
              [3, 4]])
det_A = np.linalg.det(A)  # -2.0
```

**Properties:**
- If $\det(A) = 0$, matrix is singular (not invertible)
- If $|\det(A)| > 1$, transformation expands space
- If $|\det(A)| < 1$, transformation contracts space

## Rank and Linear Independence

The **rank** is the number of linearly independent rows/columns:

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
rank = np.linalg.matrix_rank(A)  # 2 (not full rank!)
```

**ML Application:** Identifying redundant features.

## Matrix Decompositions

### LU Decomposition

Decomposes $A$ into lower and upper triangular matrices:

$$A = LU$$

```python
from scipy.linalg import lu

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
P, L, U = lu(A)
```

### QR Decomposition

Decomposes $A$ into orthogonal and upper triangular matrices:

$$A = QR$$

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])
Q, R = np.linalg.qr(A)
```

**ML Application:** Gram-Schmidt process, solving least squares.

## Solving Systems of Linear Equations

### The Problem

Solve $Ax = b$ for $x$:

```python
A = np.array([[3, 1],
              [1, 2]])
b = np.array([9, 8])

# Method 1: Using solve (most efficient)
x = np.linalg.solve(A, b)  # [2, 3]

# Method 2: Using inverse (less efficient)
x = np.linalg.inv(A) @ b

# Method 3: Using least squares (works for non-square matrices)
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
```

**ML Application:** This is exactly what linear regression does!

## Pseudo-Inverse (Moore-Penrose Inverse)

For non-square or singular matrices:

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])  # 3x2 matrix (not square)

A_pinv = np.linalg.pinv(A)  # 2x3 pseudo-inverse

# Property: A @ A_pinv @ A ≈ A
```

**ML Application:** Solving overdetermined systems in linear regression.

## Key Takeaways

!!! success "Important Concepts"
    - Matrix inverse: $AA^{-1} = I$ (when it exists)
    - Determinant: Measures scaling factor of transformation
    - Rank: Number of linearly independent rows/columns
    - Decompositions: LU, QR for efficient computation
    - Solving $Ax = b$ is central to many ML algorithms

## Next Steps

[Next: Lesson 4 - Eigenvalues & Eigenvectors](04-eigenvalues.md){ .md-button .md-button--primary }

[Back: Lesson 2 - Matrices](02-matrices.md){ .md-button }
