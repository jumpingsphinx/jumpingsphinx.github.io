# Lesson 3: Advanced Matrix Operations

## Introduction

This lesson covers advanced matrix operations essential for machine learning: inverses, determinants, rank, and matrix decompositions. These concepts underpin many ML algorithms including linear regression, PCA, and neural network optimization.

## Matrix Inverse

The inverse of a matrix $A$ is denoted $A^{-1}$ and satisfies:

$$AA^{-1} = A^{-1}A = I$$

Think of it as the matrix equivalent of reciprocal: just like $a \cdot \frac{1}{a} = 1$ for numbers, we have $A \cdot A^{-1} = I$ for matrices.

### When Does an Inverse Exist?

A matrix has an inverse **only if**:
1. It's a **square matrix** (same number of rows and columns)
2. It's **non-singular** (determinant ≠ 0, full rank)

### Computing and Verifying Inverse

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Example matrix
A = np.array([[1, 2],
              [3, 4]])

# Compute inverse
A_inv = np.linalg.inv(A)

print("Matrix A:")
print(A)
print("\nInverse A^-1:")
print(A_inv)

# Verify: A @ A_inv should be identity
result = A @ A_inv
print("\nA @ A^-1 (should be identity):")
print(result)
print("\nRounded (cleaner view):")
print(np.round(result, 10))

# Also verify A_inv @ A = I
result2 = A_inv @ A
print("\nA^-1 @ A (should also be identity):")
print(np.round(result2, 10))
```
</div>

### ML Application: Solving Normal Equations

The inverse is used in the **closed-form solution** for linear regression:

$$\hat{w} = (X^TX)^{-1}X^Ty$$

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Simple linear regression problem
# Data: y = 2*x1 + 3*x2 + noise
X = np.array([[1, 1],
              [1, 2],
              [2, 2],
              [2, 3]])  # 4 samples, 2 features

y = np.array([5, 8, 9, 13])  # Target values

print("Data matrix X:")
print(X)
print("\nTarget y:")
print(y)

# Solve using normal equations: w = (X^T X)^-1 X^T y
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
Xty = X.T @ y
w = XtX_inv @ Xty

print("\nOptimal weights w:")
print(w)
print(f"This means: y ≈ {w[0]:.2f}*x1 + {w[1]:.2f}*x2")

# Verify predictions
y_pred = X @ w
print("\nPredictions vs Actual:")
for i in range(len(y)):
    print(f"  Sample {i+1}: pred={y_pred[i]:.2f}, actual={y[i]}")
```
</div>

!!! warning "Computational Considerations"
    In practice, we rarely compute the inverse explicitly. Methods like `np.linalg.solve()` are faster and more numerically stable!

## Determinant

The **determinant** is a scalar value that encodes important information about a matrix. It tells us:

1. **Whether a matrix is invertible** (det ≠ 0 means invertible)
2. **How much the matrix scales volumes** (area in 2D, volume in 3D)
3. **Whether the transformation preserves orientation** (sign of det)

### Visual Understanding of the Determinant

Before diving into the calculations, watch this intuitive explanation of what the determinant really represents geometrically:

<div style="text-align: center; margin: 20px 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/Ip3X9LOh2dk" title="The determinant" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

### Geometric Interpretation

For a 2×2 matrix representing a transformation, $|\det(A)|$ is the **area scaling factor**. For 3×3, it's the **volume scaling factor**.

### How to Calculate the Determinant

**For 2×2 matrices:**

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**For 3×3 matrices** (cofactor expansion along first row):

$$\det\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = a(ei-fh) - b(di-fg) + c(dh-eg)$$

For larger matrices, we use recursive methods or specialized algorithms.

### Computing Determinants

<div class="python-interactive" markdown="1">
```python
import numpy as np

# 2x2 matrix
A = np.array([[1, 2],
              [3, 4]])

det_A = np.linalg.det(A)
print("2x2 Matrix A:")
print(A)
print(f"\nDeterminant: {det_A}")

# Manual calculation for 2x2: ad - bc
manual_det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
print(f"Manual: {A[0,0]}*{A[1,1]} - {A[0,1]}*{A[1,0]} = {manual_det}")

# 3x3 matrix
B = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

det_B = np.linalg.det(B)
print(f"\n3x3 Matrix B:")
print(B)
print(f"Determinant: {det_B:.1f}")
```
</div>

### Interpreting the Determinant

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Test different scenarios
matrices = {
    "Identity (no scaling)": np.array([[1, 0], [0, 1]]),
    "Scale by 2": np.array([[2, 0], [0, 2]]),
    "Scale by 0.5": np.array([[0.5, 0], [0, 0.5]]),
    "Singular (not invertible)": np.array([[1, 2], [2, 4]]),
    "Reflection (flips orientation)": np.array([[1, 0], [0, -1]]),
}

for name, M in matrices.items():
    det = np.linalg.det(M)
    print(f"\n{name}:")
    print(M)
    print(f"  Determinant: {det:.2f}")

    if abs(det) < 1e-10:
        print("  → Singular! Not invertible, collapses space")
    elif det > 0:
        print(f"  → Scales area by {abs(det):.2f}x, preserves orientation")
    else:
        print(f"  → Scales area by {abs(det):.2f}x, flips orientation")
```
</div>

### Properties and ML Applications

**Key Properties:**
- $\det(I) = 1$ (identity matrix)
- $\det(AB) = \det(A) \cdot \det(B)$
- $\det(A^T) = \det(A)$
- $\det(A^{-1}) = \frac{1}{\det(A)}$
- If $\det(A) = 0$: matrix is **singular** (not invertible, columns/rows are linearly dependent)
- If $|\det(A)| > 1$: transformation **expands** space
- If $|\det(A)| < 1$: transformation **contracts** space

**ML Applications:**
- **Checking invertibility**: Before computing $A^{-1}$, check if $\det(A) \neq 0$
- **Numerical stability**: Small determinants indicate ill-conditioned matrices
- **Volume calculations**: Used in probability (multivariate Gaussian)
- **Eigenvalue relationships**: $\det(A) = \prod \lambda_i$ (product of eigenvalues)

## Rank and Linear Independence

The **rank** of a matrix is the **number of linearly independent rows (or columns)**. It tells us the dimension of the vector space spanned by the rows/columns.

### Visual Understanding of Column Space and Rank

To build intuition about what rank means and how it relates to the column space of a matrix, watch this visual explanation:

<div style="text-align: center; margin: 20px 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/uQhTuRlWMxw" title="Inverse matrices, column space and null space" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

### What Does Rank Tell Us?

- **Full rank**: $\text{rank}(A) = \min(\text{rows}, \text{cols})$ → All rows/columns are independent
- **Rank deficient**: $\text{rank}(A) < \min(\text{rows}, \text{cols})$ → Some rows/columns are redundant
- For square matrices: **Full rank ⟺ Invertible ⟺ det ≠ 0**

### Computing Rank

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Full rank matrix
A_full = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])

# Rank deficient matrix (3rd row = 1st row + 2nd row)
A_deficient = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [5, 7, 9]])  # [5,7,9] = [1,2,3] + [4,5,6]

# Another rank deficient (columns are dependent)
A_deficient2 = np.array([[1, 2, 3],
                         [2, 4, 6],
                         [3, 6, 9]])  # Each col is multiple of col 1

print("Full rank matrix:")
print(A_full)
print(f"Rank: {np.linalg.matrix_rank(A_full)} (out of {min(A_full.shape)})")
print(f"Determinant: {np.linalg.det(A_full):.1f}\n")

print("Rank deficient matrix 1 (dependent row):")
print(A_deficient)
print(f"Rank: {np.linalg.matrix_rank(A_deficient)} (out of {min(A_deficient.shape)})")
print(f"Determinant: {np.linalg.det(A_deficient):.10f}\n")

print("Rank deficient matrix 2 (dependent columns):")
print(A_deficient2)
print(f"Rank: {np.linalg.matrix_rank(A_deficient2)} (out of {min(A_deficient2.shape)})")
print(f"Determinant: {np.linalg.det(A_deficient2):.10f}")

print("\n💡 Notice: Rank deficient ⟺ Determinant = 0")
```
</div>

### ML Application: Feature Redundancy

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Dataset with redundant features
# Feature 1: height in cm
# Feature 2: weight in kg
# Feature 3: height in meters (redundant! = feature 1 / 100)
# Feature 4: BMI (not redundant, calculated from 1 and 2)

X = np.array([[170, 70, 1.70, 24.2],  # Sample 1
              [180, 80, 1.80, 24.7],  # Sample 2
              [165, 60, 1.65, 22.0],  # Sample 3
              [175, 75, 1.75, 24.5]]) # Sample 4

print("Dataset shape:", X.shape)  # 4 samples, 4 features
print("Dataset:\n", X)

rank = np.linalg.matrix_rank(X)
print(f"\nRank: {rank} (out of {min(X.shape)} possible)")

if rank < min(X.shape):
    print(f"⚠️ Only {rank} linearly independent features!")
    print("   Feature 3 (height_m) is redundant with Feature 1 (height_cm)")
    print("   This can cause problems in regression!")
else:
    print("✓ All features are linearly independent")
```
</div>

**Why This Matters in ML:**
- **Multicollinearity**: Redundant features cause numerical instability in regression
- **Dimensionality reduction**: Low rank suggests we can use fewer features
- **Matrix inversion**: Can't invert rank-deficient matrices (used in linear regression)
- **Feature selection**: Helps identify which features are truly independent

## Matrix Decompositions

Matrix decompositions (factorizations) break a matrix into simpler components. They're fundamental to numerical linear algebra and ML algorithms.

### LU Decomposition

**LU decomposition** factors a matrix into:
- **L**: Lower triangular matrix (zeros above diagonal)
- **U**: Upper triangular matrix (zeros below diagonal)

$$A = LU$$

Or more commonly with pivoting: $PA = LU$ where $P$ is a permutation matrix.

**Why is this useful?**
- Solving $Ax = b$ becomes easier: solve $Ly = b$, then $Ux = y$
- Triangular systems are fast to solve (forward/backward substitution)
- Computing determinant: $\det(A) = \det(L) \cdot \det(U) = \prod U_{ii}$

<div class="python-interactive" markdown="1">
```python
import numpy as np
from scipy.linalg import lu

# Original matrix
A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

print("Original matrix A:")
print(A)

# LU decomposition with pivoting
P, L, U = lu(A)

print("\nPermutation matrix P:")
print(P)
print("\nLower triangular L:")
print(L)
print("\nUpper triangular U:")
print(U)

# Verify: PA = LU
PA = P @ A
LU_product = L @ U
print("\nP @ A:")
print(PA)
print("\nL @ U:")
print(LU_product)
print("\nAre they equal?", np.allclose(PA, LU_product))

# Use LU to solve Ax = b
b = np.array([4, 10, 24])
from scipy.linalg import lu_solve
x = lu_solve((L @ np.linalg.inv(P), U), b)
print(f"\nSolving Ax = b for b = {b}")
print(f"Solution x: {x}")
print(f"Verification A @ x = {A @ x}")
```
</div>

**ML Application:** LU decomposition is used internally by `np.linalg.solve()` for efficient system solving.

### QR Decomposition

**QR decomposition** factors a matrix into:
- **Q**: Orthogonal/orthonormal matrix ($Q^TQ = I$, columns are perpendicular unit vectors)
- **R**: Upper triangular matrix

$$A = QR$$

**What is an Orthogonal/Orthonormal Matrix?**

A matrix $Q$ is **orthonormal** if:
1. Its columns are **unit vectors** (length 1): $\|q_i\| = 1$
2. Its columns are **orthogonal** (perpendicular): $q_i^T q_j = 0$ for $i \neq j$

This means: $Q^T Q = I$ (and $Q^T = Q^{-1}$ for square matrices)

**Properties of orthonormal matrices:**
- Preserve lengths: $\|Qx\| = \|x\|$
- Preserve angles and distances
- Numerically stable (condition number = 1)
- Easy to invert: just transpose!

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Original matrix
A = np.array([[1, 1],
              [1, 2],
              [1, 3]], dtype=float)

print("Original matrix A:")
print(A)
print(f"Shape: {A.shape}")

# QR decomposition
Q, R = np.linalg.qr(A)

print("\nOrthonormal matrix Q:")
print(Q)
print(f"Shape: {Q.shape}")

print("\nUpper triangular R:")
print(R)
print(f"Shape: {R.shape}")

# Verify: A = QR
QR_product = Q @ R
print("\nQ @ R:")
print(QR_product)
print(f"Equal to A? {np.allclose(A, QR_product)}")

# Verify Q is orthonormal: Q^T Q = I
QtQ = Q.T @ Q
print("\nQ^T @ Q (should be identity):")
print(QtQ)
print(f"Is Q orthonormal? {np.allclose(QtQ, np.eye(Q.shape[1]))}")

# Check individual columns are unit vectors
print("\nColumn norms (should all be 1):")
for i in range(Q.shape[1]):
    norm = np.linalg.norm(Q[:, i])
    print(f"  Column {i}: {norm:.10f}")

# Check columns are orthogonal
print("\nDot products between columns (should be 0):")
for i in range(Q.shape[1]):
    for j in range(i+1, Q.shape[1]):
        dot = np.dot(Q[:, i], Q[:, j])
        print(f"  Column {i} · Column {j}: {dot:.10f}")
```
</div>

### The Gram-Schmidt Process

QR decomposition is intimately related to the **Gram-Schmidt process**, which converts a set of linearly independent vectors into an orthonormal set.

**Algorithm:**
1. Take first vector, normalize it to unit length
2. Take second vector, subtract its projection onto first, then normalize
3. Continue: subtract projections onto all previous vectors, then normalize

<div class="python-interactive" markdown="1">
```python
import numpy as np

def gram_schmidt(vectors):
    """Gram-Schmidt orthogonalization"""
    orthonormal = []

    for v in vectors:
        # Subtract projections onto all previous orthonormal vectors
        u = v.copy()
        for q in orthonormal:
            projection = np.dot(v, q) * q
            u = u - projection

        # Normalize to unit length
        u = u / np.linalg.norm(u)
        orthonormal.append(u)

    return np.array(orthonormal).T

# Start with linearly independent vectors
v1 = np.array([1, 1, 1], dtype=float)
v2 = np.array([1, 2, 0], dtype=float)
v3 = np.array([1, 0, 3], dtype=float)

A = np.column_stack([v1, v2, v3])
print("Original vectors (as columns):")
print(A)

# Apply Gram-Schmidt
Q_manual = gram_schmidt([v1, v2, v3])
print("\nOrthonormalized vectors:")
print(Q_manual)

# Compare with NumPy's QR
Q_numpy, _ = np.linalg.qr(A)
print("\nNumPy QR's Q (might differ by sign):")
print(Q_numpy)

# Verify orthonormality
print("\nQ^T @ Q (should be identity):")
print(Q_manual.T @ Q_manual)
```
</div>

### ML Applications of QR Decomposition

**1. Solving Least Squares** (most important for ML!)

Linear regression minimizes $\|Ax - b\|^2$. The solution is:

$$x = (A^TA)^{-1}A^Tb$$

But computing $A^TA$ can be numerically unstable. With QR:

$$A = QR \Rightarrow x = R^{-1}Q^Tb$$

This is more stable because $Q$ is orthonormal.

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Least squares problem: find x that minimizes ||Ax - b||²
# (Overdetermined system: more equations than unknowns)

A = np.array([[1, 1],
              [1, 2],
              [1, 3],
              [1, 4]], dtype=float)

b = np.array([2.1, 3.9, 6.2, 7.8])

print("System Ax = b:")
print(f"A shape: {A.shape} (overdetermined: 4 equations, 2 unknowns)")
print(f"b: {b}\n")

# Method 1: Normal equations (less stable)
x_normal = np.linalg.inv(A.T @ A) @ A.T @ b
print("Method 1 - Normal equations:")
print(f"x = {x_normal}")
residual1 = np.linalg.norm(A @ x_normal - b)
print(f"Residual: {residual1:.6f}")

# Method 2: QR decomposition (more stable)
Q, R = np.linalg.qr(A)
x_qr = np.linalg.inv(R) @ Q.T @ b
print("\nMethod 2 - QR decomposition:")
print(f"x = {x_qr}")
residual2 = np.linalg.norm(A @ x_qr - b)
print(f"Residual: {residual2:.6f}")

# Method 3: NumPy's lstsq (uses SVD, even better)
x_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
print("\nMethod 3 - lstsq (SVD-based):")
print(f"x = {x_lstsq}")
residual3 = np.linalg.norm(A @ x_lstsq - b)
print(f"Residual: {residual3:.6f}")

print(f"\n💡 All methods give same answer, but QR/SVD are more numerically stable!")
```
</div>

**2. Computing Determinant:**
$$\det(A) = \det(Q) \cdot \det(R) = \pm \prod R_{ii}$$

**3. Eigenvalue Algorithms:** QR algorithm iteratively computes eigenvalues

**4. Feature Orthogonalization:** Create independent features for better model performance

## Solving Systems of Linear Equations

Solving $Ax = b$ for $x$ is **everywhere** in machine learning. It's the foundation of:
- Linear regression (finding optimal weights)
- Neural network training (computing gradients)
- Least squares optimization
- Computer graphics (transformations)

### Three Types of Systems

1. **Exactly determined** ($m = n$, square $A$): Unique solution if $A$ is invertible
2. **Overdetermined** ($m > n$, tall $A$): More equations than unknowns → least squares solution
3. **Underdetermined** ($m < n$, wide $A$): More unknowns than equations → infinite solutions

### Exactly Determined Systems

<div class="python-interactive" markdown="1">
```python
import numpy as np

# System: 3x + y = 9
#         x + 2y = 8
# Solution: x=2, y=3

A = np.array([[3, 1],
              [1, 2]], dtype=float)
b = np.array([9, 8], dtype=float)

print("System of equations:")
print("  3x + y = 9")
print("  x + 2y = 8")
print(f"\nMatrix A:\n{A}")
print(f"Vector b: {b}")

# Method 1: Using solve (most efficient - uses LU)
x_solve = np.linalg.solve(A, b)
print(f"\nMethod 1 - np.linalg.solve:")
print(f"Solution: x = {x_solve}")

# Method 2: Using inverse (less efficient)
A_inv = np.linalg.inv(A)
x_inv = A_inv @ b
print(f"\nMethod 2 - Inverse:")
print(f"A^-1:\n{A_inv}")
print(f"Solution: x = {x_inv}")

# Method 3: Manual (Cramer's rule for 2x2)
det_A = np.linalg.det(A)
x_cramer = np.array([
    np.linalg.det(np.column_stack([b, A[:, 1]])) / det_A,
    np.linalg.det(np.column_stack([A[:, 0], b])) / det_A
])
print(f"\nMethod 3 - Cramer's rule:")
print(f"Solution: x = {x_cramer}")

# Verify the solution
verification = A @ x_solve
print(f"\nVerification: A @ x = {verification}")
print(f"Expected b = {b}")
print(f"Match: {np.allclose(verification, b)}")

print("\n💡 Prefer np.linalg.solve - it's fastest and most stable!")
```
</div>

### Overdetermined Systems (Least Squares)

When there are **more equations than unknowns**, there's usually no exact solution. Instead, we find the $x$ that **minimizes the error** $\|Ax - b\|^2$.

This is **linear regression**!

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Overdetermined system: fit a line y = mx + c to data points
# We have 5 points, but only 2 parameters (m, c)

# Data points (x, y)
x_data = np.array([1, 2, 3, 4, 5], dtype=float)
y_data = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

# Build system: [1 x] @ [c; m] = y
A = np.column_stack([np.ones_like(x_data), x_data])
b = y_data

print("Fitting line y = mx + c to data:")
print(f"x: {x_data}")
print(f"y: {y_data}")
print(f"\nMatrix A (augmented with 1s for intercept):\n{A}")
print(f"Shape: {A.shape} - overdetermined!")

# Solve least squares
params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
c, m = params

print(f"\nBest fit line: y = {m:.2f}x + {c:.2f}")
print(f"Residual sum of squares: {residuals[0]:.4f}")

# Compute predictions and error
y_pred = A @ params
errors = b - y_pred
print(f"\nPredictions: {y_pred}")
print(f"Errors: {errors}")
print(f"RMS error: {np.sqrt(np.mean(errors**2)):.4f}")

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(x_data, y_data, color='blue', s=100, label='Data points', zorder=3)
plt.plot(x_data, y_pred, color='red', linewidth=2, label=f'Best fit: y={m:.2f}x+{c:.2f}')

# Show residuals
for i in range(len(x_data)):
    plt.plot([x_data[i], x_data[i]], [y_data[i], y_pred[i]],
             'gray', linestyle='--', alpha=0.5)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Linear Regression via Least Squares')
plt.show()

print("\n🎯 This IS machine learning - fitting a model to data!")
```
</div>

### Underdetermined Systems

When there are **more unknowns than equations**, there are infinitely many solutions. We typically find the **minimum norm solution** (smallest $\|x\|$).

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Underdetermined: 1 equation, 2 unknowns
# x + 2y = 5
# Infinitely many solutions: y = (5-x)/2

A = np.array([[1, 2]], dtype=float)  # 1x2 matrix
b = np.array([5], dtype=float)

print("Underdetermined system:")
print("  x + 2y = 5")
print(f"A shape: {A.shape} (1 equation, 2 unknowns)")

# Find minimum norm solution using lstsq
x_min, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
print(f"\nMinimum norm solution: {x_min}")
print(f"Norm: {np.linalg.norm(x_min):.4f}")

# Verify it satisfies the equation
print(f"A @ x = {A @ x_min}, b = {b}")
print(f"Satisfies equation: {np.allclose(A @ x_min, b)}")

# Show other solutions have larger norm
other_solutions = [
    np.array([1, 2]),
    np.array([3, 1]),
    np.array([5, 0]),
    np.array([-3, 4])
]

print("\nOther valid solutions (infinite possibilities):")
for sol in other_solutions:
    if np.allclose(A @ sol, b):
        print(f"  {sol} → norm = {np.linalg.norm(sol):.4f}")

print(f"\n💡 Minimum norm solution has smallest magnitude!")
```
</div>

## Pseudo-Inverse (Moore-Penrose Inverse)

The **pseudo-inverse** $A^+$ generalizes the inverse to **non-square** and **singular** matrices.

For any matrix $A$ (any shape!):
- $A A^+ A = A$ (always holds)
- If $A$ is invertible: $A^+ = A^{-1}$
- If $A$ is overdetermined: $A^+ = (A^T A)^{-1} A^T$ (left inverse)
- If $A$ is underdetermined: $A^+ = A^T(A A^T)^{-1}$ (right inverse)

### Computing the Pseudo-Inverse

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Example 1: Tall matrix (overdetermined)
A_tall = np.array([[1, 2],
                   [3, 4],
                   [5, 6]], dtype=float)

print("Tall matrix (overdetermined):")
print(A_tall)
print(f"Shape: {A_tall.shape}")

A_tall_pinv = np.linalg.pinv(A_tall)
print(f"\nPseudo-inverse shape: {A_tall_pinv.shape}")
print(A_tall_pinv)

# Verify property: A @ A+ @ A = A
result = A_tall @ A_tall_pinv @ A_tall
print(f"\nA @ A+ @ A =\n{result}")
print(f"Equal to A? {np.allclose(result, A_tall)}")

# Example 2: Wide matrix (underdetermined)
A_wide = np.array([[1, 2, 3],
                   [4, 5, 6]], dtype=float)

print("\n" + "="*50)
print("Wide matrix (underdetermined):")
print(A_wide)
print(f"Shape: {A_wide.shape}")

A_wide_pinv = np.linalg.pinv(A_wide)
print(f"\nPseudo-inverse shape: {A_wide_pinv.shape}")
print(A_wide_pinv)

# Verify
result = A_wide @ A_wide_pinv @ A_wide
print(f"\nA @ A+ @ A =\n{result}")
print(f"Equal to A? {np.allclose(result, A_wide)}")

# Example 3: Singular square matrix
A_singular = np.array([[1, 2],
                       [2, 4]], dtype=float)

print("\n" + "="*50)
print("Singular square matrix:")
print(A_singular)
print(f"Determinant: {np.linalg.det(A_singular)} (not invertible!)")

A_singular_pinv = np.linalg.pinv(A_singular)
print(f"\nPseudo-inverse:")
print(A_singular_pinv)

# Can't use regular inverse, but pseudo-inverse works!
result = A_singular @ A_singular_pinv @ A_singular
print(f"\nA @ A+ @ A =\n{result}")
print(f"Equal to A? {np.allclose(result, A_singular)}")
```
</div>

### ML Application: Solving Any Linear System

The pseudo-inverse provides a **unified way** to solve $Ax = b$ for **any** matrix:

$$x = A^+ b$$

- If exactly determined and invertible: gives exact solution
- If overdetermined: gives least squares solution
- If underdetermined: gives minimum norm solution

<div class="python-interactive" markdown="1">
```python
import numpy as np

print("Demo: Solving Ax = b with pseudo-inverse\n")

# Case 1: Overdetermined (regression problem)
print("Case 1: Overdetermined (4 equations, 2 unknowns)")
A1 = np.array([[1, 1],
               [1, 2],
               [1, 3],
               [1, 4]], dtype=float)
b1 = np.array([2.1, 3.9, 6.2, 7.8])

x1 = np.linalg.pinv(A1) @ b1
print(f"Solution: {x1}")
print(f"Residual norm: {np.linalg.norm(A1 @ x1 - b1):.4f}")

# Case 2: Exactly determined
print("\nCase 2: Exactly determined (2 equations, 2 unknowns)")
A2 = np.array([[3, 1],
               [1, 2]], dtype=float)
b2 = np.array([9, 8], dtype=float)

x2 = np.linalg.pinv(A2) @ b2
print(f"Solution: {x2}")
print(f"Residual norm: {np.linalg.norm(A2 @ x2 - b2):.10f} (exact!)")

# Case 3: Underdetermined
print("\nCase 3: Underdetermined (1 equation, 2 unknowns)")
A3 = np.array([[1, 2]], dtype=float)
b3 = np.array([5], dtype=float)

x3 = np.linalg.pinv(A3) @ b3
print(f"Solution: {x3}")
print(f"Solution norm: {np.linalg.norm(x3):.4f} (minimum!)")
print(f"Residual norm: {np.linalg.norm(A3 @ x3 - b3):.10f}")

print("\n🎯 Pseudo-inverse handles ALL cases!")
```
</div>

**Why Pseudo-Inverse Matters in ML:**
- Works for **rank-deficient** data matrices (when features are correlated)
- Handles **overdetermined** systems (more data than parameters - typical in ML!)
- Provides **best-fit** solutions when no exact solution exists
- Used in **ridge regression**, **PCA**, and **matrix completion**

## Key Takeaways

!!! success "Important Concepts"
    **Matrix Inverse:**
    - $AA^{-1} = I$ for square, non-singular matrices
    - Used in normal equations: $w = (X^TX)^{-1}X^Ty$
    - Prefer `np.linalg.solve()` over computing inverse directly

    **Determinant:**
    - Measures volume scaling: $\det(A) = 0$ means singular
    - For 2×2: $\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$
    - Sign indicates orientation preservation

    **Rank:**
    - Number of linearly independent rows/columns
    - Rank deficiency indicates feature redundancy
    - Full rank ⟺ Invertible (for square matrices)

    **LU Decomposition:**
    - $A = LU$ (lower × upper triangular)
    - Efficient for solving systems
    - Used internally by `solve()`

    **QR Decomposition:**
    - $A = QR$ (orthonormal × upper triangular)
    - $Q$ preserves lengths and angles
    - Most stable method for least squares
    - Gram-Schmidt creates orthonormal vectors

    **Solving Linear Systems:**
    - Exactly determined: unique solution if invertible
    - Overdetermined: least squares solution (linear regression!)
    - Underdetermined: minimum norm solution
    - Use pseudo-inverse for any case: $x = A^+b$

## Practice Problems

Try these to test your understanding:

1. Compute the determinant and inverse of $\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$
2. Check if $\begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & 1 \end{bmatrix}$ is full rank
3. Use QR decomposition to solve a least squares problem
4. Find the pseudo-inverse of a 3×2 matrix

## Next Steps

Now that you understand advanced matrix operations, let's explore eigenvalues and eigenvectors - the foundation of PCA and many other ML algorithms!

[Next: Lesson 4 - Eigenvalues & Eigenvectors](04-eigenvalues.md){ .md-button .md-button--primary }

[Back: Lesson 2 - Matrices](02-matrices.md){ .md-button }
