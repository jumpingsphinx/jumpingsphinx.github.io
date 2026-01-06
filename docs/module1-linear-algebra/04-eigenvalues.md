# Lesson 4: Eigenvalues and Eigenvectors

## Introduction

Eigenvalues and eigenvectors are among the **most profound concepts** in linear algebra and machine learning. They reveal the fundamental "directions" and "scaling factors" hidden within matrices, and they're the mathematical foundation for:

- **Principal Component Analysis (PCA)** - dimensionality reduction
- **Spectral clustering** - grouping similar data
- **Google's PageRank** - ranking web pages
- **Quantum mechanics** - describing particle states
- **Stability analysis** - understanding dynamic systems

## What Are Eigenvalues and Eigenvectors?

### The Definition

For a square matrix $A$, an **eigenvector** $\vec{v}$ and its corresponding **eigenvalue** $\lambda$ satisfy:

$$A\vec{v} = \lambda\vec{v}$$

In words: **When you multiply matrix $A$ by eigenvector $\vec{v}$, you get the same vector back, just scaled by $\lambda$.**

### Why This Is Special

Most vectors **change direction** when transformed by a matrix. But eigenvectors are **special directions** that remain on the same line - they only get stretched or shrunk.

- **Eigenvector** ($\vec{v}$): The special direction
- **Eigenvalue** ($\lambda$): How much it gets scaled
  - $\lambda > 1$: vector gets stretched
  - $0 < \lambda < 1$: vector gets shrunk
  - $\lambda < 0$: vector gets flipped and scaled
  - $\lambda = 0$: vector collapses to zero (matrix is singular!)

### Geometric Interpretation

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Diagonal matrix - eigenvectors are axis-aligned
A = np.array([[2, 0],
              [0, 3]])

# First eigenvector: along x-axis
v1 = np.array([1, 0])
result1 = A @ v1

print("First eigenvector v1 =", v1)
print("A @ v1 =", result1)
print(f"Is it just scaled? {result1[0]}/{v1[0]} = {result1[0]/v1[0]}")
print(f"Eigenvalue λ1 = {result1[0]/v1[0]}\n")

# Second eigenvector: along y-axis
v2 = np.array([0, 1])
result2 = A @ v2

print("Second eigenvector v2 =", v2)
print("A @ v2 =", result2)
print(f"Is it just scaled? {result2[1]}/{v2[1]} = {result2[1]/v2[1]}")
print(f"Eigenvalue λ2 = {result2[1]/v2[1]}")

print("\n💡 Diagonal matrices have eigenvectors along coordinate axes!")
```
</div>

### The Characteristic Equation

To find eigenvalues, we solve:

$$\det(A - \lambda I) = 0$$

This gives us a polynomial whose roots are the eigenvalues.

**For a 2×2 matrix:**

$$A = \begin{bmatrix} a & b \\\\ c & d \end{bmatrix}$$

The characteristic equation is:

$$\lambda^2 - (a+d)\lambda + (ad-bc) = 0$$

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Solve characteristic equation manually for 2x2
A = np.array([[4, 2],
              [1, 3]])

# Coefficients: λ² - trace(A)·λ + det(A) = 0
trace = np.trace(A)  # a + d = 7
det = np.linalg.det(A)  # ad - bc = 10

print(f"Matrix A:\n{A}")
print(f"\nCharacteristic equation: λ² - {trace}λ + {det} = 0")

# Solve using quadratic formula
discriminant = trace**2 - 4*det
lambda1 = (trace + np.sqrt(discriminant)) / 2
lambda2 = (trace - np.sqrt(discriminant)) / 2

print(f"\nManually computed eigenvalues:")
print(f"  λ1 = {lambda1}")
print(f"  λ2 = {lambda2}")

# Verify with NumPy
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"\nNumPy computed eigenvalues:")
print(f"  {eigenvalues}")
print(f"\nThey match! ✓")
```
</div>

## Computing Eigenvalues and Eigenvectors

NumPy provides `np.linalg.eig()` to compute eigenvalues and eigenvectors. The eigenvectors are returned as **columns** of a matrix.

<div class="python-interactive" markdown="1">
```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvectors (as columns):\n{eigenvectors}")

# Verify the eigenvalue equation: A*v = λ*v
print("\n--- Verification for first eigenvector ---")
v1 = eigenvectors[:, 0]  # First column
lambda1 = eigenvalues[0]

Av1 = A @ v1
lambda_v1 = lambda1 * v1

print(f"A @ v1 = {Av1}")
print(f"λ1 * v1 = {lambda_v1}")
print(f"Equal? {np.allclose(Av1, lambda_v1)}")

# Verify for second eigenvector
print("\n--- Verification for second eigenvector ---")
v2 = eigenvectors[:, 1]
lambda2 = eigenvalues[1]

Av2 = A @ v2
lambda_v2 = lambda2 * v2

print(f"A @ v2 = {Av2}")
print(f"λ2 * v2 = {lambda_v2}")
print(f"Equal? {np.allclose(Av2, lambda_v2)}")
```
</div>

### Finding Eigenvectors from Eigenvalues

Once you have an eigenvalue $\lambda$, find its eigenvector by solving:

$$(A - \lambda I)\vec{v} = \vec{0}$$

This is a system of linear equations with infinitely many solutions (any scalar multiple works).

<div class="python-interactive" markdown="1">
```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

# We know λ1 = 5 from above
lambda1 = 5

# Solve (A - λI)v = 0
A_minus_lambda_I = A - lambda1 * np.eye(2)
print(f"A - λ1*I =\n{A_minus_lambda_I}")

# This matrix should be singular (determinant = 0)
print(f"\nDeterminant: {np.linalg.det(A_minus_lambda_I):.10f}")
print("(approximately zero - the matrix is singular)")

# The null space gives us the eigenvector
# For this 2x2, we can see: [-1*v1 + 2*v2 = 0], so v2 = v1/2
# One eigenvector: [2, 1] (or any multiple)
v1_manual = np.array([2, 1])

print(f"\nManual eigenvector: {v1_manual}")
print(f"A @ v1 = {A @ v1_manual}")
print(f"λ1 * v1 = {lambda1 * v1_manual}")
print(f"Equal? {np.allclose(A @ v1_manual, lambda1 * v1_manual)}")
```
</div>

## Properties of Eigenvalues

Eigenvalues have several beautiful mathematical properties that connect to other matrix concepts.

### Key Properties

1. **Sum of eigenvalues = Trace** (sum of diagonal elements)
   $$\sum_{i=1}^{n} \lambda_i = \text{trace}(A) = \sum_{i=1}^{n} a_{ii}$$

2. **Product of eigenvalues = Determinant**
   $$\prod_{i=1}^{n} \lambda_i = \det(A)$$

3. **Symmetric matrices have real eigenvalues**
   - If $A = A^T$, all eigenvalues are real numbers (no complex numbers)

4. **Symmetric matrices have orthogonal eigenvectors**
   - If $A = A^T$, eigenvectors corresponding to different eigenvalues are perpendicular

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Create a matrix
A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print(f"\nEigenvalues: {eigenvalues}")

# Property 1: Sum = Trace
print(f"\n--- Property 1: Sum = Trace ---")
print(f"Sum of eigenvalues: {eigenvalues.sum():.6f}")
print(f"Trace of A: {np.trace(A):.6f}")
print(f"Equal? {np.allclose(eigenvalues.sum(), np.trace(A))}")

# Property 2: Product = Determinant
print(f"\n--- Property 2: Product = Determinant ---")
print(f"Product of eigenvalues: {np.prod(eigenvalues):.6f}")
print(f"Determinant of A: {np.linalg.det(A):.6f}")
print(f"Equal? {np.allclose(np.prod(eigenvalues), np.linalg.det(A))}")

# Property 3 & 4: Symmetric matrix
print(f"\n--- Properties 3 & 4: Symmetric Matrix ---")
S = np.array([[2, 1],
              [1, 2]])
print(f"Symmetric matrix S:\n{S}")

eigenvalues_s, eigenvectors_s = np.linalg.eig(S)
print(f"\nEigenvalues: {eigenvalues_s}")
print(f"All real? {all(np.isreal(eigenvalues_s))}")

# Check orthogonality of eigenvectors
v1 = eigenvectors_s[:, 0]
v2 = eigenvectors_s[:, 1]
dot_product = np.dot(v1, v2)
print(f"\nEigenvectors:\n{eigenvectors_s}")
print(f"Dot product of eigenvectors: {dot_product:.10f}")
print(f"Orthogonal? {np.allclose(dot_product, 0)}")
```
</div>

### Why These Properties Matter

- **Trace = Sum of eigenvalues**: Quick way to check eigenvalue computation
- **Determinant = Product**: If any eigenvalue is zero, the matrix is singular
- **Symmetric → Real eigenvalues**: No complex numbers to deal with in ML
- **Symmetric → Orthogonal eigenvectors**: Natural basis for the space (PCA uses this!)

## Eigendecomposition

**Eigendecomposition** (also called **spectral decomposition**) breaks a matrix into its eigenvectors and eigenvalues.

### The Formula

For a **diagonalizable matrix** $A$:

$$A = Q\Lambda Q^{-1}$$

where:
- $Q$: Matrix of eigenvectors as columns
- $\Lambda$: Diagonal matrix of eigenvalues
- $Q^{-1}$: Inverse of eigenvector matrix

**Important:** Not all matrices can be eigendecomposed (they must be diagonalizable).

### What Does This Mean?

Eigendecomposition says: **Any matrix $A$ can be broken down into:**
1. **Change of basis** → $Q^{-1}$ (transform to eigenvector space)
2. **Simple scaling** → $\Lambda$ (scale along each eigenvector)
3. **Change back** → $Q$ (return to original space)

In the eigenvector basis, matrix multiplication is just **scaling** - the simplest operation!

### Geometric Interpretation

<div class="python-interactive" markdown="1">
```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Original matrix A:")
print(A)
print(f"\nEigenvalues: {eigenvalues}")
print(f"\nEigenvector matrix Q:")
print(eigenvectors)

# Diagonal matrix of eigenvalues
Lambda = np.diag(eigenvalues)
print(f"\nDiagonal matrix Λ:")
print(Lambda)

# Reconstruct A = Q Λ Q^(-1)
Q_inv = np.linalg.inv(eigenvectors)
A_reconstructed = eigenvectors @ Lambda @ Q_inv

print(f"\nReconstructed A:")
print(A_reconstructed)
print(f"\nOriginal and reconstructed equal? {np.allclose(A, A_reconstructed)}")

# Show what each part does
print("\n--- What each transformation does ---")
v = np.array([1, 0])  # Test vector
print(f"Original vector: {v}")
print(f"After Q^(-1) (to eigen-basis): {Q_inv @ v}")
print(f"After Λ (scaling): {Lambda @ Q_inv @ v}")
print(f"After Q (back to original): {eigenvectors @ Lambda @ Q_inv @ v}")
print(f"Direct A @ v: {A @ v}")
print("They're the same!")
```
</div>

### Power of Eigendecomposition: Matrix Powers

Computing $A^n$ directly is expensive. With eigendecomposition, it's easy!

$$A^n = (Q\Lambda Q^{-1})^n = Q\Lambda^n Q^{-1}$$

And $\Lambda^n$ is trivial - just raise each eigenvalue to the $n$-th power:

$$\Lambda^n = \begin{bmatrix} \lambda_1^n & 0 \\\\ 0 & \lambda_2^n \end{bmatrix}$$

<div class="python-interactive" markdown="1">
```python
import numpy as np

A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)
Q = eigenvectors
Q_inv = np.linalg.inv(Q)

# Compute A^10 the hard way
A_10_direct = np.linalg.matrix_power(A, 10)

# Compute A^10 using eigendecomposition
Lambda_10 = np.diag(eigenvalues ** 10)
A_10_eigen = Q @ Lambda_10 @ Q_inv

print("A^10 (direct computation):")
print(A_10_direct)
print("\nA^10 (via eigendecomposition):")
print(A_10_eigen)
print(f"\nBoth methods equal? {np.allclose(A_10_direct, A_10_eigen)}")

# Show the power of eigenvalues
print(f"\nλ1^10 = {eigenvalues[0]**10:.2f}")
print(f"λ2^10 = {eigenvalues[1]**10:.2f}")
print("Eigendecomposition makes computing matrix powers trivial!")
```
</div>

### ML Application: Markov Chains

Eigendecomposition is used to find the **steady state** of Markov chains (e.g., PageRank).

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Transition matrix for a simple Markov chain
# State 1 → State 2 with prob 0.3, stays in State 1 with prob 0.7
# State 2 → State 1 with prob 0.4, stays in State 2 with prob 0.6
P = np.array([[0.7, 0.4],
              [0.3, 0.6]])

print("Transition matrix P:")
print(P)

# Find eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(P)

print(f"\nEigenvalues: {eigenvalues}")

# The steady state corresponds to eigenvalue = 1
steady_idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)
steady_state = eigenvectors[:, steady_idx]
steady_state = np.real(steady_state)  # Remove tiny imaginary parts
steady_state = steady_state / steady_state.sum()  # Normalize to sum to 1

print(f"\nSteady state: {steady_state}")
print(f"In the long run: {steady_state[0]*100:.1f}% in State 1, {steady_state[1]*100:.1f}% in State 2")

# Verify by running the Markov chain
initial = np.array([1, 0])  # Start in State 1
state = initial
for i in range(100):
    state = P @ state

print(f"\nAfter 100 steps from State 1: {state}")
print("It converges to the steady state!")
```
</div>

## Singular Value Decomposition (SVD)

**SVD is one of the most important matrix factorizations in machine learning.** Unlike eigendecomposition (which only works for square matrices), SVD works for **any matrix** - square, tall, wide, whatever!

### The SVD Formula

For **any** $m \times n$ matrix $A$:

$$A = U\Sigma V^T$$

where:
- $U$: $m \times m$ orthogonal matrix (left singular vectors)
  - Columns are eigenvectors of $AA^T$
- $\Sigma$: $m \times n$ diagonal matrix (singular values)
  - Diagonal entries are $\sigma_i = \sqrt{\lambda_i}$ where $\lambda_i$ are eigenvalues of $A^TA$
- $V^T$: $n \times n$ orthogonal matrix (right singular vectors)
  - Rows are eigenvectors of $A^TA$

### What Does SVD Mean?

SVD breaks **any linear transformation** into three simple steps:

1. **Rotate** (via $V^T$) → Align input with special directions
2. **Scale** (via $\Sigma$) → Stretch/shrink along each direction
3. **Rotate** (via $U$) → Rotate to output space

This is the **fundamental theorem** of linear algebra: every matrix is rotation + scaling + rotation!

### Relationship to Eigendecomposition

**Eigendecomposition**: $A = Q\Lambda Q^{-1}$ (square matrices only)
**SVD**: $A = U\Sigma V^T$ (any matrix!)

- SVD generalizes eigendecomposition to non-square matrices
- If $A$ is square and symmetric: $U = V = Q$ and $\Sigma = |\Lambda|$
- Singular values are always **real and non-negative**
- $U$ and $V$ are always **orthogonal** (columns are orthonormal)

### Computing SVD

<div class="python-interactive" markdown="1">
```python
import numpy as np

# A non-square matrix (4 rows, 3 columns)
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]], dtype=float)

print("Original matrix A (4×3):")
print(A)

# Compute SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)

print(f"\nU shape: {U.shape}   (left singular vectors)")
print(f"S shape: {S.shape}   (singular values)")
print(f"VT shape: {VT.shape}  (right singular vectors)")

print(f"\nSingular values: {S}")

# Reconstruct A = U Σ V^T
Sigma = np.diag(S)
A_reconstructed = U @ Sigma @ VT

print(f"\nReconstructed A:")
print(A_reconstructed)
print(f"\nReconstruction successful? {np.allclose(A, A_reconstructed)}")

# Check orthogonality of U and V
print("\n--- Checking orthogonality ---")
print(f"U^T U =\n{U.T @ U}")
print(f"Is U orthogonal? {np.allclose(U.T @ U, np.eye(U.shape[1]))}")

print(f"\nV^T V =\n{VT.T @ VT}")
print(f"Is V orthogonal? {np.allclose(VT.T @ VT, np.eye(VT.shape[0]))}")
```
</div>

### How to Calculate SVD

SVD is computed via eigendecomposition:

1. Compute $A^TA$ (this is $n \times n$ and symmetric)
2. Find eigenvalues $\lambda_i$ and eigenvectors $v_i$ of $A^TA$
   - The $v_i$ form the columns of $V$
   - Singular values: $\sigma_i = \sqrt{\lambda_i}$
3. Compute $u_i = \frac{1}{\sigma_i}Av_i$ to get columns of $U$

<div class="python-interactive" markdown="1">
```python
import numpy as np

A = np.array([[3, 1],
              [1, 3],
              [1, 1]], dtype=float)

print("Matrix A:")
print(A)

# Step 1: Compute A^T A
ATA = A.T @ A
print(f"\nA^T A:")
print(ATA)

# Step 2: Eigendecomposition of A^T A
eigenvalues, V = np.linalg.eig(ATA)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
V = V[:, idx]

print(f"\nEigenvalues of A^T A: {eigenvalues}")
print(f"V (eigenvectors):\n{V}")

# Singular values = sqrt(eigenvalues)
singular_values = np.sqrt(eigenvalues)
print(f"\nSingular values: {singular_values}")

# Step 3: Compute U = A V Σ^(-1)
Sigma_inv = np.diag(1.0 / singular_values)
U = A @ V @ Sigma_inv

print(f"\nU (computed manually):\n{U}")

# Verify with NumPy's SVD
U_np, S_np, VT_np = np.linalg.svd(A, full_matrices=False)

print(f"\nNumPy's U:\n{U_np}")
print(f"NumPy's singular values: {S_np}")
print(f"\nManual and NumPy match? {np.allclose(np.abs(U), np.abs(U_np))}")
print("(Signs may differ - both are valid)")
```
</div>

### Geometric Interpretation: Image Compression

SVD can be visualized as finding the "principal directions" in your data. Here's a practical example with a simple 2D dataset:

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate correlated data
np.random.seed(42)
mean = [0, 0]
cov = [[3, 1.5], [1.5, 1]]  # Covariance matrix
data = np.random.multivariate_normal(mean, cov, 200)

# Compute SVD of data matrix
U, S, VT = np.linalg.svd(data.T, full_matrices=False)

# Plot the data and principal directions
plt.figure(figsize=(10, 8))
plt.scatter(data[:, 0], data[:, 1], alpha=0.5)

# Plot singular vectors scaled by singular values
origin = np.zeros(2)
for i in range(2):
    direction = VT[i] * S[i]
    plt.arrow(0, 0, direction[0], direction[1],
              head_width=0.2, head_length=0.3, fc=f'C{i+1}', ec=f'C{i+1}',
              linewidth=2, label=f'σ{i+1} = {S[i]:.2f}')

plt.axhline(0, color='k', linewidth=0.5)
plt.axvline(0, color='k', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('SVD: Singular vectors show principal directions\n(length = singular value magnitude)')
plt.legend()
plt.show()

print(f"Singular values: {S}")
print(f"First direction captures {S[0]**2 / (S**2).sum() * 100:.1f}% of variance")
print(f"Second direction captures {S[1]**2 / (S**2).sum() * 100:.1f}% of variance")
```
</div>

### Low-Rank Approximation

SVD's killer feature: **compress matrices by keeping only the largest singular values!**

$$A \approx U_k \Sigma_k V_k^T$$

where $k < \min(m,n)$ is the number of singular values we keep.

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Create a 100x50 matrix
np.random.seed(42)
A = np.random.randn(100, 50)

print(f"Original matrix shape: {A.shape}")
print(f"Original matrix size: {A.size} numbers")

# Full SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)

# Keep only top k singular values
k = 10
U_k = U[:, :k]
S_k = S[:k]
VT_k = VT[:k, :]

# Reconstruct
A_approx = U_k @ np.diag(S_k) @ VT_k

print(f"\nRank-{k} approximation:")
print(f"  U_k shape: {U_k.shape}")
print(f"  S_k shape: {S_k.shape}")
print(f"  VT_k shape: {VT_k.shape}")
print(f"  Total numbers stored: {U_k.size + S_k.size + VT_k.size}")
print(f"  Compression: {(U_k.size + S_k.size + VT_k.size) / A.size * 100:.1f}% of original")

# Reconstruction error
error = np.linalg.norm(A - A_approx, 'fro') / np.linalg.norm(A, 'fro')
print(f"\nRelative reconstruction error: {error*100:.2f}%")

# Variance explained
variance_explained = S_k.sum()**2 / (S**2).sum()
print(f"Variance explained by top {k} components: {variance_explained*100:.1f}%")
```
</div>

### ML Applications of SVD

#### 1. Recommender Systems (Netflix Prize)

SVD factors a user-item matrix to find latent features:

<div class="python-interactive" markdown="1">
```python
import numpy as np

# User-movie rating matrix (rows=users, cols=movies)
# 0 means not rated
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
], dtype=float)

print("User-Movie ratings (0 = not rated):")
print(ratings)

# Replace 0s with mean rating for SVD
mean_rating = ratings[ratings > 0].mean()
ratings_filled = ratings.copy()
ratings_filled[ratings == 0] = mean_rating

# SVD to find latent features
U, S, VT = np.linalg.svd(ratings_filled, full_matrices=False)

# Use rank-2 approximation (2 latent features)
k = 2
U_k = U[:, :k]
S_k = np.diag(S[:k])
VT_k = VT[:k, :]

# Predicted ratings
predictions = U_k @ S_k @ VT_k

print(f"\nPredicted ratings:")
print(predictions)

# Show a prediction for missing rating
user_idx, movie_idx = 0, 2
print(f"\nUser 0, Movie 2 (was unrated):")
print(f"  Predicted rating: {predictions[user_idx, movie_idx]:.2f}")
```
</div>

#### 2. Image Compression

Keep only the top singular values to compress images:

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Create a simple "image" (gradient pattern)
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
X, Y = np.meshgrid(x, y)
image = np.sin(5 * X) * np.cos(5 * Y)

print(f"Image shape: {image.shape}")

# SVD
U, S, VT = np.linalg.svd(image, full_matrices=False)

# Reconstruct with different ranks
for k in [1, 5, 10, 20]:
    compressed = U[:, :k] @ np.diag(S[:k]) @ VT[:k, :]
    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    compression = (k * (U.shape[0] + VT.shape[1]) + k) / image.size
    print(f"\nRank {k}:")
    print(f"  Compression: {compression*100:.1f}% of original")
    print(f"  Error: {error*100:.2f}%")
```
</div>

#### 3. Pseudoinverse (from Lesson 3)

The pseudoinverse $A^+ = V\Sigma^+ U^T$ where $\Sigma^+$ inverts non-zero singular values:

<div class="python-interactive" markdown="1">
```python
import numpy as np

A = np.array([[1, 2],
              [3, 4],
              [5, 6]], dtype=float)

# Compute pseudoinverse via SVD
U, S, VT = np.linalg.svd(A, full_matrices=False)

# Invert non-zero singular values
S_inv = np.diag(1.0 / S)

A_pinv_manual = VT.T @ S_inv @ U.T

print("A^+ via SVD:")
print(A_pinv_manual)

# Compare with NumPy
A_pinv_np = np.linalg.pinv(A)
print("\nA^+ via NumPy:")
print(A_pinv_np)

print(f"\nMatch? {np.allclose(A_pinv_manual, A_pinv_np)}")
```
</div>

### SVD vs Eigendecomposition Summary

| Feature | Eigendecomposition | SVD |
|---------|-------------------|-----|
| **Works for** | Square matrices only | Any matrix |
| **Formula** | $A = Q\Lambda Q^{-1}$ | $A = U\Sigma V^T$ |
| **Vectors** | May not be orthogonal | Always orthogonal |
| **Values** | Can be complex | Always real & ≥ 0 |
| **Always exists?** | No (needs to be diagonalizable) | Yes! |
| **Interpretation** | Natural modes of system | Principal directions |

## Principal Component Analysis (PCA)

**PCA is the most widely used dimensionality reduction technique in machine learning.** It uses eigenvalues and eigenvectors to find the directions of maximum variance in your data.

### The Problem PCA Solves

Machine learning often deals with **high-dimensional data**:
- Images: 1000s of pixels → 1000s of features
- Text: 10,000s of words → 10,000s of features
- Genomics: millions of DNA positions → millions of features

**Problems with high dimensions:**
- Slow training and prediction
- Curse of dimensionality (data becomes sparse)
- Overfitting
- Hard to visualize

**PCA's solution**: Find a few directions that capture most of the variation in the data.

### The Goal

Reduce dimensions while preserving maximum variance:
- **Original**: 100 features
- **Reduced**: 10 principal components
- **Retain**: 95% of information!

### What Are Principal Components?

**Principal components** are the eigenvectors of the covariance matrix, ordered by eigenvalue magnitude.

- **First PC**: Direction of maximum variance
- **Second PC**: Direction of maximum variance orthogonal to first PC
- **Third PC**: Direction of maximum variance orthogonal to first two PCs
- ...and so on

Each PC is a **linear combination** of original features:

$$\text{PC}_1 = w_{11}x_1 + w_{12}x_2 + \cdots + w_{1n}x_n$$

### How PCA Works: Step-by-Step

The algorithm is beautifully simple:

1. **Center the data** - Subtract the mean from each feature
   $$X_{\text{centered}} = X - \bar{X}$$

2. **Compute covariance matrix** - Measure how features vary together
   $$C = \frac{1}{n-1}X_{\text{centered}}^T X_{\text{centered}}$$

3. **Find eigenvectors** - These are the principal components
   $$C v_i = \lambda_i v_i$$

4. **Sort by eigenvalue** - Larger eigenvalue = more important direction
   $$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$$

5. **Project data** - Transform to new coordinate system
   $$X_{\text{pca}} = X_{\text{centered}} \cdot V_k$$

where $V_k$ contains the top $k$ eigenvectors.

### Complete Example: Manual Implementation

<div class="python-interactive" markdown="1">
```python
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data  # 150 samples, 4 features
y = iris.target

print(f"Original data shape: {X.shape}")
print(f"Features: sepal length, sepal width, petal length, petal width")

# Step 1: Center the data (subtract mean)
mean = X.mean(axis=0)
X_centered = X - mean

print(f"\n--- Step 1: Center the data ---")
print(f"Original means: {mean}")
print(f"Centered means: {X_centered.mean(axis=0)}")

# Step 2: Compute covariance matrix
cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)

print(f"\n--- Step 2: Covariance matrix ---")
print(f"Shape: {cov_matrix.shape}")
print(f"Covariance matrix:\n{cov_matrix}")

# Step 3: Compute eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print(f"\n--- Step 3: Eigendecomposition ---")
print(f"Eigenvalues (before sorting): {eigenvalues}")

# Step 4: Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\n--- Step 4: Sort by importance ---")
print(f"Sorted eigenvalues: {eigenvalues}")

# Explained variance ratio
explained_variance = eigenvalues / eigenvalues.sum()
cumulative_variance = np.cumsum(explained_variance)

print(f"\nExplained variance ratio:")
for i, (ev, cv) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"  PC{i+1}: {ev*100:5.2f}% (cumulative: {cv*100:5.2f}%)")

# Step 5: Project onto first 2 principal components
PC1 = eigenvectors[:, 0]
PC2 = eigenvectors[:, 1]

print(f"\n--- Step 5: Project onto top 2 PCs ---")
print(f"PC1 weights: {PC1}")
print(f"PC2 weights: {PC2}")

X_pca = X_centered @ np.column_stack([PC1, PC2])
print(f"Projected data shape: {X_pca.shape}")

# Visualize
plt.figure(figsize=(12, 5))

# Original data (first 2 features)
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Original Data (First 2 Features)')
plt.colorbar(label='Species')
plt.grid(True, alpha=0.3)

# PCA projection
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel(f'First PC ({explained_variance[0]*100:.1f}% variance)')
plt.ylabel(f'Second PC ({explained_variance[1]*100:.1f}% variance)')
plt.title('PCA Projection')
plt.colorbar(label='Species')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n💡 PCA rotates the data to align with maximum variance directions!")
```
</div>

### Understanding the Covariance Matrix

The **covariance matrix** tells us how features vary together:

- **Diagonal elements**: Variance of each feature
- **Off-diagonal elements**: Covariance between features

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Simple 2D example
np.random.seed(42)
X = np.random.randn(100, 2)
X[:, 1] = X[:, 0] + 0.5 * np.random.randn(100)  # Make y correlated with x

# Center and compute covariance
X_centered = X - X.mean(axis=0)
cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)

print("Covariance matrix:")
print(cov)
print(f"\nVar(x) = {cov[0,0]:.3f}")
print(f"Var(y) = {cov[1,1]:.3f}")
print(f"Cov(x,y) = {cov[0,1]:.3f}")
print("\nPositive covariance → features are correlated")

# PCA finds directions that diagonalize this!
eigenvalues, eigenvectors = np.linalg.eig(cov)
print(f"\nEigenvalues: {eigenvalues}")
print(f"First PC captures most variance: {eigenvalues[0]:.3f}")
print(f"Second PC captures remaining variance: {eigenvalues[1]:.3f}")
```
</div>

### Scree Plot: Choosing Number of Components

A **scree plot** shows explained variance vs number of components:

<div class="python-interactive" markdown="1">
```python
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data

# PCA analysis
X_centered = X - X.mean(axis=0)
cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]

# Variance explained
explained_variance = eigenvalues / eigenvalues.sum()
cumulative_variance = np.cumsum(explained_variance)

# Scree plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(eigenvalues) + 1), explained_variance, alpha=0.7, color='steelblue')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot: Variance per Component')
plt.xticks(range(1, len(eigenvalues) + 1))
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(eigenvalues) + 1), cumulative_variance, 'o-', linewidth=2)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Variance Explained')
plt.xticks(range(1, len(eigenvalues) + 1))
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Explained variance by each PC:")
for i, ev in enumerate(explained_variance):
    print(f"  PC{i+1}: {ev*100:.2f}%")

print(f"\nTo retain 95% variance: use {np.argmax(cumulative_variance >= 0.95) + 1} components")
```
</div>

### PCA with Scikit-Learn

In practice, use sklearn's optimized implementation:

<div class="python-interactive" markdown="1">
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import numpy as np

# Load data
iris = load_iris()
X = iris.data

# Create PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("PCA with sklearn:")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

print(f"\nPrincipal Components (eigenvectors):")
print(pca.components_)

print(f"\nMean (centering): {pca.mean_}")

# Alternative: Choose components to explain 95% variance
pca_auto = PCA(n_components=0.95)
X_pca_auto = pca_auto.fit_transform(X)

print(f"\nAuto-selected {pca_auto.n_components_} components to explain 95% variance")
```
</div>

### Inverse Transform: Reconstruction

You can transform back to original space (with some information loss):

<div class="python-interactive" markdown="1">
```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X = iris.data

# PCA with 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Reconstruct
X_reconstructed = pca.inverse_transform(X_pca)

print(f"Original shape: {X.shape}")
print(f"Compressed shape: {X_pca.shape}")
print(f"Reconstructed shape: {X_reconstructed.shape}")

# Reconstruction error
error = np.mean((X - X_reconstructed)**2)
print(f"\nMean squared reconstruction error: {error:.6f}")

# Compare one sample
print(f"\nOriginal sample 0: {X[0]}")
print(f"Reconstructed:     {X_reconstructed[0]}")
print(f"Difference:        {X[0] - X_reconstructed[0]}")
```
</div>

### ML Pipeline Example: PCA for Classification

<div class="python-interactive" markdown="1">
```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

# Without PCA
model_no_pca = LogisticRegression(max_iter=200)
scores_no_pca = cross_val_score(model_no_pca, X, y, cv=5)

print("Without PCA:")
print(f"  Features: {X.shape[1]}")
print(f"  Accuracy: {scores_no_pca.mean():.3f} ± {scores_no_pca.std():.3f}")

# With PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

model_with_pca = LogisticRegression(max_iter=200)
scores_with_pca = cross_val_score(model_with_pca, X_pca, y, cv=5)

print(f"\nWith PCA (2 components, {pca.explained_variance_ratio_.sum()*100:.1f}% variance):")
print(f"  Features: {X_pca.shape[1]}")
print(f"  Accuracy: {scores_with_pca.mean():.3f} ± {scores_with_pca.std():.3f}")

print("\n💡 PCA reduced features from 4 to 2 with minimal accuracy loss!")
```
</div>

### PCA vs SVD

**PCA and SVD are mathematically equivalent!**

- PCA: Eigendecomposition of covariance matrix $C = \frac{1}{n-1}X^TX$
- SVD: Direct decomposition of data matrix $X = U\Sigma V^T$

The principal components are the right singular vectors $V$, and eigenvalues relate to singular values:

$$\lambda_i = \frac{\sigma_i^2}{n-1}$$

<div class="python-interactive" markdown="1">
```python
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

# Center the data
X_centered = X - X.mean(axis=0)

# Method 1: PCA via eigendecomposition
cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
eigenvalues, eigenvectors = np.linalg.eig(cov)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Method 2: PCA via SVD
U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
eigenvalues_svd = (S**2) / (X.shape[0] - 1)

print("Eigenvalues (via eigendecomposition):")
print(eigenvalues)

print("\nEigenvalues (via SVD):")
print(eigenvalues_svd)

print("\nPrincipal components (eigenvectors):")
print(eigenvectors[:, 0])

print("\nPrincipal components (SVD right singular vectors):")
print(VT[0])

print(f"\nSame up to sign? {np.allclose(np.abs(eigenvectors[:, 0]), np.abs(VT[0]))}")
```
</div>

## When to Use PCA

**Use PCA when:**
- ✅ You have too many features (curse of dimensionality)
- ✅ Features are correlated
- ✅ You need visualization (reduce to 2D/3D)
- ✅ You want to speed up training
- ✅ You need noise reduction

**Don't use PCA when:**
- ❌ Features are already uncorrelated
- ❌ You need interpretable features (PCA creates combinations)
- ❌ You have very few features
- ❌ Non-linear relationships matter (use kernel PCA instead)

## Key Takeaways

!!! success "Important Concepts"
    - Eigenvectors are special directions preserved by matrix transformation
    - Eigenvalues measure scaling along eigenvector directions
    - SVD decomposes any matrix into singular vectors and values
    - PCA finds directions of maximum variance using eigenvectors
    - PCA is fundamental for dimensionality reduction

## Practical Tips

!!! tip "PCA Best Practices"
    - Always standardize features before PCA (zero mean, unit variance)
    - Choose number of components based on explained variance
    - Visualize explained variance with a scree plot
    - Remember: PCA is sensitive to outliers

## Next Steps

Congratulations! You've completed the linear algebra lessons. Now it's time to practice what you've learned.

[Complete the Exercises](exercises.md){ .md-button .md-button--primary }

[Back: Lesson 3 - Advanced Matrix Operations](03-matrix-operations.md){ .md-button }
