# Lesson 4: Eigenvalues and Eigenvectors

## Introduction

Eigenvalues and eigenvectors are among the most important concepts in linear algebra for machine learning. They're the foundation of Principal Component Analysis (PCA), one of the most widely used dimensionality reduction techniques.

## What Are Eigenvalues and Eigenvectors?

For a square matrix $A$, an **eigenvector** $\vec{v}$ and its corresponding **eigenvalue** $\lambda$ satisfy:

$$A\vec{v} = \lambda\vec{v}$$

### Geometric Interpretation

When you multiply a matrix by its eigenvector, the vector only gets scaled (not rotated). The eigenvalue is the scaling factor.

```python
import numpy as np

A = np.array([[2, 0],
              [0, 3]])
v = np.array([1, 0])  # Eigenvector

result = A @ v  # [2, 0] = 2 * [1, 0]
# v is scaled by eigenvalue λ = 2
```

## Computing Eigenvalues and Eigenvectors

```python
A = np.array([[4, 2],
              [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)  # [5, 2]
print("Eigenvectors:\n", eigenvectors)

# Verify for first eigenvector
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]

print("A @ v1:", A @ v1)
print("lambda1 * v1:", lambda1 * v1)
# They should be equal!
```

## Properties of Eigenvalues

1. **Sum of eigenvalues = Trace** (sum of diagonal elements)
2. **Product of eigenvalues = Determinant**
3. Symmetric matrices have real eigenvalues
4. Symmetric matrices have orthogonal eigenvectors

## Eigendecomposition

For a diagonalizable matrix $A$:

$$A = Q\Lambda Q^{-1}$$

where:
- $Q$: Matrix of eigenvectors (columns)
- $\Lambda$: Diagonal matrix of eigenvalues

```python
A = np.array([[4, 2],
              [1, 3]])

eigenvalues, eigenvectors = np.linalg.eig(A)

# Reconstruct A
Lambda = np.diag(eigenvalues)
A_reconstructed = eigenvectors @ Lambda @ np.linalg.inv(eigenvectors)

print(np.allclose(A, A_reconstructed))  # True
```

## Singular Value Decomposition (SVD)

SVD works for **any** matrix (not just square):

$$A = U\Sigma V^T$$

where:
- $U$: Left singular vectors (eigenvectors of $AA^T$)
- $\Sigma$: Singular values (square roots of eigenvalues)
- $V^T$: Right singular vectors (eigenvectors of $A^TA$)

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])  # 4x3 matrix

U, S, VT = np.linalg.svd(A, full_matrices=False)

print(U.shape)   # (4, 3)
print(S.shape)   # (3,)
print(VT.shape)  # (3, 3)

# Reconstruct A
Sigma = np.diag(S)
A_reconstructed = U @ Sigma @ VT
print(np.allclose(A, A_reconstructed))  # True
```

## Principal Component Analysis (PCA)

PCA uses eigenvalues and eigenvectors to find the most important directions in your data.

### The Goal

Reduce dimensions while preserving maximum variance:
- Original: 100 features
- Reduced: 10 features (principal components)
- Retain 95% of information!

### How PCA Works

1. **Center the data** (subtract mean)
2. **Compute covariance matrix**: $C = \frac{1}{n}X^TX$
3. **Find eigenvectors** of covariance matrix
4. **Project data** onto top eigenvectors

```python
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X = iris.data  # 150 samples, 4 features

# Step 1: Center the data
X_centered = X - X.mean(axis=0)

# Step 2: Compute covariance matrix
cov_matrix = (X_centered.T @ X_centered) / (X.shape[0] - 1)

# Step 3: Compute eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort by eigenvalue (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Step 4: Project onto first 2 principal components
PC1 = eigenvectors[:, 0]
PC2 = eigenvectors[:, 1]

X_pca = X_centered @ np.column_stack([PC1, PC2])

# Visualize
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=iris.target, cmap='viridis')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('PCA of Iris Dataset')
plt.colorbar(scatter, label='Species')
plt.grid(True)
plt.show()

# Explained variance
explained_variance = eigenvalues / eigenvalues.sum()
print("Explained variance ratio:", explained_variance)
# [0.92, 0.05, 0.02, 0.005]
# First PC captures 92% of variance!
```

### Using sklearn's PCA

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Components:\n", pca.components_)
```

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
