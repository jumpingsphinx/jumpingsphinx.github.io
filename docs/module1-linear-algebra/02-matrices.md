# Lesson 2: Matrices

## Introduction

Matrices are 2D arrays that represent collections of vectors, transformations, and entire datasets. Understanding matrices is crucial for machine learning since your data is stored as matrices and models perform matrix operations.

## What is a Matrix?

A **matrix** is a rectangular array of numbers arranged in rows and columns:

$$A = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23}
\end{bmatrix}$$

This is a $2 \times 3$ matrix (2 rows, 3 columns).

### In Machine Learning

- **Each row**: A data sample (observation)
- **Each column**: A feature
- **Entire matrix**: Your dataset!

**Example:** A dataset of 3 houses with 4 features:

$$X = \begin{bmatrix}
1500 & 3 & 2 & 2005 \\
2000 & 4 & 3 & 2010 \\
1200 & 2 & 1 & 1995
\end{bmatrix}$$

Rows = houses, Columns = [sqft, bedrooms, bathrooms, year_built]

## Creating Matrices in NumPy

!!! tip "Try It Yourself"
    Run this code to see how matrices are created in NumPy!

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Create a matrix
A = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"Matrix A:\n{A}")
print(f"Shape: {A.shape}")  # (2, 3) - 2 rows, 3 columns

# Special matrices
zeros = np.zeros((3, 4))       # 3x4 matrix of zeros
ones = np.ones((2, 2))         # 2x2 matrix of ones
identity = np.eye(3)           # 3x3 identity matrix

print(f"\nZeros matrix:\n{zeros}")
print(f"\nIdentity matrix:\n{identity}")

# From a dataset - houses example
data = [[1500, 3, 2],
        [2000, 4, 3],
        [1200, 2, 1]]
X = np.array(data)
print(f"\nHouse dataset:\n{X}")
print(f"Features: [sqft, bedrooms, bathrooms]")
```
</div>

**Try modifying:** Change the matrix dimensions or create your own dataset!

## Types of Matrices

### Square Matrix
Equal number of rows and columns ($n \times n$):

$$A = \begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}$$

### Identity Matrix
Square matrix with 1s on diagonal, 0s elsewhere:

$$I = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{bmatrix}$$

```python
I = np.eye(3)
```

**Property:** $A \cdot I = I \cdot A = A$ (identity for multiplication)

### Diagonal Matrix
Non-zero elements only on the diagonal:

$$D = \begin{bmatrix}
d_1 & 0 & 0 \\
0 & d_2 & 0 \\
0 & 0 & d_3
\end{bmatrix}$$

```python
D = np.diag([1, 2, 3])
```

### Symmetric Matrix
Equal to its transpose ($A = A^T$):

$$A = \begin{bmatrix}
1 & 2 & 3 \\
2 & 5 & 6 \\
3 & 6 & 9
\end{bmatrix}$$

**ML Application:** Covariance matrices are symmetric.

### Sparse Matrix
Most elements are zero:

$$S = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 0 & 2 & 0 \\
0 & 0 & 0 & 3
\end{bmatrix}$$

**ML Application:** Text data (document-term matrices) are often sparse.

## Matrix Operations

### Transpose

Flip rows and columns:

$$A = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix} \quad \Rightarrow \quad A^T = \begin{bmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{bmatrix}$$

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])
A_T = A.T  # or np.transpose(A)
```

**Properties:**
- $(A^T)^T = A$
- $(A + B)^T = A^T + B^T$
- $(AB)^T = B^T A^T$

### Addition and Subtraction

Element-wise operations (matrices must have same shape):

$$A + B = \begin{bmatrix}
a_{11} + b_{11} & a_{12} + b_{12} \\
a_{21} + b_{21} & a_{22} + b_{22}
\end{bmatrix}$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A + B  # [[6, 8], [10, 12]]
```

### Scalar Multiplication

Multiply every element by a scalar:

$$c \cdot A = \begin{bmatrix}
c \cdot a_{11} & c \cdot a_{12} \\
c \cdot a_{21} & c \cdot a_{22}
\end{bmatrix}$$

```python
A = np.array([[1, 2], [3, 4]])
B = 2 * A  # [[2, 4], [6, 8]]
```

## Matrix-Vector Multiplication

This is fundamental to understanding how ML models make predictions!

$$A\vec{x} = \begin{bmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{bmatrix} \begin{bmatrix}
x_1 \\
x_2
\end{bmatrix} = \begin{bmatrix}
a_{11}x_1 + a_{12}x_2 \\
a_{21}x_1 + a_{22}x_2
\end{bmatrix}$$

Each element of the result is the **dot product** of a matrix row with the vector.

!!! example "ML Application: Making Predictions"
    This is exactly how a linear model makes predictions! Try it:

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Weight matrix (model parameters)
W = np.array([[0.5, 0.3],
              [0.2, 0.7]])

# Input features (e.g., house: [size_normalized, age_normalized])
x = np.array([0.8, 0.6])

# Make prediction: y = W @ x
y = W @ x
print(f"Weight matrix W:\n{W}")
print(f"\nInput features x: {x}")
print(f"\nPrediction y = W @ x: {y}")
print(f"\nManual computation:")
print(f"  y[0] = {W[0,0]}*{x[0]} + {W[0,1]}*{x[1]} = {W[0,0]*x[0] + W[0,1]*x[1]:.2f}")
print(f"  y[1] = {W[1,0]}*{x[0]} + {W[1,1]}*{x[1]} = {W[1,0]*x[0] + W[1,1]*x[1]:.2f}")
```
</div>

**Mathematical form:** $\hat{y} = Wx + b$

where $W$ is a weight matrix, $x$ is input features, $b$ is bias.

## Matrix-Matrix Multiplication

Multiplying two matrices combines their transformations:

$$C = AB$$

where $C_{ij} = \sum_{k} A_{ik} B_{kj}$

**Rule:** For $A_{m \times n}$ and $B_{n \times p}$, result is $C_{m \times p}$

!!! warning "Dimension Compatibility"
    Number of columns in $A$ must equal number of rows in $B$!

```python
A = np.array([[1, 2],
              [3, 4]])  # 2x2
B = np.array([[5, 6],
              [7, 8]])  # 2x2

C = A @ B  # 2x2
# [[19, 22],
#  [43, 50]]

# Element-by-element:
# C[0,0] = 1*5 + 2*7 = 19
# C[0,1] = 1*6 + 2*8 = 22
# C[1,0] = 3*5 + 4*7 = 43
# C[1,1] = 3*6 + 4*8 = 50
```

**Properties:**
- **Not commutative:** $AB \neq BA$ (usually)
- **Associative:** $(AB)C = A(BC)$
- **Distributive:** $A(B + C) = AB + AC$

**ML Application:** Stacking neural network layers:

$$output = W_3(W_2(W_1 x))$$

## Practical Example: Dataset Operations

!!! example "Working with Real Data"
    This is what you'll do in every ML project - manipulate datasets!

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Dataset: 4 samples, 3 features
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

print(f"Dataset shape: {X.shape}")  # (4, 3)
print(f"Dataset:\n{X}\n")

# Mean of each feature (column)
feature_means = X.mean(axis=0)
print(f"Feature means: {feature_means}")

# Mean of each sample (row)
sample_means = X.mean(axis=1)
print(f"Sample means: {sample_means}")

# Center the data (subtract mean) - important preprocessing!
X_centered = X - feature_means
print(f"\nCentered data:\n{X_centered}")

# Get specific samples and features
first_sample = X[0, :]      # First row
second_feature = X[:, 1]    # Second column
subset = X[0:2, 1:3]        # First 2 rows, columns 1-2

print(f"\nFirst sample: {first_sample}")
print(f"Second feature (all rows): {second_feature}")
print(f"Subset [0:2, 1:3]:\n{subset}")
```
</div>

**Try it:** Change the indexing to extract different parts of the dataset!

## Visualization

Matrices can transform vectors:

```python
import matplotlib.pyplot as plt
import numpy as np

# Original vector
v = np.array([1, 1])

# Transformation matrices
rotation = np.array([[0, -1],
                      [1, 0]])  # 90° rotation
scaling = np.array([[2, 0],
                     [0, 0.5]])  # Scale x by 2, y by 0.5
shear = np.array([[1, 0.5],
                   [0, 1]])  # Shear transformation

# Apply transformations
v_rot = rotation @ v
v_scale = scaling @ v
v_shear = shear @ v

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, v_transformed, title in zip(axes,
                                     [v_rot, v_scale, v_shear],
                                     ['Rotation', 'Scaling', 'Shear']):
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.01, label='Original')
    ax.quiver(0, 0, v_transformed[0], v_transformed[1],
              angles='xy', scale_units='xy', scale=1,
              color='red', width=0.01, label='Transformed')
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    ax.set_title(title)

plt.tight_layout()
plt.show()
```

## Key Takeaways

!!! success "Important Concepts"
    - Matrices represent datasets (rows = samples, columns = features)
    - Transpose flips rows and columns
    - Matrix multiplication combines transformations
    - Matrix-vector product is how models make predictions
    - Matrix dimensions must be compatible for multiplication

## Next Steps

[Next: Lesson 3 - Advanced Matrix Operations](03-matrix-operations.md){ .md-button .md-button--primary }

[Back: Lesson 1 - Vectors](01-vectors.md){ .md-button }
