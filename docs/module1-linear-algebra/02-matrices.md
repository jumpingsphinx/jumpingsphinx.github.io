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

Matrix multiplication is **one of the most important operations in machine learning**. It's how neural networks process data, how we transform features, and how we compute predictions for entire datasets at once.

### The Mechanics

Multiplying two matrices combines their transformations:

$$C = AB$$

where each element is computed as:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

This means: **element $(i,j)$ in $C$ is the dot product of row $i$ from $A$ with column $j$ from $B$**.

**Dimension Rule:** For $A_{m \times n}$ and $B_{n \times p}$, the result is $C_{m \times p}$

!!! warning "Dimension Compatibility"
    Number of columns in $A$ must equal number of rows in $B$!

    ✅ $(3 \times 2) \times (2 \times 4) = (3 \times 4)$ - Valid!

    ❌ $(3 \times 2) \times (3 \times 4)$ - Invalid! (2 ≠ 3)

### Step-by-Step Example

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Simple 2x2 example
A = np.array([[1, 2],
              [3, 4]])  # 2x2
B = np.array([[5, 6],
              [7, 8]])  # 2x2

C = A @ B  # Matrix multiplication

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)
print("\nResult C = A @ B:")
print(C)

# Let's compute each element manually to understand
print("\n--- Manual computation ---")
print(f"C[0,0] = A[0,:]·B[:,0] = {A[0,0]}*{B[0,0]} + {A[0,1]}*{B[1,0]} = {C[0,0]}")
print(f"C[0,1] = A[0,:]·B[:,1] = {A[0,0]}*{B[0,1]} + {A[0,1]}*{B[1,1]} = {C[0,1]}")
print(f"C[1,0] = A[1,:]·B[:,0] = {A[1,0]}*{B[0,0]} + {A[1,1]}*{B[1,0]} = {C[1,0]}")
print(f"C[1,1] = A[1,:]·B[:,1] = {A[1,0]}*{B[0,1]} + {A[1,1]}*{B[1,1]} = {C[1,1]}")
```
</div>

### Important Properties

**1. Not Commutative** - Order matters!

<div class="python-interactive" markdown="1">
```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

AB = A @ B
BA = B @ A

print("A @ B:")
print(AB)
print("\nB @ A:")
print(BA)
print(f"\nAre they equal? {np.array_equal(AB, BA)}")
print("This is why order matters in neural networks!")
```
</div>

**2. Associative** - $(AB)C = A(BC)$

**3. Distributive** - $A(B + C) = AB + AC$

### ML Application 1: Batch Predictions

This is **crucial**: matrix multiplication lets us make predictions for an **entire dataset at once**!

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Weight matrix for a linear model (2 outputs, 3 inputs)
W = np.array([[0.5, 0.3, 0.1],    # Weights for output 1
              [0.2, 0.4, 0.6]])    # Weights for output 2

# Dataset: 4 samples, 3 features each
X = np.array([[1.0, 2.0, 3.0],    # Sample 1
              [4.0, 5.0, 6.0],    # Sample 2
              [7.0, 8.0, 9.0],    # Sample 3
              [2.5, 3.5, 4.5]])   # Sample 4

print(f"Weight matrix W shape: {W.shape}")  # (2, 3)
print(f"Data matrix X shape: {X.shape}")    # (4, 3)

# Make predictions for ALL samples at once!
# Y = X @ W.T  (transpose W so dimensions align)
Y = X @ W.T

print(f"\nPredictions Y shape: {Y.shape}")  # (4, 2)
print("Predictions for all 4 samples:")
print(Y)

print("\n--- Verify first sample manually ---")
sample1 = X[0]  # [1.0, 2.0, 3.0]
pred1_out1 = np.dot(sample1, W[0])  # Prediction for output 1
pred1_out2 = np.dot(sample1, W[1])  # Prediction for output 2
print(f"Sample 1 predictions: [{pred1_out1:.2f}, {pred1_out2:.2f}]")
print(f"Matches Y[0]: {Y[0]}")
```
</div>

**Key Insight:** Instead of looping through samples one-by-one, matrix multiplication processes the entire dataset in one operation - this is **vectorization**, the foundation of fast ML!

### ML Application 2: Neural Network Layers

Each layer in a neural network is a matrix multiplication:

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Simulate a 2-layer neural network
# Layer 1: 3 inputs → 4 hidden units
# Layer 2: 4 hidden units → 2 outputs

# Weight matrices
W1 = np.random.randn(3, 4) * 0.5  # Shape: (3, 4)
W2 = np.random.randn(4, 2) * 0.5  # Shape: (4, 2)

# Input: 5 samples, 3 features each
X = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0],
              [7.0, 8.0, 9.0],
              [2.5, 3.5, 4.5],
              [1.5, 2.5, 3.5]])

print(f"Input X shape: {X.shape}")  # (5, 3)
print(f"W1 shape: {W1.shape}")      # (3, 4)
print(f"W2 shape: {W2.shape}")      # (4, 2)

# Forward pass through network
hidden = X @ W1  # (5, 3) @ (3, 4) = (5, 4)
output = hidden @ W2  # (5, 4) @ (4, 2) = (5, 2)

print(f"\nHidden layer shape: {hidden.shape}")  # (5, 4)
print(f"Output shape: {output.shape}")          # (5, 2)

print("\nFirst sample:")
print(f"  Input: {X[0]}")
print(f"  Hidden activations: {hidden[0]}")
print(f"  Output: {output[0]}")

print("\n🔥 This is the core of deep learning!")
print(f"   We processed {X.shape[0]} samples through a 2-layer network")
print(f"   using just 2 matrix multiplications!")
```
</div>

**Why This Matters:**

1. **Speed**: GPUs are optimized for matrix multiplication - this is why deep learning is fast
2. **Parallel Processing**: All samples computed simultaneously
3. **Composability**: Stack layers by chaining matrix multiplications: $Y = X W_1 W_2 W_3 ...$

### Visual Understanding of Matrix Transformations

To deepen your understanding of what matrix multiplication actually does geometrically, watch this visual explanation:

<div style="text-align: center; margin: 20px 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/XkY2DOUCWMU" title="Matrix multiplication as composition" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

### ML Application 3: Feature Transformation

Transform all features at once:

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Original features: [height_cm, weight_kg, age_years]
X = np.array([[170, 70, 25],
              [180, 80, 30],
              [165, 60, 22]])

# Transformation matrix - create new features
# New features: [BMI-like, age_scaled, combined_metric]
T = np.array([[0.01,  0.00, 0.0],   # Scale height
              [0.00,  0.01, 0.0],   # Scale weight
              [0.34, -0.34, 0.1]])  # Combined feature

X_transformed = X @ T.T

print("Original features:")
print(X)
print("\nTransformation matrix:")
print(T)
print("\nTransformed features:")
print(X_transformed)
print("\nAll 3 samples transformed in one operation!")
```
</div>

### Dimension Troubleshooting

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Common scenarios
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # 2x3

B = np.array([[1, 2],
              [3, 4],
              [5, 6]])     # 3x2

C = np.array([[1, 2, 3]])  # 1x3

print(f"A shape: {A.shape}")  # (2, 3)
print(f"B shape: {B.shape}")  # (3, 2)
print(f"C shape: {C.shape}")  # (1, 3)

# Valid multiplications
AB = A @ B  # (2,3) @ (3,2) = (2,2) ✅
print(f"\nA @ B shape: {AB.shape}")
print(AB)

BA = B @ A  # (3,2) @ (2,3) = (3,3) ✅
print(f"\nB @ A shape: {BA.shape}")
print(BA)

# Transpose to fix dimension mismatch
# CA would fail: (1,3) @ (2,3) ❌
# But C @ A.T works: (1,3) @ (3,2) ✅
CA_T = C @ A.T
print(f"\nC @ A.T shape: {CA_T.shape}")
print(CA_T)

print("\n💡 Tip: Use .T to transpose when dimensions don't match!")
```
</div>

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

## Visualization: Matrix Transformations

Matrices can transform vectors - this is how computer graphics work and how neural networks transform data!

<div class="python-interactive" markdown="1">
```python
import matplotlib.pyplot as plt
import numpy as np

# Original vector
v = np.array([1, 1])

# Transformation matrices
rotation = np.array([[0, -1],
                      [1, 0]])  # 90° counter-clockwise rotation
scaling = np.array([[2, 0],
                     [0, 0.5]])  # Scale x by 2, y by 0.5
shear = np.array([[1, 0.5],
                   [0, 1]])  # Shear transformation

# Apply transformations using matrix multiplication
v_rot = rotation @ v
v_scale = scaling @ v
v_shear = shear @ v

print(f"Original vector: {v}")
print(f"After rotation: {v_rot}")
print(f"After scaling: {v_scale}")
print(f"After shear: {v_shear}")

# Plot transformations
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

transformations = [
    (v_rot, 'Rotation (90°)', rotation),
    (v_scale, 'Scaling (2x, 0.5y)', scaling),
    (v_shear, 'Shear', shear)
]

for ax, (v_transformed, title, matrix) in zip(axes, transformations):
    # Plot original vector
    ax.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1,
              color='blue', width=0.01, label='Original', zorder=3)

    # Plot transformed vector
    ax.quiver(0, 0, v_transformed[0], v_transformed[1],
              angles='xy', scale_units='xy', scale=1,
              color='red', width=0.01, label='Transformed', zorder=3)

    # Formatting
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend(loc='upper right')
    ax.set_title(f'{title}\n{matrix[0]}\n{matrix[1]}', fontsize=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
fig  # Display the figure in browser

print("\n🎨 Try changing the vector v or the transformation matrices!")
```
</div>

**Try modifying:**
- Change `v = np.array([1, 1])` to `[2, 1]` or `[1, 3]`
- Create your own transformation matrix
- Combine transformations: `combined = rotation @ scaling @ v`

## Key Takeaways

!!! success "Important Concepts"
    - Matrices represent datasets (rows = samples, columns = features)
    - Transpose flips rows and columns
    - Matrix multiplication combines transformations
    - Matrix-vector product is how models make predictions
    - Matrix dimensions must be compatible for multiplication

## Bonus: 3D Transformations

Want to see how these concepts extend to three dimensions and beyond? This optional video visualizes linear transformations in 3D space:

<div style="text-align: center; margin: 20px 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/rHLEWRxRGiM" title="Three-dimensional linear transformations" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

## Next Steps

[Next: Lesson 3 - Advanced Matrix Operations](03-matrix-operations.md){ .md-button .md-button--primary }

[Back: Lesson 1 - Vectors](01-vectors.md){ .md-button }
