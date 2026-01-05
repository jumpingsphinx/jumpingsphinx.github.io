# Lesson 1: Vectors

## Introduction

Vectors are fundamental building blocks in machine learning. Every data sample you work with is represented as a vector, and understanding how to manipulate vectors is essential for understanding ML algorithms.

## What is a Vector?

A **vector** is an ordered list of numbers. You can think of vectors in two complementary ways:

### Geometric Interpretation

A vector represents a point in space or a direction with magnitude. For example, in 2D:

$$\vec{v} = \begin{bmatrix} 3 \\ 2 \end{bmatrix}$$

This vector points from the origin (0, 0) to the point (3, 2).

### Algebraic Interpretation

A vector is simply a collection of numbers in a specific order. In machine learning:
- Each number represents a **feature**
- The vector represents a **data sample**

**Example:** A house might be represented as:
$$\vec{house} = \begin{bmatrix} 1500 \\ 3 \\ 2 \\ 250000 \end{bmatrix}$$

where features are: [square feet, bedrooms, bathrooms, price]

## Vector Notation

Vectors can be written in different ways:

- **Column vector** (most common in ML):
$$\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

- **Row vector**:
$$\vec{v}^T = \begin{bmatrix} v_1 & v_2 & \cdots & v_n \end{bmatrix}$$

- **Python/NumPy**:
```python
v = np.array([v1, v2, ..., vn])
```

## Creating Vectors in NumPy

!!! tip "Try It Yourself"
    Click the **▶ Run Code** button below to execute Python code directly in your browser - no installation needed!

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Create a vector
v = np.array([1, 2, 3, 4])
print(f"Vector: {v}")
print(f"Shape: {v.shape}")

# Create a column vector (2D array)
v_col = np.array([[1], [2], [3], [4]])
print(f"\nColumn vector shape: {v_col.shape}")

# Create specific vectors
zeros = np.zeros(5)
ones = np.ones(3)
print(f"\nZeros: {zeros}")
print(f"Ones: {ones}")
```
</div>

**Try modifying the code:** Change the numbers in the vector and run it again!

!!! tip "1D vs 2D Arrays"
    In NumPy, `np.array([1, 2, 3])` creates a 1D array with shape `(3,)`, while `np.array([[1], [2], [3]])` creates a 2D column vector with shape `(3, 1)`. Both work for most operations, but be aware of the difference.

## Vector Operations

### 1. Vector Addition

Add vectors element-wise:

$$\vec{a} + \vec{b} = \begin{bmatrix} a_1 \\ a_2 \end{bmatrix} + \begin{bmatrix} b_1 \\ b_2 \end{bmatrix} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \end{bmatrix}$$

**Geometric interpretation:** Place vectors tip-to-tail.

```python
a = np.array([1, 2])
b = np.array([3, 1])
c = a + b  # [4, 3]
```

**ML Application:** Combining features or model updates.

### 2. Scalar Multiplication

Multiply each element by a scalar:

$$c \cdot \vec{v} = c \cdot \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \end{bmatrix}$$

**Geometric interpretation:** Scales the vector (stretches or shrinks).

```python
v = np.array([1, 2, 3])
scaled = 2 * v  # [2, 4, 6]
```

**ML Application:** Learning rate scaling in gradient descent.

### 3. Dot Product (Inner Product)

The dot product combines two vectors into a single number:

$$\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i \cdot b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Three ways to compute dot product
dot1 = np.dot(a, b)      # 32
dot2 = a @ b             # 32 (@ is matrix multiplication operator)
dot3 = (a * b).sum()     # 32 (element-wise multiply, then sum)
```

**Geometric interpretation:**

$$\vec{a} \cdot \vec{b} = \|\vec{a}\| \|\vec{b}\| \cos(\theta)$$

where $\theta$ is the angle between vectors.

- If $\vec{a} \cdot \vec{b} > 0$: vectors point in similar directions
- If $\vec{a} \cdot \vec{b} = 0$: vectors are perpendicular (orthogonal)
- If $\vec{a} \cdot \vec{b} < 0$: vectors point in opposite directions

**ML Application:**
- Measuring similarity between data samples
- Computing neural network activations: $\vec{w} \cdot \vec{x} + b$

### 4. Element-wise Operations

Multiply or divide elements one-by-one (Hadamard product):

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

elementwise_mult = a * b  # [4, 10, 18]
elementwise_div = a / b   # [0.25, 0.4, 0.5]
```

!!! warning "Dot Product vs Element-wise Multiplication"
    - `np.dot(a, b)` or `a @ b`: dot product → single number
    - `a * b`: element-wise multiplication → vector

## Vector Norms

A **norm** measures the "size" or "length" of a vector.

### L2 Norm (Euclidean Norm)

The most common norm - the straight-line distance:

$$\|\vec{v}\|_2 = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

```python
v = np.array([3, 4])
l2_norm = np.linalg.norm(v)  # 5.0
# Or manually:
l2_norm_manual = np.sqrt((v ** 2).sum())  # 5.0
```

### L1 Norm (Manhattan Norm)

Sum of absolute values:

$$\|\vec{v}\|_1 = |v_1| + |v_2| + \cdots + |v_n|$$

```python
v = np.array([3, -4])
l1_norm = np.linalg.norm(v, ord=1)  # 7.0
# Or manually:
l1_norm_manual = np.abs(v).sum()  # 7.0
```

### L∞ Norm (Maximum Norm)

The largest absolute value:

$$\|\vec{v}\|_\infty = \max(|v_1|, |v_2|, \ldots, |v_n|)$$

```python
v = np.array([3, -7, 2])
linf_norm = np.linalg.norm(v, ord=np.inf)  # 7.0
# Or manually:
linf_norm_manual = np.abs(v).max()  # 7.0
```

**ML Applications:**
- **L2 norm**: Euclidean distance, L2 regularization
- **L1 norm**: Manhattan distance, L1 regularization (encourages sparsity)
- **L∞ norm**: Measuring worst-case error

## Unit Vectors and Normalization

A **unit vector** has length 1. Normalizing a vector means converting it to a unit vector:

$$\hat{v} = \frac{\vec{v}}{\|\vec{v}\|}$$

```python
v = np.array([3, 4])
v_normalized = v / np.linalg.norm(v)  # [0.6, 0.8]
print(np.linalg.norm(v_normalized))   # 1.0
```

**ML Application:** Feature normalization to ensure all features have similar scales.

## Distance Between Vectors

### Euclidean Distance

$$d(\vec{a}, \vec{b}) = \|\vec{a} - \vec{b}\|_2 = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

distance = np.linalg.norm(a - b)  # 5.196
```

**ML Application:** K-nearest neighbors, clustering algorithms.

### Cosine Similarity

Measures angle between vectors (independent of magnitude):

$$\text{similarity} = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|}$$

$$\text{similarity} \in [-1, 1]$$

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(cosine_sim)  # 0.9746 (very similar direction)
```

**ML Application:** Text similarity, recommendation systems.

## Practical Example: Working with Real Data

Let's represent and compare houses:

```python
import numpy as np

# House features: [square_feet, bedrooms, bathrooms]
house1 = np.array([1500, 3, 2])
house2 = np.array([2000, 4, 3])
house3 = np.array([1200, 2, 1])

# Normalize features (mean 0, std 1)
houses = np.array([house1, house2, house3])
mean = houses.mean(axis=0)
std = houses.std(axis=0)

house1_norm = (house1 - mean) / std
house2_norm = (house2 - mean) / std
house3_norm = (house3 - mean) / std

# Find which house is most similar to house1
dist_1_2 = np.linalg.norm(house1_norm - house2_norm)
dist_1_3 = np.linalg.norm(house1_norm - house3_norm)

print(f"Distance from house1 to house2: {dist_1_2:.2f}")
print(f"Distance from house1 to house3: {dist_1_3:.2f}")

if dist_1_2 < dist_1_3:
    print("House 2 is more similar to house 1")
else:
    print("House 3 is more similar to house 1")
```

## Visualization

Visualizing vectors helps build geometric intuition:

```python
import matplotlib.pyplot as plt

# Create vectors
v1 = np.array([3, 2])
v2 = np.array([1, 3])
v_sum = v1 + v2

# Plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot vectors from origin
ax.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
          color='red', width=0.006, label='v1')
ax.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
          color='blue', width=0.006, label='v2')
ax.quiver(0, 0, v_sum[0], v_sum[1], angles='xy', scale_units='xy', scale=1,
          color='green', width=0.006, label='v1 + v2')

# Formatting
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 6)
ax.set_aspect('equal')
ax.grid(True)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.legend()
ax.set_title('Vector Addition')
plt.show()
```

## Key Takeaways

!!! success "Important Concepts"
    - Vectors represent both data samples and directions in space
    - Vector addition combines vectors element-wise
    - Dot product measures similarity and projection
    - Norms measure vector magnitude (L1, L2, L∞)
    - Normalization creates unit vectors for fair comparison
    - Distance metrics (Euclidean, cosine) measure similarity

## Common Patterns in ML

| Operation | ML Use Case |
|-----------|-------------|
| `x + y` | Combining features, model updates |
| `np.dot(w, x)` | Linear model prediction: $y = w^T x + b$ |
| `np.linalg.norm(x)` | Regularization penalty |
| `x / np.linalg.norm(x)` | Feature normalization |
| `np.linalg.norm(x - y)` | Distance for clustering, KNN |

## Practice Problems

Before moving to the exercises, try these quick problems:

1. Create a vector `v = [1, 2, 3, 4]` and compute its L2 norm
2. Normalize this vector to unit length
3. Create two vectors and compute their dot product
4. Plot two 2D vectors and their sum

## Next Steps

Now that you understand vectors, let's move on to matrices - collections of vectors that represent entire datasets!

[Next: Lesson 2 - Matrices](02-matrices.md){ .md-button .md-button--primary }

[Or complete the exercises first](exercises.md){ .md-button }

---

**Questions?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues).
