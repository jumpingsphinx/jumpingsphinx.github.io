# Mathematics Primer

A quick reference for the mathematical concepts used throughout ML101.

## Linear Algebra

### Vectors

A vector is an ordered list of numbers:
$$\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

**Dot Product:**
$$\vec{a} \cdot \vec{b} = \sum_{i=1}^{n} a_i b_i$$

**Norm (length):**
$$\|\vec{v}\|_2 = \sqrt{\sum_{i=1}^{n} v_i^2}$$

### Matrices

A matrix is a 2D array of numbers:
$$A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}$$

**Matrix Multiplication:**
$$C = AB \quad \text{where} \quad C_{ij} = \sum_{k} A_{ik} B_{kj}$$

**Transpose:**
$$(A^T)_{ij} = A_{ji}$$

## Calculus

### Derivatives

The derivative measures rate of change:
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Common Derivatives:**
- $\frac{d}{dx}(x^n) = nx^{n-1}$
- $\frac{d}{dx}(e^x) = e^x$
- $\frac{d}{dx}(\ln x) = \frac{1}{x}$

### Partial Derivatives

For functions of multiple variables:
$$\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}$$

### Gradient

The gradient is a vector of partial derivatives:
$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \end{bmatrix}$$

### Chain Rule

For composed functions:
$$\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$$

**ML Application:** Backpropagation!

## Probability

### Basic Probability

$$P(A) = \frac{\text{favorable outcomes}}{\text{total outcomes}}$$

**Properties:**
- $0 \leq P(A) \leq 1$
- $P(A) + P(\text{not } A) = 1$

### Conditional Probability

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

### Expectation

$$E[X] = \sum_{i} x_i \cdot P(X = x_i)$$

Or for continuous: $E[X] = \int x \cdot p(x) dx$

### Variance

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

## Statistics

### Mean (Average)

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

### Standard Deviation

$$\sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^2}$$

### Covariance

$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])]$$

### Correlation

$$\rho = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

## Functions Used in ML

### Sigmoid (Logistic Function)

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- **Range:** $(0, 1)$
- **Use:** Binary classification, probabilities

### Softmax

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

- **Use:** Multi-class classification

### ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

- **Use:** Neural network activation

### Log Loss (Cross-Entropy)

$$L = -\sum_{i} y_i \log(\hat{y}_i)$$

- **Use:** Classification loss function

## Optimization

### Gradient Descent

Iteratively update parameters to minimize loss:
$$\theta := \theta - \alpha \nabla J(\theta)$$

where:
- $\theta$: parameters
- $\alpha$: learning rate
- $\nabla J$: gradient of cost function

## Matrix Calculus

### Gradient of Vector Function

$$\nabla_x (w^T x) = w$$

### Gradient of Quadratic Form

$$\nabla_x (x^T A x) = (A + A^T)x$$

For symmetric $A$: $\nabla_x (x^T A x) = 2Ax$

## Quick Reference

| Operation | Notation | NumPy |
|-----------|----------|-------|
| Dot product | $\vec{a} \cdot \vec{b}$ | `np.dot(a, b)` or `a @ b` |
| Matrix mult | $AB$ | `A @ B` |
| Transpose | $A^T$ | `A.T` |
| Inverse | $A^{-1}$ | `np.linalg.inv(A)` |
| Norm | $\|\vec{v}\|$ | `np.linalg.norm(v)` |
| Exponential | $e^x$ | `np.exp(x)` |
| Natural log | $\ln x$ | `np.log(x)` |
| Sum | $\sum$ | `np.sum()` |
| Mean | $\bar{x}$ | `np.mean()` |

## Don't Worry!

!!! info "You Don't Need to Memorize"
    - Focus on **intuition** over formulas
    - Use this as a **reference** when needed
    - The course explains concepts from first principles
    - Practice builds understanding naturally

## Further Reading

- [Khan Academy - Linear Algebra](https://www.khanacademy.org/math/linear-algebra)
- [Khan Academy - Calculus](https://www.khanacademy.org/math/calculus-1)
- [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- [3Blue1Brown - Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)

[Back to Home](../index.md){ .md-button }
