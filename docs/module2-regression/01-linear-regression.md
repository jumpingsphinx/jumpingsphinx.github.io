# Lesson 1: Linear Regression

## Introduction

Linear regression is the **foundation of supervised machine learning**. It's often the first algorithm you should try on a new regression problem, and understanding it deeply will help you understand more complex models.

In this lesson, you'll learn:
- How to formulate regression problems mathematically
- Simple linear regression (one feature)
- Multiple linear regression (many features)
- The normal equation (closed-form solution)
- How to implement linear regression from scratch
- Model evaluation metrics

## What is Linear Regression?

### Visual Introduction to Linear Regression

Before diving into the mathematics, watch this excellent intuitive explanation of how linear regression works:

<div style="text-align: center; margin: 20px 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/nk2CQITm_eo" title="Linear Regression by StatQuest" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

**Linear regression** models the relationship between input features and a continuous target variable using a linear function:

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n$$

Or in vector notation:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b = \mathbf{x}^T \mathbf{w} + b$$

where $\hat{y}$ is the predicted value, $\mathbf{x} = [x_1, x_2, \ldots, x_n]^T$ is the feature vector, $\mathbf{w} = [w_1, w_2, \ldots, w_n]^T$ is the weight vector (parameters to learn), and $b$ (or $w_0$) is the bias term (intercept).

### The Goal

Find the weights $\mathbf{w}$ and bias $b$ that **minimize the difference** between predicted values $\hat{y}$ and actual values $y$.

### Real-World Applications

Linear regression is everywhere:
- **House price prediction**: Price from square footage, bedrooms, location
- **Sales forecasting**: Revenue from advertising spend
- **Medical research**: Drug dosage effects
- **Economics**: GDP from various economic indicators
- **Sports analytics**: Player performance prediction

## Simple Linear Regression

Let's start with the simplest case: **one input feature**.

$$\hat{y} = w x + b$$

This represents a **straight line** in 2D space.

### Geometric Intuition

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data: y = 2x + 1 + noise
np.random.seed(42)
X = np.random.rand(50) * 10
y = 2 * X + 1 + np.random.randn(50) * 2

# Plot data points
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.6, label='Data points')

# True relationship (what we're trying to find)
x_line = np.linspace(0, 10, 100)
y_true = 2 * x_line + 1
plt.plot(x_line, y_true, 'r-', linewidth=2, label='True relationship: y = 2x + 1')

plt.xlabel('Feature (x)', fontsize=12)
plt.ylabel('Target (y)', fontsize=12)
plt.title('Linear Regression: Finding the Best Fit Line', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Goal: Find w and b such that ŷ = wx + b fits the data best")
```
</div>

### The Loss Function

How do we measure "best fit"? We use a **loss function** that quantifies the error between predictions and actual values.

The most common choice is **Mean Squared Error (MSE)**:

$$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2 = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - (wx^{(i)} + b))^2$$

where $m$ is the number of training examples, $y^{(i)}$ is the actual value for example $i$, and $\hat{y}^{(i)}$ is the predicted value for example $i$.

!!! info "Why Squared Error?"
    - **Squaring** makes all errors positive (no cancellation)
    - **Squaring** penalizes large errors more heavily
    - **Squared error** has nice mathematical properties (differentiable, convex)

### Visualizing the Loss Function

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Compute MSE for different values of w (with b=0 for simplicity)
w_values = np.linspace(-1, 3, 100)
mse_values = []

for w in w_values:
    predictions = w * X
    mse = np.mean((y - predictions) ** 2)
    mse_values.append(mse)

# Find optimal w
optimal_w = w_values[np.argmin(mse_values)]
min_mse = min(mse_values)

plt.figure(figsize=(10, 6))
plt.plot(w_values, mse_values, 'b-', linewidth=2)
plt.scatter([optimal_w], [min_mse], color='red', s=100, zorder=5, label=f'Minimum at w={optimal_w:.2f}')
plt.xlabel('Weight (w)', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.title('Loss Function: MSE vs Weight', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Optimal weight: {optimal_w:.2f}")
print(f"Minimum MSE: {min_mse:.2f}")
print("\n💡 The goal of training is to find the weight that minimizes this curve!")
```
</div>

### Closed-Form Solution for Simple Linear Regression

For simple linear regression, we can derive **exact formulas** for the optimal parameters:

$$w = \frac{\sum_{i=1}^{m}(x^{(i)} - \bar{x})(y^{(i)} - \bar{y})}{\sum_{i=1}^{m}(x^{(i)} - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}$$

$$b = \bar{y} - w\bar{x}$$

where $\bar{x}$ and $\bar{y}$ are the means of $x$ and $y$.

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.rand(50) * 10
y = 2 * X + 1 + np.random.randn(50) * 2

# Calculate parameters using closed-form solution
x_mean = np.mean(X)
y_mean = np.mean(y)

# Numerator: covariance
numerator = np.sum((X - x_mean) * (y - y_mean))
# Denominator: variance of X
denominator = np.sum((X - x_mean) ** 2)

w = numerator / denominator
b = y_mean - w * x_mean

print("=== Simple Linear Regression ===")
print(f"Mean of X: {x_mean:.4f}")
print(f"Mean of y: {y_mean:.4f}")
print(f"\nOptimal weight (w): {w:.4f}")
print(f"Optimal bias (b): {b:.4f}")
print(f"\nEquation: ŷ = {w:.4f}x + {b:.4f}")
print(f"True equation: y = 2x + 1")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='Data')

x_line = np.linspace(0, 10, 100)
y_pred = w * x_line + b
plt.plot(x_line, y_pred, 'r-', linewidth=2, label=f'Fitted: ŷ = {w:.2f}x + {b:.2f}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Linear Regression: Closed-Form Solution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
</div>

## Multiple Linear Regression

In practice, we have **many features**. Multiple linear regression extends simple linear regression:

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n$$

### Matrix Notation

Let's express this compactly using matrices. For $m$ samples with $n$ features:

**Design Matrix $\mathbf{X}$** (with bias column):

$$\mathbf{X} = \begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\\\
1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\\\
\vdots & \vdots & \vdots & \ddots & \vdots \\\\
1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)}
\end{bmatrix}$$

Shape: $(m \times (n+1))$

**Weight Vector $\mathbf{w}$**:

$$\mathbf{w} = \begin{bmatrix} w_0 \\\\ w_1 \\\\ \vdots \\\\ w_n \end{bmatrix}$$

Shape: $((n+1) \times 1)$

**Predictions**:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$$

Shape: $(m \times 1)$

### Why Matrix Notation Matters

Using matrices allows us to:
1. **Compute all predictions at once** (vectorization)
2. **Derive elegant closed-form solutions**
3. **Leverage optimized linear algebra libraries** (NumPy, BLAS)

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Create sample data: 5 houses, 3 features each
# Features: [square_feet, bedrooms, age]
X_raw = np.array([
    [1500, 3, 10],
    [2000, 4, 5],
    [1200, 2, 15],
    [1800, 3, 8],
    [2500, 5, 2]
], dtype=float)

# Add bias column (column of 1s)
m = X_raw.shape[0]
X = np.column_stack([np.ones(m), X_raw])

print("Design Matrix X (with bias column):")
print(X)
print(f"\nShape: {X.shape}")
print("  - 5 samples (rows)")
print("  - 4 features including bias (columns)")

# Suppose we have learned weights
w = np.array([50000, 100, 20000, -2000])  # [bias, price_per_sqft, price_per_bedroom, age_discount]

# Make predictions for ALL samples at once
predictions = X @ w  # Matrix multiplication!

print("\n=== Predictions ===")
print("Weights: [bias, sqft, bedrooms, age]")
print(f"w = {w}")
print("\nPredicted house prices:")
for i, (features, pred) in enumerate(zip(X_raw, predictions)):
    print(f"  House {i+1}: {features} → ${pred:,.0f}")
```
</div>

## The Normal Equation

For multiple linear regression, we can derive a **closed-form solution** called the **Normal Equation**:

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

This gives us the **optimal weights** that minimize MSE in one step!

### Derivation (Optional)

The MSE loss in matrix form is:

$$J(\mathbf{w}) = \frac{1}{m}(\mathbf{y} - \mathbf{X}\mathbf{w})^T(\mathbf{y} - \mathbf{X}\mathbf{w})$$

Taking the gradient and setting it to zero:

$$\nabla_\mathbf{w} J = -\frac{2}{m}\mathbf{X}^T(\mathbf{y} - \mathbf{X}\mathbf{w}) = 0$$

Solving for $\mathbf{w}$:

$$\mathbf{X}^T\mathbf{y} = \mathbf{X}^T\mathbf{X}\mathbf{w}$$

$$\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

### Implementation from Scratch

<div class="python-interactive" markdown="1">
```python
import numpy as np

def linear_regression_normal_equation(X, y):
    """
    Fit linear regression using the normal equation.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix, shape (m, n)
    y : np.ndarray
        Target vector, shape (m,)
    
    Returns:
    --------
    np.ndarray
        Optimal weights, shape (n+1,)
    """
    # Add bias column
    m = X.shape[0]
    X_b = np.column_stack([np.ones(m), X])
    
    # Normal equation: w = (X^T X)^(-1) X^T y
    XtX = X_b.T @ X_b
    Xty = X_b.T @ y
    w = np.linalg.inv(XtX) @ Xty
    
    return w

# Generate synthetic data
np.random.seed(42)
m = 100  # samples
n = 3    # features

# True relationship: y = 1 + 2*x1 + 3*x2 + 4*x3 + noise
X = np.random.randn(m, n)
true_w = np.array([1, 2, 3, 4])  # [bias, w1, w2, w3]
y = X @ true_w[1:] + true_w[0] + np.random.randn(m) * 0.5

# Fit model
w_learned = linear_regression_normal_equation(X, y)

print("=== Normal Equation Results ===")
print(f"True weights:    {true_w}")
print(f"Learned weights: {w_learned.round(4)}")
print(f"\nError: {np.abs(true_w - w_learned).round(4)}")

# Make predictions
X_b = np.column_stack([np.ones(m), X])
y_pred = X_b @ w_learned
mse = np.mean((y - y_pred) ** 2)
print(f"\nMean Squared Error: {mse:.6f}")
```
</div>

### When to Use Normal Equation vs Gradient Descent

| Aspect | Normal Equation | Gradient Descent |
|--------|-----------------|------------------|
| **Computation** | $O(n^3)$ for matrix inverse | $O(kmn)$ for k iterations |
| **Features** | Slow for large $n$ (>10,000) | Works with many features |
| **Iterations** | One step | Many iterations needed |
| **Learning rate** | Not needed | Must be tuned |
| **Scalability** | Limited | Scales to big data |

!!! tip "Rule of Thumb"
    - Use **Normal Equation** when $n < 10,000$ features
    - Use **Gradient Descent** when $n > 10,000$ or data doesn't fit in memory

### Using NumPy's Built-in Solver

In practice, use `np.linalg.lstsq()` instead of computing the inverse directly - it's more numerically stable:

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Same data as before
np.random.seed(42)
m, n = 100, 3
X = np.random.randn(m, n)
true_w = np.array([1, 2, 3, 4])
y = X @ true_w[1:] + true_w[0] + np.random.randn(m) * 0.5

# Add bias column
X_b = np.column_stack([np.ones(m), X])

# Method 1: Normal equation with inverse (less stable)
w_inv = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Method 2: np.linalg.lstsq (more stable, recommended)
w_lstsq, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=None)

# Method 3: np.linalg.solve (if X^T X is well-conditioned)
w_solve = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)

print("Comparison of methods:")
print(f"Inverse method:  {w_inv.round(4)}")
print(f"lstsq method:    {w_lstsq.round(4)}")
print(f"solve method:    {w_solve.round(4)}")
print(f"True weights:    {true_w}")

print("\n💡 np.linalg.lstsq is most robust - use it in practice!")
```
</div>

## Using Scikit-Learn

For production code, use scikit-learn's `LinearRegression`:

<div class="python-interactive" markdown="1">
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate data
np.random.seed(42)
m = 200
X = np.random.randn(m, 3)
true_w = np.array([1, 2, 3, 4])  # [bias, w1, w2, w3]
y = X @ true_w[1:] + true_w[0] + np.random.randn(m) * 0.5

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Get learned parameters
print("=== Scikit-Learn LinearRegression ===")
print(f"Learned intercept (bias): {model.intercept_:.4f}")
print(f"Learned coefficients: {model.coef_.round(4)}")
print(f"\nTrue: bias=1, coefficients=[2, 3, 4]")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Test Set Performance ===")
print(f"Mean Squared Error: {mse:.6f}")
print(f"R² Score: {r2:.6f}")
```
</div>

## Model Evaluation Metrics

Evaluating regression models requires different metrics than classification. Here are the most important ones:

### Mean Squared Error (MSE)

$$\text{MSE} = \frac{1}{m}\sum_{i=1}^{m}(y^{(i)} - \hat{y}^{(i)})^2$$

- Average of squared errors
- **Lower is better** (minimum is 0)
- Units are squared (hard to interpret)

### Root Mean Squared Error (RMSE)

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(y^{(i)} - \hat{y}^{(i)})^2}$$

- Square root of MSE
- **Same units as target** (easier to interpret)
- "Average error magnitude"

### Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{m}\sum_{i=1}^{m}|y^{(i)} - \hat{y}^{(i)}|$$

- Average of absolute errors
- **More robust to outliers** than MSE
- Same units as target

### R² Score (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_{i=1}^{m}(y^{(i)} - \hat{y}^{(i)})^2}{\sum_{i=1}^{m}(y^{(i)} - \bar{y})^2} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}$$

- Measures proportion of variance explained by the model
- **Range**: Can be negative (terrible model), 0 (baseline), 1 (perfect)
- $R^2 = 1$: Perfect predictions
- $R^2 = 0$: Model predicts the mean
- $R^2 < 0$: Model is worse than predicting the mean

### Computing All Metrics

<div class="python-interactive" markdown="1">
```python
import numpy as np

def evaluate_regression(y_true, y_pred):
    """
    Calculate all regression metrics.
    """
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R² Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }

# Generate sample predictions
np.random.seed(42)
y_true = np.array([3.0, -0.5, 2.0, 7.0, 4.5, 6.0, 2.5])

# Good predictions
y_pred_good = np.array([2.8, -0.3, 2.2, 7.1, 4.3, 5.8, 2.6])

# Bad predictions
y_pred_bad = np.array([1.0, 1.0, 1.0, 4.0, 3.0, 3.0, 3.0])

# Mean predictions (baseline)
y_pred_mean = np.full_like(y_true, np.mean(y_true))

print("=== Comparison of Model Quality ===\n")

print("Good Model:")
metrics = evaluate_regression(y_true, y_pred_good)
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")

print("\nBad Model:")
metrics = evaluate_regression(y_true, y_pred_bad)
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")

print("\nBaseline (Mean Prediction):")
metrics = evaluate_regression(y_true, y_pred_mean)
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")

print("\n💡 Notice: R² = 0 when predicting the mean (baseline)")
```
</div>

### Choosing the Right Metric

| Metric | When to Use |
|--------|-------------|
| **MSE/RMSE** | When large errors are especially bad; standard choice |
| **MAE** | When outliers are present; want interpretable units |
| **R²** | When comparing models; want proportion of variance explained |

!!! warning "R² Caveats"
    - R² can be misleading for non-linear relationships
    - Can be artificially inflated by adding more features
    - Always look at residual plots, not just R²

## Visualizing Model Fit

### Predictions vs Actual

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate data
np.random.seed(42)
X = np.random.randn(100, 1) * 2
y = 3 * X.ravel() + 2 + np.random.randn(100) * 1.5

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Predictions vs Actual
ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.6)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2, label='Perfect prediction')
ax1.set_xlabel('Actual Values', fontsize=12)
ax1.set_ylabel('Predicted Values', fontsize=12)
ax1.set_title('Predictions vs Actual', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residuals
ax2 = axes[1]
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, alpha=0.6)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Values', fontsize=12)
ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
ax2.set_title('Residual Plot', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Good residual plot characteristics:")
print("  ✓ Points randomly scattered around 0")
print("  ✓ No clear patterns")
print("  ✓ Roughly constant spread (homoscedasticity)")
```
</div>

### Residual Analysis

Residuals ($e = y - \hat{y}$) should be:
1. **Randomly distributed** around zero
2. **No patterns** (patterns suggest missing features or wrong model)
3. **Constant variance** (homoscedasticity)
4. **Approximately normal** (for statistical inference)

## Assumptions of Linear Regression

Linear regression makes several assumptions. Violating them can lead to poor predictions or misleading conclusions.

### 1. Linearity
The relationship between features and target is linear.

**Check:** Look for patterns in residual plots.

### 2. Independence
Observations are independent of each other.

**Violation example:** Time series data (today's value depends on yesterday's).

### 3. Homoscedasticity
Constant variance of residuals across all predictions.

**Check:** Residuals should have similar spread at all prediction levels.

### 4. No Multicollinearity
Features should not be highly correlated with each other.

**Problem:** Unstable weight estimates, hard to interpret.

<div class="python-interactive" markdown="1">
```python
import numpy as np

# Example of multicollinearity
np.random.seed(42)
n = 100

# Feature 1: original
x1 = np.random.randn(n)

# Feature 2: almost identical to feature 1 (high correlation!)
x2 = x1 + np.random.randn(n) * 0.01

# Feature 3: independent
x3 = np.random.randn(n)

X = np.column_stack([x1, x2, x3])

# Correlation matrix
corr_matrix = np.corrcoef(X.T)

print("Correlation Matrix:")
print(corr_matrix.round(4))

print("\n⚠️ Features 1 and 2 are nearly identical (correlation ≈ 1)")
print("   This is multicollinearity - avoid in your models!")

print("\n💡 Solutions:")
print("   - Remove one of the correlated features")
print("   - Use regularization (Ridge, Lasso)")
print("   - Use PCA to create uncorrelated features")
```
</div>

## Complete Example: California Housing

Let's apply everything to a real dataset:

<div class="python-interactive" markdown="1">
```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

print("=== California Housing Dataset ===")
print(f"Features: {feature_names}")
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"\nTarget: Median house value (in $100,000s)")
print(f"Target range: ${y.min()*100000:,.0f} to ${y.max()*100000:,.0f}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (important for comparing coefficients!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Model Performance ===")
print(f"RMSE: ${rmse * 100000:,.0f}")
print(f"R² Score: {r2:.4f}")
print(f"\n→ Model explains {r2*100:.1f}% of house price variance")

# Feature importance (by coefficient magnitude)
print(f"\n=== Feature Importance (Standardized Coefficients) ===")
coef_df = list(zip(feature_names, model.coef_))
coef_df.sort(key=lambda x: abs(x[1]), reverse=True)
for name, coef in coef_df:
    sign = "+" if coef > 0 else ""
    print(f"  {name:12s}: {sign}{coef:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Predictions vs Actual
ax1 = axes[0]
ax1.scatter(y_test, y_pred, alpha=0.3, s=10)
ax1.plot([0, 5], [0, 5], 'r--', linewidth=2)
ax1.set_xlabel('Actual Price ($100,000s)')
ax1.set_ylabel('Predicted Price ($100,000s)')
ax1.set_title(f'Predictions vs Actual (R² = {r2:.3f})')
ax1.grid(True, alpha=0.3)

# Feature coefficients
ax2 = axes[1]
colors = ['green' if c > 0 else 'red' for c in model.coef_]
ax2.barh(feature_names, model.coef_, color=colors, alpha=0.7)
ax2.set_xlabel('Coefficient (Standardized)')
ax2.set_title('Feature Coefficients')
ax2.axvline(x=0, color='black', linewidth=0.5)
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
```
</div>

## Key Takeaways

!!! success "Important Concepts"
    - **Linear regression** models the relationship $\hat{y} = \mathbf{X}\mathbf{w}$
    - The **goal** is to minimize MSE (Mean Squared Error)
    - **Normal equation** gives closed-form solution: $\mathbf{w} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$
    - Use `np.linalg.lstsq()` or sklearn for numerical stability
    - **Evaluation metrics**: MSE, RMSE, MAE, R²
    - **Always visualize** predictions and residuals
    - Check **assumptions**: linearity, independence, homoscedasticity, no multicollinearity

## Common Patterns in ML

| Concept | Formula | Implementation |
|---------|---------|----------------|
| Prediction | $\hat{y} = \mathbf{X}\mathbf{w}$ | `y_pred = X @ w` |
| MSE Loss | $\frac{1}{m}\|\mathbf{y} - \hat{\mathbf{y}}\|^2$ | `np.mean((y - y_pred)**2)` |
| Normal Equation | $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ | `np.linalg.lstsq(X, y)` |
| R² Score | $1 - \frac{SS_{res}}{SS_{tot}}$ | `r2_score(y, y_pred)` |

## Practice Problems

Before moving to the next lesson, try these:

1. Implement simple linear regression from scratch (without using the closed-form formula for multiple features)
2. Create a dataset where linear regression performs poorly (hint: non-linear relationship)
3. Calculate R² manually and verify against sklearn
4. Plot residuals for a well-fitted vs poorly-fitted model

## What's Next?

In the next lesson, you'll learn **Gradient Descent** - the optimization algorithm that powers most of machine learning, including neural networks!

[Next: Lesson 2 - Gradient Descent](02-gradient-descent.md){ .md-button .md-button--primary }

[Complete the Exercises](exercises.md){ .md-button }

[Back to Module Overview](index.md){ .md-button }

---

**Questions?** Open an issue on [GitHub](https://github.com/jumpingsphinx/jumpingsphinx.github.io/issues).