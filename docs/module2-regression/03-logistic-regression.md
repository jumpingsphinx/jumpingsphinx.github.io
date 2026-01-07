# Lesson 3: Logistic Regression

## Introduction

Despite its name, **Logistic Regression** is actually a **classification algorithm**, not a regression algorithm. It is one of the most popular and widely used algorithms in machine learning, serving as the "Hello World" of classification tasks.

While Linear Regression predicts a continuous value (like a house price), Logistic Regression predicts the **probability** that an instance belongs to a specific class (like "Spam" or "Not Spam").

In this lesson, you'll learn:
- Why linear regression fails for classification
- The Sigmoid function and probability interpretation
- The Log Loss (Binary Cross-Entropy) cost function
- How to implement logistic regression from scratch
- Decision boundaries
- Multi-class classification (One-vs-Rest)

## Regression vs. Classification

Before diving into the math, let's clarify the difference:

* **Regression**: Predicting a continuous number (e.g., temperature, price, age).
* **Classification**: Predicting a discrete label (e.g., Cat/Dog, Yes/No, Red/Blue/Green).

### Why not use Linear Regression?

Imagine you are trying to predict if a tumor is malignant (1) or benign (0) based on its size.

If you fit a linear regression line $h(x) = wx + b$, the output can be anything: $-10$, $1.5$, or $500$.
1.  **Unbounded Output**: Probability must be between 0 and 1. Linear regression goes to $\pm \infty$.
2.  **Sensitivity to Outliers**: Adding a single extreme data point can shift the regression line drastically, changing the classification threshold for all other points.

We need a function that "squashes" the output to be strictly between 0 and 1. Enter the **Sigmoid Function**.

## The Sigmoid Function

To convert the linear output $z = wx + b$ into a probability, we pass it through the **sigmoid function** (also called the logistic function):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Our hypothesis function becomes:

$$h_\theta(x) = \sigma(wx + b) = \frac{1}{1 + e^{-(wx + b)}}$$

### Properties of Sigmoid
- If $z \to \infty$, $\sigma(z) \to 1$
- If $z \to -\infty$, $\sigma(z) \to 0$
- If $z = 0$, $\sigma(z) = 0.5$

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate values for z
z = np.linspace(-10, 10, 100)
sigma = sigmoid(z)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(z, sigma, 'b-', linewidth=3, label=r'$\sigma(z) = \frac{1}{1+e^{-z}}$')
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.axhline(y=0.5, color='r', linestyle=':', label='Threshold (0.5)')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.1)
plt.axhline(y=1, color='k', linestyle='-', alpha=0.1)

plt.title('The Sigmoid Function', fontsize=16)
plt.xlabel('z (Linear Output: wx + b)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

print(f"Sigmoid(0) = {sigmoid(0)}")
print(f"Sigmoid(5) = {sigmoid(5):.4f} (High probability)")
print(f"Sigmoid(-5) = {sigmoid(-5):.4f} (Low probability)")

```

</div>

### Interpreting the Output

The output of the logistic regression model is the **probability that y = 1** given input x.

If , it means there is a **70% chance** the example is a positive case (e.g., spam) and a **30% chance** it is negative.

## Decision Boundary

To make a final classification (0 or 1), we typically use a threshold of 0.5:

* Predict **1** if 
* Predict **0** if 

Since  exactly when , this means we predict class 1 whenever:

The line  is called the **Decision Boundary**. It separates the region where we predict class 1 from the region where we predict class 0.

<div class="python-interactive" markdown="1">

```python
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic data (2 classes)

np.random.seed(42)

# Class 0: centered at (2, 2)

X0 = np.random.randn(20, 2) + [2, 2]

# Class 1: centered at (6, 6)

X1 = np.random.randn(20, 2) + [6, 6]

# Combine

X = np.vstack([X0, X1])
y = np.hstack([np.zeros(20), np.ones(20)])

# Let's assume we found these weights (w) and bias (b)

# This defines a line: w1*x1 + w2*x2 + b = 0

# x2 = -(w1\*x1 + b) / w2

w = np.array([1, 1])
b = -8

# Plotting the data

plt.figure(figsize=(8, 6))
plt.scatter(X0[:, 0], X0[:, 1], color='blue', label='Class 0')
plt.scatter(X1[:, 0], X1[:, 1], color='red', marker='x', label='Class 1')

# Plotting the Decision Boundary

x1\_boundary = np.linspace(0, 8, 100)
x2\_boundary = -(w[0] \* x1\_boundary + b) / w[1]

plt.plot(x1\_boundary, x2\_boundary, 'g--', linewidth=2, label='Decision Boundary')
plt.fill\_between(x1\_boundary, x2\_boundary, 10, color='red', alpha=0.1)
plt.fill\_between(x1\_boundary, -2, x2\_boundary, color='blue', alpha=0.1)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Linear Decision Boundary')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 9)
plt.ylim(0, 9)
plt.show()

print("The green line represents z = 0.")
print("Above the line, z \> 0 (Prediction: 1).")
print("Below the line, z \< 0 (Prediction: 0).")


```

## Cost Function: Log Loss

For Linear Regression, we used Mean Squared Error (MSE). However, if we put the sigmoid function into MSE, the resulting cost function becomes **non-convex** (wavy), meaning Gradient Descent might get stuck in local minima.

Instead, we use **Log Loss** (also called Binary Cross-Entropy).

### The Formula

This looks complex, but it's intuitive:

1. **If actual y = 1**: The cost is .
* If prediction , cost  (Correct!)
* If prediction , cost  (Huge penalty!)


2. **If actual y = 0**: The cost is .
* If prediction , cost  (Correct!)
* If prediction , cost  (Huge penalty!)



h = np.linspace(0.001, 0.999, 100)

# Cost when y = 1

cost_y1 = -np.log(h)

# Cost when y = 0

cost_y0 = -np.log(1 - h)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(h, cost_y1, 'b-', linewidth=2)
plt.title('Cost when True Class y = 1')
plt.xlabel('Predicted Probability h(x)')
plt.ylabel('Cost')
plt.grid(True, alpha=0.3)
plt.text(0.2, 3, 'Penalty increases\nas prediction approaches 0', fontsize=10)

plt.subplot(1, 2, 2)
plt.plot(h, cost_y0, 'r-', linewidth=2)
plt.title('Cost when True Class y = 0')
plt.xlabel('Predicted Probability h(x)')
plt.ylabel('Cost')
plt.grid(True, alpha=0.3)
plt.text(0.2, 3, 'Penalty increases\nas prediction approaches 1', fontsize=10)

plt.tight_layout()
plt.show()

```

&lt;/div&gt;

## Gradient Descent for Logistic Regression

To minimize the Log Loss, we use Gradient Descent. Amazingly, the **update rule looks exactly the same** as Linear Regression\!

$$w := w - \alpha \frac{\partial J}{\partial w}$$
$$b := b - \alpha \frac{\partial J}{\partial b}$$

However, the definition of $h(x)$ has changed:

  - **Linear Regression**: $h(x) = wx + b$
  - **Logistic Regression**: $h(x) = \sigma(wx + b)$

### Derivatives

$$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})$$

These derivatives are identical in form to linear regression, but the value of $h_\theta(x^{(i)})$ is calculated using the sigmoid.

## Implementation from Scratch

Let's build a `LogisticRegression` class using standard NumPy, similar to how we built our Linear Regression model.

&lt;div class=&quot;python-interactive&quot; markdown=&quot;1&quot;&gt;
```python
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
def **init**(self, learning\_rate=0.01, num\_iterations=1000):
self.learning\_rate = learning\_rate
self.num\_iterations = num\_iterations
self.w = None
self.b = None
self.cost\_history = []

```
def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

def fit(self, X, y):
    m, n = X.shape
    self.w = np.zeros(n)
    self.b = 0
    
    for i in range(self.num_iterations):
        # 1. Forward pass (Prediction)
        z = np.dot(X, self.w) + self.b
        predictions = self.sigmoid(z)
        
        # 2. Compute Cost (Log Loss)
        # Add epsilon to prevent log(0)
        epsilon = 1e-15 
        cost = (-1/m) * np.sum(y * np.log(predictions + epsilon) + 
                               (1 - y) * np.log(1 - predictions + epsilon))
        self.cost_history.append(cost)
        
        # 3. Backward pass (Gradients)
        errors = predictions - y
        dw = (1/m) * np.dot(X.T, errors)
        db = (1/m) * np.sum(errors)
        
        # 4. Update parameters
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
        
def predict_proba(self, X):
    z = np.dot(X, self.w) + self.b
    return self.sigmoid(z)

def predict(self, X, threshold=0.5):
    return (self.predict_proba(X) >= threshold).astype(int)
```

# \--- Test the Implementation ---

# Generate synthetic data

np.random.seed(0)
X = np.r\_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = np.array([0] \* 20 + [1] \* 20)

# Train

model = LogisticRegression(learning\_rate=0.1, num\_iterations=1000)
model.fit(X, y)

# Visualize Cost

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(model.cost\_history)
plt.title('Training Cost')
plt.xlabel('Iteration')
plt.ylabel('Log Loss')

# Visualize Decision Boundary

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=50)

# Create grid to draw boundary

x\_min, x\_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y\_min, y\_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x\_min, x\_max, 0.1),
np.arange(y\_min, y\_max, 0.1))

# Predict for every point on the grid

Z = model.predict(np.c\_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.title(f'Decision Boundary (Acc: {np.mean(model.predict(X) == y):.2f})')

plt.tight\_layout()
plt.show()

print(f"Learned Weights: {model.w}")
print(f"Learned Bias: {model.b}")


```

## Multi-Class Classification

What if we have 3 classes (e.g., Cat, Dog, Bird) instead of just 2?

### One-vs-Rest (OvR)

This is the most common strategy for extending binary classifiers to multi-class problems.

1. Train a separate binary classifier for each class.
* **Classifier 1**: Is it a Cat? (Yes vs. No)
* **Classifier 2**: Is it a Dog? (Yes vs. No)
* **Classifier 3**: Is it a Bird? (Yes vs. No)


2. When predicting, run all 3 classifiers.
3. Choose the class with the **highest probability**.

### Softmax Regression

Alternatively, we can generalize the Sigmoid function to the **Softmax function**. Softmax outputs a probability distribution across  classes such that they all sum to 1.

This is typically used in Neural Networks, but is essentially "Logistic Regression for multiple classes".

## Using Scikit-Learn

In production, you'll use sklearn's optimized implementation. It handles regularization (L2 by default) and multi-class support automatically.

# Generate dataset

X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
n_redundant=0, n_classes=2, random_state=42)

# Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train

# sklearn uses 'C' for Inverse Regularization Strength (smaller C = stronger regularization)

clf = LogisticRegression(C=1.0, random_state=42)
clf.fit(X_train, y_train)

# Predict

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("=== Scikit-Learn Logistic Regression ===")
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plotting the decision boundary

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
np.arange(y_min, y_max, 0.02))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap='coolwarm', s=50)
plt.title(f'Sklearn Logistic Regression (Accuracy: {acc:.2f})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

```

&lt;/div&gt;

## Key Takeaways

\!\!\! success "Essential Concepts"
\- **Classification**: Logistic Regression is used for predicting categorical labels (0 or 1).
\- **Sigmoid**: Transforms linear output $z$ into a probability between 0 and 1.
\- **Log Loss**: The convex cost function used for optimization. It heavily penalizes confident wrong predictions.
\- **Decision Boundary**: The line (or hyperplane) where probability is 0.5.
\- **Multi-class**: Can be handled using One-vs-Rest or Softmax.

\!\!\! warning "Common Pitfall"
Logistic Regression constructs a **linear** decision boundary. It cannot separate data that is not linearly separable (like a circle of points inside another circle) unless you add polynomial features\!

## What's Next?

We've now covered the two main pillars of supervised learning: Linear Regression and Logistic Regression. However, if we add too many features, our models might memorize the noise in the training data. This is called **Overfitting**.

In the next lesson, we'll learn how to fix this using **Regularization**.

[Next: Lesson 4 - Regularization](04-regularization.md){ .md-button .md-button--primary }

[Complete the Exercises](exercises.md){ .md-button }

[Back to Module Overview](index.md){ .md-button }

-----

**Questions?** Open an issue on [GitHub](https://github.com/jumpingsphinx/ML101/issues).

```
```

```