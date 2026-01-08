# Lesson 4: Boosting

## Introduction

Boosting is one of the most powerful and theoretically grounded ensemble methods in machine learning. Unlike Random Forest, which builds trees independently and reduces variance through averaging, boosting builds trees **sequentially**, with each new tree focusing on correcting the mistakes of the previous ones. This iterative error-correction process allows boosting to reduce **bias** while maintaining low variance, often achieving state-of-the-art performance.

The key insight behind boosting is deceptively simple: **combine many weak learners to create a strong learner**. A "weak learner" is a model that performs only slightly better than random guessing. By carefully combining these weak models, boosting can create an ensemble that rivals or exceeds the performance of much more complex individual models.

### Why Boosting Matters

**Historical Impact:**

- Boosting algorithms (AdaBoost, Gradient Boosting, XGBoost) dominate machine learning competitions
- Won the prestigious Gödel Prize in 2003 for theoretical contributions
- Forms the basis of many production systems at tech companies
- Consistently outperforms other algorithms on tabular data

**Real-World Applications:**

- **Search Ranking**: Google, Bing use gradient boosting for search result ranking
- **Click Prediction**: Facebook uses boosting for ad click-through rate prediction
- **Fraud Detection**: Financial institutions use boosting to detect fraudulent transactions
- **Risk Assessment**: Insurance and lending use boosting for risk scoring
- **Computer Vision**: Face detection (Viola-Jones algorithm) uses boosted classifiers

**Key Advantages:**

1. **High Accuracy**: Often achieves best performance on structured/tabular data
2. **Bias Reduction**: Focuses on hard examples, reducing underfitting
3. **Feature Interactions**: Automatically learns complex feature interactions
4. **Handles Mixed Data**: Works with continuous, categorical, and missing values
5. **Theoretically Grounded**: Strong mathematical foundations and guarantees
6. **Versatile**: Many variants (AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost)

## Boosting vs Bagging: Fundamental Differences

### Visual Introduction to Boosting

Before getting into the details, watch this excellent explanation of AdaBoost and the boosting concept:

<div style="text-align: center; margin: 20px 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/LsK-xG1cLYA" title="AdaBoost by StatQuest" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

Before getting into specific algorithms, let's understand how boosting differs from bagging (Random Forest).

### Philosophical Difference

**Bagging (Random Forest):**
- **Philosophy**: "Wisdom of crowds" - average many independent opinions
- **Training**: Build trees in parallel, independently
- **Diversity**: Through bootstrap sampling and feature randomness
- **Focus**: All data points treated equally
- **Error Reduction**: Reduces variance by averaging
- **Weak Point**: Cannot reduce bias of base learners

**Boosting:**
- **Philosophy**: "Learn from mistakes" - iteratively focus on errors
- **Training**: Build trees sequentially, each depends on previous
- **Diversity**: Through adaptive reweighting of training examples
- **Focus**: Hard-to-predict examples get more attention
- **Error Reduction**: Reduces bias by focusing on difficult cases
- **Weak Point**: Can overfit if too many iterations

### Mathematical Perspective

**Bagging Prediction:**

All trees have equal weight:

$$\hat{f}(x) = \frac{1}{B}\sum_{b=1}^{B}f_b(x)$$

**Boosting Prediction:**

Trees have different weights $\alpha_b$ based on their performance:

$$\hat{f}(x) = \sum_{b=1}^{B}\alpha_b f_b(x)$$

The weights $\alpha_b$ are larger for more accurate trees and smaller for less accurate ones.

### Visual Comparison

Imagine you're learning to play darts:

**Bagging Approach:**
1. Take 100 practice throws independently
2. For each new dart, throw it where the average of all 100 practice throws landed
3. Result: Smooth, stable performance, but can't improve on systematic errors

**Boosting Approach:**
1. Throw a dart
2. Notice where you missed (e.g., consistently too far left)
3. Next throw, explicitly correct for that error (aim more right)
4. Repeat: each throw learns from previous mistakes
5. Result: Progressively improve, eventually hit bullseye

### When to Use Each

| Scenario | Bagging (Random Forest) | Boosting |
|----------|------------------------|----------|
| **Goal** | Reduce variance, robust baseline | Maximum accuracy, reduce bias |
| **Data Quality** | Noisy data | Clean data (sensitive to outliers) |
| **Model Complexity** | Base models are complex (deep trees) | Base models are simple (shallow trees) |
| **Overfitting Risk** | Low (averaging reduces overfit) | Higher (needs careful tuning) |
| **Training Speed** | Fast (parallelizable) | Slower (sequential) |
| **Interpretability** | Moderate (feature importance) | Moderate to Low |
| **Hyperparameter Sensitivity** | Low (robust defaults) | High (requires tuning) |

## AdaBoost: Adaptive Boosting

AdaBoost (Adaptive Boosting), introduced by Freund and Schapire in 1997, was the first practical boosting algorithm. It's elegant, theoretically sound, and provides an excellent introduction to boosting concepts.

### The Core Idea

AdaBoost maintains a **weight** for each training example. These weights determine how much each example influences the next tree:

1. **Start**: All examples have equal weight
2. **Train**: Build a weak learner on weighted data
3. **Evaluate**: Calculate how well the learner performed
4. **Adapt**: Increase weights of misclassified examples, decrease weights of correctly classified ones
5. **Repeat**: Next learner focuses more on previously misclassified examples

Over time, the algorithm forces itself to learn the hard cases.

### AdaBoost Algorithm (Classification)

**Input:**
- Training data: $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ where $y_i \in \{-1, +1\}$
- Number of iterations: $T$
- Weak learner algorithm (e.g., decision stump)

**Initialize:**

$$w_i^{(1)} = \frac{1}{n} \quad \text{for } i = 1, \ldots, n$$

All examples start with equal weight.

**For t = 1 to T:**

1. **Train weak learner** $h_t$ on data weighted by $w^{(t)}$:

   $$h_t = \text{argmin}_{h} \sum_{i=1}^{n} w_i^{(t)} \mathbb{1}(h(x_i) \neq y_i)$$

2. **Calculate weighted error**:

   $$\epsilon_t = \frac{\sum_{i=1}^{n} w_i^{(t)} \mathbb{1}(h_t(x_i) \neq y_i)}{\sum_{i=1}^{n} w_i^{(t)}}$$

3. **Calculate learner weight** (higher weight for more accurate learners):

   $$\alpha_t = \frac{1}{2}\ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$

4. **Update example weights**:

   $$w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(x_i))$$

   - If correctly classified ($y_i h_t(x_i) = 1$): weight decreases (multiply by $e^{-\alpha_t} < 1$)
   - If misclassified ($y_i h_t(x_i) = -1$): weight increases (multiply by $e^{\alpha_t} > 1$)

5. **Normalize weights**:

   $$w_i^{(t+1)} = \frac{w_i^{(t+1)}}{\sum_{j=1}^{n} w_j^{(t+1)}}$$

**Final prediction:**

$$H(x) = \text{sign}\left(\sum_{t=1}^{T}\alpha_t h_t(x)\right)$$

### Understanding the Weight Update

The weight update formula is the heart of AdaBoost. Let's understand it:

$$w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(x_i))$$

**Case 1: Correct prediction** ($y_i = h_t(x_i)$, so $y_i h_t(x_i) = 1$)

$$w_i^{(t+1)} = w_i^{(t)} \cdot e^{-\alpha_t}$$

Since $\alpha_t > 0$ and $e^{-\alpha_t} < 1$, the weight **decreases**. The algorithm pays less attention to this example next round.

**Case 2: Incorrect prediction** ($y_i \neq h_t(x_i)$, so $y_i h_t(x_i) = -1$)

$$w_i^{(t+1)} = w_i^{(t)} \cdot e^{\alpha_t}$$

Since $e^{\alpha_t} > 1$, the weight **increases**. The algorithm pays more attention to this example next round.

**Magnitude of $\alpha_t$:**

$$\alpha_t = \frac{1}{2}\ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$

- If $\epsilon_t = 0.5$ (random guessing): $\alpha_t = 0$ (learner ignored)
- If $\epsilon_t = 0.1$ (good learner): $\alpha_t = 1.1$ (high weight)
- If $\epsilon_t = 0.01$ (excellent learner): $\alpha_t = 2.3$ (very high weight)

### Interactive Example: AdaBoost in Action

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Create a complex dataset
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Dataset: Two interleaving half circles")
print(f"Training: {len(X_train)} samples")
print(f"Testing: {len(X_test)} samples")

# Single weak learner (decision stump: tree with max_depth=1)
weak_learner = DecisionTreeClassifier(max_depth=1, random_state=42)
weak_learner.fit(X_train, y_train)

# AdaBoost with 50 stumps
adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    algorithm='SAMME',
    random_state=42
)
adaboost.fit(X_train, y_train)

# Visualize decision boundaries
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
                     np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200))

# Predictions
Z_weak = weak_learner.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_boost = adaboost.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Single weak learner
axes[0].contourf(xx, yy, Z_weak, alpha=0.4, cmap='RdYlBu')
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', edgecolor='black')
axes[0].set_title(f'Single Weak Learner (Stump)\nTrain: {weak_learner.score(X_train, y_train):.3f}, Test: {weak_learner.score(X_test, y_test):.3f}')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# AdaBoost
axes[1].contourf(xx, yy, Z_boost, alpha=0.4, cmap='RdYlBu')
axes[1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', edgecolor='black')
axes[1].set_title(f'AdaBoost (50 Stumps)\nTrain: {adaboost.score(X_train, y_train):.3f}, Test: {adaboost.score(X_test, y_test):.3f}')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

# Show learning progression
print("\n" + "="*50)
print("Learning Progression:")
print("="*50)

n_estimators_range = [1, 5, 10, 20, 50]
for n in n_estimators_range:
    ada_temp = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=n,
        algorithm='SAMME',
        random_state=42
    )
    ada_temp.fit(X_train, y_train)
    train_acc = ada_temp.score(X_train, y_train)
    test_acc = ada_temp.score(X_test, y_test)
    print(f"  {n:3d} weak learners: Train={train_acc:.3f}, Test={test_acc:.3f}")

# Analyze estimator weights
print("\n" + "="*50)
print("Estimator Weights (first 10):")
print("="*50)
print("Higher weights = better performing weak learners")
for i in range(min(10, len(adaboost.estimator_weights_))):
    print(f"  Estimator {i+1}: weight = {adaboost.estimator_weights_[i]:.3f}")

print("\nWeighted combination of 50 weak learners creates a strong learner!")
```
</div>

**Expected Output:**
- Single stump creates simple linear decision boundary (underfits)
- AdaBoost creates complex, non-linear boundary that fits the moons
- Accuracy improves progressively with more weak learners
- Different weak learners have different weights based on performance

### AdaBoost Properties and Theory

**Theoretical Guarantees:**

AdaBoost has remarkable theoretical properties. The training error of the final ensemble is bounded:

$$\text{Training Error} \leq \exp\left(-2\sum_{t=1}^{T}(\frac{1}{2} - \epsilon_t)^2\right)$$

This shows that if each weak learner is better than random ($\epsilon_t < 0.5$), the training error decreases **exponentially** with the number of iterations!

**Margin Theory:**

AdaBoost doesn't just minimize classification error—it maximizes the **margin**. The margin of an example is how confident the ensemble is in its prediction:

$$\text{margin}(x_i, y_i) = y_i \sum_{t=1}^{T}\alpha_t h_t(x_i)$$

- Large positive margin: confident correct prediction
- Large negative margin: confident wrong prediction
- Near zero: uncertain

AdaBoost tends to increase the margins of all examples, which improves generalization.

**Relationship to Exponential Loss:**

AdaBoost can be viewed as performing gradient descent on the exponential loss function:

$$L(y, f(x)) = \exp(-y f(x))$$

This connection links AdaBoost to broader optimization frameworks and leads to Gradient Boosting.

## Gradient Boosting Machines (GBM)

### Visual Introduction to Gradient Boosting

For a deeper understanding of gradient boosting, watch this excellent explanation:

<div style="text-align: center; margin: 20px 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/3CC4N4z3GJc" title="Gradient Boost Part 1 by StatQuest" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

Gradient Boosting is a more general and powerful framework than AdaBoost. Instead of reweighting examples, it directly optimizes a loss function using gradient descent in function space.

### The Core Idea

**Traditional Gradient Descent:**

In standard machine learning, we optimize parameters $\theta$ of a model $f_\theta$:

$$\theta_{t+1} = \theta_t - \eta \frac{\partial L}{\partial \theta}$$

**Gradient Boosting:**

Instead of optimizing parameters, we optimize the **function** itself by adding new functions:

$$f_{t+1}(x) = f_t(x) + \eta h_t(x)$$

where $h_t$ is trained to approximate the **negative gradient** of the loss function.

### Intuition: Residual Fitting

For regression with squared error loss, gradient boosting has a beautiful interpretation:

1. **Start** with initial prediction $f_0(x) = \bar{y}$ (mean of targets)
2. **Calculate residuals**: $r_i = y_i - f_0(x_i)$ (what we got wrong)
3. **Train new tree** $h_1$ to predict residuals $r_i$
4. **Update model**: $f_1(x) = f_0(x) + \eta h_1(x)$ (correct our mistakes)
5. **Repeat**: Each new tree predicts the residuals of the previous model

This is like a team of experts, each correcting the mistakes of the previous team.

### Gradient Boosting Algorithm

**Input:**
- Training data: $\{(x_1, y_1), \ldots, (x_n, y_n)\}$
- Differentiable loss function: $L(y, f(x))$
- Number of iterations: $T$
- Learning rate: $\eta$ (shrinkage)
- Tree depth: $J$ (typically 3-10)

**Initialize:**

$$f_0(x) = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, \gamma)$$

For squared loss: $f_0(x) = \bar{y}$ (mean)
For log loss: $f_0(x) = \log\left(\frac{p}{1-p}\right)$ where $p = \frac{\#\text{positive}}{\#\text{total}}$

**For t = 1 to T:**

1. **Compute pseudo-residuals** (negative gradient of loss):

   $$r_{it} = -\frac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\Bigg|_{f=f_{t-1}}$$

   For squared loss: $r_{it} = y_i - f_{t-1}(x_i)$

   For log loss: $r_{it} = y_i - p_{t-1}(x_i)$ where $p = \frac{1}{1+e^{-f}}$

2. **Fit regression tree** $h_t$ to pseudo-residuals:

   $$h_t = \arg\min_h \sum_{i=1}^{n}(r_{it} - h(x_i))^2$$

3. **Update model**:

   $$f_t(x) = f_{t-1}(x) + \eta h_t(x)$$

**Final model:**

$$f(x) = f_0(x) + \eta\sum_{t=1}^{T}h_t(x)$$

### Loss Functions for Different Tasks

Gradient Boosting can optimize any differentiable loss function:

**Regression:**

1. **Squared Loss** (L2):
   $$L(y, f) = \frac{1}{2}(y - f)^2$$
   $$\text{Gradient: } \frac{\partial L}{\partial f} = f - y$$

2. **Absolute Loss** (L1):
   $$L(y, f) = |y - f|$$
   $$\text{Gradient: } \frac{\partial L}{\partial f} = \text{sign}(f - y)$$

3. **Huber Loss** (robust):
   $$L(y, f) = \begin{cases} \frac{1}{2}(y-f)^2 & \text{if } |y-f| \leq \delta \\ \delta|y-f| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

**Binary Classification:**

1. **Logistic Loss** (log loss, cross-entropy):
   $$L(y, f) = \log(1 + e^{-yf}) \quad \text{where } y \in \{-1, +1\}$$
   $$\text{Gradient: } \frac{\partial L}{\partial f} = -\frac{y}{1 + e^{yf}}$$

2. **Exponential Loss** (AdaBoost):
   $$L(y, f) = e^{-yf}$$
   $$\text{Gradient: } \frac{\partial L}{\partial f} = -ye^{-yf}$$

### Interactive Example: Gradient Boosting Regression

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Create synthetic regression data with non-linear pattern
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.3, X.shape[0])

print("Synthetic Non-linear Regression Problem")
print(f"Samples: {len(X)}")
print("Target: y = sin(x) + noise")

# Single deep tree (tends to overfit)
single_tree = DecisionTreeRegressor(max_depth=10, random_state=42)
single_tree.fit(X, y)

# Gradient Boosting with shallow trees
gb_shallow = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
gb_shallow.fit(X, y)

# Make predictions
X_plot = np.linspace(0, 10, 200).reshape(-1, 1)
y_single = single_tree.predict(X_plot)
y_gb = gb_shallow.predict(X_plot)
y_true = np.sin(X_plot).ravel()

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Single tree
axes[0].scatter(X, y, alpha=0.6, label='Training data')
axes[0].plot(X_plot, y_true, 'g--', linewidth=2, label='True function')
axes[0].plot(X_plot, y_single, 'r-', linewidth=2, label='Single tree prediction')
axes[0].set_xlabel('X')
axes[0].set_ylabel('y')
axes[0].set_title(f'Single Deep Tree (depth=10)\nTrain R²: {single_tree.score(X, y):.3f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Gradient Boosting
axes[1].scatter(X, y, alpha=0.6, label='Training data')
axes[1].plot(X_plot, y_true, 'g--', linewidth=2, label='True function')
axes[1].plot(X_plot, y_gb, 'b-', linewidth=2, label='Gradient Boosting')
axes[1].set_xlabel('X')
axes[1].set_ylabel('y')
axes[1].set_title(f'Gradient Boosting (100 trees, depth=3)\nTrain R²: {gb_shallow.score(X, y):.3f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Demonstrate residual fitting progression
print("\n" + "="*50)
print("Residual Fitting Progression")
print("="*50)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

stages = [0, 1, 5, 10, 50, 100]
for idx, stage in enumerate(stages):
    if stage == 0:
        # Initial prediction (mean)
        y_pred = np.full(len(X), y.mean())
        title = f"Stage 0: Initial (mean)\nR² = {1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2):.3f}"
    else:
        # Partial fit with 'stage' estimators
        gb_partial = GradientBoostingRegressor(
            n_estimators=stage,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        gb_partial.fit(X, y)
        y_pred = gb_partial.predict(X)
        title = f"Stage {stage}: {stage} trees\nR² = {gb_partial.score(X, y):.3f}"

    axes[idx].scatter(X, y, alpha=0.6, s=20)
    axes[idx].plot(X_plot, np.sin(X_plot), 'g--', linewidth=1.5)

    if stage == 0:
        axes[idx].axhline(y=y.mean(), color='r', linewidth=2, label='Initial prediction')
    else:
        X_plot_pred = gb_partial.predict(X_plot)
        axes[idx].plot(X_plot, X_plot_pred, 'r-', linewidth=2)

    axes[idx].set_title(title, fontsize=10)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlabel('X')
    axes[idx].set_ylabel('y')

plt.tight_layout()
plt.show()

print("\nEach stage adds a tree that predicts the residuals,")
print("progressively fitting the true function!")
```
</div>

**Expected Output:**
- Single deep tree overfits with jagged predictions
- Gradient Boosting smoothly approximates the true function
- Progression shows gradual improvement from mean to final fit
- R² increases monotonically as trees are added

### Shrinkage (Learning Rate)

The learning rate $\eta$ (also called shrinkage) controls how much each tree contributes:

$$f_t(x) = f_{t-1}(x) + \eta h_t(x)$$

**Why Shrinkage Helps:**

1. **Prevents Overfitting**: Small $\eta$ means each tree makes a small correction
2. **Improves Generalization**: More trees with small contributions → better ensemble
3. **Regularization**: Similar to regularization in neural networks

**Trade-off:**

- **Small $\eta$ (0.01-0.1)**: Better generalization, but needs many trees (slow)
- **Large $\eta$ (0.3-1.0)**: Faster training, but may overfit with too many trees
- **Typical**: $\eta = 0.1$ with 100-1000 trees

**Relationship between $\eta$ and $T$:**

There's a trade-off between learning rate and number of trees:
- Low $\eta$ + high $T$: Best performance, but slow
- High $\eta$ + low $T$: Fast, but worse performance
- Rule of thumb: $\eta \times T \approx \text{constant}$

### Interactive Example: Effect of Learning Rate

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print("Classification Dataset")
print(f"Training: {len(X_train)} samples")
print(f"Testing: {len(X_test)} samples")
print(f"Features: {X.shape[1]}")

# Test different learning rates
learning_rates = [0.01, 0.05, 0.1, 0.3, 1.0]
n_estimators = 200

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for lr in learning_rates:
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train, y_train)

    # Track performance as trees are added
    train_scores = []
    test_scores = []

    for i, y_pred_train in enumerate(gb.staged_predict(X_train)):
        train_scores.append(np.mean(y_pred_train == y_train))

    for i, y_pred_test in enumerate(gb.staged_predict(X_test)):
        test_scores.append(np.mean(y_pred_test == y_test))

    # Plot
    axes[0].plot(range(1, n_estimators+1), train_scores, label=f'LR={lr}', linewidth=2)
    axes[1].plot(range(1, n_estimators+1), test_scores, label=f'LR={lr}', linewidth=2)

axes[0].set_xlabel('Number of Trees', fontsize=11)
axes[0].set_ylabel('Accuracy', fontsize=11)
axes[0].set_title('Training Accuracy vs Learning Rate', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Number of Trees', fontsize=11)
axes[1].set_ylabel('Accuracy', fontsize=11)
axes[1].set_title('Test Accuracy vs Learning Rate', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print final accuracies
print("\n" + "="*50)
print("Final Test Accuracies:")
print("="*50)

for lr in learning_rates:
    gb = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=lr,
        max_depth=3,
        random_state=42
    )
    gb.fit(X_train, y_train)
    test_acc = gb.score(X_test, y_test)
    print(f"  Learning Rate {lr:.2f}: {test_acc:.4f}")

print("\nKey Observations:")
print("• High LR (1.0): Fast learning, but overfits early")
print("• Low LR (0.01): Slow learning, needs many trees")
print("• Medium LR (0.1): Good balance of speed and performance")
```
</div>

**Expected Output:**
- High learning rate converges fast but plateaus or overfits
- Low learning rate improves slowly and steadily
- Medium learning rate (0.1) often provides best test performance
- Training and test curves show the bias-variance tradeoff

## Weak Learners in Boosting

### Decision Stumps vs Shallow Trees

**Decision Stump:**
- Tree with max_depth=1 (single split)
- Very weak learner (barely better than random)
- AdaBoost typically uses stumps
- Fast to train

**Shallow Tree:**
- Tree with max_depth=3-10
- Can capture feature interactions
- Gradient Boosting typically uses shallow trees
- Balance between expressiveness and regularization

### Why Use Weak Learners?

It seems counterintuitive: why use weak models?

1. **Bias-Variance Trade-off**: Weak learners have high bias but low variance
2. **Complementarity**: Many weak learners can cover different parts of input space
3. **Regularization**: Prevents individual trees from overfitting
4. **Interpretability**: Shallow trees are easier to understand
5. **Computational Efficiency**: Faster to train many small trees than few large ones

**Mathematical Insight:**

A single complex tree might memorize training data (high variance). But 100 simple trees, each handling one aspect of the problem, can collectively model complex patterns while maintaining stability.

## Hyperparameters in Gradient Boosting

Gradient Boosting has many hyperparameters. Understanding them is crucial for good performance.

### Key Hyperparameters

| Parameter | Description | Typical Values | Effect |
|-----------|-------------|----------------|--------|
| `n_estimators` | Number of trees | 100-1000 | More trees → better fit (but can overfit) |
| `learning_rate` | Shrinkage factor | 0.01-0.3 | Lower → more trees needed, better generalization |
| `max_depth` | Tree depth | 3-10 | Deeper → more complex interactions (but can overfit) |
| `min_samples_split` | Min samples to split | 2-20 | Higher → simpler trees |
| `min_samples_leaf` | Min samples in leaf | 1-20 | Higher → smoother predictions |
| `subsample` | Fraction of samples per tree | 0.5-1.0 | <1.0 → stochastic gradient boosting (like bagging) |
| `max_features` | Features per split | sqrt(M) or M | Lower → more diversity |

### Tuning Strategy

**Start with these defaults:**
```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,
    random_state=42
)
```

**Then tune in this order:**

1. **n_estimators and learning_rate together**:
   - Try (100, 0.1), (200, 0.05), (500, 0.02)
   - Monitor validation error to detect overfitting

2. **max_depth**:
   - Try 3, 5, 7, 10
   - Deeper trees for complex datasets

3. **min_samples_split and min_samples_leaf**:
   - Increase if overfitting
   - Try 5, 10, 20

4. **subsample** (if overfitting):
   - Try 0.8, 0.5
   - Adds randomness like Random Forest

## Comparison: AdaBoost vs Gradient Boosting

| Aspect | AdaBoost | Gradient Boosting |
|--------|----------|-------------------|
| **Optimization** | Minimizes exponential loss | Minimizes any differentiable loss |
| **Weighting** | Reweights training examples | Fits residuals/pseudo-residuals |
| **Weak Learner** | Usually stumps (depth=1) | Usually shallow trees (depth=3-10) |
| **Loss Function** | Exponential loss only | Any differentiable loss |
| **Outlier Sensitivity** | High (exponential loss) | Lower (can use robust losses) |
| **Flexibility** | Less flexible | More flexible |
| **Use Case** | Binary classification | Classification and regression |
| **Modern Use** | Less common now | More common (XGBoost, LightGBM) |

## Practical Tips

### When to Use Boosting

**Ideal Scenarios:**
- Structured/tabular data with complex patterns
- Need maximum accuracy (Kaggle competitions)
- Large datasets where overfitting is less concern
- Clean data without too many outliers

**When to Avoid:**
- Very noisy data (use Random Forest)
- Real-time predictions (trees are sequential, slower)
- Need quick training (bagging is parallelizable)
- Limited compute resources

### Common Pitfalls

1. **Too Many Trees**: Monitor validation error, stop when it plateaus or increases
2. **Learning Rate Too High**: Causes overfitting and instability
3. **Trees Too Deep**: Individual trees overfit, reducing ensemble benefit
4. **Ignoring Outliers**: Boosting is sensitive to outliers, clean data first
5. **Not Using Validation**: Always use holdout or cross-validation to tune

## Summary

**Key Takeaways:**

1. **Boosting builds trees sequentially**, each correcting errors of previous ones
2. **AdaBoost** reweights examples, focusing on misclassified cases
3. **Gradient Boosting** fits residuals using gradient descent in function space
4. **Weak learners** (stumps or shallow trees) combine to form strong ensemble
5. **Learning rate** controls regularization through shrinkage
6. **Hyperparameter tuning** is crucial for best performance

**Boosting vs Bagging:**
- Boosting: Reduces bias, sequential, higher accuracy potential
- Bagging: Reduces variance, parallel, more robust to noise

**Next Steps:**

In the next lesson, we'll explore **XGBoost**, the industry-standard implementation of gradient boosting that adds:
- Advanced regularization techniques
- Optimized algorithms for speed
- Handling of missing values
- Built-in cross-validation
- Feature importance calculation

XGBoost and its variants (LightGBM, CatBoost) dominate Kaggle competitions and production ML systems.

## Practice Exercises

Ready to apply boosting? Head to the exercises to:

1. Implement AdaBoost from scratch
2. Compare AdaBoost and Gradient Boosting
3. Tune hyperparameters for optimal performance
4. Apply to real-world classification and regression problems
5. Understand when boosting outperforms Random Forest

[Start Exercises](exercises.md){ .md-button .md-button--primary }

## Additional Resources

- **Original AdaBoost Paper**: [Freund & Schapire (1997). "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting"](https://www.sciencedirect.com/science/article/pii/S002200009791504X)
- **Gradient Boosting Paper**: [Friedman (2001). "Greedy Function Approximation: A Gradient Boosting Machine"](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)
- **Scikit-learn Documentation**: [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html), [GradientBoosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- **Book**: "The Elements of Statistical Learning" - Chapter 10

---

**Navigation:**

- [← Previous: Random Forest](03-random-forest.md)
- [Next: XGBoost →](05-xgboost.md)
- [Module Overview](index.md)
