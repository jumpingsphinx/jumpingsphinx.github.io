# Decision Trees: Fundamentals and Intuition

## Introduction

Decision trees are one of the most intuitive and interpretable machine learning algorithms. Unlike the linear models we've studied so far, decision trees can naturally model complex, non-linear relationships by recursively partitioning the feature space into regions. Think of them as a series of yes/no questions that guide you to a prediction—much like the game "20 Questions" or a troubleshooting flowchart.

In this lesson, we'll build decision trees from the ground up, understanding not just how they work, but why they work and when to use them.

## Why Decision Trees?

Before diving into the mathematics, let's understand why decision trees are so popular in practice:

### Advantages

1. **Interpretability**: Trees are easy to visualize and explain to non-technical stakeholders
2. **No feature scaling needed**: Unlike linear models or neural networks, trees don't require normalization
3. **Handle non-linearity**: Capture complex, non-linear decision boundaries naturally
4. **Mixed data types**: Work with both numerical and categorical features
5. **Feature interactions**: Automatically discover interactions between features
6. **Minimal data preprocessing**: Handle missing values and outliers relatively well

### Limitations

1. **Overfitting**: Deep trees can memorize training data
2. **Instability**: Small changes in data can lead to very different trees
3. **Biased toward dominant classes**: In imbalanced datasets
4. **Greedy algorithm**: Local optimization doesn't guarantee global optimum
5. **Poor extrapolation**: Cannot predict beyond the range of training data

!!! tip "When to Use Decision Trees"
    Decision trees excel when:

    - You need model interpretability
    - Features have complex, non-linear relationships
    - You have mixed data types (categorical + numerical)
    - You want a quick baseline model

    Avoid when:

    - You need the absolute highest accuracy (use ensembles instead)
    - Data has strong linear relationships (linear models may be better)
    - You need reliable probability estimates

---

## The Core Idea: Recursive Partitioning

### Visual Introduction to Decision Trees

Before diving into the mathematics, watch this excellent visual explanation of how decision trees work:

<div style="text-align: center; margin: 20px 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/7VeUPuFGJHk" title="Decision Trees by StatQuest" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

At its heart, a decision tree recursively splits the feature space into rectangular regions. Each split is chosen to maximize the "purity" of the resulting groups—we want samples in each region to be as similar as possible in terms of their target variable.

### A Simple Example

Imagine predicting whether someone will buy a product based on their age and income:

```
Dataset:
Age | Income | Bought?
----|--------|--------
 25 |  30K   |   No
 45 |  80K   |   Yes
 35 |  50K   |   Yes
 20 |  25K   |   No
 50 | 100K   |   Yes
 30 |  40K   |   No
```

A decision tree might learn:

```
                [Root: All samples]
                        |
            Income > $45K?
           /                \
         Yes                 No
         /                     \
    Predict: Yes          Age > 27?
    (3/3 correct)        /         \
                       Yes          No
                       /              \
                 Predict: No      Predict: No
                 (2/2 correct)    (2/2 correct)
```

Each internal node asks a question about a feature, each branch represents an answer, and each leaf node makes a prediction.

---

## Mathematical Foundation

### Binary Classification Trees

For a binary classification problem, at each node we have a set of samples $S$ with labels $y_i \in \{0, 1\}$. Our goal is to split $S$ into two subsets $S_{\text{left}}$ and $S_{\text{right}}$ that are as "pure" as possible.

#### Impurity Measures

We measure purity using **impurity functions**. A node is pure (impurity = 0) when all samples belong to the same class, and maximally impure when classes are evenly distributed.

### 1. Gini Impurity

The **Gini impurity** measures the probability of incorrectly classifying a randomly chosen sample if we randomly assign a label according to the class distribution in the node:

$$
\text{Gini}(S) = 1 - \sum_{k=1}^{K} p_k^2
$$

Where:
- $K$ is the number of classes
- $p_k$ is the proportion of samples belonging to class $k$ in set $S$

**For binary classification** ($K = 2$):

$$
\text{Gini}(S) = 1 - p_1^2 - p_0^2 = 1 - p_1^2 - (1-p_1)^2 = 2p_1(1-p_1)
$$

**Properties:**
- Minimum: $\text{Gini} = 0$ (perfectly pure node)
- Maximum: $\text{Gini} = 0.5$ (50/50 split in binary case)
- Fast to compute (no logarithms)

**Example:** If a node has 70 class-0 samples and 30 class-1 samples:

$$
\text{Gini} = 1 - (0.7)^2 - (0.3)^2 = 1 - 0.49 - 0.09 = 0.42
$$

### 2. Entropy (Information Theory)

**Entropy** measures the amount of "disorder" or "uncertainty" in a set:

$$
\text{Entropy}(S) = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

Where $p_k \log_2(p_k) = 0$ if $p_k = 0$ (by convention).

**For binary classification**:

$$
\text{Entropy}(S) = -p_1 \log_2(p_1) - p_0 \log_2(p_0)
$$

**Properties:**
- Minimum: $\text{Entropy} = 0$ (pure node)
- Maximum: $\text{Entropy} = 1$ (50/50 split in binary case)
- More computationally expensive than Gini
- Rooted in information theory

**Example:** Same 70/30 split:

$$
\text{Entropy} = -0.7 \log_2(0.7) - 0.3 \log_2(0.3) \approx 0.88
$$

### 3. Information Gain

**Information Gain** measures the reduction in entropy (or impurity) from splitting:

$$
\text{IG}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)
$$

Where:
- $S$ is the current set of samples
- $A$ is the feature we're splitting on
- $S_v$ is the subset of $S$ where feature $A$ has value $v$
- $|S|$ denotes the size of set $S$

The second term is the **weighted average entropy** of the child nodes.

**Example:** Suppose we have a parent node with 100 samples (60 class-0, 40 class-1) and we split on a feature that creates:
- Left child: 70 samples (50 class-0, 20 class-1)
- Right child: 30 samples (10 class-0, 20 class-1)

$$
\begin{align}
\text{Entropy}_{\text{parent}} &= -0.6 \log_2(0.6) - 0.4 \log_2(0.4) \approx 0.971 \\
\text{Entropy}_{\text{left}} &= -\frac{50}{70} \log_2(\frac{50}{70}) - \frac{20}{70} \log_2(\frac{20}{70}) \approx 0.863 \\
\text{Entropy}_{\text{right}} &= -\frac{10}{30} \log_2(\frac{10}{30}) - \frac{20}{30} \log_2(\frac{20}{30}) \approx 0.918 \\
\text{Entropy}_{\text{weighted}} &= \frac{70}{100}(0.863) + \frac{30}{100}(0.918) \approx 0.879 \\
\text{IG} &= 0.971 - 0.879 = 0.092
\end{align}
$$

A higher information gain means a better split.

### Regression Trees: Variance Reduction

For **regression problems**, we predict continuous values. Instead of class purity, we measure how well samples in a node agree on their target value.

The impurity measure for regression is typically the **variance** or **mean squared error (MSE)**:

$$
\text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2
$$

Where $\bar{y} = \frac{1}{|S|} \sum_{i \in S} y_i$ is the mean target value in set $S$.

The **variance reduction** from a split is:

$$
\text{VR}(S, A) = \text{MSE}(S) - \sum_{v} \frac{|S_v|}{|S|} \text{MSE}(S_v)
$$

The prediction for a leaf node in regression is simply the **mean** of all target values in that leaf:

$$
\hat{y} = \frac{1}{|S_{\text{leaf}}|} \sum_{i \in S_{\text{leaf}}} y_i
$$

---

## The Splitting Algorithm

Building a decision tree involves recursively choosing the best split at each node. Here's the high-level algorithm:

### CART Algorithm (Classification and Regression Trees)

```
function BuildTree(S, max_depth, min_samples_split):
    # Stopping criteria
    if max_depth == 0 or |S| < min_samples_split or S is pure:
        return LeafNode(majority_class(S))  # or mean(S) for regression

    # Find best split
    best_feature, best_threshold = None, None
    best_gain = -infinity

    for each feature f in features:
        for each possible threshold t on feature f:
            S_left = {samples where f <= t}
            S_right = {samples where f > t}

            gain = InformationGain(S, S_left, S_right)

            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = t

    # Recursively build subtrees
    S_left = {samples where best_feature <= best_threshold}
    S_right = {samples where best_feature > best_threshold}

    left_subtree = BuildTree(S_left, max_depth - 1, min_samples_split)
    right_subtree = BuildTree(S_right, max_depth - 1, min_samples_split)

    return DecisionNode(best_feature, best_threshold, left_subtree, right_subtree)
```

### Key Hyperparameters

Decision trees have several hyperparameters to control their complexity:

1. **max_depth**: Maximum depth of the tree
   - Deeper trees = more complex models = higher risk of overfitting
   - Typical values: 3-10 for interpretable models, 10-20 for performance

2. **min_samples_split**: Minimum samples required to split a node
   - Higher values = simpler trees
   - Typical values: 2-20

3. **min_samples_leaf**: Minimum samples required in a leaf node
   - Prevents tiny leaves that may be noise
   - Typical values: 1-10

4. **max_features**: Number of features to consider for each split
   - Introduces randomness (used in Random Forests)
   - Options: None (all features), "sqrt", "log2", or integer

5. **criterion**: Impurity measure
   - Classification: "gini" or "entropy"
   - Regression: "mse" or "mae"

---

## Handling Different Feature Types

### Continuous Features

For continuous (numerical) features, we need to find the optimal threshold $t$ such that splitting on $\text{feature} \leq t$ maximizes information gain.

**Algorithm:**
1. Sort all unique values of the feature
2. For each adjacent pair of values, consider the midpoint as a threshold
3. Evaluate information gain for each threshold
4. Choose the threshold with maximum gain

**Example:** If a feature has values [1.2, 3.5, 3.7, 7.1], we'd test thresholds at:
- $(1.2 + 3.5)/2 = 2.35$
- $(3.5 + 3.7)/2 = 3.6$
- $(3.7 + 7.1)/2 = 5.4$

### Categorical Features

For categorical features with $K$ categories, there are $2^{K-1} - 1$ possible binary splits.

**For binary classification**, we can order categories by their proportion of positive class and treat them as ordinal, reducing complexity to $O(K \log K)$.

**Example:** For feature "Color" with values {Red, Blue, Green}:
- Possible splits: {Red} vs {Blue, Green}
- {Blue} vs {Red, Green}
- {Green} vs {Red, Blue}
- {Red, Blue} vs {Green}
- {Red, Green} vs {Blue}

---

## Interactive Code Examples

Let's build intuition with working code examples you can run and modify.

### Example 1: Computing Impurity Measures

<div class="python-interactive" markdown="1">
```python
import numpy as np

def gini_impurity(labels):
    """
    Calculate Gini impurity for a set of labels.

    Args:
        labels: Array of class labels

    Returns:
        Gini impurity (0 = pure, 0.5 = maximally impure for binary)
    """
    if len(labels) == 0:
        return 0

    # Count occurrences of each class
    classes, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)

    # Gini = 1 - sum(p_i^2)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

def entropy(labels):
    """
    Calculate entropy for a set of labels.

    Args:
        labels: Array of class labels

    Returns:
        Entropy (0 = pure, 1 = maximally impure for binary)
    """
    if len(labels) == 0:
        return 0

    classes, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)

    # Entropy = -sum(p_i * log2(p_i))
    # Use np.where to avoid log(0)
    entropy_val = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy_val

# Test with different class distributions
print("Pure node (all same class):")
pure_labels = np.array([1, 1, 1, 1, 1])
print(f"  Gini: {gini_impurity(pure_labels):.4f}")
print(f"  Entropy: {entropy(pure_labels):.4f}")

print("\n50/50 split (maximally impure for binary):")
balanced = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
print(f"  Gini: {gini_impurity(balanced):.4f}")
print(f"  Entropy: {entropy(balanced):.4f}")

print("\n70/30 split:")
imbalanced = np.array([0]*70 + [1]*30)
print(f"  Gini: {gini_impurity(imbalanced):.4f}")
print(f"  Entropy: {entropy(imbalanced):.4f}")
```
</div>

**Expected Output:**
```
Pure node (all same class):
  Gini: 0.0000
  Entropy: 0.0000

50/50 split (maximally impure for binary):
  Gini: 0.5000
  Entropy: 1.0000

70/30 split:
  Gini: 0.4200
  Entropy: 0.8813
```

### Example 2: Finding the Best Split

<div class="python-interactive" markdown="1">
```python
import numpy as np

def information_gain(parent_labels, left_labels, right_labels, criterion='gini'):
    """
    Calculate information gain from a split.

    Args:
        parent_labels: Labels before split
        left_labels: Labels in left child
        right_labels: Labels in right child
        criterion: 'gini' or 'entropy'

    Returns:
        Information gain (higher is better)
    """
    if criterion == 'gini':
        impurity_func = gini_impurity
    else:
        impurity_func = entropy

    n_parent = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)

    # Impurity before split
    parent_impurity = impurity_func(parent_labels)

    # Weighted average impurity after split
    child_impurity = (n_left / n_parent) * impurity_func(left_labels) + \
                     (n_right / n_parent) * impurity_func(right_labels)

    # Information gain
    gain = parent_impurity - child_impurity
    return gain

def find_best_split(X, y, feature_idx, criterion='gini'):
    """
    Find the best threshold to split on for a given feature.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        feature_idx: Index of feature to split on
        criterion: 'gini' or 'entropy'

    Returns:
        best_threshold, best_gain
    """
    feature_values = X[:, feature_idx]
    unique_values = np.unique(feature_values)

    best_gain = -np.inf
    best_threshold = None

    # Try each possible threshold (midpoint between consecutive values)
    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i+1]) / 2

        # Split data
        left_mask = feature_values <= threshold
        right_mask = ~left_mask

        # Calculate information gain
        gain = information_gain(y, y[left_mask], y[right_mask], criterion)

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain

# Example: Simple 2D dataset
np.random.seed(42)
X = np.random.randn(100, 2)
# Create labels based on a simple rule: x1 > 0
y = (X[:, 0] > 0).astype(int)

# Find best split on feature 0
threshold, gain = find_best_split(X, y, feature_idx=0, criterion='gini')
print(f"Best split on feature 0:")
print(f"  Threshold: {threshold:.4f}")
print(f"  Information gain: {gain:.4f}")

# Find best split on feature 1
threshold, gain = find_best_split(X, y, feature_idx=1, criterion='gini')
print(f"\nBest split on feature 1:")
print(f"  Threshold: {threshold:.4f}")
print(f"  Information gain: {gain:.4f}")
```
</div>

### Example 3: Simple Decision Tree Implementation

<div class="python-interactive" markdown="1">
```python
import numpy as np
from collections import Counter

class SimpleDecisionTree:
    """
    A simple decision tree implementation for binary classification.
    """
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        """Build the decision tree."""
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """Recursively build the tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            # Leaf node: return most common class
            most_common_class = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'class': most_common_class}

        # Find best split
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate gain
                gain = information_gain(y, y[left_mask], y[right_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        # If no good split found, create leaf
        if best_feature is None:
            most_common_class = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'class': most_common_class}

        # Create split
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Recursively build subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def predict(self, X):
        """Make predictions."""
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, node):
        """Predict a single sample."""
        if node['leaf']:
            return node['class']

        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])

# Test on a simple dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train our simple tree
tree = SimpleDecisionTree(max_depth=3)
tree.fit(X_train, y_train)

# Evaluate
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Simple Decision Tree Accuracy: {accuracy:.4f}")
```
</div>

---

## Visualizing Decision Trees

Visual understanding is crucial for decision trees. Let's see how to create various visualizations.

### Example 4: Tree Visualization with sklearn

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# Load iris dataset (using only 2 features for visualization)
iris = load_iris()
X = iris.data[:, :2]  # Use only sepal length and width
y = iris.target

# Binary classification: setosa vs non-setosa
y_binary = (y == 0).astype(int)

# Train a shallow tree for interpretability
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X, y_binary)

# Visualize the tree
plt.figure(figsize=(15, 10))
plot_tree(tree, feature_names=['Sepal Length', 'Sepal Width'],
          class_names=['Not Setosa', 'Setosa'],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Visualization (max_depth=3)")
plt.tight_layout()
plt.show()

# Print tree structure
print("Tree depth:", tree.get_depth())
print("Number of leaves:", tree.get_n_leaves())
print("Feature importances:", tree.feature_importances_)
```
</div>

### Example 5: Decision Boundary Visualization

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """Plot the decision boundary of a classifier."""
    h = 0.02  # Step size in mesh

    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k', cmap='RdYlBu')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(label='Class')
    plt.show()

# Generate data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Compare different tree depths
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, depth in enumerate([2, 5, 10]):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X, y)

    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes[i].contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    axes[i].scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors='k', cmap='RdYlBu')
    axes[i].set_title(f'max_depth={depth}')
    axes[i].set_xlabel('Feature 1')
    axes[i].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("Notice how deeper trees create more complex, rectangular decision boundaries")
print("Depth 2: Simple, smooth boundaries")
print("Depth 5: More detailed boundaries")
print("Depth 10: Very complex, potentially overfitting")
```
</div>

---

## Real-World Example: Titanic Survival Prediction

Let's apply decision trees to a real problem: predicting Titanic passenger survival.

<div class="python-interactive" markdown="1">
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Create a simplified Titanic-like dataset
np.random.seed(42)
n_samples = 500

data = {
    'Age': np.random.randint(1, 80, n_samples),
    'Fare': np.random.exponential(30, n_samples),
    'Sex': np.random.choice([0, 1], n_samples),  # 0: male, 1: female
    'Pclass': np.random.choice([1, 2, 3], n_samples),  # Ticket class
}

# Create survival based on rules (women and children first, higher class better)
df = pd.DataFrame(data)
survival_prob = 0.2  # Base survival rate

# Adjust based on features
survival_prob += (df['Sex'] == 1) * 0.5  # Women more likely to survive
survival_prob += (df['Age'] < 16) * 0.3  # Children more likely
survival_prob -= (df['Pclass'] == 3) * 0.2  # Third class less likely
survival_prob = np.clip(survival_prob, 0, 1)

df['Survived'] = (np.random.random(n_samples) < survival_prob).astype(int)

print("Dataset Overview:")
print(df.head())
print("\nSurvival Rate:", df['Survived'].mean())
print("\nClass Distribution:")
print(df.groupby('Survived').size())

# Prepare data
X = df[['Age', 'Fare', 'Sex', 'Pclass']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train decision tree
tree = DecisionTreeClassifier(max_depth=4, min_samples_split=20, random_state=42)
tree.fit(X_train, y_train)

# Evaluate
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# Feature importance
feature_names = ['Age', 'Fare', 'Sex', 'Pclass']
importances = tree.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importances:")
for i in range(len(feature_names)):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances in Titanic Survival Prediction')
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```
</div>

---

## Overfitting and Tree Pruning

One of the biggest challenges with decision trees is **overfitting**—when the tree becomes so complex that it memorizes the training data rather than learning generalizable patterns.

### Pre-Pruning (Early Stopping)

Stop growing the tree based on criteria:

- `max_depth`: Limit tree depth
- `min_samples_split`: Require minimum samples to split
- `min_samples_leaf`: Require minimum samples in leaves
- `max_leaf_nodes`: Limit total number of leaves

### Post-Pruning (Cost-Complexity Pruning)

Build a full tree, then prune it back. The idea is to trade off tree size against fit to data:

$$
\text{Cost}_\alpha(T) = \text{Error}(T) + \alpha \cdot |T|
$$

Where:
- $\text{Error}(T)$ is the training error
- $|T|$ is the number of leaf nodes
- $\alpha$ is the complexity parameter (higher $\alpha$ = more pruning)

### Example 6: Comparing Tree Complexity

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Generate dataset with noise
X, y = make_classification(n_samples=300, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1,
                           flip_y=0.1, random_state=42)  # 10% label noise

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train trees with different max_depth
depths = range(1, 15)
train_scores = []
test_scores = []

for depth in depths:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'o-', label='Training Accuracy', linewidth=2)
plt.plot(depths, test_scores, 's-', label='Test Accuracy', linewidth=2)
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree: Training vs Test Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Find optimal depth
optimal_depth = depths[np.argmax(test_scores)]
print(f"Optimal max_depth: {optimal_depth}")
print(f"Best test accuracy: {max(test_scores):.4f}")
print("\nNotice the overfitting:")
print(f"- At depth {depths[-1]}: train={train_scores[-1]:.4f}, test={test_scores[-1]:.4f}")
print(f"- At optimal depth {optimal_depth}: train={train_scores[optimal_depth-1]:.4f}, test={test_scores[optimal_depth-1]:.4f}")
```
</div>

---

## Feature Importance

Decision trees naturally provide **feature importance** scores, which measure how much each feature contributes to reducing impurity across all splits.

### Calculation

For each feature $i$, importance is computed as:

$$
\text{Importance}(i) = \sum_{t \in \text{splits on } i} \frac{n_t}{N} \cdot \Delta \text{Impurity}_t
$$

Where:
- $n_t$ is the number of samples at node $t$
- $N$ is the total number of samples
- $\Delta \text{Impurity}_t$ is the impurity reduction from split $t$

These are normalized to sum to 1.

**Interpretation:**
- Higher importance = feature is used higher in tree and reduces impurity more
- Importance = 0 means feature is never used
- Can be misleading with correlated features

---

## Practical Guidelines

### When to Use Decision Trees

✅ **Good for:**
- Exploratory analysis and quick baselines
- Problems requiring interpretability
- Mixed feature types (numerical + categorical)
- Non-linear relationships
- Feature selection (via importances)

❌ **Not ideal for:**
- When you need the highest possible accuracy (use ensembles)
- Small datasets (prone to overfitting)
- Problems with strong linear relationships
- When you need probability estimates (trees give crude probabilities)

### Hyperparameter Tuning Strategy

1. **Start with defaults**: Get a baseline
2. **Control overfitting**: Set `max_depth=5-10` or `min_samples_leaf=5-10`
3. **Use cross-validation**: Don't tune on test set
4. **Try both criteria**: `gini` is faster, `entropy` sometimes better
5. **Ensemble if needed**: Single trees rarely win; Random Forest/XGBoost usually better

### Common Pitfalls

!!! warning "Avoid These Mistakes"
    1. **No pruning**: Deep trees always overfit
    2. **Imbalanced data**: Use `class_weight='balanced'` or stratified sampling
    3. **Treating importances as causal**: Correlation ≠ causation
    4. **Ignoring tree structure**: Visualize to check if tree makes sense
    5. **Using default parameters**: Always tune hyperparameters

---

## Summary

In this lesson, you learned:

✅ **Core concepts**: Recursive partitioning, impurity measures, splitting criteria
✅ **Mathematics**: Gini impurity, entropy, information gain, variance reduction
✅ **Algorithm**: CART and the greedy splitting process
✅ **Implementation**: How to code a decision tree from scratch
✅ **Visualization**: Tree plots and decision boundaries
✅ **Overfitting**: Pre-pruning and post-pruning strategies
✅ **Feature importance**: How trees rank features
✅ **Practical skills**: When to use trees and how to tune them

### Key Takeaways

1. Decision trees partition the feature space using recursive binary splits
2. Splits are chosen greedily to maximize impurity reduction
3. Trees are interpretable but prone to overfitting
4. Pruning (depth limits, min samples) is essential
5. Single trees are rarely optimal—ensembles (Random Forest, XGBoost) usually perform better

### Next Steps

Now that you understand individual trees, you're ready to learn about **ensemble methods** that combine multiple trees for superior performance:

- **Random Forest**: Bagging multiple trees with feature randomness
- **Gradient Boosting**: Sequentially building trees to correct errors
- **XGBoost**: Optimized boosting with regularization

[Continue to Lesson 2: Tree Algorithms](02-tree-algorithms.md){ .md-button .md-button--primary }

---

## Additional Resources

### Further Reading

- **Books**:
  - *Introduction to Statistical Learning* (Chapter 8)
  - *Pattern Recognition and Machine Learning* by Bishop (Chapter 14)

- **Papers**:
  - Breiman et al. (1984): *Classification and Regression Trees*
  - Quinlan (1986): *Induction of Decision Trees*

### Practice Exercises

Ready to apply what you learned? Work through the hands-on exercises:

[Go to Exercises](exercises.md){ .md-button }

---

**Questions or feedback?** Open an issue on [GitHub](https://github.com/jumpingsphinx/jumpingsphinx.github.io/issues) or contribute improvements!
