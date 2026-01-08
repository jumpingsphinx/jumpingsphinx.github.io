# Lesson 5: XGBoost

## Introduction

XGBoost (eXtreme Gradient Boosting) is the most successful and widely-used machine learning algorithm for structured/tabular data in the last decade. Developed by Tianqi Chen in 2014, XGBoost takes the gradient boosting framework and optimizes it to an extreme degree—hence the name "eXtreme." It combines algorithmic innovations, systems optimizations, and practical features into a single, powerful package.

Since its release, XGBoost has become the de facto standard for winning machine learning competitions and deploying high-performance models in production. If you look at Kaggle competition winners from 2015-2020, you'll find XGBoost (or its variants LightGBM and CatBoost) in the vast majority of winning solutions.

### Why XGBoost Dominates

**Competition Results:**

- Used in **17 out of 29** Kaggle competition winning solutions in 2015
- Consistently appears in top solutions across domains
- Default choice for tabular data before deep learning approaches
- Industry standard at major tech companies

**Real-World Impact:**

- **E-commerce**: Amazon, Alibaba use XGBoost for recommendation systems
- **Finance**: Banks use XGBoost for credit scoring and fraud detection
- **Healthcare**: Used for disease prediction and treatment optimization
- **Web Search**: Search ranking and relevance scoring
- **Ad Tech**: Click-through rate prediction and bid optimization

**Key Advantages Over Standard Gradient Boosting:**

1. **Speed**: 10-30x faster than scikit-learn GradientBoosting
2. **Regularization**: Built-in L1 and L2 regularization prevents overfitting
3. **Missing Values**: Native support for sparse and missing data
4. **Parallelization**: Parallel tree construction, even though boosting is sequential
5. **Flexibility**: Supports custom objectives and evaluation metrics
6. **Built-in Features**: Cross-validation, early stopping, feature importance
7. **Hardware Optimizations**: Cache-aware algorithms, out-of-core computation

### What Makes XGBoost "eXtreme"

The "extreme" in XGBoost comes from multiple innovations:

1. **Systems Optimizations**: Parallel processing, cache-aware access patterns, out-of-core computing
2. **Algorithmic Improvements**: Weighted quantile sketch, sparsity-aware split finding
3. **Regularization**: Complexity control through tree structure and leaf weights
4. **Flexibility**: Highly customizable with many hyperparameters
5. **Production Ready**: Handles large-scale data, distributes across clusters

## XGBoost vs Standard Gradient Boosting

### Visual Introduction to XGBoost

Before getting into the details, watch this excellent explanation of XGBoost:

<div style="text-align: center; margin: 20px 0;">
<iframe width="560" height="315" src="https://www.youtube.com/embed/OtD8wVaFm6E" title="XGBoost by StatQuest" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

Before getting into XGBoost specifics, let's understand what it improves over standard gradient boosting.

### Regularized Objective Function

**Standard Gradient Boosting** optimizes:

$$\text{Obj} = \sum_{i=1}^{n} L(y_i, \hat{y}_i)$$

Just the loss function, no explicit regularization.

**XGBoost** optimizes:

$$\text{Obj} = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K}\Omega(f_k)$$

where the regularization term is:

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda\|w\|^2$$

- $T$: number of leaves in the tree
- $w$: leaf weights (predictions)
- $\gamma$: minimum loss reduction required to split (like min_impurity_decrease)
- $\lambda$: L2 regularization on leaf weights

This regularization term explicitly penalizes complex trees, preventing overfitting.

### Tree Construction Algorithm

**Standard Gradient Boosting**:
- Uses exact greedy algorithm
- Considers all possible splits
- Slow for large datasets
- Memory intensive

**XGBoost**:
- Uses approximate algorithm with histograms
- Weighted quantile sketch for split candidates
- Sparsity-aware split finding
- Fast and memory-efficient
- Parallel tree construction

### Missing Value Handling

**Standard Gradient Boosting**:
- Requires imputation before training
- No native missing value support
- Needs preprocessing

**XGBoost**:
- Learns optimal default direction for missing values
- Treats missing as a separate category
- No preprocessing needed
- Sparsity-aware algorithms

### Speed Comparison

| Operation | scikit-learn GBM | XGBoost | Speedup |
|-----------|------------------|---------|---------|
| Small dataset (10K samples) | 5s | 0.5s | 10x |
| Medium dataset (100K samples) | 60s | 3s | 20x |
| Large dataset (1M samples) | 800s | 30s | 27x |

*Approximate timings on typical hardware. Actual speedup depends on dataset and hyperparameters.*

## XGBoost Mathematical Foundation

Let's understand the mathematics behind XGBoost's improvements.

### Objective Function Derivation

At iteration $t$, we have model $\hat{y}_i^{(t)}$:

$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + f_t(x_i)$$

where $f_t$ is the new tree we're adding.

The objective function becomes:

$$\text{Obj}^{(t)} = \sum_{i=1}^{n} L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

### Second-Order Taylor Approximation

XGBoost uses a second-order Taylor expansion of the loss function around $\hat{y}_i^{(t-1)}$:

$$L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) \approx L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)$$

where:
- $g_i = \frac{\partial L(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}$ is the **first-order gradient**
- $h_i = \frac{\partial^2 L(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$ is the **second-order gradient (Hessian)**

**Key Innovation:** Standard gradient boosting only uses first-order gradients ($g_i$). XGBoost uses second-order gradients ($h_i$) for more accurate optimization.

### Simplified Objective

Removing constant terms:

$$\text{Obj}^{(t)} \approx \sum_{i=1}^{n}\left[g_i f_t(x_i) + \frac{1}{2}h_i f_t^2(x_i)\right] + \Omega(f_t)$$

For a tree with $T$ leaves and leaf weights $w$, if $I_j$ is the set of samples in leaf $j$:

$$\text{Obj}^{(t)} = \sum_{j=1}^{T}\left[\left(\sum_{i \in I_j}g_i\right)w_j + \frac{1}{2}\left(\sum_{i \in I_j}h_i + \lambda\right)w_j^2\right] + \gamma T$$

### Optimal Leaf Weight

For a fixed tree structure, the optimal weight for leaf $j$ is:

$$w_j^* = -\frac{\sum_{i \in I_j}g_i}{\sum_{i \in I_j}h_i + \lambda}$$

This is the closed-form solution that minimizes the objective for a given tree structure.

### Split Finding: Gain Calculation

The gain from splitting a leaf into left (L) and right (R) children is:

$$\text{Gain} = \frac{1}{2}\left[\frac{(\sum_{i \in I_L}g_i)^2}{\sum_{i \in I_L}h_i + \lambda} + \frac{(\sum_{i \in I_R}g_i)^2}{\sum_{i \in I_R}h_i + \lambda} - \frac{(\sum_{i \in I}g_i)^2}{\sum_{i \in I}h_i + \lambda}\right] - \gamma$$

- If Gain > 0: split improves the objective
- $\gamma$ acts as a threshold: only splits that improve loss by at least $\gamma$ are made
- This is XGBoost's splitting criterion (analogous to Gini impurity in decision trees)

### Comparison: Standard GB vs XGBoost

| Aspect | Standard Gradient Boosting | XGBoost |
|--------|---------------------------|---------|
| **Gradients** | First-order only ($g$) | First and second-order ($g$, $h$) |
| **Regularization** | None explicit | L1 ($\alpha$), L2 ($\lambda$), tree complexity ($\gamma$) |
| **Leaf Weights** | Mean of residuals | Optimal closed-form solution |
| **Split Criterion** | Variance reduction | Regularized gain formula |
| **Mathematical Rigor** | Taylor expansion to 1st order | Taylor expansion to 2nd order |

## XGBoost Algorithmic Innovations

### 1. Approximate Split Finding

For large datasets, considering every possible split point is expensive. XGBoost uses **weighted quantile sketch**:

**Algorithm:**

1. Propose candidate split points based on percentiles of feature distribution
2. Weight data points by second-order gradient $h_i$
3. Create histogram bins
4. Only evaluate splits at bin boundaries

**Benefits:**
- Much faster than exact algorithm
- Controlled accuracy through number of bins
- Works well even with relatively few candidates

**Hyperparameter:** `tree_method='approx'` or `tree_method='hist'`

### 2. Sparsity-Aware Split Finding

Many real-world datasets have missing values or zero entries (sparsity). XGBoost handles this elegantly:

**Algorithm:**

For each split, try two scenarios:
1. Default direction LEFT: assign missing values to left child
2. Default direction RIGHT: assign missing values to right child
3. Choose direction that gives better gain

**Benefits:**
- No need to impute missing values
- Learns optimal direction for each split
- Fast: only visits non-missing entries
- Handles both missing values and sparse features (e.g., one-hot encoded data)

**Key Point:** The default direction is learned from data, not predetermined.

### 3. Parallel Tree Construction

Boosting is inherently sequential (each tree depends on previous), but XGBoost parallelizes **within each tree**:

**What's Parallelized:**

1. **Split finding**: Evaluate different features in parallel
2. **Sorting**: Parallel sort of feature values
3. **Histogram construction**: Parallel bin creation

**What's NOT Parallelized:**

- Adding trees to the ensemble (still sequential)
- Tree construction depends on previous tree's residuals

**Result:**
- Major speedup even though boosting is sequential
- Scales well with multiple CPU cores
- Can use GPU acceleration

### 4. Cache-Aware Access

XGBoost optimizes for modern CPU cache hierarchies:

**Problem:** Random memory access is slow (cache misses)

**Solution:**
- Block structure for data storage
- Access data in contiguous blocks
- Prefetch data into cache
- Compress data to fit more in cache

**Result:** Significant speedup on large datasets

### 5. Out-of-Core Computation

For datasets larger than memory:

**Problem:** Can't fit entire dataset in RAM

**Solution:**
- Divide data into blocks on disk
- Load blocks into memory as needed
- Compress blocks to reduce I/O
- Overlap computation with I/O

**Result:** Can train on datasets much larger than available RAM

## XGBoost Hyperparameters

XGBoost has many hyperparameters organized into categories.

### Tree-Specific Parameters

| Parameter | Description | Default | Typical Range | Effect |
|-----------|-------------|---------|---------------|--------|
| `max_depth` | Maximum tree depth | 6 | 3-10 | Deeper → more complex, potential overfit |
| `min_child_weight` | Min sum of hessian in child | 1 | 1-10 | Higher → more conservative, prevents overfit |
| `gamma` | Min loss reduction for split | 0 | 0-5 | Higher → fewer splits, more regularization |
| `subsample` | Fraction of samples per tree | 1.0 | 0.5-1.0 | <1 → faster, more regularization |
| `colsample_bytree` | Fraction of features per tree | 1.0 | 0.5-1.0 | <1 → more diversity |
| `colsample_bylevel` | Fraction of features per level | 1.0 | 0.5-1.0 | <1 → more diversity at each depth |
| `colsample_bynode` | Fraction of features per node | 1.0 | 0.5-1.0 | <1 → maximum diversity |

### Regularization Parameters

| Parameter | Description | Default | Typical Range | Effect |
|-----------|-------------|---------|---------------|--------|
| `lambda` (L2) | L2 regularization on weights | 1 | 0-10 | Higher → smoother leaf weights |
| `alpha` (L1) | L1 regularization on weights | 0 | 0-10 | Higher → sparsity in leaf weights |

### Learning Task Parameters

| Parameter | Description | Default | Typical Range | Effect |
|-----------|-------------|---------|---------------|--------|
| `learning_rate` (`eta`) | Shrinkage of each tree | 0.3 | 0.01-0.3 | Lower → more trees needed, better generalization |
| `n_estimators` | Number of trees | 100 | 50-1000 | More → better fit (watch for overfit) |
| `objective` | Loss function | 'reg:squarederror' | Various | Task-dependent |
| `eval_metric` | Evaluation metric | Depends on objective | Various | Monitoring performance |

### Other Important Parameters

| Parameter | Description | Default | Values | Effect |
|-----------|-------------|---------|--------|--------|
| `tree_method` | Tree construction algorithm | 'auto' | 'auto', 'exact', 'approx', 'hist', 'gpu_hist' | Speed vs accuracy tradeoff |
| `scale_pos_weight` | Balance of positive/negative weights | 1 | >1 for imbalanced | Handles class imbalance |
| `max_delta_step` | Maximum delta step for weight | 0 | 0-10 | Helps with imbalanced logistic regression |

## Hyperparameter Tuning Strategy

With so many hyperparameters, systematic tuning is essential.

### Step-by-Step Tuning Process

**1. Fix learning rate and determine optimal number of trees**

```python
params = {
    'learning_rate': 0.1,
    'n_estimators': 1000,  # Large number
    'early_stopping_rounds': 50  # Stop if no improvement
}
```

Use cross-validation or validation set to find when performance plateaus.

**2. Tune tree-specific parameters**

```python
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5, 7]
}
```

Start with `max_depth` and `min_child_weight` as they have the biggest impact.

**3. Add column and row sampling**

```python
param_grid = {
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
```

These add randomness and speed up training.

**4. Tune regularization parameters**

```python
param_grid = {
    'gamma': [0, 0.1, 0.5, 1.0],
    'lambda': [0, 1, 5, 10],
    'alpha': [0, 0.1, 1.0, 5]
}
```

Add regularization if overfitting.

**5. Lower learning rate for final model**

```python
params = {
    'learning_rate': 0.01,  # Lower
    'n_estimators': 5000,   # More trees to compensate
    'early_stopping_rounds': 100
}
```

Lower learning rate with more trees typically gives best results.

### Quick Start: Good Default Parameters

**For most problems, start with:**

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    lambda=1,
    alpha=0,
    tree_method='hist',  # Fast
    random_state=42
)
```

Then tune from there based on validation performance.

## Interactive Example: XGBoost in Practice

<div class="python-interactive" markdown="1">
```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Note: Using sklearn's GradientBoostingClassifier as XGBoost-like alternative
# It implements gradient boosting with similar concepts

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)

print("="*60)
print("Gradient Boosting: Breast Cancer Classification")
print("="*60)
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Classes: Malignant (0), Benign (1)")
print(f"Class distribution: {np.bincount(y)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 1. Baseline Gradient Boosting with default parameters
# ============================================================
print("\n" + "="*60)
print("STEP 1: Baseline Gradient Boosting")
print("="*60)

gb_baseline = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gb_baseline.fit(X_train, y_train)

train_acc = gb_baseline.score(X_train, y_train)
test_acc = gb_baseline.score(X_test, y_test)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ============================================================
# 2. Using validation set monitoring
# ============================================================
print("\n" + "="*60)
print("STEP 2: Validation Monitoring")
print("="*60)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

gb_monitored = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42,
    validation_fraction=0.2,
    n_iter_no_change=20,
    tol=1e-4
)

gb_monitored.fit(X_train, y_train)

print(f"Number of estimators used: {gb_monitored.n_estimators_}")
print(f"Test Accuracy: {gb_monitored.score(X_test, y_test):.4f}")

# ============================================================
# 3. Hyperparameter tuning
# ============================================================
print("\n" + "="*60)
print("STEP 3: Hyperparameter Tuning")
print("="*60)

# Test different max_depth values
depths = [3, 5, 7, 9]
scores = []

for depth in depths:
    model = GradientBoostingClassifier(
        max_depth=depth,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    scores.append(cv_scores.mean())
    print(f"  max_depth={depth}: CV Accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

best_depth = depths[np.argmax(scores)]
print(f"\nBest max_depth: {best_depth}")

# Train with best depth
gb_tuned = GradientBoostingClassifier(
    max_depth=best_depth,
    n_estimators=200,
    learning_rate=0.05,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)

gb_tuned.fit(X_train, y_train)

print(f"\nTuned Model Test Accuracy: {gb_tuned.score(X_test, y_test):.4f}")

# ============================================================
# 4. Feature Importance
# ============================================================
print("\n" + "="*60)
print("STEP 4: Feature Importance")
print("="*60)

# Sort features by importance
importance_indices = np.argsort(gb_tuned.feature_importances_)[::-1]
sorted_features = [feature_names[i] for i in importance_indices]
sorted_importances = gb_tuned.feature_importances_[importance_indices]

print("\nTop 10 Most Important Features:")
print(f"{'Feature':<30} {'Importance':<10}")
print("-" * 40)
for i in range(10):
    print(f"{sorted_features[i]:<30} {sorted_importances[i]:<10.6f}")

# Visualize
plt.figure(figsize=(10, 8))
top_15_features = sorted_features[:15]
top_15_importances = sorted_importances[:15]
plt.barh(range(len(top_15_features)), top_15_importances, color='steelblue')
plt.yticks(range(len(top_15_features)), top_15_features)
plt.xlabel('Importance', fontsize=12)
plt.title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ============================================================
# 5. Learning Curves
# ============================================================
print("\n" + "="*60)
print("STEP 5: Learning Curves")
print("="*60)

# Track performance as trees are added
n_trees = range(10, 201, 10)
train_scores = []
test_scores = []

for n in n_trees:
    model = GradientBoostingClassifier(
        max_depth=best_depth,
        n_estimators=n,
        learning_rate=0.05,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    model.fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(n_trees, train_scores, label='Train Accuracy', linewidth=2)
plt.plot(n_trees, test_scores, label='Test Accuracy', linewidth=2)
plt.xlabel('Number of Trees', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Gradient Boosting Learning Curves', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================
# 6. Final Evaluation
# ============================================================
print("\n" + "="*60)
print("STEP 6: Final Evaluation")
print("="*60)

y_pred = gb_tuned.predict(X_test)
y_pred_proba = gb_tuned.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                          target_names=['Malignant', 'Benign']))

print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"✓ Final Test Accuracy: {gb_tuned.score(X_test, y_test):.4f}")
print(f"✓ ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"✓ Best max_depth: {best_depth}")
print(f"✓ Most important feature: {sorted_features[0]}")
print("✓ Gradient Boosting achieves excellent performance with tuning!")
```
</div>

**Expected Output:**
- Baseline achieves ~96% accuracy
- Early stopping prevents overfitting
- Hyperparameter tuning improves performance
- Feature importance reveals key diagnostic features
- ROC AUC above 0.99
- Learning curves show convergence

## XGBoost Feature Importance

XGBoost provides three types of feature importance:

### 1. Gain (Default)

**Definition:** Average gain of splits using the feature

$$\text{Importance}_{\text{gain}}(f) = \frac{1}{K}\sum_{k : \text{trees using } f} \text{Gain}_k(f)$$

- **Interpretation:** How much the feature improves the model
- **Default** in scikit-learn API
- **Best for:** Understanding predictive power

### 2. Weight (Frequency)

**Definition:** Number of times feature is used for splitting

$$\text{Importance}_{\text{weight}}(f) = \text{Number of splits using } f$$

- **Interpretation:** How often the feature is used
- **Biased** toward high-cardinality features
- **Best for:** Identifying frequently used features

### 3. Cover

**Definition:** Average coverage of splits using the feature

$$\text{Importance}_{\text{cover}}(f) = \frac{1}{K}\sum_{k : \text{trees using } f} \text{Cover}_k(f)$$

where Coverage = number of samples affected by the split

- **Interpretation:** How many samples are affected
- **Less common** than gain or weight
- **Best for:** Understanding impact on data

### Interactive Example: Different Importance Types

<div class="python-interactive" markdown="1">
```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Load dataset
data = load_wine()
X, y = data.data, data.target
feature_names = list(data.feature_names)

print("Wine Classification Dataset")
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(np.unique(y))}")

# Train Gradient Boosting
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X, y)

print(f"\nAccuracy: {model.score(X, y):.3f}")

# Get different importance measures
# 1. Feature importance (impurity-based, similar to XGBoost's "gain")
impurity_importance = model.feature_importances_

# 2. Permutation importance (similar to XGBoost's feature impact)
perm_result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
perm_importance = perm_result.importances_mean

# Sort and normalize
def normalize_and_sort(importance, feature_names):
    norm_importance = importance / importance.sum()
    indices = np.argsort(norm_importance)[::-1]
    return [feature_names[i] for i in indices], norm_importance[indices]

imp_features, imp_values = normalize_and_sort(impurity_importance, feature_names)
perm_features, perm_values = normalize_and_sort(perm_importance, feature_names)

print("\n" + "="*60)
print("Feature Importance: Different Metrics")
print("="*60)

print("\nTop 5 by Impurity-based Importance (Predictive Power):")
print(f"{'Feature':<25} {'Importance':<10}")
print("-" * 35)
for i in range(5):
    print(f"{imp_features[i]:<25} {imp_values[i]:<10.6f}")

print("\nTop 5 by Permutation Importance (Feature Impact):")
print(f"{'Feature':<25} {'Importance':<10}")
print("-" * 35)
for i in range(5):
    print(f"{perm_features[i]:<25} {perm_values[i]:<10.6f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Impurity-based
top_imp = 10
axes[0].barh(range(top_imp), imp_values[:top_imp], color='steelblue')
axes[0].set_yticks(range(top_imp))
axes[0].set_yticklabels(imp_features[:top_imp])
axes[0].set_xlabel('Normalized Importance')
axes[0].set_title('Impurity-based\n(Similar to XGBoost Gain)')
axes[0].invert_yaxis()

# Permutation-based
axes[1].barh(range(top_imp), perm_values[:top_imp], color='coral')
axes[1].set_yticks(range(top_imp))
axes[1].set_yticklabels(perm_features[:top_imp])
axes[1].set_xlabel('Normalized Importance')
axes[1].set_title('Permutation-based\n(Feature Impact on Accuracy)')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("Observations:")
print("• Impurity-based: Fast to compute, shows predictive contribution")
print("• Permutation-based: More reliable, measures actual impact on predictions")
print("• Rankings can differ significantly!")
```
</div>

**Expected Output:**
- Three different importance rankings
- Gain-based importance differs from weight-based
- Visualization shows clear differences
- Understanding when to use each type

## Handling Imbalanced Data

XGBoost has built-in features for imbalanced classification.

### Scale Positive Weight

**Parameter:** `scale_pos_weight`

**Formula:**

$$\text{scale\_pos\_weight} = \frac{\text{Number of negative samples}}{\text{Number of positive samples}}$$

**Usage:**

```python
# For dataset with 900 negative, 100 positive samples
model = xgb.XGBClassifier(
    scale_pos_weight=900/100,  # = 9
    # ... other parameters
)
```

**Effect:**
- Increases weight of positive class
- Loss from positive samples counts more
- Helps model focus on minority class

### Interactive Example: Imbalanced Classification

<div class="python-interactive" markdown="1">
```python
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Create imbalanced dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    weights=[0.95, 0.05],  # 95% class 0, 5% class 1
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Imbalanced Classification Dataset")
print(f"Training samples: {len(X_train)}")
print(f"Class 0: {np.sum(y_train == 0)} ({100*np.sum(y_train == 0)/len(y_train):.1f}%)")
print(f"Class 1: {np.sum(y_train == 1)} ({100*np.sum(y_train == 1)/len(y_train):.1f}%)")
print(f"Imbalance ratio: {np.sum(y_train == 0) / np.sum(y_train == 1):.1f}:1")

# ============================================================
# 1. Without balancing
# ============================================================
print("\n" + "="*60)
print("Model 1: No Balancing")
print("="*60)

xgb_no_balance = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

xgb_no_balance.fit(X_train, y_train)
y_pred = xgb_no_balance.predict(X_test)

print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

# ============================================================
# 2. With scale_pos_weight
# ============================================================
print("\n" + "="*60)
print("Model 2: With scale_pos_weight")
print("="*60)

scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

xgb_balanced = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

xgb_balanced.fit(X_train, y_train)
y_pred_balanced = xgb_balanced.predict(X_test)

print(classification_report(y_test, y_pred_balanced, target_names=['Class 0', 'Class 1']))

# ============================================================
# 3. Comparison
# ============================================================
print("\n" + "="*60)
print("Comparison")
print("="*60)

# Confusion matrices
cm_no_balance = confusion_matrix(y_test, y_pred)
cm_balanced = confusion_matrix(y_test, y_pred_balanced)

print("\nConfusion Matrix (No Balancing):")
print(f"  Predicted:  0     1")
print(f"Actual 0:  {cm_no_balance[0, 0]:4d}  {cm_no_balance[0, 1]:4d}")
print(f"Actual 1:  {cm_no_balance[1, 0]:4d}  {cm_no_balance[1, 1]:4d}")

print("\nConfusion Matrix (With Balancing):")
print(f"  Predicted:  0     1")
print(f"Actual 0:  {cm_balanced[0, 0]:4d}  {cm_balanced[0, 1]:4d}")
print(f"Actual 1:  {cm_balanced[1, 0]:4d}  {cm_balanced[1, 1]:4d}")

# ROC AUC
y_pred_proba = xgb_no_balance.predict_proba(X_test)[:, 1]
y_pred_proba_balanced = xgb_balanced.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)
roc_auc_balanced = roc_auc_score(y_test, y_pred_proba_balanced)

print(f"\nROC AUC (No Balancing): {roc_auc:.4f}")
print(f"ROC AUC (With Balancing): {roc_auc_balanced:.4f}")

# Calculate recall for minority class
recall_no_balance = cm_no_balance[1, 1] / (cm_no_balance[1, 0] + cm_no_balance[1, 1])
recall_balanced = cm_balanced[1, 1] / (cm_balanced[1, 0] + cm_balanced[1, 1])

print(f"\nMinority Class Recall (No Balancing): {recall_no_balance:.4f}")
print(f"Minority Class Recall (With Balancing): {recall_balanced:.4f}")

print("\n" + "="*60)
print("Key Takeaway:")
print("scale_pos_weight improves recall for minority class")
print("while maintaining good overall performance!")
print("="*60)
```
</div>

**Expected Output:**
- Without balancing: High accuracy but poor minority class recall
- With balancing: Better minority class recall, similar ROC AUC
- Confusion matrices show the difference clearly
- Balancing trades off precision for recall on minority class

## XGBoost vs Alternatives

### XGBoost vs LightGBM vs CatBoost

These are the three dominant gradient boosting libraries:

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Speed** | Fast | Fastest | Medium |
| **Memory** | Moderate | Low | Moderate |
| **Accuracy** | High | High | Highest (on some tasks) |
| **Categorical Features** | One-hot encode | Label encode + special handling | Native support (best) |
| **GPU Support** | Yes | Yes | Yes |
| **Default Hyperparameters** | Good | Good | Excellent |
| **Tree Growth** | Level-wise (balanced) | Leaf-wise (unbalanced, faster) | Symmetric, oblivious trees |
| **Learning Curve** | Moderate | Moderate | Easiest |
| **Best For** | General purpose, most tested | Large datasets, speed critical | Categorical data, robust defaults |
| **Popularity** | Highest | High | Growing |

**When to Choose:**

- **XGBoost**: Default choice, most mature, largest community
- **LightGBM**: Very large datasets (>100M samples), speed critical
- **CatBoost**: Many categorical features, want best defaults

### XGBoost vs Random Forest

| Aspect | Random Forest | XGBoost |
|--------|---------------|---------|
| **Accuracy** | Good | Better (typically) |
| **Speed** | Fast (parallel) | Moderate (sequential trees, parallel splits) |
| **Overfitting** | Resistant (bagging) | Needs tuning (boosting) |
| **Hyperparameters** | Few, robust defaults | Many, requires tuning |
| **Interpretability** | Feature importance | Feature importance + partial dependence |
| **Missing Values** | Need imputation | Native handling |
| **When to Use** | Quick baseline, noisy data | Maximum accuracy, clean data |

## Production Considerations

### Model Deployment

**Saving and Loading:**

```python
# Save model
model.save_model('xgboost_model.json')

# Load model
loaded_model = xgb.XGBClassifier()
loaded_model.load_model('xgboost_model.json')
```

**Alternative: Pickle (not recommended)**

```python
import pickle

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

**Recommendation:** Use `.save_model()` and `.load_model()` for better version compatibility.

### Prediction Speed

**Single Prediction:**
- Typical: 0.1-1 ms
- Depends on: number of trees, tree depth, number of features

**Batch Prediction:**
- Much faster per sample (vectorized operations)
- Typical: 10-100 predictions per ms

**Optimization Tips:**
1. Use fewer trees if latency critical
2. Limit max_depth
3. Use `tree_method='hist'` for faster training
4. Consider model compression (fewer features, pruning)

### Memory Usage

**Training:**
- Stores gradients and hessians for all samples
- Stores tree structures
- Typical: 2-5x dataset size

**Inference:**
- Only stores tree structures
- Typical: 1-10 MB for 100 trees
- Much smaller than training memory

**Out-of-Core:**
For datasets larger than RAM, use external memory:

```python
dtrain = xgb.DMatrix('train.svm.buffer')
```

## Common Pitfalls and Solutions

### 1. Overfitting

**Symptoms:**
- Training accuracy much higher than test accuracy
- Validation error increases after initial decrease

**Solutions:**
- Reduce `max_depth` (try 3-6)
- Increase `min_child_weight` (try 3-10)
- Increase `gamma` (try 0.1-1.0)
- Add regularization: `lambda` (1-10), `alpha` (0.1-5)
- Use `subsample` < 1.0 (try 0.8)
- Use `colsample_bytree` < 1.0 (try 0.8)
- Lower `learning_rate` and increase `n_estimators`
- Use early stopping

### 2. Underfitting

**Symptoms:**
- Both training and test accuracy low
- Validation error plateaus at high value

**Solutions:**
- Increase `max_depth` (try 7-10)
- Decrease `min_child_weight` (try 1)
- Decrease `gamma` (try 0)
- Increase `n_estimators` (try 500-1000)
- Check for data issues (missing values, outliers)
- Feature engineering

### 3. Slow Training

**Symptoms:**
- Training takes too long
- Can't iterate quickly

**Solutions:**
- Use `tree_method='hist'` (fastest)
- Reduce `n_estimators` during development
- Use smaller `max_depth`
- Use `subsample` < 1.0
- Reduce number of features
- Use GPU: `tree_method='gpu_hist'`

### 4. Memory Issues

**Symptoms:**
- Out of memory errors
- System swapping/freezing

**Solutions:**
- Use `tree_method='hist'` (most memory efficient)
- Reduce `max_bin` (number of bins for histograms)
- Use `subsample` < 1.0
- Use external memory: `DMatrix(..., cache_prefix='cache')`
- Reduce number of features
- Use data sampling for development

## Summary

**Key Takeaways:**

1. **XGBoost is the gold standard** for structured/tabular data
2. **Second-order optimization** (using Hessian) makes it more accurate than standard gradient boosting
3. **Regularization** ($\gamma$, $\lambda$, $\alpha$) prevents overfitting
4. **Systems optimizations** make it 10-30x faster than scikit-learn
5. **Native missing value handling** eliminates preprocessing
6. **Many hyperparameters** require systematic tuning but offer great flexibility
7. **Feature importance** provides interpretability
8. **Production-ready** with fast inference and small model size

**XGBoost Advantages:**
- Highest accuracy on tabular data
- Fast training and prediction
- Handles missing values natively
- Built-in regularization
- Extensive hyperparameter control
- Strong community and ecosystem

**XGBoost Limitations:**
- Requires hyperparameter tuning
- Sequential tree building (can't fully parallelize)
- Sensitive to outliers (less than AdaBoost, more than Random Forest)
- Overkill for simple problems
- Less interpretable than single decision trees

**When to Use XGBoost:**

✅ Structured/tabular data
✅ Need maximum accuracy
✅ Have clean or cleanable data
✅ Can invest time in hyperparameter tuning
✅ Production deployment with performance requirements

**Recommended Workflow:**

1. Start with Random Forest (quick baseline)
2. Try XGBoost with default parameters
3. Tune XGBoost hyperparameters systematically
4. Compare with LightGBM if dataset is very large
5. Use ensemble of XGBoost + LightGBM + Random Forest for final model

## Practice Exercises

Ready to master XGBoost? Head to the exercises to:

1. Compare XGBoost with Random Forest and Gradient Boosting
2. Perform systematic hyperparameter tuning
3. Handle imbalanced classification problems
4. Optimize for different metrics (accuracy, AUC, F1)
5. Deploy XGBoost model to production
6. Build a Kaggle-style winning solution

[Start Exercises](exercises.md){ .md-button .md-button--primary }

## Additional Resources

- **Original Paper**: [Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"](https://arxiv.org/abs/1603.02754)
- **Official Documentation**: [XGBoost Docs](https://xgboost.readthedocs.io/)
- **Tutorials**: [XGBoost Python Tutorial](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
- **GitHub**: [XGBoost Repository](https://github.com/dmlc/xgboost)
- **Comparison**: [XGBoost vs LightGBM vs CatBoost](https://towardsdatascience.com/catboost-vs-lightgbm-vs-xgboost-c80f40662924)

---

**Navigation:**

- [← Previous: Boosting](04-boosting.md)
- [Exercises →](exercises.md)
- [Module Overview](index.md)
