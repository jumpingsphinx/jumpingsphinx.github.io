# Tree Algorithms: ID3, C4.5, and CART

## Introduction

In the previous lesson, we learned the fundamental concepts of decision trees. Now we'll dive deeper into the specific algorithms that have shaped the field: **ID3** (Iterative Dichotomiser 3), **C4.5** (its successor), and **CART** (Classification and Regression Trees). Each algorithm has unique characteristics, strengths, and historical significance.

Understanding these algorithms will give you insight into:
- How different splitting criteria affect tree structure
- Techniques for handling continuous and categorical data
- Methods for pruning overgrown trees
- The evolution of tree-based learning

## Historical Context

Decision tree algorithms have evolved significantly since their inception:

**Timeline:**
- **1960s-1970s**: Early work by Hunt, Marin, and Stone
- **1986**: Ross Quinlan introduces **ID3**
- **1993**: Quinlan releases **C4.5**, improving upon ID3
- **1984**: Breiman, Friedman, Olshen, and Stone publish **CART**
- **2000s**: Modern implementations in scikit-learn and other libraries

!!! info "Why Study Classic Algorithms?"
    Even though modern libraries implement optimized versions, understanding ID3, C4.5, and CART teaches you:

    - **The evolution of ideas**: How each algorithm improved upon its predecessor
    - **Design trade-offs**: Why certain choices were made
    - **Problem-solving approaches**: Different ways to handle the same challenges
    - **Implementation details**: What happens under the hood in sklearn

---

## ID3: Iterative Dichotomiser 3

### Overview

**ID3** was one of the first practical decision tree algorithms, introduced by Ross Quinlan in 1986. It uses **information gain** based on **entropy** to select the best attribute at each node.

### Key Characteristics

- **Splitting criterion**: Information gain (entropy-based)
- **Feature types**: Categorical features only
- **Tree structure**: Multi-way splits (not just binary)
- **Pruning**: No pruning mechanism
- **Missing values**: Not handled natively

### The Algorithm

#### Entropy and Information Gain Review

Recall from Lesson 1 that **entropy** measures uncertainty:

$$
H(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
$$

Where:
- $S$ is a set of samples
- $c$ is the number of classes
- $p_i$ is the proportion of samples in class $i$

**Information Gain** measures the reduction in entropy from splitting on attribute $A$:

$$
\text{IG}(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)
$$

Where $S_v$ is the subset of $S$ where attribute $A$ has value $v$.

#### ID3 Pseudocode

```python
def ID3(examples, attributes, target_attribute):
    """
    Build a decision tree using ID3 algorithm.

    Args:
        examples: Training data
        attributes: List of attributes to consider
        target_attribute: The attribute to predict

    Returns:
        Decision tree (node)
    """
    # Create a root node
    root = Node()

    # If all examples are positive, return single-node tree Root, with label = +
    if all examples are positive:
        root.label = '+'
        return root

    # If all examples are negative, return single-node tree Root, with label = -
    if all examples are negative:
        root.label = '-'
        return root

    # If attributes is empty, return single-node tree Root,
    # with label = most common value of target_attribute in examples
    if attributes is empty:
        root.label = most_common_value(examples, target_attribute)
        return root

    # Otherwise, begin
    # A ← attribute from attributes that best classifies examples
    A = attribute_with_max_information_gain(examples, attributes)
    root.attribute = A

    # For each possible value v_i of A:
    for each value v_i of A:
        # Add a new branch below Root, corresponding to A = v_i
        examples_v_i = subset of examples where A = v_i

        if examples_v_i is empty:
            # Add a leaf node with label = most common value of target_attribute
            child = Node()
            child.label = most_common_value(examples, target_attribute)
        else:
            # Add subtree ID3(examples_v_i, attributes - {A}, target_attribute)
            child = ID3(examples_v_i, attributes - {A}, target_attribute)

        root.add_child(v_i, child)

    return root
```

### Example: Playing Tennis

Let's work through the classic ID3 example of deciding whether to play tennis based on weather conditions.

**Dataset:**

| Outlook  | Temperature | Humidity | Windy | Play Tennis |
|----------|-------------|----------|-------|-------------|
| Sunny    | Hot         | High     | False | No          |
| Sunny    | Hot         | High     | True  | No          |
| Overcast | Hot         | High     | False | Yes         |
| Rainy    | Mild        | High     | False | Yes         |
| Rainy    | Cool        | Normal   | False | Yes         |
| Rainy    | Cool        | Normal   | True  | No          |
| Overcast | Cool        | Normal   | True  | Yes         |
| Sunny    | Mild        | High     | False | No          |
| Sunny    | Cool        | Normal   | False | Yes         |
| Rainy    | Mild        | Normal   | False | Yes         |
| Sunny    | Mild        | Normal   | True  | Yes         |
| Overcast | Mild        | High     | True  | Yes         |
| Overcast | Hot         | Normal   | False | Yes         |
| Rainy    | Mild        | High     | True  | No          |

**Step 1: Calculate entropy of the dataset**

Total: 14 examples (9 Yes, 5 No)

$$
H(S) = -\frac{9}{14}\log_2\frac{9}{14} - \frac{5}{14}\log_2\frac{5}{14} \approx 0.940
$$

**Step 2: Calculate information gain for each attribute**

For **Outlook**:
- Sunny: 5 examples (2 Yes, 3 No) → $H = -\frac{2}{5}\log_2\frac{2}{5} - \frac{3}{5}\log_2\frac{3}{5} \approx 0.971$
- Overcast: 4 examples (4 Yes, 0 No) → $H = 0$ (pure)
- Rainy: 5 examples (3 Yes, 2 No) → $H = -\frac{3}{5}\log_2\frac{3}{5} - \frac{2}{5}\log_2\frac{2}{5} \approx 0.971$

$$
\text{IG}(\text{Outlook}) = 0.940 - \left(\frac{5}{14} \cdot 0.971 + \frac{4}{14} \cdot 0 + \frac{5}{14} \cdot 0.971\right) \approx 0.246
$$

For **Temperature** (Hot, Mild, Cool):

$$
\text{IG}(\text{Temperature}) \approx 0.029
$$

For **Humidity** (High, Normal):

$$
\text{IG}(\text{Humidity}) \approx 0.151
$$

For **Windy** (True, False):

$$
\text{IG}(\text{Windy}) \approx 0.048
$$

**Step 3: Choose attribute with highest information gain**

**Outlook** has the highest information gain (0.246), so it becomes the root node.

**Step 4: Recursively build subtrees**

The tree continues to grow by selecting the best attribute at each node until we reach pure nodes or run out of attributes.

### Interactive Code: ID3 Implementation

<div class="python-interactive" markdown="1">
```python
import numpy as np
from collections import Counter

class ID3Node:
    """Node in an ID3 decision tree."""
    def __init__(self):
        self.attribute = None  # Attribute to split on
        self.children = {}     # Dictionary: attribute_value -> child_node
        self.label = None      # Class label for leaf nodes
        self.is_leaf = False

def entropy(labels):
    """Calculate entropy of a set of labels."""
    if len(labels) == 0:
        return 0

    counts = Counter(labels)
    probabilities = np.array(list(counts.values())) / len(labels)

    # Entropy = -sum(p * log2(p))
    entropy_val = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy_val

def get_column(data, col_name):
    """Extract a column from data dictionary."""
    return [row[col_name] for row in data]

def filter_data(data, attribute, value):
    """Filter data where attribute equals value."""
    return [row for row in data if row[attribute] == value]

def information_gain(data, attribute, target):
    """
    Calculate information gain from splitting on an attribute.

    Args:
        data: List of dictionaries (each dictionary is one example)
        attribute: Attribute name to split on
        target: Target attribute name

    Returns:
        Information gain value
    """
    # Total entropy
    target_values = get_column(data, target)
    total_entropy = entropy(target_values)

    # Weighted average entropy after split
    attr_values = get_column(data, attribute)
    unique_values = list(set(attr_values))
    weighted_entropy = 0

    for value in unique_values:
        subset = filter_data(data, attribute, value)
        weight = len(subset) / len(data)
        subset_targets = get_column(subset, target)
        weighted_entropy += weight * entropy(subset_targets)

    # Information gain
    ig = total_entropy - weighted_entropy
    return ig

def majority_class(data, target):
    """Return most common class in data."""
    targets = get_column(data, target)
    counts = Counter(targets)
    return counts.most_common(1)[0][0]

def id3(data, attributes, target):
    """
    Build decision tree using ID3 algorithm.

    Args:
        data: Training data (list of dictionaries)
        attributes: List of attribute names to consider
        target: Target attribute name

    Returns:
        ID3Node representing the tree
    """
    node = ID3Node()

    # Get target values
    targets = get_column(data, target)
    unique_targets = list(set(targets))

    # If all examples have same class, return leaf
    if len(unique_targets) == 1:
        node.is_leaf = True
        node.label = targets[0]
        return node

    # If no attributes left, return leaf with majority class
    if len(attributes) == 0:
        node.is_leaf = True
        node.label = majority_class(data, target)
        return node

    # Choose best attribute
    gains = {attr: information_gain(data, attr, target) for attr in attributes}
    best_attribute = max(gains, key=gains.get)

    node.attribute = best_attribute

    # Create branches for each value of best attribute
    attr_values = get_column(data, best_attribute)
    unique_values = list(set(attr_values))

    for value in unique_values:
        # Get subset where attribute = value
        subset = filter_data(data, best_attribute, value)

        if len(subset) == 0:
            # Empty subset: create leaf with majority class
            child = ID3Node()
            child.is_leaf = True
            child.label = majority_class(data, target)
            node.children[value] = child
        else:
            # Recursively build subtree
            remaining_attrs = [a for a in attributes if a != best_attribute]
            child = id3(subset, remaining_attrs, target)
            node.children[value] = child

    return node

def predict_id3(tree, sample):
    """Make prediction for a single sample (dictionary)."""
    if tree.is_leaf:
        return tree.label

    # Get value of splitting attribute for this sample
    attr_value = sample[tree.attribute]

    # Follow the branch
    if attr_value in tree.children:
        return predict_id3(tree.children[attr_value], sample)
    else:
        # Value not seen in training: return None
        return None

def print_tree(tree, indent=0, value="Root"):
    """Print tree structure."""
    prefix = "  " * indent

    if tree.is_leaf:
        print(f"{prefix}{value} -> Leaf: {tree.label}")
    else:
        print(f"{prefix}{value} -> Split on: {tree.attribute}")
        for attr_value, child in tree.children.items():
            print_tree(child, indent + 1, f"{tree.attribute}={attr_value}")

# Tennis example dataset (as list of dictionaries)
tennis_data = [
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Windy': False, 'PlayTennis': 'No'},
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Windy': True, 'PlayTennis': 'No'},
    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Windy': False, 'PlayTennis': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Mild', 'Humidity': 'High', 'Windy': False, 'PlayTennis': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Windy': False, 'PlayTennis': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Windy': True, 'PlayTennis': 'No'},
    {'Outlook': 'Overcast', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Windy': True, 'PlayTennis': 'Yes'},
    {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'High', 'Windy': False, 'PlayTennis': 'No'},
    {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'Normal', 'Windy': False, 'PlayTennis': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Windy': False, 'PlayTennis': 'Yes'},
    {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Windy': True, 'PlayTennis': 'Yes'},
    {'Outlook': 'Overcast', 'Temperature': 'Mild', 'Humidity': 'High', 'Windy': True, 'PlayTennis': 'Yes'},
    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'Normal', 'Windy': False, 'PlayTennis': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Mild', 'Humidity': 'High', 'Windy': True, 'PlayTennis': 'No'},
]

# Build tree
attributes = ['Outlook', 'Temperature', 'Humidity', 'Windy']
tree = id3(tennis_data, attributes, 'PlayTennis')

print("ID3 Decision Tree:")
print("=" * 60)
print_tree(tree)

# Calculate information gains
print("\n" + "=" * 60)
print("Information Gains:")
for attr in attributes:
    ig = information_gain(tennis_data, attr, 'PlayTennis')
    print(f"  {attr}: {ig:.4f}")

# Test predictions
print("\n" + "=" * 60)
print("Test Predictions:")

test_samples = [
    {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Windy': True},
    {'Outlook': 'Rainy', 'Temperature': 'Mild', 'Humidity': 'Normal', 'Windy': False},
    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Humidity': 'High', 'Windy': False},
]

for i, sample in enumerate(test_samples, 1):
    pred = predict_id3(tree, sample)
    print(f"Sample {i}: {sample}")
    print(f"  Prediction: {pred}\n")
```
</div>

### Limitations of ID3

While ID3 was groundbreaking, it has several limitations:

1. **Categorical features only**: Cannot handle continuous attributes directly
2. **No pruning**: Prone to overfitting
3. **No missing values**: Cannot handle missing data
4. **Bias toward high-cardinality attributes**: Attributes with many values get artificially high information gain
5. **Multi-way splits**: Can create many branches, making the tree wide and hard to interpret

These limitations led to the development of **C4.5**.

---

## C4.5: The Successor to ID3

### Overview

**C4.5** was introduced by Ross Quinlan in 1993 as an improvement over ID3. It addresses most of ID3's limitations and became one of the most influential machine learning algorithms.

### Key Improvements Over ID3

1. **Handles continuous attributes**: Discretizes continuous values
2. **Uses gain ratio**: Addresses bias toward high-cardinality attributes
3. **Pruning**: Includes post-pruning to reduce overfitting
4. **Missing values**: Can handle missing attribute values
5. **Rule generation**: Can convert trees to rule sets

### Gain Ratio: Addressing Information Gain Bias

**Problem with Information Gain:**

Information gain is biased toward attributes with many distinct values. For example, an attribute like "ID number" (unique for each instance) would have perfect information gain but zero predictive power.

**Example:**

Consider a dataset with 14 examples and an attribute "Date" with 14 unique values (one per sample). This attribute would split the data into 14 pure subsets, giving it maximum information gain, even though it's useless for prediction.

**Solution: Gain Ratio**

C4.5 uses **gain ratio** instead of raw information gain:

$$
\text{GainRatio}(S, A) = \frac{\text{IG}(S, A)}{\text{SplitInfo}(S, A)}
$$

Where **split information** measures the entropy of the split itself:

$$
\text{SplitInfo}(S, A) = -\sum_{i=1}^{v} \frac{|S_i|}{|S|} \log_2 \frac{|S_i|}{|S|}
$$

Where $v$ is the number of distinct values of attribute $A$.

**Intuition:**

- Split information is large when the split creates many subsets of similar size
- It penalizes attributes that fragment the data into many small pieces
- Gain ratio normalizes information gain by this penalty

**Example:**

For the tennis dataset:

**Outlook** (3 values):
$$
\text{SplitInfo}(\text{Outlook}) = -\left(\frac{5}{14}\log_2\frac{5}{14} + \frac{4}{14}\log_2\frac{4}{14} + \frac{5}{14}\log_2\frac{5}{14}\right) \approx 1.577
$$

$$
\text{GainRatio}(\text{Outlook}) = \frac{0.246}{1.577} \approx 0.156
$$

**Day** (14 unique values, hypothetical):
$$
\text{SplitInfo}(\text{Day}) = -\sum_{i=1}^{14} \frac{1}{14}\log_2\frac{1}{14} = \log_2(14) \approx 3.807
$$

Even if IG(Day) = 0.940 (maximum), gain ratio would be:
$$
\text{GainRatio}(\text{Day}) = \frac{0.940}{3.807} \approx 0.247
$$

This is comparable to Outlook, preventing the useless attribute from dominating.

### Handling Continuous Attributes

C4.5 handles continuous attributes through **binary discretization**:

1. **Sort** the continuous values
2. **Identify candidate thresholds**: midpoints between consecutive values with different classes
3. **Evaluate each threshold**: treat it as a binary split (≤ threshold vs > threshold)
4. **Choose threshold** with highest gain ratio

**Algorithm:**

For continuous attribute $A$ with sorted values $v_1, v_2, \ldots, v_n$:

1. For each adjacent pair $(v_i, v_{i+1})$ where the class changes:
   - Compute threshold $t = (v_i + v_{i+1}) / 2$
   - Calculate gain ratio for split: $A \leq t$ vs $A > t$

2. Select threshold with maximum gain ratio

**Example:**

Suppose we have attribute "Temperature" with continuous values and class labels:

| Temperature | Class |
|-------------|-------|
| 64          | No    |
| 65          | No    |
| 68          | Yes   |
| 69          | Yes   |
| 70          | Yes   |
| 71          | Yes   |
| 72          | No    |
| 75          | Yes   |
| 80          | No    |
| 85          | No    |

Candidate thresholds where class changes:
- $t_1 = (65 + 68) / 2 = 66.5$ (No to Yes)
- $t_2 = (71 + 72) / 2 = 71.5$ (Yes to No)
- $t_3 = (72 + 75) / 2 = 73.5$ (No to Yes)
- $t_4 = (75 + 80) / 2 = 77.5$ (Yes to No)

Evaluate gain ratio for each and select the best.

### Pruning in C4.5

C4.5 uses **post-pruning** based on **error-based pruning**:

**Algorithm:**

1. **Build full tree** using gain ratio
2. **For each non-leaf node**, calculate:
   - **Error if node becomes leaf**: misclassified examples using majority class
   - **Error if node stays internal**: sum of errors in subtrees
3. **Replace with leaf** if error doesn't increase significantly
4. Use **pessimistic error estimate** with confidence intervals

**Error Estimation:**

For a leaf with $N$ examples and $E$ errors, the pessimistic error is:

$$
\text{Error}(N, E) = \frac{E + 0.5}{N}
$$

The 0.5 is a continuity correction. This gives a more conservative estimate than the training error $E/N$.

**Pruning Decision:**

For a subtree $T$ with $L$ leaves:

- Error if kept: $\sum_{\text{leaves}} N_i \cdot \text{Error}(N_i, E_i)$
- Error if pruned: $N \cdot \text{Error}(N, E)$

Prune if the second is lower or not significantly higher.

### Handling Missing Values

C4.5 handles missing values in three ways:

**1. During Training (choosing splits):**

When calculating gain ratio for attribute $A$:
- Use only examples where $A$ is known
- Adjust weights to account for missing values

**2. During Training (creating branches):**

When an example has missing value for the chosen split attribute:
- Assign it to **multiple branches** with fractional weights
- Weight = probability of each branch based on known values

**3. During Prediction:**

For a test example with missing value:
- Follow all branches
- Weight each prediction by branch probability
- Return weighted majority vote

### Interactive Code: C4.5 Concepts

<div class="python-interactive" markdown="1">
```python
import numpy as np
from collections import Counter

def split_info(data, attribute):
    """
    Calculate split information for an attribute.

    Split info measures how evenly the split divides the data.
    High split info = many small subsets (penalized)
    """
    attr_values = get_column(data, attribute)
    counts = Counter(attr_values)
    total = len(data)

    proportions = np.array([count / total for count in counts.values()])

    # SplitInfo = -sum(p_i * log2(p_i))
    split_info_val = -np.sum(proportions * np.log2(proportions + 1e-10))
    return split_info_val

def gain_ratio(data, attribute, target):
    """
    Calculate gain ratio for an attribute.

    GainRatio = InformationGain / SplitInfo
    """
    ig = information_gain(data, attribute, target)
    si = split_info(data, attribute)

    if si == 0:
        return 0

    gr = ig / si
    return gr

def discretize_continuous(data, attribute, target):
    """
    Find best binary split for a continuous attribute.

    Returns:
        best_threshold, best_gain_ratio
    """
    # Sort by attribute value
    sorted_data = sorted(data, key=lambda x: x[attribute])
    values = [row[attribute] for row in sorted_data]
    labels = [row[target] for row in sorted_data]

    best_threshold = None
    best_gr = -np.inf

    # Try thresholds at midpoints where class changes
    for i in range(len(values) - 1):
        # Only consider points where class changes
        if labels[i] != labels[i+1]:
            threshold = (values[i] + values[i+1]) / 2

            # Create binary split
            data_with_split = []
            for row in data:
                new_row = row.copy()
                new_row['_binary_split'] = row[attribute] <= threshold
                data_with_split.append(new_row)

            # Calculate gain ratio for this split
            gr = gain_ratio(data_with_split, '_binary_split', target)

            if gr > best_gr:
                best_gr = gr
                best_threshold = threshold

    return best_threshold, best_gr

# Example: Compare information gain vs gain ratio
tennis_data = [
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Day': 1, 'PlayTennis': 'No'},
    {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Day': 2, 'PlayTennis': 'No'},
    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Day': 3, 'PlayTennis': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Mild', 'Day': 4, 'PlayTennis': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Cool', 'Day': 5, 'PlayTennis': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Cool', 'Day': 6, 'PlayTennis': 'No'},
    {'Outlook': 'Overcast', 'Temperature': 'Cool', 'Day': 7, 'PlayTennis': 'Yes'},
    {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Day': 8, 'PlayTennis': 'No'},
    {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Day': 9, 'PlayTennis': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Mild', 'Day': 10, 'PlayTennis': 'Yes'},
    {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Day': 11, 'PlayTennis': 'Yes'},
    {'Outlook': 'Overcast', 'Temperature': 'Mild', 'Day': 12, 'PlayTennis': 'Yes'},
    {'Outlook': 'Overcast', 'Temperature': 'Hot', 'Day': 13, 'PlayTennis': 'Yes'},
    {'Outlook': 'Rainy', 'Temperature': 'Mild', 'Day': 14, 'PlayTennis': 'No'},
]

print("Comparing Information Gain vs Gain Ratio:")
print("=" * 70)

attributes = ['Outlook', 'Temperature', 'Day']
for attr in attributes:
    ig = information_gain(tennis_data, attr, 'PlayTennis')
    si = split_info(tennis_data, attr)
    gr = gain_ratio(tennis_data, attr, 'PlayTennis')

    print(f"\n{attr}:")
    print(f"  Information Gain:  {ig:.4f}")
    print(f"  Split Info:        {si:.4f}")
    print(f"  Gain Ratio:        {gr:.4f}")

print("\n" + "=" * 70)
print("\nNotice how 'Day' (unique values) has:")
print("  - High information gain (would be chosen by ID3)")
print("  - High split info (penalty for fragmenting data)")
print("  - Lower gain ratio (C4.5 avoids this attribute)")

# Example: Continuous attribute handling
continuous_data = [
    {'Temperature': 64, 'PlayTennis': 'No'},
    {'Temperature': 65, 'PlayTennis': 'No'},
    {'Temperature': 68, 'PlayTennis': 'Yes'},
    {'Temperature': 69, 'PlayTennis': 'Yes'},
    {'Temperature': 70, 'PlayTennis': 'Yes'},
    {'Temperature': 71, 'PlayTennis': 'Yes'},
    {'Temperature': 72, 'PlayTennis': 'No'},
    {'Temperature': 75, 'PlayTennis': 'Yes'},
    {'Temperature': 80, 'PlayTennis': 'No'},
    {'Temperature': 85, 'PlayTennis': 'No'},
]

threshold, gr = discretize_continuous(continuous_data, 'Temperature', 'PlayTennis')
print(f"\nBest threshold for Temperature: {threshold:.1f}")
print(f"Gain ratio: {gr:.4f}")
print(f"Split: Temperature ≤ {threshold:.1f} vs Temperature > {threshold:.1f}")
```
</div>

---

## CART: Classification and Regression Trees

### Overview

**CART** was developed by Breiman, Friedman, Olshen, and Stone in 1984. Unlike ID3/C4.5 which focus on classification, CART handles both **classification** and **regression** problems.

### Key Characteristics

- **Splitting criterion**:
  - Classification: **Gini impurity**
  - Regression: **Variance reduction** (MSE)
- **Tree structure**: **Binary splits** only
- **Pruning**: **Cost-complexity pruning** with cross-validation
- **Feature types**: Handles both continuous and categorical
- **Missing values**: Surrogate splits

### Why Gini Instead of Entropy?

CART uses **Gini impurity** instead of entropy for several reasons:

**1. Computational Efficiency:**

$$
\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2 \quad \text{(no logarithms)}
$$

$$
\text{Entropy}(S) = -\sum_{i=1}^{c} p_i \log_2(p_i) \quad \text{(requires logarithms)}
$$

**2. Similar Results:**

For binary classification, Gini and entropy are highly correlated:

| $p$ (proportion of positive class) | Gini | Entropy |
|-------------------------------------|------|---------|
| 0.0                                 | 0.00 | 0.00    |
| 0.1                                 | 0.18 | 0.47    |
| 0.3                                 | 0.42 | 0.88    |
| 0.5                                 | 0.50 | 1.00    |
| 0.7                                 | 0.42 | 0.88    |
| 0.9                                 | 0.18 | 0.47    |
| 1.0                                 | 0.00 | 0.00    |

Both are maximized at $p = 0.5$ and minimized at $p = 0$ or $p = 1$.

**3. Slightly Different Behavior:**

- **Gini** tends to isolate the most frequent class in its own branch
- **Entropy** tends to create more balanced trees
- In practice, the difference is usually negligible

### Binary Splits

CART always creates **binary splits**, even for categorical variables:

**For continuous variables:**
- Find threshold $t$: $X \leq t$ vs $X > t$

**For categorical variables:**
- Group categories into two sets: $X \in \{A, B\}$ vs $X \in \{C, D, E\}$

**Example:**

For "Color" ∈ {Red, Blue, Green, Yellow}:
- Possible splits: {Red} vs {Blue, Green, Yellow}
- {Red, Blue} vs {Green, Yellow}
- {Red, Green} vs {Blue, Yellow}
- etc.

For $k$ categories, there are $2^{k-1} - 1$ possible splits.

**Optimization for Binary Classification:**

For binary classification, we can order categories by proportion of positive class, reducing complexity from $O(2^k)$ to $O(k \log k)$.

### Regression with CART

For **regression**, CART minimizes **mean squared error (MSE)**:

$$
\text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y})^2
$$

Where $\bar{y} = \frac{1}{|S|}\sum_{i \in S} y_i$ is the mean of the target variable in set $S$.

**Splitting Criterion:**

The **variance reduction** from a split is:

$$
\Delta\text{MSE} = \text{MSE}(S) - \left(\frac{|S_L|}{|S|}\text{MSE}(S_L) + \frac{|S_R|}{|S|}\text{MSE}(S_R)\right)
$$

Choose the split that maximizes $\Delta\text{MSE}$.

**Prediction:**

For a leaf node, the prediction is the **mean** of all target values in that leaf:

$$
\hat{y} = \frac{1}{|S_{\text{leaf}}|} \sum_{i \in S_{\text{leaf}}} y_i
$$

### Cost-Complexity Pruning

CART uses **cost-complexity pruning** (also called **weakest link pruning**):

**Idea:**

Find the right balance between tree size and fit to data using a complexity parameter $\alpha$.

**Cost-Complexity Measure:**

For a tree $T$, the cost-complexity is:

$$
R_\alpha(T) = R(T) + \alpha |T|
$$

Where:
- $R(T)$ is the **resubstitution error** (misclassification rate on training data)
- $|T|$ is the **number of leaf nodes**
- $\alpha \geq 0$ is the **complexity parameter**

**Algorithm:**

1. **Build full tree** $T_0$
2. **For increasing values of** $\alpha$:
   - Find subtree $T(\alpha)$ that minimizes $R_\alpha(T)$
   - This creates a sequence: $T_0 \supset T_1 \supset \cdots \supset T_n$ (root only)
3. **Use cross-validation** to select best $\alpha$
4. **Return tree** $T(\alpha^*)$ with optimal $\alpha^*$

**How to Find** $T(\alpha)$:

For each internal node $t$ in tree $T$:
1. Calculate $g(t)$, the **per-leaf increase in error** from pruning node $t$:

$$
g(t) = \frac{R(t) - R(T_t)}{|T_t| - 1}
$$

Where:
- $R(t)$ = error if node $t$ becomes a leaf
- $R(T_t)$ = error of subtree rooted at $t$
- $|T_t|$ = number of leaves in subtree

2. Prune the node with smallest $g(t)$
3. Repeat until only root remains

**Why This Works:**

- Small $\alpha$ (≈ 0): favors large trees (low training error)
- Large $\alpha$: favors small trees (high regularization)
- Cross-validation finds the $\alpha$ that generalizes best

### Interactive Code: CART Concepts

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Example 1: Gini vs Entropy comparison
def gini_impurity(p):
    """Gini impurity as a function of positive class proportion."""
    return 2 * p * (1 - p)

def entropy_func(p):
    """Entropy as a function of positive class proportion."""
    p = np.clip(p, 1e-10, 1 - 1e-10)  # Avoid log(0)
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

# Plot comparison
p_values = np.linspace(0, 1, 100)
gini_values = [gini_impurity(p) for p in p_values]
entropy_values = [entropy_func(p) for p in p_values]

plt.figure(figsize=(10, 6))
plt.plot(p_values, gini_values, label='Gini Impurity', linewidth=2)
plt.plot(p_values, entropy_values, label='Entropy', linewidth=2)
plt.xlabel('Proportion of Positive Class (p)')
plt.ylabel('Impurity')
plt.title('Gini Impurity vs Entropy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Both measures:")
print("  - Are maximized at p = 0.5 (maximum uncertainty)")
print("  - Are minimized at p = 0 or p = 1 (pure nodes)")
print("  - Have similar shapes (highly correlated)")

# Example 2: Regression Tree
diabetes = load_diabetes()
X = diabetes.data[:, [2]]  # Use only one feature for visualization
y = diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train trees with different depths
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
depths = [2, 5, 10]

for idx, depth in enumerate(depths):
    # Train model
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    # Predictions
    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_plot = tree.predict(X_plot)

    # Plot
    axes[idx].scatter(X_train, y_train, alpha=0.3, s=20, label='Training data')
    axes[idx].plot(X_plot, y_plot, 'r-', linewidth=2, label='Tree prediction')
    axes[idx].set_xlabel('Feature')
    axes[idx].set_ylabel('Target')
    axes[idx].set_title(f'Regression Tree (depth={depth})')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

    # Score
    train_score = tree.score(X_train, y_train)
    test_score = tree.score(X_test, y_test)
    axes[idx].text(0.05, 0.95, f'Train R²: {train_score:.3f}\nTest R²: {test_score:.3f}',
                   transform=axes[idx].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

print("\nNotice how regression trees create piecewise constant predictions:")
print("  - Depth 2: Very simple, high bias")
print("  - Depth 5: Reasonable fit")
print("  - Depth 10: Overfitting, memorizing noise")

# Example 3: Cost-complexity pruning
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build full tree
tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(X_train, y_train)

# Get pruning path
path = tree_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

print("\n" + "=" * 60)
print("Cost-Complexity Pruning:")
print("=" * 60)
print(f"Number of α values: {len(ccp_alphas)}")
print(f"α range: [{ccp_alphas[0]:.6f}, {ccp_alphas[-1]:.6f}]")

# Train trees for different alphas
trees = []
train_scores = []
test_scores = []

for alpha in ccp_alphas[:-1]:  # Exclude last (only root)
    tree = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
    tree.fit(X_train, y_train)
    trees.append(tree)
    train_scores.append(tree.score(X_train, y_train))
    test_scores.append(tree.score(X_test, y_test))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Tree size vs alpha
node_counts = [tree.tree_.node_count for tree in trees]
axes[0].plot(ccp_alphas[:-1], node_counts, marker='o')
axes[0].set_xlabel('α (complexity parameter)')
axes[0].set_ylabel('Number of nodes')
axes[0].set_title('Tree Size vs α')
axes[0].grid(True, alpha=0.3)

# Accuracy vs alpha
axes[1].plot(ccp_alphas[:-1], train_scores, marker='o', label='Training')
axes[1].plot(ccp_alphas[:-1], test_scores, marker='s', label='Testing')
axes[1].set_xlabel('α (complexity parameter)')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy vs α')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find best alpha
best_idx = np.argmax(test_scores)
best_alpha = ccp_alphas[best_idx]
best_test_score = test_scores[best_idx]

print(f"\nOptimal α: {best_alpha:.6f}")
print(f"Best test accuracy: {best_test_score:.4f}")
print(f"Number of nodes: {node_counts[best_idx]}")
```
</div>

---

## Comparison: ID3 vs C4.5 vs CART

### Summary Table

| Feature | ID3 | C4.5 | CART |
|---------|-----|------|------|
| **Splitting Criterion** | Information Gain (Entropy) | Gain Ratio | Gini (classification)<br>MSE (regression) |
| **Continuous Attributes** | ❌ No | ✅ Yes (binary discretization) | ✅ Yes |
| **Categorical Attributes** | ✅ Yes (multi-way) | ✅ Yes (multi-way) | ✅ Yes (binary) |
| **Split Type** | Multi-way | Multi-way | Binary only |
| **Regression** | ❌ No | ❌ No | ✅ Yes |
| **Pruning** | ❌ None | ✅ Error-based | ✅ Cost-complexity |
| **Missing Values** | ❌ No | ✅ Fractional weights | ✅ Surrogate splits |
| **Overfitting Control** | Weak | Good | Excellent |
| **Computational Cost** | Low | Medium | Medium-High |
| **Output** | Tree only | Tree + rules | Tree only |
| **Implementation** | Rare | Rare (patented until 2006) | sklearn (default) |

### When to Use Each

**ID3:**
- Historical interest / educational purposes
- Simple categorical classification problems
- When interpretability is paramount and data is clean

**C4.5:**
- When you need rule generation
- Mixed continuous/categorical data
- When gain ratio is theoretically important
- **Note**: Use sklearn's implementation or J48 (Weka's C4.5 port)

**CART:**
- **Most practical choice** (used by sklearn)
- Regression problems
- When binary splits are preferred
- Need robust pruning
- Production systems

!!! tip "In Practice"
    **Use sklearn's DecisionTreeClassifier/Regressor**, which implements an optimized version of CART:

    ```python
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    # Classification (uses Gini by default)
    clf = DecisionTreeClassifier(max_depth=5)

    # Can also use entropy (like ID3/C4.5)
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)

    # Regression (uses MSE)
    reg = DecisionTreeRegressor(max_depth=5)
    ```

---

## Advanced Topics

### Surrogate Splits (CART)

When the primary split variable has missing values, CART uses **surrogate splits**:

**Algorithm:**

1. Find best split on attribute $A$ (primary split)
2. For each other attribute $B$:
   - Find best split on $B$ that **mimics** the primary split
   - Measure agreement with primary split
3. Order attributes by agreement (surrogates)
4. When predicting with missing $A$, use best available surrogate

**Example:**

Primary: "Income > $50K" → Left: 70, Right: 30

Surrogates:
1. "Education = Bachelor+" → Agrees 85% (best)
2. "Age > 35" → Agrees 75%
3. "Hours/week > 40" → Agrees 65%

If Income is missing, use Education; if that's also missing, use Age, etc.

### Multi-Output Trees

Decision trees can handle **multiple targets** simultaneously:

**Multi-Output Classification:**
- Each leaf predicts a **vector of class labels**
- Example: Predict both "will buy" and "will churn"

**Multi-Output Regression:**
- Each leaf predicts a **vector of continuous values**
- Example: Predict (x, y) coordinates

**Implementation:**

sklearn supports this natively:

```python
# Multi-output regression
tree = DecisionTreeRegressor()
tree.fit(X, Y)  # Y is (n_samples, n_outputs)
predictions = tree.predict(X_new)  # Returns (n_samples, n_outputs)
```

### Extremely Randomized Trees

**Extra-Trees** (Extremely Randomized Trees) introduce additional randomness:

**Differences from standard trees:**
1. **Thresholds are random**: Instead of finding optimal threshold, sample it randomly
2. **Faster training**: No need to search for best threshold
3. **More variance reduction**: Random splits decorrelate trees

**When to use:**
- As base learners in ensemble methods
- When training time is critical
- With very high-dimensional data

---

## Practical Implementation

### Complete Example: Building and Comparing Trees

<div class="python-interactive" markdown="1">
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
wine = load_wine()
X, y = wine.data, wine.target
feature_names = wine.feature_names
target_names = wine.target_names

print("Wine Dataset:")
print(f"  Samples: {X.shape[0]}")
print(f"  Features: {X.shape[1]}")
print(f"  Classes: {len(target_names)} {target_names}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Compare different configurations
configs = [
    {'name': 'CART (Gini)', 'criterion': 'gini', 'splitter': 'best'},
    {'name': 'Entropy (C4.5-like)', 'criterion': 'entropy', 'splitter': 'best'},
    {'name': 'Extra Trees', 'criterion': 'gini', 'splitter': 'random'},
]

results = []

for config in configs:
    # Train model
    model = DecisionTreeClassifier(
        criterion=config['criterion'],
        splitter=config['splitter'],
        max_depth=5,
        min_samples_split=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    results.append({
        'name': config['name'],
        'train_acc': train_score,
        'test_acc': test_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_nodes': model.tree_.node_count,
        'n_leaves': model.tree_.n_leaves
    })

    print(f"\n{config['name']}:")
    print(f"  Train accuracy: {train_score:.4f}")
    print(f"  Test accuracy:  {test_score:.4f}")
    print(f"  CV accuracy:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Nodes: {model.tree_.node_count}, Leaves: {model.tree_.n_leaves}")

# Visualize best model (Gini)
best_model = DecisionTreeClassifier(
    criterion='gini', max_depth=3, min_samples_split=10, random_state=42
)
best_model.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(best_model,
          feature_names=feature_names,
          class_names=target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Wine Classification Tree (CART with Gini, depth=3)")
plt.tight_layout()
plt.show()

# Feature importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)),
           [feature_names[i] for i in indices],
           rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importances (CART with Gini)')
plt.tight_layout()
plt.show()

print("\nTop 5 Most Important Features:")
for i in range(min(5, len(importances))):
    print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Confusion matrix
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Add text annotations
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

plt.xticks(range(len(target_names)), target_names)
plt.yticks(range(len(target_names)), target_names)
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
```
</div>

---

## Summary

In this lesson, you learned:

✅ **ID3 Algorithm**: Information gain, multi-way splits, categorical features only
✅ **C4.5 Algorithm**: Gain ratio, continuous attributes, pruning, missing values
✅ **CART Algorithm**: Gini impurity, binary splits, regression support, cost-complexity pruning
✅ **Key Differences**: Splitting criteria, feature handling, pruning methods
✅ **Practical Considerations**: When to use each, sklearn implementation
✅ **Advanced Topics**: Surrogate splits, multi-output trees, extra-trees

### Key Takeaways

1. **ID3** introduced information gain but has significant limitations
2. **C4.5** improved ID3 with gain ratio, continuous attributes, and pruning
3. **CART** is the most widely used (sklearn's implementation)
4. **Gini vs Entropy**: Similar results, Gini is faster
5. **Binary vs Multi-way**: CART's binary splits are simpler and often better
6. **Pruning is essential**: Cost-complexity pruning (CART) is most principled

### Next Steps

Now that you understand individual tree algorithms, you're ready to learn how combining multiple trees leads to dramatic improvements:

[Continue to Lesson 3: Random Forest](03-random-forest.md){ .md-button .md-button--primary }

---

## Additional Resources

### Further Reading

- **Original Papers**:
  - Quinlan, J. R. (1986). "Induction of Decision Trees"
  - Quinlan, J. R. (1993). "C4.5: Programs for Machine Learning"
  - Breiman et al. (1984). "Classification and Regression Trees"

- **Books**:
  - Hastie, Tibshirani, Friedman - *Elements of Statistical Learning* (Chapter 9)
  - Witten, Frank, Hall - *Data Mining: Practical Machine Learning Tools*

### Implementation Notes

**sklearn's DecisionTreeClassifier:**
- Implements an optimized version of CART
- Can use either 'gini' or 'entropy'
- Supports both classification and regression
- Includes cost-complexity pruning via `ccp_alpha`

**Other Implementations:**
- **Weka**: J48 (C4.5 implementation)
- **R**: rpart (CART), C50 (C4.5)
- **XGBoost/LightGBM**: Use tree algorithms as base learners

---

**Questions or feedback?** Open an issue on [GitHub](https://github.com/jumpingsphinx/jumpingsphinx.github.io/issues) or contribute improvements!
