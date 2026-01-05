# Datasets

This page documents all datasets used in ML101 and how to access them.

## Built-in Datasets

All datasets are available through scikit-learn or built-in libraries - no downloads required!

### Classification Datasets

#### Iris Dataset
**Module:** 1, 2
**Task:** Multi-class classification (3 classes)
**Samples:** 150
**Features:** 4 (sepal length, sepal width, petal length, petal width)

```python
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
```

#### Breast Cancer Wisconsin
**Module:** 2
**Task:** Binary classification
**Samples:** 569
**Features:** 30 (computed from digitized images)

```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
```

#### Digits Dataset
**Module:** 1, 4
**Task:** Digit recognition (10 classes)
**Samples:** 1,797
**Features:** 64 (8x8 pixel images)

```python
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target
```

### Regression Datasets

#### California Housing
**Module:** 2, 3
**Task:** Regression (predict median house value)
**Samples:** 20,640
**Features:** 8

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X, y = housing.data, housing.target
```

#### Boston Housing (Deprecated but Educational)
**Module:** 2
**Task:** Regression (predict house prices)
**Samples:** 506
**Features:** 13

```python
# Note: Deprecated in sklearn, included for learning
from sklearn.datasets import load_boston
boston = load_boston()
```

### Deep Learning Datasets

#### MNIST
**Module:** 4
**Task:** Handwritten digit recognition
**Samples:** 70,000 (60k train, 10k test)
**Features:** 28x28 grayscale images

```python
from torchvision import datasets
mnist = datasets.MNIST(root='./data', download=True)
```

#### Fashion-MNIST
**Module:** 4
**Task:** Clothing classification
**Samples:** 70,000
**Features:** 28x28 grayscale images (10 classes)

```python
from torchvision import datasets
fashion = datasets.FashionMNIST(root='./data', download=True)
```

#### CIFAR-10
**Module:** 4
**Task:** Image classification
**Samples:** 60,000
**Features:** 32x32 color images (10 classes)

```python
from torchvision import datasets
cifar10 = datasets.CIFAR10(root='./data', download=True)
```

## Generating Synthetic Data

### Linear Regression Data

```python
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=20,
    random_state=42
)
```

### Classification Data

```python
from sklearn.datasets import make_classification, make_moons, make_circles

# Linearly separable
X, y = make_classification(n_samples=100, n_features=2,
                           n_redundant=0, n_clusters_per_class=1)

# Non-linear (moons)
X, y = make_moons(n_samples=100, noise=0.1)

# Non-linear (circles)
X, y = make_circles(n_samples=100, noise=0.05, factor=0.5)
```

### Clustering Data

```python
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=3, n_features=2)
```

## Dataset Information

### Accessing Metadata

```python
# Get feature names
print(iris.feature_names)

# Get target names
print(iris.target_names)

# Get description
print(iris.DESCR)
```

### Dataset Properties

```python
import numpy as np

print(f"Shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")
```

## Data Preprocessing

### Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Normalization

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

## External Datasets (Optional)

For advanced projects beyond the course:

- **Kaggle**: [kaggle.com/datasets](https://www.kaggle.com/datasets)
- **UCI ML Repository**: [archive.ics.uci.edu/ml](https://archive.ics.uci.edu/ml)
- **HuggingFace Datasets**: [huggingface.co/datasets](https://huggingface.co/datasets)
- **TensorFlow Datasets**: [tensorflow.org/datasets](https://www.tensorflow.org/datasets)

## Tips

!!! tip "Always Explore First"
    Before modeling, always:
    - Check dataset shape and types
    - Look for missing values
    - Visualize distributions
    - Check for class imbalance

!!! warning "Reproducibility"
    Use `random_state` parameter for reproducible results:
    ```python
    train_test_split(X, y, random_state=42)
    ```

[Back to Home](../index.md){ .md-button }
