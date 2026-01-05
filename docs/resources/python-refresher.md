# Python Refresher

Quick review of Python concepts needed for ML101.

## Python Basics

### Variables and Types

```python
# Numbers
x = 5           # int
y = 3.14        # float

# Strings
name = "ML101"
message = 'Hello'

# Booleans
is_learning = True
is_difficult = False

# Lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", True, 3.14]

# Dictionaries
person = {"name": "Alice", "age": 25}

# Check type
print(type(x))  # <class 'int'>
```

### Functions

```python
def add(a, b):
    """Add two numbers."""
    return a + b

result = add(3, 5)  # 8

# Default arguments
def greet(name="World"):
    return f"Hello, {name}!"

# Lambda functions
square = lambda x: x ** 2
```

### Control Flow

```python
# If statements
if x > 0:
    print("Positive")
elif x == 0:
    print("Zero")
else:
    print("Negative")

# For loops
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

for item in [1, 2, 3]:
    print(item)

# While loops
count = 0
while count < 5:
    print(count)
    count += 1

# List comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(10) if x % 2 == 0]
```

## NumPy Essentials

### Creating Arrays

```python
import numpy as np

# From list
a = np.array([1, 2, 3, 4])

# Zeros and ones
zeros = np.zeros(5)
ones = np.ones((3, 4))

# Range
r = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Linspace
l = np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1]

# Random
rand = np.random.rand(3, 3)
randn = np.random.randn(3, 3)  # Standard normal
```

### Array Operations

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Element-wise operations
c = a + b      # [5, 7, 9]
c = a * 2      # [2, 4, 6]
c = a * b      # [4, 10, 18]
c = a ** 2     # [1, 4, 9]

# Mathematical functions
np.exp(a)      # Exponential
np.log(a)      # Natural log
np.sqrt(a)     # Square root
np.sin(a)      # Sine

# Aggregations
np.sum(a)      # 6
np.mean(a)     # 2.0
np.max(a)      # 3
np.min(a)      # 1
```

### Indexing and Slicing

```python
a = np.array([1, 2, 3, 4, 5])

# Indexing
print(a[0])     # 1
print(a[-1])    # 5

# Slicing
print(a[1:4])   # [2, 3, 4]
print(a[:3])    # [1, 2, 3]
print(a[2:])    # [3, 4, 5]

# 2D arrays
A = np.array([[1, 2, 3],
              [4, 5, 6]])

print(A[0, 0])  # 1
print(A[1, 2])  # 6
print(A[0, :])  # [1, 2, 3] (first row)
print(A[:, 1])  # [2, 5] (second column)
```

### Broadcasting

```python
# Add scalar to array
a = np.array([1, 2, 3])
b = a + 10  # [11, 12, 13]

# Add arrays of different shapes
A = np.array([[1, 2, 3],
              [4, 5, 6]])
v = np.array([10, 20, 30])

B = A + v  # [[11, 22, 33],
           #  [14, 25, 36]]
```

## Matplotlib Basics

### Line Plots

```python
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.show()
```

### Scatter Plots

```python
x = np.random.rand(50)
y = np.random.rand(50)

plt.scatter(x, y, c='blue', alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Scatter')
plt.show()
```

### Multiple Subplots

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x, y)
axes[0].set_title('Plot 1')

axes[1].scatter(x, y)
axes[1].set_title('Plot 2')

plt.tight_layout()
plt.show()
```

## Pandas Basics

### DataFrames

```python
import pandas as pd

# Create DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['NY', 'LA', 'SF']
}
df = pd.DataFrame(data)

# View data
print(df.head())
print(df.info())
print(df.describe())

# Indexing
print(df['Name'])        # Column
print(df.loc[0])         # Row by index
print(df.iloc[0])        # Row by position

# Filtering
adults = df[df['Age'] >= 30]

# Adding columns
df['Age_Plus_10'] = df['Age'] + 10
```

### Reading Data

```python
# CSV
df = pd.read_csv('data.csv')

# Excel
df = pd.read_excel('data.xlsx')

# Save
df.to_csv('output.csv', index=False)
```

## Python Tips for ML

### Print Debugging

```python
# Always check shapes!
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Print intermediate values
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.2%}")
```

### Assertions

```python
# Verify assumptions
assert X.shape[0] == y.shape[0], "Number of samples must match"
assert X.ndim == 2, "X must be 2D"
```

### F-strings

```python
name = "Alice"
age = 25

# Old way
print("Name: " + name + ", Age: " + str(age))

# F-string (Python 3.6+)
print(f"Name: {name}, Age: {age}")

# Formatting
pi = 3.14159
print(f"Pi: {pi:.2f}")  # Pi: 3.14
```

### List vs NumPy Array

```python
# Lists: slow, flexible
list1 = [1, 2, 3]
list2 = [4, 5, 6]
# list1 + list2 = [1, 2, 3, 4, 5, 6]  # Concatenation!

# Arrays: fast, vectorized
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
# arr1 + arr2 = [5, 7, 9]  # Element-wise!
```

### Common Mistakes

```python
# ❌ Wrong: Modifying during iteration
for i in range(len(list)):
    list.pop(i)  # Don't do this!

# ✅ Right: Create new list
new_list = [x for x in list if condition]

# ❌ Wrong: Comparing floats
if 0.1 + 0.2 == 0.3:  # False!

# ✅ Right: Use tolerance
if abs((0.1 + 0.2) - 0.3) < 1e-10:

# ❌ Wrong: Mutable default argument
def append_to(element, to=[]):
    to.append(element)
    return to

# ✅ Right: Use None
def append_to(element, to=None):
    if to is None:
        to = []
    to.append(element)
    return to
```

## Virtual Environments

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install packages
pip install numpy pandas matplotlib

# Save requirements
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Deactivate
deactivate
```

## Jupyter Shortcuts

| Action | Shortcut |
|--------|----------|
| Run cell | `Shift + Enter` |
| Run in place | `Ctrl + Enter` |
| Insert above | `A` (command mode) |
| Insert below | `B` (command mode) |
| Delete cell | `DD` (command mode) |
| To markdown | `M` (command mode) |
| To code | `Y` (command mode) |
| Command mode | `Esc` |
| Edit mode | `Enter` |

## Resources

- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## Practice

Before starting Module 1, make sure you can:

- [ ] Create and manipulate NumPy arrays
- [ ] Perform basic array operations
- [ ] Index and slice arrays
- [ ] Plot simple graphs with matplotlib
- [ ] Write and call functions
- [ ] Use list comprehensions

[Back to Home](../index.md){ .md-button }
